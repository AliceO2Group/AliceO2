// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file standalone.cxx
/// \author David Rohr

#include "utils/qconfig.h"
#include "GPUReconstruction.h"
#include "GPUReconstructionTimeframe.h"
#include "GPUReconstructionConvert.h"
#include "GPUChainTracking.h"
#include "GPUChainTrackingGetters.inc"
#include "GPUTPCDef.h"
#include "GPUQA.h"
#include "GPUParam.h"
#include "display/GPUDisplayInterface.h"
#include "genEvents.h"

#include "TPCFastTransform.h"
#include "CorrectionMapsHelper.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUSettings.h"
#include "GPUConstantMem.h"

#include "GPUO2DataTypes.h"
#include "GPUChainITS.h"

#include "DataFormatsTPC/CompressedClusters.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <tuple>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <cfenv>
#include <clocale>
#include <sys/stat.h>
#endif
#include "utils/timer.h"
#include "utils/qmaths_helpers.h"
#include "utils/vecpod.h"

using namespace o2::gpu;

// #define BROKEN_EVENTS

namespace o2::gpu
{
extern GPUSettingsStandalone configStandalone;
}

GPUReconstruction *rec, *recAsync, *recPipeline;
uint32_t syncAsyncDecodedClusters = 0;
GPUChainTracking *chainTracking, *chainTrackingAsync, *chainTrackingPipeline;
GPUChainITS *chainITS, *chainITSAsync, *chainITSPipeline;
std::string eventsDir;
void unique_ptr_aligned_delete(char* v)
{
  operator delete(v, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
}
std::unique_ptr<char, void (*)(char*)> outputmemory(nullptr, unique_ptr_aligned_delete), outputmemoryPipeline(nullptr, unique_ptr_aligned_delete), inputmemory(nullptr, unique_ptr_aligned_delete);
std::unique_ptr<GPUDisplayFrontendInterface> eventDisplay;
std::unique_ptr<GPUReconstructionTimeframe> tf;
int32_t nEventsInDirectory = 0;
std::atomic<uint32_t> nIteration, nIterationEnd;

std::vector<GPUTrackingInOutPointers> ioPtrEvents;
std::vector<GPUChainTracking::InOutMemory> ioMemEvents;

int32_t ReadConfiguration(int argc, char** argv)
{
  int32_t qcRet = qConfigParse(argc, (const char**)argv);
  if (qcRet) {
    if (qcRet != qConfig::qcrHelp) {
      printf("Error parsing command line parameters\n");
    }
    return 1;
  }
  if (configStandalone.printSettings > 1) {
    printf("Config Dump before ReadConfiguration\n");
    qConfigPrint();
  }
  if (configStandalone.proc.debugLevel == -1) {
    configStandalone.proc.debugLevel = 0;
  }
#ifndef _WIN32
  setlocale(LC_ALL, "en_US.utf-8");
  setlocale(LC_NUMERIC, "en_US.utf-8");
  if (configStandalone.cpuAffinity != -1) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(configStandalone.cpuAffinity, &mask);

    printf("Setting affinitiy to restrict on CPU core %d\n", configStandalone.cpuAffinity);
    if (0 != sched_setaffinity(0, sizeof(mask), &mask)) {
      printf("Error setting CPU affinity\n");
      return 1;
    }
  }
  if (configStandalone.fifoScheduler) {
    printf("Setting FIFO scheduler\n");
    sched_param param;
    sched_getparam(0, &param);
    param.sched_priority = 1;
    if (0 != sched_setscheduler(0, SCHED_FIFO, &param)) {
      printf("Error setting scheduler\n");
      return 1;
    }
  }
#ifdef __FAST_MATH__
  if (configStandalone.fpe == 1) {
#else
  if (configStandalone.fpe) {
#endif
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  }
  if (configStandalone.flushDenormals) {
    disable_denormals();
  }

#else
  if (configStandalone.cpuAffinity != -1) {
    printf("Affinity setting not supported on Windows\n");
    return 1;
  }
  if (configStandalone.fifoScheduler) {
    printf("FIFO Scheduler setting not supported on Windows\n");
    return 1;
  }
  if (configStandalone.fpe == 1) {
    printf("FPE not supported on Windows\n");
    return 1;
  }
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#error Why was configStandalone.rec.tpc.mergerReadFromTrackerDirectly = 0 needed?
  configStandalone.proc.inKernelParallel = false;
  configStandalone.proc.createO2Output = 0;
  if (configStandalone.rundEdx == -1) {
    configStandalone.rundEdx = 0;
  }
#endif
#ifndef GPUCA_BUILD_QA
  if (configStandalone.proc.runQA || configStandalone.eventGenerator) {
    printf("QA not enabled in build\n");
    return 1;
  }
#endif
  if (configStandalone.proc.doublePipeline && configStandalone.testSyncAsync) {
    printf("Cannot run asynchronous processing with double pipeline\n");
    return 1;
  }
  if (configStandalone.proc.doublePipeline && (configStandalone.runs < 4 || !configStandalone.outputcontrolmem)) {
    printf("Double pipeline mode needs at least 4 runs per event and external output. To cycle though multiple events, use --preloadEvents and --runs n for n iterations round-robin\n");
    return 1;
  }
  if (configStandalone.TF.bunchSim && configStandalone.TF.nMerge) {
    printf("Cannot run --MERGE and --SIMBUNCHES togeterh\n");
    return 1;
  }
  if (configStandalone.TF.bunchSim > 1) {
    configStandalone.TF.timeFrameLen = 1.e9 * configStandalone.TF.bunchSim / configStandalone.TF.interactionRate;
  }
  if (configStandalone.TF.nMerge) {
    double len = configStandalone.TF.nMerge - 1;
    if (configStandalone.TF.randomizeDistance) {
      len += 0.5;
    }
    if (configStandalone.TF.shiftFirstEvent) {
      len += 0.5;
    }
    configStandalone.TF.timeFrameLen = (len * configStandalone.TF.averageDistance / GPUReconstructionTimeframe::TPCZ + 1) * GPUReconstructionTimeframe::DRIFT_TIME;
  }
  if (configStandalone.QA.inputHistogramsOnly && configStandalone.QA.compareInputs.size() == 0) {
    printf("Can only produce QA pdf output when input files are specified!\n");
    return 1;
  }
  if (configStandalone.QA.enableLocalOutput && !configStandalone.QA.inputHistogramsOnly && configStandalone.QA.output == "" && configStandalone.QA.plotsDir != "") {
    configStandalone.QA.output = configStandalone.QA.plotsDir + "/output.root";
  }
  if (configStandalone.QA.inputHistogramsOnly) {
    configStandalone.rundEdx = false;
    configStandalone.noEvents = true;
  }
  if (configStandalone.QA.dumpToROOT) {
    configStandalone.proc.outputSharedClusterMap = true;
  }
  if (configStandalone.eventDisplay) {
    configStandalone.noprompt = 1;
  }
  if (configStandalone.proc.debugLevel >= 4) {
    if (configStandalone.proc.inKernelParallel) {
      configStandalone.proc.inKernelParallel = 1;
    } else {
      configStandalone.proc.nHostThreads = 1;
    }
  }
  if (configStandalone.setO2Settings) {
    if (configStandalone.runGPU && configStandalone.proc.debugLevel <= 1) {
      if (!(configStandalone.inputcontrolmem && configStandalone.outputcontrolmem)) {
        printf("setO2Settings requires the usage of --inputMemory and --outputMemory as in O2\n");
        return 1;
      }
      configStandalone.proc.forceHostMemoryPoolSize = 1024 * 1024 * 1024;
    }
    configStandalone.rec.tpc.trackReferenceX = 83;
    configStandalone.proc.outputSharedClusterMap = 1;
    configStandalone.proc.clearO2OutputFromGPU = 1;
    configStandalone.QA.clusterRejectionHistograms = 1;
    configStandalone.proc.tpcIncreasedMinClustersPerRow = 500000;
    configStandalone.proc.ignoreNonFatalGPUErrors = 1;
    // TODO: rundEdx=1
    // GPU_proc.qcRunFraction=$TPC_TRACKING_QC_RUN_FRACTION;"
    // [[ $CTFINPUT == 1 ]] && GPU_CONFIG_KEY+="GPU_proc.tpcInputWithClusterRejection=1;"
    // double pipeline / rtc
  }

  if (configStandalone.outputcontrolmem) {
    bool forceEmptyMemory = getenv("LD_PRELOAD") && strstr(getenv("LD_PRELOAD"), "valgrind") != nullptr;
    outputmemory.reset((char*)operator new(configStandalone.outputcontrolmem, std::align_val_t(GPUCA_BUFFER_ALIGNMENT)));
    if (forceEmptyMemory) {
      printf("Valgrind detected, emptying GPU output memory to avoid false positive undefined reads");
      memset(outputmemory.get(), 0, configStandalone.outputcontrolmem);
    }
    if (configStandalone.proc.doublePipeline) {
      outputmemoryPipeline.reset((char*)operator new(configStandalone.outputcontrolmem, std::align_val_t(GPUCA_BUFFER_ALIGNMENT)));
      if (forceEmptyMemory) {
        memset(outputmemoryPipeline.get(), 0, configStandalone.outputcontrolmem);
      }
    }
  }
  if (configStandalone.inputcontrolmem) {
    inputmemory.reset((char*)operator new(configStandalone.inputcontrolmem, std::align_val_t(GPUCA_BUFFER_ALIGNMENT)));
  }

  configStandalone.proc.showOutputStat = true;

  if (configStandalone.runGPU && configStandalone.gpuType == "AUTO") {
    if (GPUReconstruction::CheckInstanceAvailable(GPUReconstruction::DeviceType::CUDA, configStandalone.proc.debugLevel >= 2)) {
      configStandalone.gpuType = "CUDA";
    } else if (GPUReconstruction::CheckInstanceAvailable(GPUReconstruction::DeviceType::HIP, configStandalone.proc.debugLevel >= 2)) {
      configStandalone.gpuType = "HIP";
    } else if (GPUReconstruction::CheckInstanceAvailable(GPUReconstruction::DeviceType::OCL, configStandalone.proc.debugLevel >= 2)) {
      configStandalone.gpuType = "OCL";
    } else if (GPUReconstruction::CheckInstanceAvailable(GPUReconstruction::DeviceType::OCL, configStandalone.proc.debugLevel >= 2)) {
      configStandalone.gpuType = "OCL";
    } else {
      if (configStandalone.runGPU > 1 && configStandalone.runGPUforce) {
        printf("No GPU backend / device found, running on CPU is disabled due to runGPUforce\n");
        return 1;
      }
      configStandalone.runGPU = false;
      configStandalone.gpuType = "CPU";
    }
  }

  if (configStandalone.printSettings) {
    configStandalone.proc.printSettings = true;
  }
  if (configStandalone.printSettings > 1) {
    printf("Config Dump after ReadConfiguration\n");
    qConfigPrint();
  }

  return (0);
}

int32_t SetupReconstruction()
{
  if (!configStandalone.eventGenerator) {
    if (configStandalone.noEvents) {
      eventsDir = "NON_EXISTING";
      configStandalone.rundEdx = false;
    } else if (rec->ReadSettings(eventsDir.c_str())) {
      printf("Error reading event config file\n");
      return 1;
    }
    const char* tmptext = configStandalone.noEvents ? "Using default event settings, no event dir loaded" : "Read event settings from dir ";
    printf("%s%s (solenoidBz: %f, constBz %d, maxTimeBin %d)\n", tmptext, configStandalone.noEvents ? "" : eventsDir.c_str(), rec->GetGRPSettings().solenoidBzNominalGPU, (int32_t)rec->GetGRPSettings().constBz, rec->GetGRPSettings().grpContinuousMaxTimeBin);
    if (configStandalone.testSyncAsync) {
      recAsync->ReadSettings(eventsDir.c_str());
    }
    if (configStandalone.proc.doublePipeline) {
      recPipeline->ReadSettings(eventsDir.c_str());
    }
  }

  chainTracking->mConfigQA = &configStandalone.QA;
  chainTracking->mConfigDisplay = &configStandalone.display;
  if (configStandalone.testSyncAsync) {
    chainTrackingAsync->mConfigQA = &configStandalone.QA;
    chainTrackingAsync->mConfigDisplay = &configStandalone.display;
  }

  GPUSettingsGRP grp = rec->GetGRPSettings();
  GPUSettingsRec recSet;
  GPUSettingsProcessing procSet;
  recSet = configStandalone.rec;
  procSet = configStandalone.proc;
  GPURecoStepConfiguration steps;

  if (configStandalone.solenoidBzNominalGPU != -1e6f) {
    grp.solenoidBzNominalGPU = configStandalone.solenoidBzNominalGPU;
  }
  if (configStandalone.constBz) {
    grp.constBz = true;
  }
  if (configStandalone.TF.nMerge || configStandalone.TF.bunchSim) {
    if (grp.grpContinuousMaxTimeBin) {
      printf("ERROR: requested to overlay continuous data - not supported\n");
      return 1;
    }
    if (!configStandalone.cont) {
      printf("Continuous mode forced\n");
      configStandalone.cont = true;
    }
    if (chainTracking->GetTPCTransformHelper()) {
      grp.grpContinuousMaxTimeBin = configStandalone.TF.timeFrameLen * ((double)GPUReconstructionTimeframe::TPCZ / (double)GPUReconstructionTimeframe::DRIFT_TIME) / chainTracking->GetTPCTransformHelper()->getCorrMap()->getVDrift();
    }
  }
  if (configStandalone.setMaxTimeBin != -2) {
    grp.grpContinuousMaxTimeBin = configStandalone.setMaxTimeBin;
  } else if (configStandalone.cont && grp.grpContinuousMaxTimeBin == 0) {
    grp.grpContinuousMaxTimeBin = -1;
  }
  if (grp.grpContinuousMaxTimeBin < -1 && !configStandalone.noEvents) {
    printf("Invalid maxTimeBin %d\n", grp.grpContinuousMaxTimeBin);
    return 1;
  }
  if (rec->GetDeviceType() == GPUReconstruction::DeviceType::CPU) {
    printf("Standalone Test Framework for CA Tracker - Using CPU\n");
  } else {
    printf("Standalone Test Framework for CA Tracker - Using GPU\n");
  }

  configStandalone.proc.forceMemoryPoolSize = (configStandalone.proc.forceMemoryPoolSize == 1 && configStandalone.eventDisplay) ? 2 : configStandalone.proc.forceMemoryPoolSize;
  if (configStandalone.eventDisplay) {
    eventDisplay.reset(GPUDisplayFrontendInterface::getFrontend(configStandalone.display.displayFrontend.c_str()));
    if (eventDisplay.get() == nullptr) {
      throw std::runtime_error("Requested display not available");
    }
    printf("Enabling event display (%s backend)\n", eventDisplay->frontendName());
    procSet.eventDisplay = eventDisplay.get();
    if (!configStandalone.QA.noMC) {
      procSet.runMC = true;
    }
  }

  if (procSet.runQA && !configStandalone.QA.noMC) {
    procSet.runMC = true;
  }

  steps.steps = gpudatatypes::RecoStep::AllRecoSteps;
  if (configStandalone.runTRD != -1) {
    steps.steps.setBits(gpudatatypes::RecoStep::TRDTracking, configStandalone.runTRD > 0);
  } else if (chainTracking->GetTRDGeometry() == nullptr) {
    steps.steps.setBits(gpudatatypes::RecoStep::TRDTracking, false);
  }
  if (configStandalone.runCompression != -1) {
    steps.steps.setBits(gpudatatypes::RecoStep::TPCCompression, configStandalone.runCompression > 0);
  }
  if (configStandalone.runTransformation != -1) {
    steps.steps.setBits(gpudatatypes::RecoStep::TPCConversion, configStandalone.runTransformation > 0);
  }
  steps.steps.setBits(gpudatatypes::RecoStep::Refit, configStandalone.runRefit);
  if (!configStandalone.runMerger) {
    steps.steps.setBits(gpudatatypes::RecoStep::TPCMerging, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TRDTracking, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCdEdx, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCCompression, false);
    steps.steps.setBits(gpudatatypes::RecoStep::Refit, false);
  }

  if (configStandalone.TF.bunchSim || configStandalone.TF.nMerge) {
    steps.steps.setBits(gpudatatypes::RecoStep::TRDTracking, false);
  }
  steps.inputs.set(gpudatatypes::InOutType::TPCClusters, gpudatatypes::InOutType::TRDTracklets);
  steps.steps.setBits(gpudatatypes::RecoStep::TPCDecompression, false);
  steps.inputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, false);
  if (grp.doCompClusterDecode) {
    steps.inputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, true);
    steps.inputs.setBits(gpudatatypes::InOutType::TPCClusters, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCCompression, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCClusterFinding, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCDecompression, true);
    steps.outputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, false);
  } else if (grp.needsClusterer) {
    steps.inputs.setBits(gpudatatypes::InOutType::TPCRaw, true);
    steps.inputs.setBits(gpudatatypes::InOutType::TPCClusters, false);
  } else {
    steps.steps.setBits(gpudatatypes::RecoStep::TPCClusterFinding, false);
  }

  // Set settings for synchronous
  GPUChainTracking::ApplySyncSettings(procSet, recSet, steps.steps, configStandalone.testSyncAsync || configStandalone.testSync, configStandalone.rundEdx);
  int32_t runAsyncQA = procSet.runQA && !configStandalone.testSyncAsyncQcInSync ? procSet.runQA : 0;
  if (configStandalone.testSyncAsync) {
    procSet.eventDisplay = nullptr;
    if (!configStandalone.testSyncAsyncQcInSync) {
      procSet.runQA = false;
    }
  }

  // Apply --recoSteps flag last so it takes precedence
  // E.g. ApplySyncSettings might enable TPCdEdx, but might not be needed if only clusterizer was requested
  if (configStandalone.recoSteps >= 0) {
    steps.steps &= configStandalone.recoSteps;
  }
  if (configStandalone.recoStepsGPU >= 0) {
    steps.stepsGPUMask &= configStandalone.recoStepsGPU;
  }

  steps.outputs.clear();
  steps.outputs.setBits(gpudatatypes::InOutType::TPCMergedTracks, steps.steps.isSet(gpudatatypes::RecoStep::TPCMerging));
  steps.outputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, steps.steps.isSet(gpudatatypes::RecoStep::TPCCompression));
  steps.outputs.setBits(gpudatatypes::InOutType::TRDTracks, steps.steps.isSet(gpudatatypes::RecoStep::TRDTracking));
  steps.outputs.setBits(gpudatatypes::InOutType::TPCClusters, steps.steps.isSet(gpudatatypes::RecoStep::TPCClusterFinding));

  if (steps.steps.isSet(gpudatatypes::RecoStep::TRDTracking)) {
    if (procSet.createO2Output && !procSet.trdTrackModelO2) {
      procSet.createO2Output = 1; // Must not be 2, to make sure TPC GPU tracks are still available for TRD
    }
  }

  rec->SetSettings(&grp, &recSet, &procSet, &steps);
  if (configStandalone.proc.doublePipeline) {
    recPipeline->SetSettings(&grp, &recSet, &procSet, &steps);
  }
  if (configStandalone.testSyncAsync) { // TODO: Add --async mode / flag
    // Set settings for asynchronous
    steps.steps.setBits(gpudatatypes::RecoStep::TPCDecompression, true);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCdEdx, true);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCCompression, false);
    steps.steps.setBits(gpudatatypes::RecoStep::TPCClusterFinding, false);
    steps.inputs.setBits(gpudatatypes::InOutType::TPCRaw, false);
    steps.inputs.setBits(gpudatatypes::InOutType::TPCClusters, false);
    steps.inputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, true);
    steps.outputs.setBits(gpudatatypes::InOutType::TPCCompressedClusters, false);
    procSet.runMC = false;
    procSet.runQA = runAsyncQA;
    procSet.eventDisplay = eventDisplay.get();
    procSet.runCompressionStatistics = 0;
    if (recSet.tpc.rejectionStrategy >= GPUSettings::RejectionStrategyB) {
      procSet.tpcInputWithClusterRejection = 1;
    }
    recSet.tpc.disableRefitAttachment = 0xFF;
    recSet.maxTrackQPtB5 = CAMath::Min(recSet.maxTrackQPtB5, recSet.tpc.rejectQPtB5);
    GPUChainTracking::ApplySyncSettings(procSet, recSet, steps.steps, false, configStandalone.rundEdx);
    recAsync->SetSettings(&grp, &recSet, &procSet, &steps);
  }

  if (configStandalone.outputcontrolmem) {
    rec->SetOutputControl(outputmemory.get(), configStandalone.outputcontrolmem);
    if (configStandalone.proc.doublePipeline) {
      recPipeline->SetOutputControl(outputmemoryPipeline.get(), configStandalone.outputcontrolmem);
    }
  }

  o2::base::Propagator* prop = nullptr;
  prop = o2::base::Propagator::Instance(true);
  prop->setGPUField(&rec->GetParam().polynomialField);
  prop->setNominalBz(rec->GetParam().bzkG);
  prop->setMatLUT(chainTracking->GetMatLUT());
  chainTracking->SetO2Propagator(prop);
  if (chainTrackingAsync) {
    chainTrackingAsync->SetO2Propagator(prop);
  }
  if (chainTrackingPipeline) {
    chainTrackingPipeline->SetO2Propagator(prop);
  }
  procSet.o2PropagatorUseGPUField = true;

  if (rec->Init()) {
    printf("Error initializing GPUReconstruction!\n");
    return 1;
  }
  if (configStandalone.outputcontrolmem && rec->IsGPU()) {
    if (rec->registerMemoryForGPU(outputmemory.get(), configStandalone.outputcontrolmem) || (configStandalone.proc.doublePipeline && recPipeline->registerMemoryForGPU(outputmemoryPipeline.get(), configStandalone.outputcontrolmem))) {
      printf("ERROR registering memory for the GPU!!!\n");
      return 1;
    }
  }
  if (configStandalone.inputcontrolmem && rec->IsGPU()) {
    if (rec->registerMemoryForGPU(inputmemory.get(), configStandalone.inputcontrolmem)) {
      printf("ERROR registering input memory for the GPU!!!\n");
      return 1;
    }
  }
  if (configStandalone.proc.debugLevel >= 4) {
    rec->PrintKernelOccupancies();
  }
  return (0);
}

int32_t ReadEvent(int32_t n)
{
  if (configStandalone.inputcontrolmem && !configStandalone.preloadEvents) {
    rec->SetInputControl(inputmemory.get(), configStandalone.inputcontrolmem);
  }
  int32_t r = chainTracking->ReadData((eventsDir + GPUCA_EVDUMP_FILE "." + std::to_string(n) + ".dump").c_str());
  if (r) {
    return r;
  }
#if defined(GPUCA_TPC_GEOMETRY_O2) && defined(GPUCA_BUILD_QA) && !defined(GPUCA_O2_LIB)
  if ((configStandalone.proc.runQA || configStandalone.eventDisplay) && !configStandalone.QA.noMC) {
    chainTracking->ForceInitQA();
    chainTracking->GetQA()->UpdateChain(chainTracking);
    if (chainTracking->GetQA()->ReadO2MCData((eventsDir + "mc." + std::to_string(n) + ".dump").c_str()) &&
        chainTracking->GetQA()->ReadO2MCData((eventsDir + "mc.0.dump").c_str()) && configStandalone.proc.runQA) {
      throw std::runtime_error("Error reading O2 MC dump");
    }
  }
#endif
  if (chainTracking->mIOPtrs.clustersNative && (configStandalone.TF.bunchSim || configStandalone.TF.nMerge || !configStandalone.runTransformation)) {
    if (configStandalone.proc.debugLevel >= 2) {
      printf("Converting Native to Legacy ClusterData for overlaying - WARNING: No raw clusters produced - Compression etc will not run!!!\n");
    }
    chainTracking->ConvertNativeToClusterDataLegacy();
  }
  return 0;
}

int32_t LoadEvent(int32_t iEvent, int32_t x)
{
  if (configStandalone.TF.bunchSim) {
    if (tf->LoadCreateTimeFrame(iEvent)) {
      return 1;
    }
  } else if (configStandalone.TF.nMerge) {
    if (tf->LoadMergedEvents(iEvent)) {
      return 1;
    }
  } else {
    if (ReadEvent(iEvent)) {
      return 1;
    }
  }
  bool encodeZS = configStandalone.encodeZS == -1 ? (chainTracking->mIOPtrs.tpcPackedDigits && !chainTracking->mIOPtrs.tpcZS) : (bool)configStandalone.encodeZS;
  bool zsFilter = configStandalone.zsFilter == -1 ? (!encodeZS && chainTracking->mIOPtrs.tpcPackedDigits && !chainTracking->mIOPtrs.tpcZS) : (bool)configStandalone.zsFilter;
  if (encodeZS || zsFilter) {
    if (!chainTracking->mIOPtrs.tpcPackedDigits) {
      printf("Need digit input to run ZS\n");
      return 1;
    }
    if (zsFilter) {
      chainTracking->ConvertZSFilter(configStandalone.zsVersion >= 2);
    }
    if (encodeZS) {
      chainTracking->ConvertZSEncoder(configStandalone.zsVersion);
    }
  }
  if (!configStandalone.runTransformation) {
    chainTracking->mIOPtrs.clustersNative = nullptr;
  } else {
    for (int32_t i = 0; i < chainTracking->NSECTORS; i++) {
      if (chainTracking->mIOPtrs.rawClusters[i]) {
        if (configStandalone.proc.debugLevel >= 2) {
          printf("Converting Legacy Raw Cluster to Native\n");
        }
        chainTracking->ConvertRun2RawToNative();
        break;
      }
    }
  }

  if (configStandalone.stripDumpedEvents) {
    if (chainTracking->mIOPtrs.tpcZS) {
      chainTracking->mIOPtrs.tpcPackedDigits = nullptr;
    }
  }

  if (!chainTracking->mIOPtrs.clustersNative && !chainTracking->mIOPtrs.tpcPackedDigits && !chainTracking->mIOPtrs.tpcZS && !chainTracking->mIOPtrs.tpcCompressedClusters) {
    printf("Need cluster native data for on-the-fly TPC transform\n");
    return 1;
  }

  ioPtrEvents[x] = chainTracking->mIOPtrs;
  ioMemEvents[x] = std::move(chainTracking->mIOMem);
  chainTracking->mIOMem = decltype(chainTracking->mIOMem)();
  return 0;
}

void OutputStat(GPUChainTracking* t, int64_t* nTracksTotal = nullptr, int64_t* nClustersTotal = nullptr)
{
  int32_t nTracks = 0;
  if (t->GetProcessingSettings().createO2Output) {
    nTracks += t->mIOPtrs.nOutputTracksTPCO2;
  } else {
    for (uint32_t k = 0; k < t->mIOPtrs.nMergedTracks; k++) {
      if (t->mIOPtrs.mergedTracks[k].OK()) {
        nTracks++;
      }
    }
  }
  if (nTracksTotal && nClustersTotal) {
    *nTracksTotal += nTracks;
    *nClustersTotal += t->mIOPtrs.nMergedTrackHits;
  }
}

int32_t RunBenchmark(GPUReconstruction* recUse, GPUChainTracking* chainTrackingUse, int32_t runs, int32_t iEvent, int64_t* nTracksTotal, int64_t* nClustersTotal, int32_t threadId = 0, HighResTimer* timerPipeline = nullptr)
{
  int32_t iRun = 0, iteration = 0;
  while ((iteration = nIteration.fetch_add(1)) < runs) {
    if (configStandalone.runs > 1) {
      printf("Run %d (thread %d)\n", iteration + 1, threadId);
    }
    recUse->SetResetTimers(iRun < configStandalone.runsInit);
    if (configStandalone.outputcontrolmem) {
      recUse->SetOutputControl(threadId ? outputmemoryPipeline.get() : outputmemory.get(), configStandalone.outputcontrolmem);
    }

    if (configStandalone.testSyncAsync) {
      printf("Running synchronous phase\n");
    }
    const GPUTrackingInOutPointers& ioPtrs = ioPtrEvents[!configStandalone.preloadEvents ? 0 : configStandalone.proc.doublePipeline ? (iteration % ioPtrEvents.size()) : (iEvent - configStandalone.StartEvent)];
    chainTrackingUse->mIOPtrs = ioPtrs;
    if (iteration == (configStandalone.proc.doublePipeline ? 2 : (configStandalone.runs - 1))) {
      if (configStandalone.proc.doublePipeline && timerPipeline) {
        timerPipeline->Start();
      }
      if (configStandalone.controlProfiler) {
        rec->startGPUProfiling();
      }
    }
    int32_t tmpRetVal = recUse->RunChains();
    int32_t iterationEnd = nIterationEnd.fetch_add(1);
    if (iterationEnd == configStandalone.runs - 1) {
      if (configStandalone.proc.doublePipeline && timerPipeline) {
        timerPipeline->Stop();
      }
      if (configStandalone.controlProfiler) {
        rec->endGPUProfiling();
      }
    }

    if (tmpRetVal == 0 || tmpRetVal == 2) {
      OutputStat(chainTrackingUse, iRun == 0 ? nTracksTotal : nullptr, iRun == 0 ? nClustersTotal : nullptr);
    }

    if (tmpRetVal == 0 && configStandalone.testSyncAsync) {

      vecpod<char> compressedTmpMem(chainTracking->mIOPtrs.tpcCompressedClusters->totalDataSize);
      memcpy(compressedTmpMem.data(), (const void*)chainTracking->mIOPtrs.tpcCompressedClusters, chainTracking->mIOPtrs.tpcCompressedClusters->totalDataSize);
      o2::tpc::CompressedClusters tmp(*chainTracking->mIOPtrs.tpcCompressedClusters);
      syncAsyncDecodedClusters = tmp.nAttachedClusters + tmp.nUnattachedClusters;
      printf("Running asynchronous phase from %'u compressed clusters\n", syncAsyncDecodedClusters);

      chainTrackingAsync->mIOPtrs = ioPtrs;
      chainTrackingAsync->mIOPtrs.tpcCompressedClusters = (o2::tpc::CompressedClustersFlat*)compressedTmpMem.data();
      chainTrackingAsync->mIOPtrs.tpcZS = nullptr;
      chainTrackingAsync->mIOPtrs.tpcPackedDigits = nullptr;
      chainTrackingAsync->mIOPtrs.mcInfosTPC = nullptr;
      chainTrackingAsync->mIOPtrs.nMCInfosTPC = 0;
      chainTrackingAsync->mIOPtrs.mcInfosTPCCol = nullptr;
      chainTrackingAsync->mIOPtrs.nMCInfosTPCCol = 0;
      chainTrackingAsync->mIOPtrs.mcLabelsTPC = nullptr;
      chainTrackingAsync->mIOPtrs.nMCLabelsTPC = 0;
      for (int32_t i = 0; i < chainTracking->NSECTORS; i++) {
        chainTrackingAsync->mIOPtrs.clusterData[i] = nullptr;
        chainTrackingAsync->mIOPtrs.nClusterData[i] = 0;
        chainTrackingAsync->mIOPtrs.rawClusters[i] = nullptr;
        chainTrackingAsync->mIOPtrs.nRawClusters[i] = 0;
      }
      chainTrackingAsync->mIOPtrs.clustersNative = nullptr;
      recAsync->SetResetTimers(iRun < configStandalone.runsInit);
      tmpRetVal = recAsync->RunChains();
      if (tmpRetVal == 0 || tmpRetVal == 2) {
        OutputStat(chainTrackingAsync, nullptr, nullptr);
      }
      recAsync->ClearAllocatedMemory();
    }
    if (!configStandalone.proc.doublePipeline) {
      recUse->ClearAllocatedMemory();
    }

    if (tmpRetVal == 2) {
      configStandalone.continueOnError = 0; // Forced exit from event display loop
      configStandalone.noprompt = 1;
    }
    if (tmpRetVal == 3 && configStandalone.proc.ignoreNonFatalGPUErrors) {
      printf("GPU Standalone Benchmark: Non-FATAL GPU error occured, ignoring\n");
    } else if (tmpRetVal && !configStandalone.continueOnError) {
      if (tmpRetVal != 2) {
        printf("GPU Standalone Benchmark: Error occured\n");
      }
      return 1;
    }
    iRun++;
  }
  if (configStandalone.proc.doublePipeline) {
    recUse->ClearAllocatedMemory();
  }
  nIteration.store(runs);
  return 0;
}

int32_t main(int argc, char** argv)
{
  std::unique_ptr<GPUReconstruction> recUnique, recUniqueAsync, recUniquePipeline;

  if (ReadConfiguration(argc, argv)) {
    return 1;
  }
  eventsDir = std::string(configStandalone.absoluteEventsDir ? "" : "events/") + configStandalone.eventsDir + "/";

  GPUSettingsDeviceBackend deviceSet;
  deviceSet.deviceType = configStandalone.runGPU ? gpudatatypes::GetDeviceType(configStandalone.gpuType.c_str()) : gpudatatypes::DeviceType::CPU;
  deviceSet.forceDeviceType = configStandalone.runGPUforce;
  deviceSet.master = nullptr;
  recUnique.reset(GPUReconstruction::CreateInstance(deviceSet));
  rec = recUnique.get();
  deviceSet.master = rec;
  if (configStandalone.testSyncAsync) {
    recUniqueAsync.reset(GPUReconstruction::CreateInstance(deviceSet));
    recAsync = recUniqueAsync.get();
  }
  if (configStandalone.proc.doublePipeline) {
    recUniquePipeline.reset(GPUReconstruction::CreateInstance(deviceSet));
    recPipeline = recUniquePipeline.get();
  }
  if (rec == nullptr || (configStandalone.testSyncAsync && recAsync == nullptr)) {
    printf("Error initializing GPUReconstruction\n");
    return 1;
  }
  rec->SetDebugLevelTmp(configStandalone.proc.debugLevel);
  chainTracking = rec->AddChain<GPUChainTracking>();
  if (configStandalone.testSyncAsync) {
    if (configStandalone.proc.debugLevel >= 3) {
      recAsync->SetDebugLevelTmp(configStandalone.proc.debugLevel);
    }
    chainTrackingAsync = recAsync->AddChain<GPUChainTracking>();
  }
  if (configStandalone.proc.doublePipeline) {
    if (configStandalone.proc.debugLevel >= 3) {
      recPipeline->SetDebugLevelTmp(configStandalone.proc.debugLevel);
    }
    chainTrackingPipeline = recPipeline->AddChain<GPUChainTracking>();
    chainTrackingPipeline->SetQAFromForeignChain(chainTracking);
  }
  if (!configStandalone.proc.doublePipeline) {
    chainITS = rec->AddChain<GPUChainITS>();
    if (configStandalone.testSyncAsync) {
      chainITSAsync = recAsync->AddChain<GPUChainITS>();
    }
  }

  if (SetupReconstruction()) {
    return 1;
  }

  std::unique_ptr<std::thread> pipelineThread;
  if (configStandalone.proc.doublePipeline) {
    pipelineThread.reset(new std::thread([]() { rec->RunPipelineWorker(); }));
  }

  if (configStandalone.seed == -1) {
    std::random_device rd;
    configStandalone.seed = (int32_t)rd();
    printf("Using random seed %d\n", configStandalone.seed);
  }

  srand(configStandalone.seed);

  nEventsInDirectory = 0;
  if (!configStandalone.noEvents) {
    while (true) {
      std::ifstream in;
      in.open((eventsDir + GPUCA_EVDUMP_FILE "." + std::to_string(nEventsInDirectory) + ".dump").c_str(), std::ifstream::binary);
      if (in.fail()) {
        break;
      }
      in.close();
      nEventsInDirectory++;
    }
  }

  if (configStandalone.TF.bunchSim || configStandalone.TF.nMerge) {
    tf.reset(new GPUReconstructionTimeframe(chainTracking, ReadEvent, nEventsInDirectory));
  }

  if (configStandalone.eventGenerator) {
    genEvents::RunEventGenerator(chainTracking, eventsDir);
    return 0;
  }

  int32_t nEvents = configStandalone.nEvents;
  if (configStandalone.TF.bunchSim) {
    nEvents = configStandalone.nEvents > 0 ? configStandalone.nEvents : 1;
  } else {
    if (nEvents == -1 || nEvents > nEventsInDirectory) {
      if (nEvents >= 0) {
        printf("Only %d events available in directory %s (%d events requested)\n", nEventsInDirectory, eventsDir.c_str(), nEvents);
      }
      nEvents = nEventsInDirectory;
    }
    if (nEvents == 0 && !configStandalone.noEvents) {
      printf("No event data found in event folder\n");
    }
    if (configStandalone.TF.nMerge > 1) {
      nEvents /= configStandalone.TF.nMerge;
    }
  }

  ioPtrEvents.resize(configStandalone.preloadEvents ? (nEvents - configStandalone.StartEvent) : 1);
  ioMemEvents.resize(configStandalone.preloadEvents ? (nEvents - configStandalone.StartEvent) : 1);
  if (configStandalone.preloadEvents) {
    printf("Preloading events%s", configStandalone.proc.debugLevel >= 2 ? "\n" : "");
    fflush(stdout);
    for (int32_t i = 0; i < nEvents - configStandalone.StartEvent; i++) {
      LoadEvent(configStandalone.StartEvent + i, i);
      printf(configStandalone.proc.debugLevel >= 2 ? "Loading event %d\n" : " %d", i + configStandalone.StartEvent);
      fflush(stdout);
    }
    printf("\n");
  }

  for (int32_t iRunOuter = 0; iRunOuter < configStandalone.runs2; iRunOuter++) {
    if (configStandalone.QA.inputHistogramsOnly) {
      chainTracking->ForceInitQA();
      break;
    }
    if (configStandalone.runs2 > 1) {
      printf("\nRUN2: %d\n", iRunOuter);
    }
    int64_t nTracksTotal = 0;
    int64_t nClustersTotal = 0;
    int32_t nEventsProcessed = 0;

    if (configStandalone.noEvents) {
      nEvents = 1;
      configStandalone.StartEvent = 0;
      chainTracking->ClearIOPointers();
    }

    for (int32_t iEvent = configStandalone.StartEvent; iEvent < nEvents; iEvent++) {
      if (iEvent != configStandalone.StartEvent) {
        printf("\n");
      }
      if (!configStandalone.noEvents && !configStandalone.preloadEvents) {
        HighResTimer timerLoad;
        timerLoad.Start();
        if (LoadEvent(iEvent, 0)) {
          goto breakrun;
        }
        if (configStandalone.dumpEvents) {
          char fname[1024];
          snprintf(fname, 1024, "event.%d.dump", nEventsProcessed);
          chainTracking->DumpData(fname);
          if (nEventsProcessed == 0) {
            rec->DumpSettings();
          }
        }

        if (configStandalone.overrideMaxTimebin && (chainTracking->mIOPtrs.clustersNative || chainTracking->mIOPtrs.tpcPackedDigits || chainTracking->mIOPtrs.tpcZS)) {
          GPUSettingsGRP grp = rec->GetGRPSettings();
          if (grp.grpContinuousMaxTimeBin == 0) {
            printf("Cannot override max time bin for non-continuous data!\n");
          } else {
            grp.grpContinuousMaxTimeBin = chainTracking->mIOPtrs.tpcZS ? GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.tpcZS) : chainTracking->mIOPtrs.tpcPackedDigits ? GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.tpcPackedDigits) : GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.clustersNative);
            printf("Max time bin set to %d\n", grp.grpContinuousMaxTimeBin);
            rec->UpdateSettings(&grp);
            if (recAsync) {
              recAsync->UpdateSettings(&grp);
            }
            if (recPipeline) {
              recPipeline->UpdateSettings(&grp);
            }
          }
        }
        printf("Loading time: %'d us\n", (int32_t)(1000000 * timerLoad.GetCurrentElapsedTime()));
      }

      nIteration.store(0);
      nIterationEnd.store(0);
      double pipelineWalltime = 1.;
      if (configStandalone.noEvents) {
        printf("No processing, no events loaded\n");
      } else if (configStandalone.proc.doublePipeline) {
        printf(configStandalone.preloadEvents ? "Processing Events %d to %d in Pipeline\n" : "Processing Event %d in Pipeline %d times\n", iEvent, configStandalone.preloadEvents ? std::min(iEvent + configStandalone.runs - 1, nEvents - 1) : configStandalone.runs);
        HighResTimer timerPipeline;
        if (configStandalone.proc.debugLevel < 2 && (RunBenchmark(rec, chainTracking, 1, iEvent, &nTracksTotal, &nClustersTotal) || RunBenchmark(recPipeline, chainTrackingPipeline, 2, iEvent, &nTracksTotal, &nClustersTotal))) {
          goto breakrun;
        }
        auto pipeline1 = std::async(std::launch::async, RunBenchmark, rec, chainTracking, configStandalone.runs, iEvent, &nTracksTotal, &nClustersTotal, 0, &timerPipeline);
        auto pipeline2 = std::async(std::launch::async, RunBenchmark, recPipeline, chainTrackingPipeline, configStandalone.runs, iEvent, &nTracksTotal, &nClustersTotal, 1, &timerPipeline);
        if (pipeline1.get() || pipeline2.get()) {
          goto breakrun;
        }
        pipelineWalltime = timerPipeline.GetElapsedTime() / (configStandalone.runs - 2);
        printf("Pipeline wall time: %f, %d iterations, %f per event\n", timerPipeline.GetElapsedTime(), configStandalone.runs - 2, pipelineWalltime);
      } else {
        printf("Processing Event %d\n", iEvent);
        if (RunBenchmark(rec, chainTracking, configStandalone.runs, iEvent, &nTracksTotal, &nClustersTotal)) {
          goto breakrun;
        }
      }
      nEventsProcessed++;

      if (configStandalone.timeFrameTime) {
        double nClusters = chainTracking->GetProcessors()->tpcMerger.NMaxClusters();
        if (nClusters > 0) {
          const int32_t nOrbits = 32;
          const double colRate = 50000;
          const double orbitRate = 11245;
          const double nClsPerTF = 755851. * nOrbits * colRate / orbitRate;
          double timePerTF = (configStandalone.proc.doublePipeline ? pipelineWalltime : ((configStandalone.proc.debugLevel ? rec->GetStatKernelTime() : rec->GetStatWallTime()) / 1000000.)) * nClsPerTF / nClusters;
          const double nGPUsReq = timePerTF * orbitRate / nOrbits;
          char stat[1024];
          snprintf(stat, 1024, "Sync phase: %.2f sec per %d orbit TF, %.1f GPUs required", timePerTF, nOrbits, nGPUsReq);
          if (configStandalone.testSyncAsync) {
            timePerTF = (configStandalone.proc.debugLevel ? recAsync->GetStatKernelTime() : recAsync->GetStatWallTime()) / 1000000. * nClsPerTF / nClusters;
            snprintf(stat + strlen(stat), 1024 - strlen(stat), " - Async phase: %f sec per TF", timePerTF);
          }
          printf("%s (Measured %s time - Extrapolated from %d clusters to %d)\n", stat, configStandalone.proc.debugLevel ? "kernel" : "wall", (int32_t)nClusters, (int32_t)nClsPerTF);
        }
      }
      if (configStandalone.testSyncAsync && chainTracking->mIOPtrs.clustersNative && chainTrackingAsync->mIOPtrs.clustersNative) {
        uint32_t rejected = chainTracking->mIOPtrs.clustersNative->nClustersTotal - syncAsyncDecodedClusters;
        float rejectionPercentage = (rejected) * 100.f / chainTracking->mIOPtrs.clustersNative->nClustersTotal;
        printf("Cluster Rejection: Sync: %'u, Compressed %'u, Async %'u, Rejected %'u (%7.2f%%)\n", chainTracking->mIOPtrs.clustersNative->nClustersTotal, syncAsyncDecodedClusters, chainTrackingAsync->mIOPtrs.clustersNative->nClustersTotal, rejected, rejectionPercentage);
      }

      if (configStandalone.preloadEvents && configStandalone.proc.doublePipeline) {
        break;
      }
    }
    if (nEventsProcessed > 1) {
      printf("Total: %ld clusters, %ld tracks\n", nClustersTotal, nTracksTotal);
    }
  }

breakrun:
  if (rec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    rec->PrintMemoryMax();
  }

#ifndef _WIN32
  if (configStandalone.proc.runQA && configStandalone.fpe) {
    fedisableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  }
#endif

  if (configStandalone.proc.doublePipeline) {
    rec->TerminatePipelineWorker();
    pipelineThread->join();
  }

  rec->Finalize();
  if (configStandalone.testSyncAsync) {
    recAsync->Finalize();
  }
  if (configStandalone.outputcontrolmem && rec->IsGPU()) {
    if (rec->unregisterMemoryForGPU(outputmemory.get()) || (configStandalone.proc.doublePipeline && recPipeline->unregisterMemoryForGPU(outputmemoryPipeline.get()))) {
      printf("Error unregistering memory\n");
    }
  }
  rec->Exit();

  if (!configStandalone.noprompt) {
    printf("Press a key to exit!\n");
    getchar();
  }
  return (0);
}
