// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "GPUTPCDef.h"
#include "GPUQA.h"
#include "GPUDisplayBackend.h"
#include "genEvents.h"

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

#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fenv.h>
#include <clocale>
#include <sys/stat.h>
#endif
#include "utils/timer.h"
#include "utils/qmaths_helpers.h"
#include "utils/vecpod.h"

#include "TPCFastTransform.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUSettings.h"
#include <vector>
#include <xmmintrin.h>

#include "GPUO2DataTypes.h"
#ifdef HAVE_O2HEADERS
#include "GPUChainITS.h"
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY
#ifdef _WIN32
#include "GPUDisplayBackendWindows.h"
#else
#include "GPUDisplayBackendX11.h"
#include "GPUDisplayBackendGlfw.h"
#endif
#include "GPUDisplayBackendGlut.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

//#define BROKEN_EVENTS

namespace GPUCA_NAMESPACE::gpu
{
extern GPUSettingsStandalone configStandalone;
}

GPUReconstruction *rec, *recAsync, *recPipeline;
GPUChainTracking *chainTracking, *chainTrackingAsync, *chainTrackingPipeline;
#ifdef HAVE_O2HEADERS
GPUChainITS *chainITS, *chainITSAsync, *chainITSPipeline;
#endif
std::unique_ptr<char[]> outputmemory, outputmemoryPipeline, inputmemory;
std::unique_ptr<GPUDisplayBackend> eventDisplay;
std::unique_ptr<GPUReconstructionTimeframe> tf;
int nEventsInDirectory = 0;
std::atomic<unsigned int> nIteration, nIterationEnd;

std::vector<GPUTrackingInOutPointers> ioPtrEvents;
std::vector<GPUChainTracking::InOutMemory> ioMemEvents;

void SetCPUAndOSSettings()
{
#ifdef FE_DFL_DISABLE_SSE_DENORMS_ENV // Flush and load denormals to zero in any case
  fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
#else
#ifndef _MM_FLUSH_ZERO_ON
#define _MM_FLUSH_ZERO_ON 0x8000
#endif
#ifndef _MM_DENORMALS_ZERO_ON
#define _MM_DENORMALS_ZERO_ON 0x0040
#endif
  _mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
#endif
}

int ReadConfiguration(int argc, char** argv)
{
  int qcRet = qConfigParse(argc, (const char**)argv);
  if (qcRet) {
    if (qcRet != qConfig::qcrHelp) {
      printf("Error parsing command line parameters\n");
    }
    return 1;
  }
  if (configStandalone.printSettings) {
    qConfigPrint();
  }
  if (configStandalone.proc.debugLevel < 0) {
    configStandalone.proc.debugLevel = 0;
  }
#ifndef _WIN32
  setlocale(LC_ALL, "");
  setlocale(LC_NUMERIC, "");
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
  if (configStandalone.fpe) {
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
  if (configStandalone.fpe) {
    printf("FPE not supported on Windows\n");
    return 1;
  }
#endif
#ifndef HAVE_O2HEADERS
  configStandalone.runTRD = configStandalone.rundEdx = configStandalone.runCompression = configStandalone.runTransformation = configStandalone.testSyncAsync = configStandalone.testSync = 0;
  configStandalone.rec.ForceEarlyTPCTransform = 1;
  configStandalone.runRefit = false;
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
  configStandalone.rec.mergerReadFromTrackerDirectly = 0;
  configStandalone.proc.ompKernels = false;
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
  if (configStandalone.proc.runQA) {
    if (getenv("LC_NUMERIC")) {
      printf("Please unset the LC_NUMERIC env variable, otherwise ROOT will not be able to fit correctly\n"); // BUG: ROOT Problem
      return 1;
    }
  }
#ifndef GPUCA_BUILD_EVENT_DISPLAY
  if (configStandalone.eventDisplay) {
    printf("EventDisplay not enabled in build\n");
    return 1;
  }
#endif
  if (configStandalone.proc.doublePipeline && configStandalone.testSyncAsync) {
    printf("Cannot run asynchronous processing with double pipeline\n");
    return 1;
  }
  if (configStandalone.proc.doublePipeline && (configStandalone.runs < 4 || !configStandalone.outputcontrolmem)) {
    printf("Double pipeline mode needs at least 3 runs per event and external output\n");
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
  if (configStandalone.QA.inputHistogramsOnly) {
    configStandalone.rundEdx = false;
  }
  if (configStandalone.eventDisplay) {
    configStandalone.noprompt = 1;
  }
  if (configStandalone.proc.debugLevel >= 4 && !configStandalone.proc.ompKernels) {
    configStandalone.proc.ompThreads = 1;
  }

  if (configStandalone.outputcontrolmem) {
    bool forceEmptyMemory = getenv("LD_PRELOAD") && strstr(getenv("LD_PRELOAD"), "valgrind") != nullptr;
    outputmemory.reset(new char[configStandalone.outputcontrolmem]);
    if (forceEmptyMemory) {
      printf("Valgrind detected, emptying GPU output memory to avoid false positive undefined reads");
      memset(outputmemory.get(), 0, configStandalone.outputcontrolmem);
    }
    if (configStandalone.proc.doublePipeline) {
      outputmemoryPipeline.reset(new char[configStandalone.outputcontrolmem]);
      if (forceEmptyMemory) {
        memset(outputmemoryPipeline.get(), 0, configStandalone.outputcontrolmem);
      }
    }
  }
  if (configStandalone.inputcontrolmem) {
    inputmemory.reset(new char[configStandalone.inputcontrolmem]);
  }

#if !(defined(CUDA_ENABLED) || defined(OPENCL1_ENABLED) || defined(HIP_ENABLED))
  if (configStandalone.runGPU) {
    printf("GPU disables at build time!\n");
    printf("Press a key to exit!\n");
    getchar();
    return 1;
  }
#endif

  configStandalone.proc.showOutputStat = true;
  return (0);
}

int SetupReconstruction()
{
  if (!configStandalone.eventGenerator) {
    char filename[256];
    snprintf(filename, 256, "events/%s/", configStandalone.EventsDir);
    if (rec->ReadSettings(filename)) {
      printf("Error reading event config file\n");
      return 1;
    }
    printf("Read event settings from dir %s (solenoidBz: %f, home-made events %d, constBz %d, maxTimeBin %d)\n", filename, rec->GetEventSettings().solenoidBz, (int)rec->GetEventSettings().homemadeEvents, (int)rec->GetEventSettings().constBz, rec->GetEventSettings().continuousMaxTimeBin);
    if (configStandalone.testSyncAsync) {
      recAsync->ReadSettings(filename);
    }
    if (configStandalone.proc.doublePipeline) {
      recPipeline->ReadSettings(filename);
    }
  }

  chainTracking->mConfigQA = &configStandalone.QA;
  chainTracking->mConfigDisplay = &configStandalone.GL;

  GPUSettingsEvent ev = rec->GetEventSettings();
  GPUSettingsRec recSet;
  GPUSettingsProcessing devProc;
  memcpy((void*)&recSet, (void*)&configStandalone.rec, sizeof(GPUSettingsRec));
  memcpy((void*)&devProc, (void*)&configStandalone.proc, sizeof(GPUSettingsProcessing));
  GPURecoStepConfiguration steps;

  if (configStandalone.eventGenerator) {
    ev.homemadeEvents = true;
  }
  if (configStandalone.solenoidBz != -1e6f) {
    ev.solenoidBz = configStandalone.solenoidBz;
  }
  if (configStandalone.constBz) {
    ev.constBz = true;
  }
  if (configStandalone.TF.nMerge || configStandalone.TF.bunchSim) {
    if (ev.continuousMaxTimeBin) {
      printf("ERROR: requested to overlay continuous data - not supported\n");
      return 1;
    }
    if (!configStandalone.cont) {
      printf("Continuous mode forced\n");
      configStandalone.cont = true;
    }
    if (chainTracking->GetTPCTransform()) {
      ev.continuousMaxTimeBin = configStandalone.TF.timeFrameLen * ((double)GPUReconstructionTimeframe::TPCZ / (double)GPUReconstructionTimeframe::DRIFT_TIME) / chainTracking->GetTPCTransform()->getVDrift();
    }
  }
  if (configStandalone.cont && ev.continuousMaxTimeBin == 0) {
    ev.continuousMaxTimeBin = -1;
  }
  if (rec->GetDeviceType() == GPUReconstruction::DeviceType::CPU) {
    printf("Standalone Test Framework for CA Tracker - Using CPU\n");
  } else {
    printf("Standalone Test Framework for CA Tracker - Using GPU\n");
  }

  configStandalone.proc.forceMemoryPoolSize = (configStandalone.proc.forceMemoryPoolSize == 1 && configStandalone.eventDisplay) ? 2 : configStandalone.proc.forceMemoryPoolSize;
  if (configStandalone.eventDisplay) {
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#ifdef _WIN32
    if (configStandalone.eventDisplay == 1) {
      printf("Enabling event display (windows backend)\n");
      eventDisplay.reset(new GPUDisplayBackendWindows);
    }

#else
    if (configStandalone.eventDisplay == 1) {
      eventDisplay.reset(new GPUDisplayBackendX11);
      printf("Enabling event display (X11 backend)\n");
    }
    if (configStandalone.eventDisplay == 3) {
      eventDisplay.reset(new GPUDisplayBackendGlfw);
      printf("Enabling event display (GLFW backend)\n");
    }

#endif
    else if (configStandalone.eventDisplay == 2) {
      eventDisplay.reset(new GPUDisplayBackendGlut);
      printf("Enabling event display (GLUT backend)\n");
    }

#endif
    devProc.eventDisplay = eventDisplay.get();
  }
  if (devProc.runQA) {
    devProc.runMC = true;
  }

  steps.steps = GPUDataTypes::RecoStep::AllRecoSteps;
  if (configStandalone.runTRD != -1) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TRDTracking, configStandalone.runTRD > 0);
  } else if (chainTracking->GetTRDGeometry() == nullptr) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TRDTracking, false);
  }
  if (configStandalone.rundEdx != -1) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, configStandalone.rundEdx > 0);
  }
  if (configStandalone.runCompression != -1) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, configStandalone.runCompression > 0);
  }
  if (configStandalone.runTransformation != -1) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCConversion, configStandalone.runTransformation > 0);
  }
  steps.steps.setBits(GPUDataTypes::RecoStep::Refit, configStandalone.runRefit);
  if (!configStandalone.runMerger) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCMerging, false);
    steps.steps.setBits(GPUDataTypes::RecoStep::TRDTracking, false);
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, false);
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, false);
    steps.steps.setBits(GPUDataTypes::RecoStep::Refit, false);
  }

  if (configStandalone.TF.bunchSim || configStandalone.TF.nMerge) {
    steps.steps.setBits(GPUDataTypes::RecoStep::TRDTracking, false);
  }
  steps.inputs.set(GPUDataTypes::InOutType::TPCClusters, GPUDataTypes::InOutType::TRDTracklets);
  if (ev.needsClusterer) {
    steps.inputs.setBits(GPUDataTypes::InOutType::TPCRaw, true);
    steps.inputs.setBits(GPUDataTypes::InOutType::TPCClusters, false);
  } else {
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCClusterFinding, false);
  }

  if (configStandalone.recoSteps >= 0) {
    steps.steps &= configStandalone.recoSteps;
  }
  if (configStandalone.recoStepsGPU >= 0) {
    steps.stepsGPUMask &= configStandalone.recoStepsGPU;
  }

  steps.outputs.clear();
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCSectorTracks, steps.steps.isSet(GPUDataTypes::RecoStep::TPCSliceTracking) && !recSet.mergerReadFromTrackerDirectly);
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCMergedTracks, steps.steps.isSet(GPUDataTypes::RecoStep::TPCMerging));
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, steps.steps.isSet(GPUDataTypes::RecoStep::TPCCompression));
  steps.outputs.setBits(GPUDataTypes::InOutType::TRDTracks, steps.steps.isSet(GPUDataTypes::RecoStep::TRDTracking));
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, steps.steps.isSet(GPUDataTypes::RecoStep::TPCClusterFinding));
  steps.steps.setBits(GPUDataTypes::RecoStep::TPCDecompression, false);
  steps.inputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, false);

  if (configStandalone.testSyncAsync || configStandalone.testSync) {
    // Set settings for synchronous
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, 0);
    recSet.useMatLUT = false;
    if (configStandalone.testSyncAsync) {
      devProc.eventDisplay = nullptr;
    }
  }

  rec->SetSettings(&ev, &recSet, &devProc, &steps);
  if (configStandalone.proc.doublePipeline) {
    recPipeline->SetSettings(&ev, &recSet, &devProc, &steps);
  }
  if (configStandalone.testSyncAsync) {
    // Set settings for asynchronous
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCDecompression, true);
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, true);
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, false);
    steps.steps.setBits(GPUDataTypes::RecoStep::TPCClusterFinding, false);
    steps.inputs.setBits(GPUDataTypes::InOutType::TPCRaw, false);
    steps.inputs.setBits(GPUDataTypes::InOutType::TPCClusters, false);
    steps.inputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, true);
    steps.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, false);
    devProc.runMC = false;
    devProc.runQA = false;
    devProc.eventDisplay = eventDisplay.get();
    devProc.runCompressionStatistics = 0;
    recSet.DisableRefitAttachment = 0xFF;
    recSet.loopInterpolationInExtraPass = 0;
    recSet.MaxTrackQPt = CAMath::Min(recSet.MaxTrackQPt, recSet.tpcRejectQPt);
    recSet.useMatLUT = true;
    recAsync->SetSettings(&ev, &recSet, &devProc, &steps);
  }

  if (configStandalone.outputcontrolmem) {
    rec->SetOutputControl(outputmemory.get(), configStandalone.outputcontrolmem);
    if (configStandalone.proc.doublePipeline) {
      recPipeline->SetOutputControl(outputmemoryPipeline.get(), configStandalone.outputcontrolmem);
    }
  }

#ifdef HAVE_O2HEADERS
  chainTracking->SetDefaultO2PropagatorForGPU();
  if (configStandalone.testSyncAsync) {
    chainTrackingAsync->SetDefaultO2PropagatorForGPU();
  }
  if (configStandalone.proc.doublePipeline) {
    chainTrackingPipeline->SetDefaultO2PropagatorForGPU();
  }
#endif

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

int ReadEvent(int n)
{
  char filename[256];
  snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, n);
  if (configStandalone.inputcontrolmem && !configStandalone.preloadEvents) {
    rec->SetInputControl(inputmemory.get(), configStandalone.inputcontrolmem);
  }
  int r = chainTracking->ReadData(filename);
  if (r) {
    return r;
  }
  if (chainTracking->mIOPtrs.clustersNative && (configStandalone.TF.bunchSim || configStandalone.TF.nMerge || !configStandalone.runTransformation)) {
    if (configStandalone.proc.debugLevel >= 2) {
      printf("Converting Native to Legacy ClusterData for overlaying - WARNING: No raw clusters produced - Compression etc will not run!!!\n");
    }
    chainTracking->ConvertNativeToClusterDataLegacy();
  }
  return 0;
}

int LoadEvent(int iEvent, int x)
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
      chainTracking->ConvertZSFilter(configStandalone.zs12bit);
    }
    if (encodeZS) {
      chainTracking->ConvertZSEncoder(configStandalone.zs12bit);
    }
  }
  if (!configStandalone.runTransformation) {
    chainTracking->mIOPtrs.clustersNative = nullptr;
  } else {
    for (int i = 0; i < chainTracking->NSLICES; i++) {
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

  if (!rec->GetParam().par.earlyTpcTransform && chainTracking->mIOPtrs.clustersNative == nullptr && chainTracking->mIOPtrs.tpcPackedDigits == nullptr && chainTracking->mIOPtrs.tpcZS == nullptr) {
    printf("Need cluster native data for on-the-fly TPC transform\n");
    return 1;
  }

  ioPtrEvents[x] = chainTracking->mIOPtrs;
  ioMemEvents[x] = std::move(chainTracking->mIOMem);
  return 0;
}

void OutputStat(GPUChainTracking* t, long long int* nTracksTotal = nullptr, long long int* nClustersTotal = nullptr)
{
  int nTracks = 0;
  if (t->GetProcessingSettings().createO2Output) {
    nTracks += t->mIOPtrs.nOutputTracksTPCO2;
  } else {
    for (unsigned int k = 0; k < t->mIOPtrs.nMergedTracks; k++) {
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

int RunBenchmark(GPUReconstruction* recUse, GPUChainTracking* chainTrackingUse, int runs, int iEvent, long long int* nTracksTotal, long long int* nClustersTotal, int threadId = 0, HighResTimer* timerPipeline = nullptr)
{
  int iRun = 0, iteration = 0;
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
      if (configStandalone.proc.doublePipeline) {
        timerPipeline->Start();
      }
      if (configStandalone.controlProfiler) {
        rec->startGPUProfiling();
      }
    }
    int tmpRetVal = recUse->RunChains();
    int iterationEnd = nIterationEnd.fetch_add(1);
    if (iterationEnd == configStandalone.runs - 1) {
      if (configStandalone.proc.doublePipeline) {
        timerPipeline->Stop();
      }
      if (configStandalone.controlProfiler) {
        rec->endGPUProfiling();
      }
    }

    if (tmpRetVal == 0 || tmpRetVal == 2) {
      OutputStat(chainTrackingUse, iRun == 0 ? nTracksTotal : nullptr, iRun == 0 ? nClustersTotal : nullptr);
      if (configStandalone.memoryStat) {
        recUse->PrintMemoryStatistics();
      } else if (configStandalone.proc.debugLevel >= 2) {
        recUse->PrintMemoryOverview();
      }
    }

#ifdef HAVE_O2HEADERS
    if (tmpRetVal == 0 && configStandalone.testSyncAsync) {
      if (configStandalone.testSyncAsync) {
        printf("Running asynchronous phase\n");
      }

      vecpod<char> compressedTmpMem(chainTracking->mIOPtrs.tpcCompressedClusters->totalDataSize);
      memcpy(compressedTmpMem.data(), (const void*)chainTracking->mIOPtrs.tpcCompressedClusters, chainTracking->mIOPtrs.tpcCompressedClusters->totalDataSize);

      chainTrackingAsync->mIOPtrs = ioPtrs;
      chainTrackingAsync->mIOPtrs.tpcCompressedClusters = (o2::tpc::CompressedClustersFlat*)compressedTmpMem.data();
      chainTrackingAsync->mIOPtrs.tpcZS = nullptr;
      chainTrackingAsync->mIOPtrs.tpcPackedDigits = nullptr;
      chainTrackingAsync->mIOPtrs.mcInfosTPC = nullptr;
      chainTrackingAsync->mIOPtrs.nMCInfosTPC = 0;
      chainTrackingAsync->mIOPtrs.mcLabelsTPC = nullptr;
      chainTrackingAsync->mIOPtrs.nMCLabelsTPC = 0;
      for (int i = 0; i < chainTracking->NSLICES; i++) {
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
        if (configStandalone.memoryStat) {
          recAsync->PrintMemoryStatistics();
        }
      }
      recAsync->ClearAllocatedMemory();
    }
#endif
    if (!configStandalone.proc.doublePipeline) {
      recUse->ClearAllocatedMemory();
    }

    if (tmpRetVal == 2) {
      configStandalone.continueOnError = 0; // Forced exit from event display loop
      configStandalone.noprompt = 1;
    }
    if (tmpRetVal && !configStandalone.continueOnError) {
      if (tmpRetVal != 2) {
        printf("Error occured\n");
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

int main(int argc, char** argv)
{
  std::unique_ptr<GPUReconstruction> recUnique, recUniqueAsync, recUniquePipeline;

  SetCPUAndOSSettings();

  if (ReadConfiguration(argc, argv)) {
    return 1;
  }

  recUnique.reset(GPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : GPUReconstruction::DEVICE_TYPE_NAMES[GPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce));
  rec = recUnique.get();
  if (configStandalone.testSyncAsync) {
    recUniqueAsync.reset(GPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : GPUReconstruction::DEVICE_TYPE_NAMES[GPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce, rec));
    recAsync = recUniqueAsync.get();
  }
  if (configStandalone.proc.doublePipeline) {
    recUniquePipeline.reset(GPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : GPUReconstruction::DEVICE_TYPE_NAMES[GPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce, rec));
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
  }
#ifdef HAVE_O2HEADERS
  if (!configStandalone.proc.doublePipeline) {
    chainITS = rec->AddChain<GPUChainITS>(0);
    if (configStandalone.testSyncAsync) {
      chainITSAsync = recAsync->AddChain<GPUChainITS>(0);
    }
  }
#endif

  if (SetupReconstruction()) {
    return 1;
  }

  std::unique_ptr<std::thread> pipelineThread;
  if (configStandalone.proc.doublePipeline) {
    pipelineThread.reset(new std::thread([]() { rec->RunPipelineWorker(); }));
  }

  if (configStandalone.seed == -1) {
    std::random_device rd;
    configStandalone.seed = (int)rd();
    printf("Using random seed %d\n", configStandalone.seed);
  }

  srand(configStandalone.seed);

  for (nEventsInDirectory = 0; true; nEventsInDirectory++) {
    std::ifstream in;
    char filename[256];
    snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, nEventsInDirectory);
    in.open(filename, std::ifstream::binary);
    if (in.fail()) {
      break;
    }
    in.close();
  }

  if (configStandalone.TF.bunchSim || configStandalone.TF.nMerge) {
    tf.reset(new GPUReconstructionTimeframe(chainTracking, ReadEvent, nEventsInDirectory));
  }

  if (configStandalone.eventGenerator) {
    genEvents::RunEventGenerator(chainTracking);
    return 0;
  }

  int nEvents = configStandalone.NEvents;
  if (configStandalone.TF.bunchSim) {
    nEvents = configStandalone.NEvents > 0 ? configStandalone.NEvents : 1;
  } else {
    if (nEvents == -1 || nEvents > nEventsInDirectory) {
      if (nEvents >= 0) {
        printf("Only %d events available in directors %s (%d events requested)\n", nEventsInDirectory, configStandalone.EventsDir, nEvents);
      }
      nEvents = nEventsInDirectory;
    }
    if (configStandalone.TF.nMerge > 1) {
      nEvents /= configStandalone.TF.nMerge;
    }
  }

  ioPtrEvents.resize(configStandalone.preloadEvents ? (nEvents - configStandalone.StartEvent) : 1);
  ioMemEvents.resize(configStandalone.preloadEvents ? (nEvents - configStandalone.StartEvent) : 1);
  if (configStandalone.preloadEvents) {
    printf("Preloading events");
    fflush(stdout);
    for (int i = 0; i < nEvents - configStandalone.StartEvent; i++) {
      LoadEvent(configStandalone.StartEvent + i, i);
      printf(" %d", i);
      fflush(stdout);
    }
    printf("\n");
  }

  for (int iRunOuter = 0; iRunOuter < configStandalone.runs2; iRunOuter++) {
    if (configStandalone.QA.inputHistogramsOnly) {
      chainTracking->ForceInitQA();
      break;
    }
    if (configStandalone.runs2 > 1) {
      printf("RUN2: %d\n", iRunOuter);
    }
    long long int nTracksTotal = 0;
    long long int nClustersTotal = 0;
    int nEventsProcessed = 0;

    for (int iEvent = configStandalone.StartEvent; iEvent < nEvents; iEvent++) {
      if (iEvent != configStandalone.StartEvent) {
        printf("\n");
      }
      if (!configStandalone.preloadEvents) {
        HighResTimer timerLoad;
        timerLoad.Start();
        if (LoadEvent(iEvent, 0)) {
          goto breakrun;
        }
        if (configStandalone.dumpEvents) {
          char fname[1024];
          sprintf(fname, "event.%d.dump", nEventsProcessed);
          chainTracking->DumpData(fname);
          if (nEventsProcessed == 0) {
            rec->DumpSettings();
          }
        }

        if (configStandalone.overrideMaxTimebin && (chainTracking->mIOPtrs.clustersNative || chainTracking->mIOPtrs.tpcPackedDigits || chainTracking->mIOPtrs.tpcZS)) {
          GPUSettingsEvent ev = rec->GetEventSettings();
          if (ev.continuousMaxTimeBin == 0) {
            printf("Cannot override max time bin for non-continuous data!\n");
          } else {
            ev.continuousMaxTimeBin = chainTracking->mIOPtrs.tpcZS ? GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.tpcZS) : chainTracking->mIOPtrs.tpcPackedDigits ? GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.tpcPackedDigits) : GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.clustersNative);
            printf("Max time bin set to %d\n", (int)ev.continuousMaxTimeBin);
            rec->UpdateEventSettings(&ev);
            if (recAsync) {
              recAsync->UpdateEventSettings(&ev);
            }
            if (recPipeline) {
              recPipeline->UpdateEventSettings(&ev);
            }
          }
        }
        printf("Loading time: %'d us\n", (int)(1000000 * timerLoad.GetCurrentElapsedTime()));
      }
      printf("Processing Event %d\n", iEvent);

      nIteration.store(0);
      nIterationEnd.store(0);
      double pipelineWalltime = 1.;
      if (configStandalone.proc.doublePipeline) {
        HighResTimer timerPipeline;
        if (RunBenchmark(rec, chainTracking, 1, iEvent, &nTracksTotal, &nClustersTotal) || RunBenchmark(recPipeline, chainTrackingPipeline, 2, iEvent, &nTracksTotal, &nClustersTotal)) {
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
        if (RunBenchmark(rec, chainTracking, configStandalone.runs, iEvent, &nTracksTotal, &nClustersTotal)) {
          goto breakrun;
        }
      }
      nEventsProcessed++;

      if (configStandalone.timeFrameTime) {
        double nClusters = chainTracking->GetTPCMerger().NMaxClusters();
        if (nClusters > 0) {
          double nClsPerTF = 550000. * 1138.3;
          double timePerTF = (configStandalone.proc.doublePipeline ? pipelineWalltime : ((configStandalone.proc.debugLevel ? rec->GetStatKernelTime() : rec->GetStatWallTime()) / 1000000.)) * nClsPerTF / nClusters;
          double nGPUsReq = timePerTF / 0.02277;
          char stat[1024];
          snprintf(stat, 1024, "Sync phase: %.2f sec per 256 orbit TF, %.1f GPUs required", timePerTF, nGPUsReq);
          if (configStandalone.testSyncAsync) {
            timePerTF = (configStandalone.proc.debugLevel ? recAsync->GetStatKernelTime() : recAsync->GetStatWallTime()) / 1000000. * nClsPerTF / nClusters;
            snprintf(stat + strlen(stat), 1024 - strlen(stat), " - Async phase: %f sec per TF", timePerTF);
          }
          printf("%s (Measured %s time - Extrapolated from %d clusters to %d)\n", stat, configStandalone.proc.debugLevel ? "kernel" : "wall", (int)nClusters, (int)nClsPerTF);
        }
      }

      if (configStandalone.preloadEvents && configStandalone.proc.doublePipeline) {
        break;
      }
    }
    if (nEventsProcessed > 1) {
      printf("Total: %lld clusters, %lld tracks\n", nClustersTotal, nTracksTotal);
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
