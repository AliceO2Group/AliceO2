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
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

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

GPUReconstruction* rec;
GPUChainTracking* chainTracking;
#ifdef HAVE_O2HEADERS
GPUChainITS* chainITS;
#endif
std::unique_ptr<char[]> outputmemory;
std::unique_ptr<GPUDisplayBackend> eventDisplay;
std::unique_ptr<GPUReconstructionTimeframe> tf;
int nEventsInDirectory = 0;

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
    return (1);
  }
  if (configStandalone.printSettings) {
    qConfigPrint();
  }

#ifndef _WIN32
  setlocale(LC_ALL, "");
  setlocale(LC_NUMERIC, "");
  if (configStandalone.affinity != -1) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(configStandalone.affinity, &mask);

    printf("Setting affinitiy to restrict on CPU %d\n", configStandalone.affinity);
    if (0 != sched_setaffinity(0, sizeof(mask), &mask)) {
      printf("Error setting CPU affinity\n");
      return (1);
    }
  }
  if (configStandalone.fifo) {
    printf("Setting FIFO scheduler\n");
    sched_param param;
    sched_getparam(0, &param);
    param.sched_priority = 1;
    if (0 != sched_setscheduler(0, SCHED_FIFO, &param)) {
      printf("Error setting scheduler\n");
      return (1);
    }
  }
  if (configStandalone.fpe) {
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  }

#else
  if (configStandalone.affinity != -1) {
    printf("Affinity setting not supported on Windows\n");
    return (1);
  }
  if (configStandalone.fifo) {
    printf("FIFO Scheduler setting not supported on Windows\n");
    return (1);
  }
  if (configStandalone.fpe) {
    printf("FPE not supported on Windows\n");
    return (1);
  }
#endif
#ifndef HAVE_O2HEADERS
  configStandalone.configRec.runTRD = configStandalone.configRec.rundEdx = configStandalone.configRec.runCompression = configStandalone.configRec.runTransformation = 0;
#endif
#ifndef GPUCA_BUILD_QA
  if (configStandalone.qa || configStandalone.eventGenerator) {
    printf("QA not enabled in build\n");
    return (1);
  }
#endif
#ifndef GPUCA_BUILD_EVENT_DISPLAY
  if (configStandalone.eventDisplay) {
    printf("EventDisplay not enabled in build\n");
    return (1);
  }
#endif
  if (configStandalone.configTF.bunchSim && configStandalone.configTF.nMerge) {
    printf("Cannot run --MERGE and --SIMBUNCHES togeterh\n");
    return (1);
  }
  if (configStandalone.configTF.bunchSim > 1) {
    configStandalone.configTF.timeFrameLen = 1.e9 * configStandalone.configTF.bunchSim / configStandalone.configTF.interactionRate;
  }
  if (configStandalone.configTF.nMerge) {
    double len = configStandalone.configTF.nMerge - 1;
    if (configStandalone.configTF.randomizeDistance) {
      len += 0.5;
    }
    if (configStandalone.configTF.shiftFirstEvent) {
      len += 0.5;
    }
    configStandalone.configTF.timeFrameLen = (len * configStandalone.configTF.averageDistance / GPUReconstructionTimeframe::TPCZ + 1) * GPUReconstructionTimeframe::DRIFT_TIME;
  }
  if (configStandalone.configQA.inputHistogramsOnly && configStandalone.configQA.compareInputs.size() == 0) {
    printf("Can only produce QA pdf output when input files are specified!\n");
    return (1);
  }
  if ((configStandalone.nways & 1) == 0) {
    printf("nWay setting musst be odd number!\n");
    return (1);
  }

  if (configStandalone.eventDisplay) {
    configStandalone.noprompt = 1;
  }
  if (configStandalone.DebugLevel >= 4) {
    configStandalone.OMPThreads = 1;
  }

#ifdef GPUCA_HAVE_OPENMP
  if (configStandalone.OMPThreads != -1) {
    omp_set_num_threads(configStandalone.OMPThreads);
  } else {
    configStandalone.OMPThreads = omp_get_max_threads();
  }
  if (configStandalone.OMPThreads != omp_get_max_threads()) {
    printf("Cannot set number of OMP threads!\n");
    return (1);
  }
#else
  configStandalone.OMPThreads = 1;
#endif
  if (configStandalone.outputcontrolmem) {
    outputmemory.reset(new char[configStandalone.outputcontrolmem]);
  }

#if !(defined(CUDA_ENABLED) || defined(OPENCL1_ENABLED) || defined(HIP_ENABLED))
  if (configStandalone.runGPU) {
    printf("GPU disables at build time!\n");
    printf("Press a key to exit!\n");
    getchar();
    return (1);
  }
#endif
  return (0);
}

int SetupReconstruction()
{
  if (!configStandalone.eventGenerator) {
    char filename[256];
    snprintf(filename, 256, "events/%s/", configStandalone.EventsDir);
    rec->ReadSettings(filename);
    printf("Read event settings from dir %s (solenoidBz: %f, home-made events %d, constBz %d, maxTimeBin %d)\n", filename, rec->GetEventSettings().solenoidBz, (int)rec->GetEventSettings().homemadeEvents, (int)rec->GetEventSettings().constBz, rec->GetEventSettings().continuousMaxTimeBin);
  }

  GPUSettingsEvent ev = rec->GetEventSettings();
  GPUSettingsRec recSet;
  GPUSettingsDeviceProcessing devProc;
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
  if (configStandalone.configTF.nMerge || configStandalone.configTF.bunchSim) {
    if (ev.continuousMaxTimeBin) {
      printf("ERROR: requested to overlay continuous data - not supported\n");
      return 1;
    }
    if (!configStandalone.cont) {
      printf("Continuous mode forced\n");
      configStandalone.cont = true;
    }
    if (chainTracking->GetTPCTransform()) {
      ev.continuousMaxTimeBin = configStandalone.configTF.timeFrameLen * ((double)GPUReconstructionTimeframe::TPCZ / (double)GPUReconstructionTimeframe::DRIFT_TIME) / chainTracking->GetTPCTransform()->getVDrift();
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

  recSet.SetMinTrackPt(GPUCA_MIN_TRACK_PT_DEFAULT);
  recSet.NWays = configStandalone.nways;
  recSet.NWaysOuter = configStandalone.nwaysouter;
  recSet.RejectMode = configStandalone.rejectMode;
  recSet.SearchWindowDZDR = configStandalone.dzdr;
  recSet.GlobalTracking = configStandalone.configRec.globalTracking;
  recSet.DisableRefitAttachment = configStandalone.configRec.disableRefitAttachment;
  recSet.ForceEarlyTPCTransform = configStandalone.configRec.ForceEarlyTPCTransform;
  if (configStandalone.referenceX < 500.) {
    recSet.TrackReferenceX = configStandalone.referenceX;
  }

  if (configStandalone.OMPThreads != -1) {
    devProc.nThreads = configStandalone.OMPThreads;
  }
  devProc.deviceNum = configStandalone.cudaDevice;
  devProc.forceMemoryPoolSize = configStandalone.forceMemorySize;
  devProc.debugLevel = configStandalone.DebugLevel;
  devProc.runQA = configStandalone.qa;
  devProc.runCompressionStatistics = configStandalone.compressionStat;
  if (configStandalone.eventDisplay) {
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#ifdef _WIN32
    if (configStandalone.eventDisplay == 1) {
      eventDisplay.reset(new GPUDisplayBackendWindows);
    }

#else
    if (configStandalone.eventDisplay == 1) {
      eventDisplay.reset(new GPUDisplayBackendX11);
    }
    if (configStandalone.eventDisplay == 3) {
      eventDisplay.reset(new GPUDisplayBackendGlfw);
    }

#endif
    else if (configStandalone.eventDisplay == 2) {
      eventDisplay.reset(new GPUDisplayBackendGlut);
    }

#endif
    devProc.eventDisplay = eventDisplay.get();
  }
  devProc.nDeviceHelperThreads = configStandalone.helperThreads;
  devProc.globalInitMutex = configStandalone.gpuInitMutex;
  devProc.gpuDeviceOnly = configStandalone.oclGPUonly;
  devProc.memoryAllocationStrategy = configStandalone.allocationStrategy;
  recSet.tpcRejectionMode = configStandalone.configRec.tpcReject;
  if (configStandalone.configRec.tpcRejectThreshold != 0.f) {
    recSet.tpcRejectQPt = 1.f / configStandalone.configRec.tpcRejectThreshold;
  }
  recSet.tpcCompressionModes = configStandalone.configRec.tpcCompression;
  recSet.tpcCompressionSortOrder = configStandalone.configRec.tpcCompressionSort;

  if (configStandalone.configProc.nStreams >= 0) {
    devProc.nStreams = configStandalone.configProc.nStreams;
  }
  if (configStandalone.configProc.constructorPipeline >= 0) {
    devProc.trackletConstructorInPipeline = configStandalone.configProc.constructorPipeline;
  }
  if (configStandalone.configProc.selectorPipeline >= 0) {
    devProc.trackletSelectorInPipeline = configStandalone.configProc.selectorPipeline;
  }

  steps.steps = GPUReconstruction::RecoStep::AllRecoSteps;
  if (configStandalone.configRec.runTRD != -1) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TRDTracking, configStandalone.configRec.runTRD > 0);
  }
  if (configStandalone.configRec.rundEdx != -1) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCdEdx, configStandalone.configRec.rundEdx > 0);
  }
  if (configStandalone.configRec.runCompression != -1) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCCompression, configStandalone.configRec.runCompression > 0);
  }
  if (configStandalone.configRec.runTransformation != -1) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCConversion, configStandalone.configRec.runTransformation > 0);
  }
  if (!configStandalone.merger) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCMerging, false);
    steps.steps.setBits(GPUReconstruction::RecoStep::TRDTracking, false);
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCdEdx, false);
    steps.steps.setBits(GPUReconstruction::RecoStep::TPCCompression, false);
  }
  if (configStandalone.configTF.bunchSim || configStandalone.configTF.nMerge) {
    steps.steps.setBits(GPUReconstruction::RecoStep::TRDTracking, false);
  }
  steps.steps.setBits(GPUReconstruction::RecoStep::TPCClusterFinding, false); // Disable cluster finding for now
  steps.inputs.set(GPUDataTypes::InOutType::TPCClusters, GPUDataTypes::InOutType::TRDTracklets);
  steps.outputs.set(GPUDataTypes::InOutType::TPCSectorTracks);
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCMergedTracks, steps.steps.isSet(GPUReconstruction::RecoStep::TPCMerging));
  steps.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, steps.steps.isSet(GPUReconstruction::RecoStep::TPCCompression));
  steps.outputs.setBits(GPUDataTypes::InOutType::TRDTracks, steps.steps.isSet(GPUReconstruction::RecoStep::TRDTracking));
  if (configStandalone.configProc.recoSteps >= 0) {
    steps.steps &= configStandalone.configProc.recoSteps;
  }
  if (configStandalone.configProc.recoStepsGPU >= 0) {
    steps.stepsGPUMask &= configStandalone.configProc.recoStepsGPU;
  }

  rec->SetSettings(&ev, &recSet, &devProc, &steps);
  if (rec->Init()) {
    printf("Error initializing GPUReconstruction!\n");
    return 1;
  }
  return (0);
}

int ReadEvent(int n)
{
  char filename[256];
  snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, n);
  int r = chainTracking->ReadData(filename);
  if (r) {
    return r;
  }
  if (chainTracking->mIOPtrs.clustersNative && (configStandalone.configTF.bunchSim || configStandalone.configTF.nMerge || !configStandalone.configRec.runTransformation)) {
    if (configStandalone.DebugLevel >= 2) {
      printf("Converting Native to Legacy ClusterData for overlaying - WARNING: No raw clusters produced - Compression etc will not run!!!\n");
    }
    chainTracking->ConvertNativeToClusterDataLegacy();
  }
  return 0;
}

int main(int argc, char** argv)
{
  std::unique_ptr<GPUReconstruction> recUnique;

  SetCPUAndOSSettings();

  if (ReadConfiguration(argc, argv)) {
    return (1);
  }

  recUnique.reset(GPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : GPUReconstruction::DEVICE_TYPE_NAMES[GPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce));
  rec = recUnique.get();
  if (rec == nullptr) {
    printf("Error initializing GPUReconstruction\n");
    return (1);
  }
  rec->SetDebugLevelTmp(configStandalone.DebugLevel);
  chainTracking = rec->AddChain<GPUChainTracking>();
#ifdef HAVE_O2HEADERS
  chainITS = rec->AddChain<GPUChainITS>(0);
#endif

  if (SetupReconstruction()) {
    return (1);
  }

  // hlt.SetRunMerger(configStandalone.merger); //TODO!

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

  if (configStandalone.configTF.bunchSim || configStandalone.configTF.nMerge) {
    tf.reset(new GPUReconstructionTimeframe(chainTracking, ReadEvent, nEventsInDirectory));
  }

  if (configStandalone.eventGenerator) {
    genEvents::RunEventGenerator(chainTracking);
    return (1);
  } else {
    int nEvents = configStandalone.NEvents;
    if (configStandalone.configTF.bunchSim) {
      nEvents = configStandalone.NEvents > 0 ? configStandalone.NEvents : 1;
    } else {
      if (nEvents == -1 || nEvents > nEventsInDirectory) {
        if (nEvents >= 0) {
          printf("Only %d events available in directors %s (%d events requested)\n", nEventsInDirectory, configStandalone.EventsDir, nEvents);
        }
        nEvents = nEventsInDirectory;
      }
      if (configStandalone.configTF.nMerge > 1) {
        nEvents /= configStandalone.configTF.nMerge;
      }
    }

    for (int j2 = 0; j2 < configStandalone.runs2; j2++) {
      if (configStandalone.configQA.inputHistogramsOnly) {
        chainTracking->ForceInitQA();
        break;
      }
      if (configStandalone.runs2 > 1) {
        printf("RUN2: %d\n", j2);
      }
      long long int nTracksTotal = 0;
      long long int nClustersTotal = 0;
      int nEventsProcessed = 0;

      for (int iEvent = configStandalone.StartEvent; iEvent < nEvents; iEvent++) {
        if (iEvent != configStandalone.StartEvent) {
          printf("\n");
        }
        HighResTimer timerLoad;
        timerLoad.Start();
        if (configStandalone.configTF.bunchSim) {
          if (tf->LoadCreateTimeFrame(iEvent)) {
            break;
          }
        } else if (configStandalone.configTF.nMerge) {
          if (tf->LoadMergedEvents(iEvent)) {
            break;
          }
        } else {
          if (ReadEvent(iEvent)) {
            break;
          }
        }
        if (!configStandalone.configRec.runTransformation) {
          chainTracking->mIOPtrs.clustersNative = nullptr;
        } else {
          for (int i = 0; i < chainTracking->NSLICES; i++) {
            if (chainTracking->mIOPtrs.rawClusters[i]) {
              if (configStandalone.DebugLevel >= 2) {
                printf("Converting Legacy Raw Cluster to Native\n");
              }
              chainTracking->ConvertRun2RawToNative();
              break;
            }
          }
        }

        if (configStandalone.overrideMaxTimebin && chainTracking->mIOPtrs.clustersNative) {
          GPUSettingsEvent ev = rec->GetEventSettings();
          ev.continuousMaxTimeBin = GPUReconstructionConvert::GetMaxTimeBin(*chainTracking->mIOPtrs.clustersNative);
          rec->UpdateEventSettings(&ev);
        }
        if (!rec->GetParam().earlyTpcTransform && chainTracking->mIOPtrs.clustersNative == nullptr) {
          printf("Need cluster native data for on-the-fly TPC transform\n");
          goto breakrun;
        }

        printf("Loading time: %'d us\n", (int)(1000000 * timerLoad.GetCurrentElapsedTime()));

        printf("Processing Event %d\n", iEvent);
        for (int j1 = 0; j1 < configStandalone.runs; j1++) {
          if (configStandalone.runs > 1) {
            printf("Run %d\n", j1 + 1);
          }
          if (configStandalone.outputcontrolmem) {
            rec->SetOutputControl(outputmemory.get(), configStandalone.outputcontrolmem);
          }
          rec->SetResetTimers(j1 <= configStandalone.runsInit);

          int tmpRetVal = rec->RunChains();

          if (tmpRetVal == 0 || tmpRetVal == 2) {
            int nTracks = 0, nClusters = 0, nAttachedClusters = 0, nAttachedClustersFitted = 0;
            for (int k = 0; k < chainTracking->GetTPCMerger().NOutputTracks(); k++) {
              if (chainTracking->GetTPCMerger().OutputTracks()[k].OK()) {
                nTracks++;
                nAttachedClusters += chainTracking->GetTPCMerger().OutputTracks()[k].NClusters();
                nAttachedClustersFitted += chainTracking->GetTPCMerger().OutputTracks()[k].NClustersFitted();
              }
            }
            nClusters = chainTracking->GetTPCMerger().NClusters();
            printf("Output Tracks: %d (%d/%d attached clusters)\n", nTracks, nAttachedClusters, nAttachedClustersFitted);
            if (j1 == 0) {
              nTracksTotal += nTracks;
              nClustersTotal += nClusters;
              nEventsProcessed++;
            }

            if (chainTracking->GetRecoSteps() & GPUReconstruction::RecoStep::TRDTracking) {
              int nTracklets = 0;
              for (int k = 0; k < chainTracking->GetTRDTracker()->NTracks(); k++) {
                auto& trk = chainTracking->GetTRDTracker()->Tracks()[k];
                nTracklets += trk.GetNtracklets();
              }
              printf("TRD Tracker reconstructed %d tracks (%d tracklets)\n", chainTracking->GetTRDTracker()->NTracks(), nTracklets);
            }
          }
          if (configStandalone.memoryStat) {
            rec->PrintMemoryStatistics();
          }
          rec->ClearAllocatedMemory();

          if (tmpRetVal == 2) {
            configStandalone.continueOnError = 0; // Forced exit from event display loop
            configStandalone.noprompt = 1;
          }
          if (tmpRetVal && !configStandalone.continueOnError) {
            if (tmpRetVal != 2) {
              printf("Error occured\n");
            }
            goto breakrun;
          }
        }
      }
      if (nEventsProcessed > 1) {
        printf("Total: %lld clusters, %lld tracks\n", nClustersTotal, nTracksTotal);
      }
    }
  }
breakrun:

#ifndef _WIN32
  if (configStandalone.qa && configStandalone.fpe) {
    fedisableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  }
#endif

  rec->Finalize();
  rec->Exit();

  if (!configStandalone.noprompt) {
    printf("Press a key to exit!\n");
    getchar();
  }
  return (0);
}
