// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2Interface.cxx
/// \author David Rohr

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCFastTransform.h"
#include <iostream>
#include <fstream>
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

using namespace GPUCA_NAMESPACE::gpu;

#ifdef BUILD_EVENT_DISPLAY
#include "GPUDisplayBackendGlfw.h"
#else
#include "GPUDisplayBackend.h"
namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackendGlfw : public GPUDisplayBackend
{
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif

#include "DataFormatsTPC/ClusterNative.h"
#include "ClusterNativeAccessExt.h"

GPUTPCO2Interface::GPUTPCO2Interface() = default;

GPUTPCO2Interface::~GPUTPCO2Interface() { Deinitialize(); }

int GPUTPCO2Interface::Initialize(const GPUO2InterfaceConfiguration& config, std::unique_ptr<TPCFastTransform>&& fastTrans)
{
  if (mInitialized) {
    return (1);
  }
  mConfig.reset(new GPUO2InterfaceConfiguration(config));
  mDumpEvents = mConfig->configInterface.dumpEvents;
  mContinuous = mConfig->configEvent.continuousMaxTimeBin != 0;
  mRec.reset(GPUReconstruction::CreateInstance(mConfig->configProcessing));
  mChain = mRec->AddChain<GPUChainTracking>();
  mChain->mConfigDisplay = &mConfig->configDisplay;
  mChain->mConfigQA = &mConfig->configQA;
  mRec->SetSettings(&mConfig->configEvent, &mConfig->configReconstruction, &mConfig->configDeviceProcessing);
  mChain->SetTPCFastTransform(std::move(fastTrans));
  if (mRec->Init()) {
    return (1);
  }
  mInitialized = true;
  return (0);
}

int GPUTPCO2Interface::Initialize(const char* options, std::unique_ptr<TPCFastTransform>&& fastTrans)
{
  if (mInitialized) {
    return (1);
  }
  float solenoidBz = -5.00668;
  float refX = 1000.;
  int nThreads = 1;
  bool useGPU = false;
  char gpuType[1024];

  if (options && *options) {
    printf("Received options %s\n", options);
    const char* optPtr = options;
    while (optPtr && *optPtr) {
      while (*optPtr == ' ') {
        optPtr++;
      }
      const char* nextPtr = strstr(optPtr, " ");
      const int optLen = nextPtr ? nextPtr - optPtr : strlen(optPtr);
      if (strncmp(optPtr, "cont", optLen) == 0) {
        mContinuous = true;
        printf("Continuous tracking mode enabled\n");
      } else if (strncmp(optPtr, "dump", optLen) == 0) {
        mDumpEvents = true;
        printf("Dumping of input events enabled\n");
      }
#ifdef BUILD_EVENT_DISPLAY
      else if (strncmp(optPtr, "display", optLen) == 0) {
        mDisplayBackend.reset(new GPUDisplayBackendGlfw);
        printf("Event display enabled\n");
      }
#endif
      else if (optLen > 3 && strncmp(optPtr, "bz=", 3) == 0) {
        sscanf(optPtr + 3, "%f", &solenoidBz);
        printf("Using solenoid field %f\n", solenoidBz);
      } else if (optLen > 5 && strncmp(optPtr, "refX=", 5) == 0) {
        sscanf(optPtr + 5, "%f", &refX);
        printf("Propagating to reference X %f\n", refX);
      } else if (optLen > 8 && strncmp(optPtr, "threads=", 8) == 0) {
        sscanf(optPtr + 8, "%d", &nThreads);
        printf("Using %d threads\n", nThreads);
      } else if (optLen > 8 && strncmp(optPtr, "gpuType=", 8) == 0) {
        int len = std::min(optLen - 8, 1023);
        memcpy(gpuType, optPtr + 8, len);
        gpuType[len] = 0;
        useGPU = true;
        printf("Using GPU Type %s\n", gpuType);
      } else {
        printf("Unknown option: %s\n", optPtr);
        return 1;
      }
      optPtr = nextPtr;
    }
  }

#ifdef GPUCA_HAVE_OPENMP
  omp_set_num_threads(nThreads);
#else
  if (nThreads != 1) {
    printf("ERROR: Compiled without OpenMP. Cannot set number of threads!\n");
  }

#endif
  mRec.reset(GPUReconstruction::CreateInstance(useGPU ? gpuType : "CPU", true));
  mChain = mRec->AddChain<GPUChainTracking>();
  if (mRec == nullptr) {
    return 1;
  }

  GPUSettingsRec rec;
  GPUSettingsEvent ev;
  GPUSettingsDeviceProcessing devProc;

  rec.SetDefaults();
  ev.SetDefaults();
  devProc.SetDefaults();

  ev.solenoidBz = solenoidBz;
  ev.continuousMaxTimeBin = mContinuous ? 0.023 * 5e6 : 0;

  rec.NWays = 3;
  rec.NWaysOuter = true;
  rec.SearchWindowDZDR = 2.5f;
  rec.TrackReferenceX = refX;

  devProc.eventDisplay = mDisplayBackend.get();

  mRec->SetSettings(&ev, &rec, &devProc);
  mChain->SetTPCFastTransform(std::move(fastTrans));
  if (mRec->Init()) {
    return 1;
  }

  mInitialized = true;
  return (0);
}

void GPUTPCO2Interface::Deinitialize()
{
  if (mInitialized) {
    mRec->Finalize();
    mRec.reset();
  }
  mInitialized = false;
}

int GPUTPCO2Interface::RunTracking(const o2::TPC::ClusterNativeAccessFullTPC* inputClusters, const GPUTPCGMMergedTrack*& outputTracks, int& nOutputTracks, const GPUTPCGMMergedTrackHit*& outputTrackClusters)
{
  if (!mInitialized) {
    return (1);
  }
  static int nEvent = 0;
  if (mDumpEvents) {
    mChain->ClearIOPointers();
    mChain->mIOPtrs.clustersNative = inputClusters;

    char fname[1024];
    sprintf(fname, "event.%d.dump", nEvent);
    mChain->DumpData(fname);
    if (nEvent == 0) {
      mRec->DumpSettings();
    }
  }

  mChain->mIOPtrs.clustersNative = inputClusters;
  mChain->ConvertNativeToClusterData();
  mChain->RunStandalone();

  outputTracks = mChain->mIOPtrs.mergedTracks;
  nOutputTracks = mChain->mIOPtrs.nMergedTracks;
  outputTrackClusters = mChain->mIOPtrs.mergedTrackHits;
  const ClusterNativeAccessExt* ext = mChain->GetClusterNativeAccessExt();
  for (int i = 0; i < mChain->mIOPtrs.nMergedTrackHits; i++) {
    GPUTPCGMMergedTrackHit& cl = (GPUTPCGMMergedTrackHit&)mChain->mIOPtrs.mergedTrackHits[i];
    cl.num -= ext->clusterOffset[cl.slice][cl.row];
  }
  nEvent++;
  return (0);
}

void GPUTPCO2Interface::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  if (!mInitialized) {
    return;
  }
  mRec->GetParam().GetClusterErrors2(row, z, sinPhi, DzDs, ErrY2, ErrZ2);
}

void GPUTPCO2Interface::Cleanup() {}
