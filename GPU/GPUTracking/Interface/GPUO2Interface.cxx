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

using namespace o2::gpu;

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

void GPUTPCO2Interface::Deinitialize()
{
  if (mInitialized) {
    mRec->Finalize();
    mRec.reset();
  }
  mInitialized = false;
}

int GPUTPCO2Interface::RunTracking(const o2::tpc::ClusterNativeAccessFullTPC* inputClusters, const GPUTPCGMMergedTrack*& outputTracks, int& nOutputTracks, const GPUTPCGMMergedTrackHit*& outputTrackClusters)
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
  mRec->RunChains();

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
