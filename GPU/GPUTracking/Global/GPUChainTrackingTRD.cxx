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

/// \file GPUChainTrackingTRD.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "GPUTrackingInputProvider.h"
#include "utils/strtag.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::trd;

int GPUChainTracking::RunTRDTracking()
{
  if (!processors()->trdTrackerGPU.IsInitialized()) {
    return 1;
  }

  GPUTRDTrackerGPU& Tracker = processors()->trdTrackerGPU;
  Tracker.Reset();
  if (mIOPtrs.nTRDTracklets == 0) {
    return 0;
  }
  Tracker.SetGenerateSpacePoints(mIOPtrs.trdSpacePoints == nullptr);

  mRec->PushNonPersistentMemory(qStr2Tag("TRDTRACK"));
  SetupGPUProcessor(&Tracker, true);

  for (unsigned int i = 0; i < mIOPtrs.nMergedTracks; i++) {
    const GPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
    if (!Tracker.PreCheckTrackTRDCandidate(trk)) {
      continue;
    }
    const GPUTRDTrackGPU& trktrd = param().rec.tpc.nWaysOuter ? (GPUTRDTrackGPU)trk.OuterParam() : (GPUTRDTrackGPU)trk;
    if (!Tracker.CheckTrackTRDCandidate(trktrd)) {
      continue;
    }

    if (Tracker.LoadTrack(trktrd, i, false)) {
      return 1;
    }
  }

  DoTRDGPUTracking<GPUTRDTrackerKernels::gpuVersion>();

  mIOPtrs.nTRDTracks = Tracker.NTracks();
  mIOPtrs.trdTracks = Tracker.Tracks();
  mRec->PopNonPersistentMemory(RecoStep::TRDTracking, qStr2Tag("TRDTRACK"));

  return 0;
}

template <int I>
int GPUChainTracking::DoTRDGPUTracking(GPUTRDTracker* externalInstance)
{
#ifdef GPUCA_HAVE_O2HEADERS
  bool doGPU = GetRecoStepsGPU() & RecoStep::TRDTracking;
  auto* Tracker = &processors()->getTRDTracker<I>();
  auto* TrackerShadow = doGPU ? &processorsShadow()->getTRDTracker<I>() : Tracker;
  if (externalInstance) {
    if constexpr (std::is_same_v<decltype(Tracker), decltype(externalInstance)>) {
      Tracker = externalInstance;
    } else {
      throw std::runtime_error("Must not provide external instance that does not match template type");
    }
  }
  Tracker->PrepareTracking(this);

  int useStream = 0;

  const auto& threadContext = GetThreadContext();
  SetupGPUProcessor(Tracker, false);
  if (doGPU) {
    TrackerShadow->OverrideGPUGeometry(reinterpret_cast<GPUTRDGeometry*>(mFlatObjectsDevice.mCalibObjects.trdGeometry));
    mInputsHost->mNTRDTracklets = mInputsShadow->mNTRDTracklets = processorsShadow()->ioPtrs.nTRDTracklets = mIOPtrs.nTRDTracklets;
    mInputsHost->mNTRDTriggerRecords = mInputsShadow->mNTRDTriggerRecords = processorsShadow()->ioPtrs.nTRDTriggerRecords = mIOPtrs.nTRDTriggerRecords;
    mInputsHost->mDoSpacepoints = mInputsShadow->mDoSpacepoints = !Tracker->GenerateSpacepoints();
    AllocateRegisteredMemory(mInputsHost->mResourceTRD);
    processorsShadow()->ioPtrs.trdTracklets = mInputsShadow->mTRDTracklets;
    processorsShadow()->ioPtrs.trdSpacePoints = Tracker->GenerateSpacepoints() ? Tracker->SpacePoints() : mInputsShadow->mTRDSpacePoints;
    if constexpr (std::is_same_v<decltype(processorsShadow()->ioPtrs.trdTracks), decltype(TrackerShadow->Tracks())>) {
      processorsShadow()->ioPtrs.trdTracks = TrackerShadow->Tracks();
    } else {
      processorsShadow()->ioPtrs.trdTracks = nullptr;
    }
    processorsShadow()->ioPtrs.nTRDTracks = mIOPtrs.nTRDTracks;
    processorsShadow()->ioPtrs.trdTriggerTimes = mInputsShadow->mTRDTriggerTimes;
    processorsShadow()->ioPtrs.trdTrackletIdxFirst = mInputsShadow->mTRDTrackletIdxFirst;
    GPUMemCpy(RecoStep::TRDTracking, mInputsShadow->mTRDTracklets, mIOPtrs.trdTracklets, sizeof(*mIOPtrs.trdTracklets) * mIOPtrs.nTRDTracklets, useStream, true);
    if (!Tracker->GenerateSpacepoints()) {
      GPUMemCpy(RecoStep::TRDTracking, mInputsShadow->mTRDSpacePoints, mIOPtrs.trdSpacePoints, sizeof(*mIOPtrs.trdSpacePoints) * mIOPtrs.nTRDTracklets, useStream, true);
    }
    GPUMemCpy(RecoStep::TRDTracking, mInputsShadow->mTRDTriggerTimes, mIOPtrs.trdTriggerTimes, sizeof(*mIOPtrs.trdTriggerTimes) * mIOPtrs.nTRDTriggerRecords, useStream, true);
    GPUMemCpy(RecoStep::TRDTracking, mInputsShadow->mTRDTrackletIdxFirst, mIOPtrs.trdTrackletIdxFirst, sizeof(*mIOPtrs.trdTrackletIdxFirst) * mIOPtrs.nTRDTriggerRecords, useStream, true);
    if (mIOPtrs.trdTrigRecMask) {
      processorsShadow()->ioPtrs.trdTrigRecMask = mInputsShadow->mTRDTrigRecMask;
      GPUMemCpy(RecoStep::TRDTracking, mInputsShadow->mTRDTrigRecMask, mIOPtrs.trdTrigRecMask, sizeof(*mIOPtrs.trdTrigRecMask) * mIOPtrs.nTRDTriggerRecords, useStream, true);
    } else {
      processorsShadow()->ioPtrs.trdTrigRecMask = nullptr;
    }
    WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), useStream);
    WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->trdTrackerGPU - (char*)processors(), TrackerShadow, sizeof(*TrackerShadow), useStream);
  }

  TransferMemoryResourcesToGPU(RecoStep::TRDTracking, Tracker, useStream);
  runKernel<GPUTRDTrackerKernels, I>(GetGridAuto(useStream), krnlRunRangeNone, krnlEventNone, externalInstance);
  TransferMemoryResourcesToHost(RecoStep::TRDTracking, Tracker, useStream);
  SynchronizeStream(useStream);

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU TRD tracker Finished");
  }
#endif
  return (0);
}

template int GPUChainTracking::DoTRDGPUTracking<0>(GPUTRDTracker*);
template int GPUChainTracking::DoTRDGPUTracking<1>(GPUTRDTracker*);
