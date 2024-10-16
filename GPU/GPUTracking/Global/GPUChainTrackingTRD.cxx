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
#include "GPUTRDTrackerKernels.h"
#include "utils/strtag.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::trd;

template <int32_t I>
int32_t GPUChainTracking::RunTRDTracking()
{
#ifndef GPUCA_ALIROOT_LIB
  auto& Tracker = processors()->getTRDTracker<I>();
  if (!Tracker.IsInitialized()) {
    return 1;
  }

  Tracker.Reset();
  if (mIOPtrs.nTRDTracklets == 0) {
    return 0;
  }

  bool isTriggeredEvent = (param().continuousMaxTimeBin == 0);

  if (!isTriggeredEvent) {
    Tracker.SetProcessPerTimeFrame(true);
  }

  Tracker.SetGenerateSpacePoints(mIOPtrs.trdSpacePoints == nullptr);

  mRec->PushNonPersistentMemory(qStr2Tag("TRDTRACK"));
  SetupGPUProcessor(&Tracker, true);

  if constexpr (I == GPUTRDTrackerKernels::gpuVersion) {
    for (uint32_t i = 0; i < mIOPtrs.nMergedTracks; i++) {
      const GPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
      if (!Tracker.PreCheckTrackTRDCandidate(trk)) {
        continue;
      }
      const GPUTRDTrackGPU& trktrd = param().rec.tpc.nWaysOuter ? (GPUTRDTrackGPU)trk.OuterParam() : (GPUTRDTrackGPU)trk;
      if (!Tracker.CheckTrackTRDCandidate(trktrd)) {
        continue;
      }
      GPUTRDTrackerGPU::HelperTrackAttributes trkAttribs, *trkAttribsPtr{nullptr};
      if (!isTriggeredEvent) {
        const float tpcTBinMUS = 0.199606f;
        trkAttribs.mTime = trk.GetParam().GetTZOffset() * tpcTBinMUS;
        trkAttribs.mTimeAddMax = 50.f; // half of a TPC drift time in us
        trkAttribs.mTimeSubMax = 50.f; // half of a TPC drift time in us
        if (!trk.CCE()) {
          if (trk.CSide()) {
            // track has only C-side clusters
            trkAttribs.mSide = 1;
          } else {
            // track has only A-side clusters
            trkAttribs.mSide = -1;
          }
        }
        trkAttribsPtr = &trkAttribs;
      }
      if (Tracker.LoadTrack(trktrd, i, false, trkAttribsPtr)) {
        return 1;
      }
    }
  } else {
#ifdef GPUCA_HAVE_O2HEADERS
    for (uint32_t i = 0; i < mIOPtrs.nOutputTracksTPCO2; i++) {
      const auto& trk = mIOPtrs.outputTracksTPCO2[i];

      if (!Tracker.PreCheckTrackTRDCandidate(trk)) {
        continue;
      }
      const GPUTRDTrack& trktrd = (GPUTRDTrack)trk;
      if (!Tracker.CheckTrackTRDCandidate(trktrd)) {
        continue;
      }

      GPUTRDTracker::HelperTrackAttributes trkAttribs, *trkAttribsPtr{nullptr};
      if (!isTriggeredEvent) {
        const float tpcTBinMUS = 0.199606f;
        trkAttribs.mTime = trk.getTime0() * tpcTBinMUS;
        trkAttribs.mTimeAddMax = trk.getDeltaTFwd() * tpcTBinMUS;
        trkAttribs.mTimeSubMax = trk.getDeltaTBwd() * tpcTBinMUS;
        if (trk.hasASideClustersOnly()) {
          trkAttribs.mSide = -1;
        } else if (trk.hasCSideClustersOnly()) {
          trkAttribs.mSide = 1;
        }
        trkAttribsPtr = &trkAttribs;
      }
      if (Tracker.LoadTrack(trktrd, i, false, trkAttribsPtr)) {
        return 1;
      }
    }
#endif
  }

  DoTRDGPUTracking<I>();

  mIOPtrs.nTRDTracks = Tracker.NTracks();
  if constexpr (I == GPUTRDTrackerKernels::gpuVersion) {
    mIOPtrs.trdTracks = Tracker.Tracks();
    mIOPtrs.trdTracksO2 = nullptr;
  } else {
#ifdef GPUCA_HAVE_O2HEADERS
    mIOPtrs.trdTracks = nullptr;
    mIOPtrs.trdTracksO2 = Tracker.Tracks();
#endif
  }
  mRec->PopNonPersistentMemory(RecoStep::TRDTracking, qStr2Tag("TRDTRACK"));

#endif // GPUCA_ALIROOT_LIB
  return 0;
}

template <int32_t I, class T>
int32_t GPUChainTracking::DoTRDGPUTracking(T* externalInstance)
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

  int32_t useStream = 0;

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
    WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->getTRDTracker<I>() - (char*)processors(), TrackerShadow, sizeof(*TrackerShadow), useStream);
  }

  TransferMemoryResourcesToGPU(RecoStep::TRDTracking, Tracker, useStream);
  runKernel<GPUTRDTrackerKernels, I>(GetGridAuto(useStream), externalInstance ? Tracker : nullptr);
  TransferMemoryResourcesToHost(RecoStep::TRDTracking, Tracker, useStream);
  SynchronizeStream(useStream);

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU TRD tracker Finished");
  }
#endif
  return (0);
}

template int32_t GPUChainTracking::RunTRDTracking<GPUTRDTrackerKernels::gpuVersion>();
template int32_t GPUChainTracking::DoTRDGPUTracking<GPUTRDTrackerKernels::gpuVersion>(GPUTRDTrackerGPU*);
template int32_t GPUChainTracking::DoTRDGPUTracking<GPUTRDTrackerKernels::gpuVersion>(GPUTRDTracker*);
template int32_t GPUChainTracking::RunTRDTracking<GPUTRDTrackerKernels::o2Version>();
template int32_t GPUChainTracking::DoTRDGPUTracking<GPUTRDTrackerKernels::o2Version>(GPUTRDTracker*);
template int32_t GPUChainTracking::DoTRDGPUTracking<GPUTRDTrackerKernels::o2Version>(GPUTRDTrackerGPU*);
