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

/// \file GPUChainTrackingMerger.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUChainTrackingDebug.h"
#include "GPULogging.h"
#include "GPUDefParametersRuntime.h"
#include "GPUO2DataTypes.h"
#include "GPUQA.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCGMMergerRebuildInternal.h"
#include "GPUConstantMem.h"
#include "GPUTPCGMMergerGPU.h"
#include "GPUTPCGMO2Output.h"
#include "GPUTPCGlobalDebugSortKernels.h"
#include "utils/strtag.h"
#include <fstream>

using namespace o2::gpu;

void GPUChainTracking::RunTPCTrackingMerger_MergeBorderTracks(uint8_t mergeMode, GPUReconstruction::krnlDeviceType deviceType)
{
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    uint32_t nBorderTracks = (mergeMode & GPUTPCGMMerger::mergeModes::mergeWithinSector) ? NSECTORS : (2 * NSECTORS);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::borderTracks>({{nBorderTracks, -WarpSize(), 0, deviceType}}, 0);
  }
  uint32_t n = (mergeMode & GPUTPCGMMerger::mergeModes::mergeAcrossCE) ? NSECTORS / 2 : NSECTORS;
  if (GetProcessingSettings().alternateBorderSort == -1 ? mRec->getGPUParameters(doGPU).par_ALTERNATE_BORDER_SORT : GetProcessingSettings().alternateBorderSort) {
    RecordMarker(&mEvents->single, 0);
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->init);
    for (uint32_t i = 0; i < n; i++) {
      int32_t stream = i % mRec->NStreams();
      runKernel<GPUTPCGMMergerMergeBorders, 0>({GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, stream && i < (uint32_t)mRec->NStreams() ? &mEvents->single : nullptr}}, i, mergeMode);
    }
    ReleaseEvent(mEvents->single);
    SynchronizeEventAndRelease(mEvents->init);
    for (uint32_t i = 0; i < n; i++) {
      int32_t stream = i % mRec->NStreams();
      int32_t n1, n2;
      GPUTPCGMBorderTrack *b1, *b2;
      int32_t jSector;
      Merger.MergeBorderTracksSetup(n1, n2, b1, b2, jSector, i, mergeMode);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = MergerShadow.BorderRange(i);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = MergerShadow.BorderRange(jSector) + *processors()->tpcTrackers[jSector].NTracks();
      runKernel<GPUTPCGMMergerMergeBorders, 3>({{1, -WarpSize(), stream, deviceType}}, range1, n1, 0);
      runKernel<GPUTPCGMMergerMergeBorders, 3>({{1, -WarpSize(), stream, deviceType}}, range2, n2, 1);
      runKernel<GPUTPCGMMergerMergeBorders, 2>({GetGridAuto(stream, deviceType)}, i, mergeMode);
    }
    int32_t ne = std::min<int32_t>(n, mRec->NStreams()) - 1; // Stream 0 must wait for all streams, Note n > 1
    for (int32_t j = 0; j < ne; j++) {
      RecordMarker(&mEvents->sector[j], j + 1);
    }
    StreamWaitForEvents(0, &mEvents->sector[0], ne);
  } else if (doGPU && !GetProcessingSettings().rtc.enable) {
    GPUFatal("GPU RTC requires alternateBorderSort!");
  } else {
    for (uint32_t i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(0, deviceType), i, mergeMode);
    }
    runKernel<GPUTPCGMMergerMergeBorders, 1>({{2 * n, -WarpSize(), 0, deviceType}}, 0, mergeMode);
    for (uint32_t i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(0, deviceType), i, mergeMode);
    }
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRanges, doGPU, Merger, &GPUTPCGMMerger::DumpMergeRanges, *mDebugFile, mergeMode);
  mRec->ReturnVolatileDeviceMemory();
}

void GPUChainTracking::RunTPCTrackingMerger_Resolve(int8_t useOrigTrackParam, int8_t mergeAll, GPUReconstruction::krnlDeviceType deviceType)
{
  runKernel<GPUTPCGMMergerResolve, 0>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerResolve, 1>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerResolve, 2>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerResolve, 3>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerResolve, 4>(GetGridAuto(0, deviceType), useOrigTrackParam, mergeAll);
}

int32_t GPUChainTracking::RunTPCTrackingMerger(bool synchronizeOutput)
{
  mRec->PushNonPersistentMemory(qStr2Tag("TPCMERGE"));
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  GPUReconstruction::krnlDeviceType deviceType = doGPU ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  uint32_t numBlocks = (!mRec->IsGPU() || doGPU) ? BlockCount() : 1;
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  const int32_t outputStream = OutputStream();
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& holdContext = GetThreadContext();

  SynchronizeGPU(); // Need to know the full number of sector tracks
  SetupGPUProcessor(&Merger, true);
  AllocateRegisteredMemory(Merger.MemoryResOutput(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracks)]);
  AllocateRegisteredMemory(Merger.MemoryResOutputState(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::sharedClusterMap)]);

  if (Merger.CheckSectors()) {
    return 1;
  }

  memset(Merger.Memory(), 0, sizeof(*Merger.Memory()));
  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  if (doGPU) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  if (GetProcessingSettings().deterministicGPUReconstruction) {
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::clearIds>(GetGridAuto(0, deviceType), 1);
  }
  const uint32_t nStreams = doGPU ? std::min<uint32_t>(mRec->NStreams(), NSECTORS) : 1;
  auto recordStreamMarkers = [&]() {
    for (uint32_t i = 0; i < nStreams; i++) {
      RecordMarker(&mEvents->stream[i], i);
    }
  };
  auto releaseStreamMarkers = [&]() {
    for (uint32_t i = 0; i < nStreams; i++) {
      ReleaseEvent(mEvents->stream[i]);
    }
  };
  RecordMarker(&mEvents->single, 0);
  for (uint32_t i = 0; i < NSECTORS; i++) {
    const int32_t stream = i % nStreams;
    runKernel<GPUTPCGMMergerUnpackResetIds>({GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, i < nStreams ? &mEvents->single : nullptr}}, i);
    runKernel<GPUTPCGMMergerSectorRefit>(GetGridAuto(stream, deviceType), i);
  }
  ReleaseEvent(mEvents->single);
  recordStreamMarkers();
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    SynchronizeGPU();
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks1>({{GPUTPCGeometry::NSECTORS, -WarpSize(), 0, deviceType}}, 0);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks2>({{1, -WarpSize(), 0, deviceType}}, 0);
    SynchronizeGPU();
  }
  for (uint32_t i = 0; i < NSECTORS; i++) {
    const int32_t stream = i % nStreams;
    runKernel<GPUTPCGMMergerUnpackGlobal>({GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, i < nStreams ? mEvents->stream : nullptr, (int32_t)nStreams}}, i);
  }
  releaseStreamMarkers();
  recordStreamMarkers();
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    SynchronizeGPU();
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks1>({{GPUTPCGeometry::NSECTORS, -WarpSize(), 0, deviceType}}, 1);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks2>({{1, -WarpSize(), 0, deviceType}}, 1);
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingSectorTracks, doGPU, Merger, &GPUTPCGMMerger::DumpSectorTracks, *mDebugFile);

  // Merge within Sectors
  runKernel<GPUTPCGMMergerClearLinks>({GetGridAuto(0, deviceType), krnlRunRangeNone, {nullptr, mEvents->stream, (int32_t)nStreams}}, false);
  releaseStreamMarkers();
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TmpCounter(), NSECTORS * sizeof(*MergerShadow.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeWithinPrepare>(GetGridAuto(0, deviceType));
  RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeWithinSector, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingMatching, doGPU, Merger, &GPUTPCGMMerger::DumpMergedWithinSectors, *mDebugFile);

  // Merge between sectors - transport to the middle of the sector and rotate vertically to the border on the left / right
  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), false);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadow.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 2, 3, 0);
  RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeBetweenSector | GPUTPCGMMerger::mergeModes::mergeAtMidRow, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadow.TmpCounter()));
  // Merge between sectors - transport to the left / right edge of the sector and rotate horizontally
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 0, 1, 0);
  RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeBetweenSector, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadow.TmpCounter()));
  // Merge between sectors - use original track param
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 0, 1, 1);
  RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeBetweenSector | GPUTPCGMMerger::mergeModes::mergeWithOriginalParameters, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingMatching, doGPU, Merger, &GPUTPCGMMerger::DumpMergedBetweenSectors, *mDebugFile);

  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadow.TmpCounter()));

  runKernel<GPUTPCGMMergerLinkExtrapolatedTracks>(GetGridAuto(0, deviceType));
  if (GetProcessingSettings().mergerSanityCheck) {
    Merger.CheckMergeGraph();
  }
  runKernel<GPUTPCGMMergerCollect>(GetGridAuto(0, deviceType));
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::mergedTracks1>({{1, -WarpSize(), 0, deviceType}}, 1);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::mergedTracks2>({{1, -WarpSize(), 0, deviceType}}, 1);
  }
  if (!doGPU && GetProcessingSettings().debug.mergerMCLabels) {
    Merger.CreateMCLabels(1, 1, 0, 0);
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingCollectedTracks, doGPU, Merger, &GPUTPCGMMerger::DumpCollected, *mDebugFile);

  if (param().rec.tpc.mergeCE) {
    runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), true);
    // Merge across CE - compare at row 63
    RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeAcrossCE, deviceType);
    // Merge across CE - compare at row of cluster
    RunTPCTrackingMerger_MergeBorderTracks(GPUTPCGMMerger::mergeModes::mergeAcrossCE | GPUTPCGMMerger::mergeModes::mergeAtCluster, deviceType);
    runKernel<GPUTPCGMMergerMergeCE>(GetGridAuto(0, deviceType));
    DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingCE, doGPU, Merger, &GPUTPCGMMerger::DumpMergeCE, *mDebugFile);
  }
  int32_t waitForTransfer = 0;
  if (doGPU) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    waitForTransfer = 1;
  }

  const bool mergerSortTracks = mRec->getGPUParameters(doGPU).par_SORT_BEFORE_FIT;
  if (mergerSortTracks) {
    runKernel<GPUTPCGMMergerSortTracksPrepare>(GetGridAuto(0, deviceType));
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracks>(GetGridAuto(0, deviceType));
  }
  if (GetProcessingSettings().mergerSanityCheck) {
    Merger.CheckCollectedTracks();
  }

  if (param().rec.tpc.rebuildTrackInFit) {
    mRec->PushNonPersistentMemory(qStr2Tag("TPCREBLD")); // The rebuild memory is freed after the fit, before the O2 output
  }
  const int32_t rebuildConstrainedRows = param().rec.tpc.rebuildTrackInFit ? mRec->getGPUParameters(doGPU).par_TPC_MERGER_REBUILD_CONSTRAINED_ROW_MEMORY : 0; // 1: row ranges estimated before the fit, 2: filled row ranges stored by the fit (host only)
  if (rebuildConstrainedRows == 1) {
    AllocateRegisteredMemory(Merger.MemoryResRebuild2());
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  }
  runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.SharedCount(), Merger.NClusters() * sizeof(*MergerShadow.SharedCount()));
  runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.ClusterAttachment(), Merger.NClusters() * sizeof(*MergerShadow.ClusterAttachment()));
  runKernel<GPUTPCGMMergerPrepareForFit, 0>(GetGridAuto(0, deviceType));
  CondWaitEvent(waitForTransfer, &mEvents->single);
  runKernel<GPUTPCGMMergerSortTracksQPt>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerPrepareForFit, 1>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerPrepareForFit, 2>(GetGridAuto(0, deviceType));
  if (doGPU && rebuildConstrainedRows == 1) { // PrepareForFit1 computed nRebuildTrackRow, needed on the host to allocate the rebuild memory, waited for below
    ReleaseEvent(mEvents->single);
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    waitForTransfer = 1;
  }

  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingPrepareFit, doGPU, Merger, &GPUTPCGMMerger::DumpFitPrepare, *mDebugFile);

  if (doGPU) {
    CondWaitEvent(waitForTransfer, &mEvents->single);
    if (waitForTransfer) {
      ReleaseEvent(mEvents->single);
    }
  } else if (doGPU) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  if (GetProcessingSettings().delayedOutput) {
    for (uint32_t i = 0; i < mOutputQueue.size(); i++) {
      GPUMemCpy(mOutputQueue[i].step, mOutputQueue[i].dst, mOutputQueue[i].src, mOutputQueue[i].size, outputStream, false);
    }
    mOutputQueue.clear();
  }

  if (param().rec.tpc.rebuildTrackInFit) {
    AllocateRegisteredMemory(Merger.MemoryResRebuild()); // With constrained rows sized by nRebuildTrackRow, which was transferred to the host above
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
    if (rebuildConstrainedRows != 2) { // 2: no cleaning, the fit stores complete rows and untouched memory shall not become physical
      const size_t nRebuildRows = rebuildConstrainedRows == 1 ? (size_t)Merger.Memory()->nRebuildTrackRow : (size_t)Merger.NMergedTracks() * GPUTPCGeometry::NROWS;
      runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.ClusterCandidates(), nRebuildRows * param().rec.tpc.rebuildTrackInFitClusterCandidates * sizeof(*MergerShadow.ClusterCandidates()));
      runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.TrackRebuildHelperRow(), nRebuildRows * sizeof(*MergerShadow.TrackRebuildHelperRow()));
    }
  }
  // mRec->SetNActiveThreads(1); // For GPUTPCGMTrackParam:: GPUCA_CADEBUG: run the host kernels from here on with 1 thread, reset below
  runKernel<GPUTPCGMMergerTrackFit, GPUTPCGMMergerTrackFit::def>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0));
  if (param().rec.tpc.rebuildTrackInFit) {
    runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadow.HitWeights(), Merger.NClusters() * sizeof(*MergerShadow.HitWeights()));
    runKernel<GPUTPCGMMergerHitWeights, GPUTPCGMMergerHitWeights::prepare>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), 0);
    for (int32_t i = 0; i < param().rec.tpc.rebuildTrackInFitClusterCandidates; i++) {
      runKernel<GPUTPCGMMergerHitWeights, GPUTPCGMMergerHitWeights::compute>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), i);
      if (i + 1 < param().rec.tpc.rebuildTrackInFitClusterCandidates) {
        runKernel<GPUTPCGMMergerHitWeights, GPUTPCGMMergerHitWeights::resolve1>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), i);
      }
    }
    runKernel<GPUTPCGMMergerHitWeights, GPUTPCGMMergerHitWeights::resolve2>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), 0);
    DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRefit, Merger, &GPUTPCGMMerger::DumpInterpolatedHits, *mDebugFile);
    runKernel<GPUTPCGMMergerHitWeights, GPUTPCGMMergerHitWeights::resolveShared>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), 0);
    DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRefit, Merger, &GPUTPCGMMerger::DumpRebuiltTracks, *mDebugFile);
    runKernel<GPUTPCGMMergerTrackFit, GPUTPCGMMergerTrackFit::rebuild>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0));
  }
  if (param().rec.tpc.retryRefit) {
    runKernel<GPUTPCGMMergerTrackFit, GPUTPCGMMergerTrackFit::retry>(GetGridAuto(0));
  }
  if (param().rec.tpc.rebuildTrackInFit) {
    mRec->PopNonPersistentMemory(RecoStep::TPCMerging, qStr2Tag("TPCREBLD"));
  }

  const bool followExtrapolation = GPUTPCGMMerger::FollowExtrapolationEnabled(param());
  const bool followExtrapolationStream = followExtrapolation && doGPU && mRec->NStreams() >= 2;
  if (followExtrapolationStream) {
    RecordMarker(&mEvents->single, 0);
  }
  runKernel<GPUTPCGMMergerFollowLoopers>(GetGridAuto(0));
  if (followExtrapolation) {
    runKernel<GPUTPCGMMergerFollowExtrapolation>({GetGridAuto(followExtrapolationStream ? 1 : 0), krnlRunRangeNone, {followExtrapolationStream ? &mEvents->init : nullptr, followExtrapolationStream ? &mEvents->single : nullptr}});
  }

  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRefit, Merger, &GPUTPCGMMerger::DumpRefit, *mDebugFile);
  runKernel<GPUTPCGMMergerFinalize, 0>({GetGridAuto(0, deviceType), krnlRunRangeNone, {nullptr, followExtrapolationStream ? &mEvents->init : nullptr}});
  if (followExtrapolationStream) {
    ReleaseEvent(mEvents->single);
    ReleaseEvent(mEvents->init);
  }
  runKernel<GPUTPCGMMergerFinalize, 1>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerFinalize, 2>(GetGridAuto(0, deviceType));
  if (param().rec.tpc.mergeLoopersAfterburner) {
    runKernel<GPUTPCGMMergerMergeLoopers, 0>(doGPU ? GetGrid(Merger.NMergedTracks(), 0, deviceType) : GetGridAuto(0, deviceType));
  }
  if (doGPU) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0);
    SynchronizeStream(0); // TODO: could probably synchronize on an event after runKernel<GPUTPCGMMergerMergeLoopers, 1>
  }
  if (param().rec.tpc.mergeLoopersAfterburner) {
    runKernel<GPUTPCGMMergerMergeLoopers, 1>(GetGridAuto(0, deviceType));
    runKernel<GPUTPCGMMergerMergeLoopers, 2>(doGPU ? GetGrid(Merger.Memory()->nLooperMatchCandidates, 0, deviceType) : GetGridAuto(0, deviceType));
    DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingLoopers, Merger, &GPUTPCGMMerger::DumpLoopers, *mDebugFile);
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRefit, doGPU, Merger, &GPUTPCGMMerger::DumpFinal, *mDebugFile);

  if (doGPU) {
    RecordMarker(&mEvents->single, 0);
    auto* waitEvent = &mEvents->single;
    if (GetProcessingSettings().keepDisplayMemory || GetProcessingSettings().createO2Output <= 1 || mFractionalQAEnabled) {
      if (!(GetProcessingSettings().keepDisplayMemory || GetProcessingSettings().createO2Output <= 1)) {
        size_t size = mRec->Res(Merger.MemoryResOutput()).Size() + constants::GPU_MEMALIGN;
        void* buffer = GetQA()->AllocateScratchBuffer(size);
        void* bufferEnd = Merger.SetPointersOutput(buffer);
        if ((size_t)((char*)bufferEnd - (char*)buffer) > size) {
          throw std::runtime_error("QA Scratch buffer exceeded");
        }
      }
      GPUMemCpy(RecoStep::TPCMerging, Merger.MergedTracks(), MergerShadow.MergedTracks(), Merger.NMergedTracks() * sizeof(*Merger.MergedTracks()), outputStream, 0, nullptr, waitEvent);
      waitEvent = nullptr;
      if (param().dodEdxEnabled) {
        GPUMemCpy(RecoStep::TPCMerging, Merger.MergedTracksdEdx(), MergerShadow.MergedTracksdEdx(), Merger.NMergedTracks() * sizeof(*Merger.MergedTracksdEdx()), outputStream, 0);
      }
      GPUMemCpy(RecoStep::TPCMerging, Merger.Clusters(), MergerShadow.Clusters(), Merger.NMergedTrackClusters() * sizeof(*Merger.Clusters()), outputStream, 0);
      GPUMemCpy(RecoStep::TPCMerging, Merger.ClusterAttachment(), MergerShadow.ClusterAttachment(), Merger.NClusters() * sizeof(*Merger.ClusterAttachment()), outputStream, 0);
    }
    if (GetProcessingSettings().outputSharedClusterMap) {
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutputState(), outputStream, nullptr, waitEvent);
      waitEvent = nullptr;
    }
    ReleaseEvent(mEvents->single);
  } else {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }
  if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
  }
  DoDebugMergerNClDump(Merger);
  // mRec->SetNActiveThreads(-1);

  mRec->ReturnVolatileDeviceMemory();
  mRec->PopNonPersistentMemory(RecoStep::TPCMerging, qStr2Tag("TPCMERGE"));

#ifndef GPUCA_RUN2
  if (GetProcessingSettings().createO2Output) {
    if (mTPCSectorScratchOnStack) {
      mRec->PopNonPersistentMemory(RecoStep::TPCSectorTracking, qStr2Tag("TPCSLCD1")); // Return the sector data memory early
      mTPCSectorScratchOnStack = false;
    }

    mRec->PushNonPersistentMemory(qStr2Tag("TPCMERG2"));
    AllocateRegisteredMemory(Merger.MemoryResOutputO2Scratch());
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
    if (!GetProcessingSettings().tpcWriteClustersAfterRejection) {
      runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::prepare>(GetGridAuto(0, deviceType));
    }
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    SynchronizeEventAndRelease(mEvents->single, doGPU); // The CUDA / HIP specialization of the sort reads nO2Tracks on the host
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(GetGridAuto(0, deviceType));
    mRec->ReturnVolatileDeviceMemory();

    if (GetProcessingSettings().clearO2OutputFromGPU) {
      mRec->MakeFutureDeviceMemoryAllocationsVolatile();
    }
    AllocateRegisteredMemory(Merger.MemoryResOutputO2(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2)]);
    AllocateRegisteredMemory(Merger.MemoryResOutputO2Clus(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2ClusRefs)]);
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::output>(GetGridAuto(0, deviceType));

    if (GetProcessingSettings().runMC && mIOPtrs.clustersNative && mIOPtrs.clustersNative->clustersMCTruth) {
      AllocateRegisteredMemory(Merger.MemoryResOutputO2MC(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2Labels)]);
      TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
      runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::mc>(GetGridAuto(0, GPUReconstruction::krnlDeviceType::CPU));
    } else if (doGPU) {
      RecordMarker(&mEvents->single, 0);
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutputO2(), outputStream, nullptr, &mEvents->single);
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutputO2Clus(), outputStream);
      ReleaseEvent(mEvents->single);
    }
    mRec->PopNonPersistentMemory(RecoStep::TPCMerging, qStr2Tag("TPCMERG2"));
  }
#endif
  if (doGPU && (synchronizeOutput || GetProcessingSettings().clearO2OutputFromGPU)) {
    SynchronizeStream(outputStream);
  }
  if (GetProcessingSettings().clearO2OutputFromGPU) {
    mRec->ReturnVolatileDeviceMemory();
  }

  mIOPtrs.mergedTracks = Merger.MergedTracks();
  mIOPtrs.nMergedTracks = Merger.NMergedTracks();
  mIOPtrs.mergedTrackHits = Merger.Clusters();
  mIOPtrs.nMergedTrackHits = Merger.NMergedTrackClusters();
  mIOPtrs.mergedTrackHitAttachment = Merger.ClusterAttachment();
  mIOPtrs.mergedTrackHitStates = Merger.ClusterStateExt();
  mIOPtrs.outputTracksTPCO2 = Merger.OutputTracksTPCO2();
  mIOPtrs.nOutputTracksTPCO2 = Merger.NOutputTracksTPCO2();
  mIOPtrs.outputClusRefsTPCO2 = Merger.OutputClusRefsTPCO2();
  mIOPtrs.nOutputClusRefsTPCO2 = Merger.NOutputClusRefsTPCO2();
  mIOPtrs.outputTracksTPCO2MC = Merger.OutputTracksTPCO2MC();

  if (doGPU) {
    processorsShadow()->ioPtrs.mergedTracks = MergerShadow.MergedTracks();
    processorsShadow()->ioPtrs.nMergedTracks = Merger.NMergedTracks();
    processorsShadow()->ioPtrs.mergedTrackHits = MergerShadow.Clusters();
    processorsShadow()->ioPtrs.nMergedTrackHits = Merger.NMergedTrackClusters();
    processorsShadow()->ioPtrs.mergedTrackHitAttachment = MergerShadow.ClusterAttachment();
    processorsShadow()->ioPtrs.mergedTrackHitStates = MergerShadow.ClusterStateExt();
    processorsShadow()->ioPtrs.outputTracksTPCO2 = MergerShadow.OutputTracksTPCO2();
    processorsShadow()->ioPtrs.nOutputTracksTPCO2 = Merger.NOutputTracksTPCO2();
    processorsShadow()->ioPtrs.outputClusRefsTPCO2 = MergerShadow.OutputClusRefsTPCO2();
    processorsShadow()->ioPtrs.nOutputClusRefsTPCO2 = Merger.NOutputClusRefsTPCO2();
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
  }

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NMergedTrackClusters(), Merger.NSectorHits());
  }
  return 0;
}
