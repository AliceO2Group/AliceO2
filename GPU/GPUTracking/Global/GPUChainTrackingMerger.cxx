// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingMerger.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include <fstream>

using namespace GPUCA_NAMESPACE::gpu;

void GPUChainTracking::RunTPCTrackingMerger_MergeBorderTracks(char withinSlice, char mergeMode, GPUReconstruction::krnlDeviceType deviceType)
{
  unsigned int n = withinSlice == -1 ? NSLICES / 2 : NSLICES;
  bool doGPUall = GetRecoStepsGPU() & RecoStep::TPCMerging && GetProcessingSettings().fullMergerOnGPU;
  if (GetProcessingSettings().alternateBorderSort && (!mRec->IsGPU() || doGPUall)) {
    GPUTPCGMMerger& Merger = processors()->tpcMerger;
    GPUTPCGMMerger& MergerShadow = doGPUall ? processorsShadow()->tpcMerger : Merger;
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->init);
    RecordMarker(&mEvents->single, 0);
    for (unsigned int i = 0; i < n; i++) {
      int stream = i % mRec->NStreams();
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, stream && i < (unsigned int)mRec->NStreams() ? &mEvents->single : nullptr}, i, withinSlice, mergeMode);
    }
    ReleaseEvent(&mEvents->single);
    SynchronizeEvents(&mEvents->init);
    ReleaseEvent(&mEvents->init);
    for (unsigned int i = 0; i < n; i++) {
      int stream = i % mRec->NStreams();
      int n1, n2;
      GPUTPCGMBorderTrack *b1, *b2;
      int jSlice;
      Merger.MergeBorderTracksSetup(n1, n2, b1, b2, jSlice, i, withinSlice, mergeMode);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = MergerShadow.BorderRange(i);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = MergerShadow.BorderRange(jSlice) + *processors()->tpcTrackers[jSlice].NTracks();
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), stream, deviceType}, krnlRunRangeNone, krnlEventNone, range1, n1, 0);
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), stream, deviceType}, krnlRunRangeNone, krnlEventNone, range2, n2, 1);
      deviceEvent** e = nullptr;
      int ne = 0;
      if (i == n - 1) { // Synchronize all execution on stream 0 with the last kernel
        ne = std::min<int>(n, mRec->NStreams());
        for (int j = 1; j < ne; j++) {
          RecordMarker(&mEvents->slice[j], j);
        }
        e = &mEvents->slice[1];
        ne--;
        stream = 0;
      }
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, e, ne}, i, withinSlice, mergeMode);
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i, withinSlice, mergeMode);
    }
    runKernel<GPUTPCGMMergerMergeBorders, 1>({2 * n, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, 0, withinSlice, mergeMode);
    for (unsigned int i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i, withinSlice, mergeMode);
    }
  }
  mRec->ReturnVolatileDeviceMemory();
}

void GPUChainTracking::RunTPCTrackingMerger_Resolve(char useOrigTrackParam, char mergeAll, GPUReconstruction::krnlDeviceType deviceType)
{
  runKernel<GPUTPCGMMergerResolve, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 3>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 4>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, useOrigTrackParam, mergeAll);
}

int GPUChainTracking::RunTPCTrackingMerger(bool synchronizeOutput)
{
  if (GetProcessingSettings().debugLevel >= 6 && GetProcessingSettings().comparableDebutOutput && param().rec.mergerReadFromTrackerDirectly) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      GPUTPCTracker& trk = processors()->tpcTrackers[i];
      TransferMemoryResourcesToHost(RecoStep::NoRecoStep, &trk);
      auto sorter = [](GPUTPCTrack& trk1, GPUTPCTrack& trk2) {
        if (trk1.NHits() == trk2.NHits()) {
          return trk1.Param().Y() > trk2.Param().Y();
        }
        return trk1.NHits() > trk2.NHits();
      };
      std::sort(trk.Tracks(), trk.Tracks() + trk.CommonMemory()->nLocalTracks, sorter);
      std::sort(trk.Tracks() + trk.CommonMemory()->nLocalTracks, trk.Tracks() + *trk.NTracks(), sorter);
      TransferMemoryResourcesToGPU(RecoStep::NoRecoStep, &trk, 0);
    }
  }
  mRec->PushNonPersistentMemory();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  bool doGPUall = doGPU && GetProcessingSettings().fullMergerOnGPU;
  GPUReconstruction::krnlDeviceType deviceType = doGPUall ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  unsigned int numBlocks = (!mRec->IsGPU() || doGPUall) ? BlockCount() : 1;
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  GPUTPCGMMerger& MergerShadowAll = doGPUall ? processorsShadow()->tpcMerger : Merger;
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& threadContext = GetThreadContext();

  SynchronizeGPU(); // Need to know the full number of slice tracks
  SetupGPUProcessor(&Merger, true);
  AllocateRegisteredMemory(Merger.MemoryResOutput(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracks)]);
  AllocateRegisteredMemory(Merger.MemoryResOutputState(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::sharedClusterMap)]);

  if (Merger.CheckSlices()) {
    return 1;
  }

  memset(Merger.Memory(), 0, sizeof(*Merger.Memory()));
  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  if (doGPUall) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackResetIds>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerSliceRefit>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, NSLICES + i);
    runKernel<GPUTPCGMMergerUnpackGlobal>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, 2 * NSLICES);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpSliceTracks, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeWithinPrepare>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  RunTPCTrackingMerger_MergeBorderTracks(1, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedWithinSlices, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 2, 3, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 1);
  RunTPCTrackingMerger_MergeBorderTracks(0, -1, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedBetweenSlices, *mDebugFile);

  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));

  runKernel<GPUTPCGMMergerLinkGlobalTracks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerCollect>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpCollected, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 1);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 1, deviceType);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 2, deviceType);
  runKernel<GPUTPCGMMergerMergeCE>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergeCE, *mDebugFile);
  int waitForTransfer = 0;
  if (doGPUall) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    waitForTransfer = 1;
  }

  if (GetProcessingSettings().mergerSortTracks) {
    runKernel<GPUTPCGMMergerSortTracksPrepare>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  unsigned int maxId = param().rec.NonConsecutiveIDs ? Merger.Memory()->nOutputTrackClusters : Merger.NMaxClusters();
  if (maxId > Merger.NMaxClusters()) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  if (!param().rec.NonConsecutiveIDs) {
    unsigned int* sharedCount = (unsigned int*)MergerShadowAll.TmpMem() + CAMath::nextMultipleOf<4>(Merger.Memory()->nOutputTracks);
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, sharedCount, maxId * sizeof(*sharedCount));
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.ClusterAttachment(), maxId * sizeof(*MergerShadowAll.ClusterAttachment()));
    runKernel<GPUTPCGMMergerPrepareClusters, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracksQPt>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFitPrepare, *mDebugFile);

  if (doGPUall) {
    CondWaitEvent(waitForTransfer, &mEvents->single);
    if (waitForTransfer) {
      ReleaseEvent(&mEvents->single);
    }
  } else if (doGPU) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  if (GetProcessingSettings().delayedOutput) {
    for (unsigned int i = 0; i < mOutputQueue.size(); i++) {
      GPUMemCpy(mOutputQueue[i].step, mOutputQueue[i].dst, mOutputQueue[i].src, mOutputQueue[i].size, mRec->NStreams() - 2, false);
    }
    mOutputQueue.clear();
  }

  runKernel<GPUTPCGMMergerTrackFit>(doGPU ? GetGrid(Merger.NOutputTracks(), 0) : GetGridAuto(0), krnlRunRangeNone, krnlEventNone, GetProcessingSettings().mergerSortTracks ? 1 : 0);
  if (param().rec.retryRefit == 1) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone, -1);
  }
  if (param().rec.loopInterpolationInExtraPass) {
    runKernel<GPUTPCGMMergerFollowLoopers>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  }
  if (doGPU && !doGPUall) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, 0);
    SynchronizeStream(0);
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, Merger, &GPUTPCGMMerger::DumpRefit, *mDebugFile);
  runKernel<GPUTPCGMMergerFinalize, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  if (!param().rec.NonConsecutiveIDs) {
    runKernel<GPUTPCGMMergerFinalize, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerFinalize, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }
  if (param().rec.tpcMergeLoopersAfterburner) {
    runKernel<GPUTPCGMMergerMergeLoopers>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFinal, *mDebugFile);

  if (doGPUall) {
    RecordMarker(&mEvents->single, 0);
    if (!GetProcessingSettings().fullMergerOnGPU) {
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutput(), mRec->NStreams() - 2, nullptr, &mEvents->single);
    } else if (GetProcessingSettings().keepDisplayMemory || GetProcessingSettings().createO2Output <= 1) {
      GPUMemCpy(RecoStep::TPCMerging, Merger.OutputTracks(), MergerShadowAll.OutputTracks(), Merger.NOutputTracks() * sizeof(*Merger.OutputTracks()), mRec->NStreams() - 2, 0, nullptr, &mEvents->single);
      GPUMemCpy(RecoStep::TPCMerging, Merger.Clusters(), MergerShadowAll.Clusters(), Merger.NOutputTrackClusters() * sizeof(*Merger.Clusters()), mRec->NStreams() - 2, 0);
      if (param().par.earlyTpcTransform) {
        GPUMemCpy(RecoStep::TPCMerging, Merger.ClustersXYZ(), MergerShadowAll.ClustersXYZ(), Merger.NOutputTrackClusters() * sizeof(*Merger.ClustersXYZ()), mRec->NStreams() - 2, 0);
      }
      GPUMemCpy(RecoStep::TPCMerging, Merger.ClusterAttachment(), MergerShadowAll.ClusterAttachment(), Merger.NMaxClusters() * sizeof(*Merger.ClusterAttachment()), mRec->NStreams() - 2, 0);
    }
    ReleaseEvent(&mEvents->single);
  } else {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }
  if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
  }

#ifdef GPUCA_TPC_GEOMETRY_O2
  if (GetProcessingSettings().createO2Output) {
    mRec->ReturnVolatileDeviceMemory();
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::prepare>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0);
    SynchronizeStream(0);
    AllocateRegisteredMemory(Merger.MemoryResOutputO2(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2)]);
    AllocateRegisteredMemory(Merger.MemoryResOutputO2Clus(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2ClusRefs)]);
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::output>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);

    if (GetProcessingSettings().runMC && mIOPtrs.clustersNative->clustersMCTruth) {
      AllocateRegisteredMemory(Merger.MemoryResOutputO2MC(), mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracksO2Labels)]);
      TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
      runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::mc>(GetGridAuto(0, GPUReconstruction::krnlDeviceType::CPU), krnlRunRangeNone, krnlEventNone);
    } else {
      RecordMarker(&mEvents->single, 0);
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutputO2(), mRec->NStreams() - 2, nullptr, &mEvents->single);
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutputO2Clus(), mRec->NStreams() - 2, nullptr, &mEvents->single);
      ReleaseEvent(&mEvents->single);
    }
  }
#endif
  if (synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 2);
  }

  mRec->ReturnVolatileDeviceMemory();
  mIOPtrs.mergedTracks = Merger.OutputTracks();
  mIOPtrs.nMergedTracks = Merger.NOutputTracks();
  mIOPtrs.mergedTrackHits = Merger.Clusters();
  mIOPtrs.nMergedTrackHits = Merger.NOutputTrackClusters();
  mIOPtrs.mergedTrackHitAttachment = Merger.ClusterAttachment();
  mIOPtrs.mergedTrackHitStates = Merger.ClusterStateExt();
  mIOPtrs.outputTracksTPCO2 = Merger.OutputTracksTPCO2();
  mIOPtrs.nOutputTracksTPCO2 = Merger.NOutputTracksTPCO2();
  mIOPtrs.outputClusRefsTPCO2 = Merger.OutputClusRefsTPCO2();
  mIOPtrs.nOutputClusRefsTPCO2 = Merger.NOutputClusRefsTPCO2();
  mIOPtrs.outputTracksTPCO2MC = Merger.OutputTracksTPCO2MC();

  if (doGPU) {
    processorsShadow()->ioPtrs.mergedTracks = MergerShadow.OutputTracks();
    processorsShadow()->ioPtrs.nMergedTracks = Merger.NOutputTracks();
    processorsShadow()->ioPtrs.mergedTrackHits = MergerShadow.Clusters();
    processorsShadow()->ioPtrs.nMergedTrackHits = Merger.NOutputTrackClusters();
    processorsShadow()->ioPtrs.mergedTrackHitAttachment = MergerShadow.ClusterAttachment();
    processorsShadow()->ioPtrs.mergedTrackHitStates = MergerShadow.ClusterStateExt();
    processorsShadow()->ioPtrs.outputTracksTPCO2 = MergerShadow.OutputTracksTPCO2();
    processorsShadow()->ioPtrs.nOutputTracksTPCO2 = Merger.NOutputTracksTPCO2();
    processorsShadow()->ioPtrs.outputClusRefsTPCO2 = MergerShadow.OutputClusRefsTPCO2();
    processorsShadow()->ioPtrs.nOutputClusRefsTPCO2 = Merger.NOutputClusRefsTPCO2();
    WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
  }

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NOutputTrackClusters(), Merger.NClusters());
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCMerging);
  return 0;
}
