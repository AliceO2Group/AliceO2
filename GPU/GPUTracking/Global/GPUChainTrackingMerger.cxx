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
#include "GPUConstantMem.h"
#include "GPUTPCGMMergerGPU.h"
#include "GPUTPCGMO2Output.h"
#include "GPUTPCGlobalDebugSortKernels.h"
#include "utils/strtag.h"
#include <fstream>

using namespace o2::gpu;

void GPUChainTracking::RunTPCTrackingMerger_MergeBorderTracks(int8_t withinSector, int8_t mergeMode, GPUReconstruction::krnlDeviceType deviceType)
{
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    uint32_t nBorderTracks = withinSector == 1 ? NSECTORS : (2 * NSECTORS);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::borderTracks>({{nBorderTracks, -WarpSize(), 0, deviceType}}, 0);
  }
  uint32_t n = withinSector == -1 ? NSECTORS / 2 : NSECTORS;
  if (GetProcessingSettings().alternateBorderSort == -1 ? mRec->getGPUParameters(doGPU).par_ALTERNATE_BORDER_SORT : GetProcessingSettings().alternateBorderSort) {
    RecordMarker(&mEvents->single, 0);
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->init);
    for (uint32_t i = 0; i < n; i++) {
      int32_t stream = i % mRec->NStreams();
      runKernel<GPUTPCGMMergerMergeBorders, 0>({GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, stream && i < (uint32_t)mRec->NStreams() ? &mEvents->single : nullptr}}, i, withinSector, mergeMode);
    }
    ReleaseEvent(mEvents->single);
    SynchronizeEventAndRelease(mEvents->init);
    for (uint32_t i = 0; i < n; i++) {
      int32_t stream = i % mRec->NStreams();
      int32_t n1, n2;
      GPUTPCGMBorderTrack *b1, *b2;
      int32_t jSector;
      Merger.MergeBorderTracksSetup(n1, n2, b1, b2, jSector, i, withinSector, mergeMode);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = MergerShadow.BorderRange(i);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = MergerShadow.BorderRange(jSector) + *processors()->tpcTrackers[jSector].NTracks();
      runKernel<GPUTPCGMMergerMergeBorders, 3>({{1, -WarpSize(), stream, deviceType}}, range1, n1, 0);
      runKernel<GPUTPCGMMergerMergeBorders, 3>({{1, -WarpSize(), stream, deviceType}}, range2, n2, 1);
      runKernel<GPUTPCGMMergerMergeBorders, 2>({GetGridAuto(stream, deviceType)}, i, withinSector, mergeMode);
    }
    int32_t ne = std::min<int32_t>(n, mRec->NStreams()) - 1; // Stream 0 must wait for all streams, Note n > 1
    for (int32_t j = 0; j < ne; j++) {
      RecordMarker(&mEvents->sector[j], j + 1);
    }
    StreamWaitForEvents(0, &mEvents->sector[0], ne);
  } else {
    for (uint32_t i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(0, deviceType), i, withinSector, mergeMode);
    }
    runKernel<GPUTPCGMMergerMergeBorders, 1>({{2 * n, -WarpSize(), 0, deviceType}}, 0, withinSector, mergeMode);
    for (uint32_t i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(0, deviceType), i, withinSector, mergeMode);
    }
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRanges, doGPU, Merger, &GPUTPCGMMerger::DumpMergeRanges, *mDebugFile, withinSector, mergeMode);
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
  GPUTPCGMMerger& MergerShadowAll = doGPU ? processorsShadow()->tpcMerger : Merger;
  const int32_t outputStream = OutputStream();
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& threadContext = GetThreadContext();

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
  for (uint32_t i = 0; i < NSECTORS; i++) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({{1, -WarpSize(), 0, deviceType}}, i);
    runKernel<GPUTPCGMMergerUnpackResetIds>(GetGridAuto(0, deviceType), i);
    runKernel<GPUTPCGMMergerSectorRefit>(GetGridAuto(0, deviceType), i); // TODO: Why all in stream 0?
  }
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({{1, -WarpSize(), 0, deviceType}}, NSECTORS);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks>({{GPUTPCGeometry::NSECTORS, -WarpSize(), 0, deviceType}}, 0);
  }
  for (uint32_t i = 0; i < NSECTORS; i++) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({{1, -WarpSize(), 0, deviceType}}, NSECTORS + i);
    runKernel<GPUTPCGMMergerUnpackGlobal>(GetGridAuto(0, deviceType), i);
  }
  runKernel<GPUTPCGMMergerUnpackSaveNumber>({{1, -WarpSize(), 0, deviceType}}, 2 * NSECTORS);
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::sectorTracks>({{GPUTPCGeometry::NSECTORS, -WarpSize(), 0, deviceType}}, 1);
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingSectorTracks, doGPU, Merger, &GPUTPCGMMerger::DumpSectorTracks, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), false);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.TmpCounter(), NSECTORS * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeWithinPrepare>(GetGridAuto(0, deviceType));
  RunTPCTrackingMerger_MergeBorderTracks(1, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingMatching, doGPU, Merger, &GPUTPCGMMerger::DumpMergedWithinSectors, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), false);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 2, 3, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 0, 1, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSectorsPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), 0, 1, 1);
  RunTPCTrackingMerger_MergeBorderTracks(0, -1, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingMatching, doGPU, Merger, &GPUTPCGMMerger::DumpMergedBetweenSectors, *mDebugFile);

  runKernel<GPUMemClean16>({{1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.TmpCounter(), 2 * NSECTORS * sizeof(*MergerShadowAll.TmpCounter()));

  runKernel<GPUTPCGMMergerLinkExtrapolatedTracks>(GetGridAuto(0, deviceType));
  if (GetProcessingSettings().mergerSanityCheck) {
    Merger.CheckMergeGraph();
  }
  runKernel<GPUTPCGMMergerCollect>(GetGridAuto(0, deviceType));
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::mergedTracks1>({{1, -WarpSize(), 0, deviceType}}, 1);
    runKernel<GPUTPCGlobalDebugSortKernels, GPUTPCGlobalDebugSortKernels::mergedTracks2>({{1, -WarpSize(), 0, deviceType}}, 1);
  }
  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingCollectedTracks, doGPU, Merger, &GPUTPCGMMerger::DumpCollected, *mDebugFile);

  if (param().rec.tpc.mergeCE) {
    runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), true);
    RunTPCTrackingMerger_MergeBorderTracks(-1, 1, deviceType);
    RunTPCTrackingMerger_MergeBorderTracks(-1, 2, deviceType);
    runKernel<GPUTPCGMMergerMergeCE>(GetGridAuto(0, deviceType));
    DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingCE, doGPU, Merger, &GPUTPCGMMerger::DumpMergeCE, *mDebugFile);
  }
  int32_t waitForTransfer = 0;
  if (doGPU) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    waitForTransfer = 1;
  }

  const bool mergerSortTracks = GetProcessingSettings().mergerSortTracks == -1 ? mRec->getGPUParameters(doGPU).par_SORT_BEFORE_FIT : GetProcessingSettings().mergerSortTracks;
  if (mergerSortTracks) {
    runKernel<GPUTPCGMMergerSortTracksPrepare>(GetGridAuto(0, deviceType));
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracks>(GetGridAuto(0, deviceType));
  }
  if (GetProcessingSettings().mergerSanityCheck) {
    Merger.CheckCollectedTracks();
  }

  uint32_t maxId = Merger.NMaxClusters();
  if (maxId > Merger.NMaxClusters()) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.SharedCount(), maxId * sizeof(*MergerShadowAll.SharedCount()));
  runKernel<GPUMemClean16>({{numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}}, MergerShadowAll.ClusterAttachment(), maxId * sizeof(*MergerShadowAll.ClusterAttachment()));
  runKernel<GPUTPCGMMergerPrepareForFit, 0>(GetGridAuto(0, deviceType));
  CondWaitEvent(waitForTransfer, &mEvents->single);
  runKernel<GPUTPCGMMergerSortTracksQPt>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerPrepareForFit, 1>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerPrepareForFit, 2>(GetGridAuto(0, deviceType));

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

  runKernel<GPUTPCGMMergerTrackFit>(doGPU ? GetGrid(Merger.NMergedTracks(), 0) : GetGridAuto(0), mergerSortTracks ? 1 : 0);
  if (param().rec.tpc.retryRefit == 1) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGridAuto(0), -1);
  }
  runKernel<GPUTPCGMMergerFollowLoopers>(GetGridAuto(0));

  DoDebugAndDump(RecoStep::TPCMerging, GPUChainTrackingDebugFlags::TPCMergingRefit, Merger, &GPUTPCGMMerger::DumpRefit, *mDebugFile);
  runKernel<GPUTPCGMMergerFinalize, 0>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerFinalize, 1>(GetGridAuto(0, deviceType));
  runKernel<GPUTPCGMMergerFinalize, 2>(GetGridAuto(0, deviceType));
  if (param().rec.tpc.mergeLoopersAfterburner) {
    runKernel<GPUTPCGMMergerMergeLoopers, 0>(doGPU ? GetGrid(Merger.NMergedTracks(), 0, deviceType) : GetGridAuto(0, deviceType));
    if (doGPU) {
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0);
      SynchronizeStream(0); // TODO: could probably synchronize on an event after runKernel<GPUTPCGMMergerMergeLoopers, 1>
    }
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
      GPUMemCpy(RecoStep::TPCMerging, Merger.MergedTracks(), MergerShadowAll.MergedTracks(), Merger.NMergedTracks() * sizeof(*Merger.MergedTracks()), outputStream, 0, nullptr, waitEvent);
      waitEvent = nullptr;
      if (param().dodEdxEnabled) {
        GPUMemCpy(RecoStep::TPCMerging, Merger.MergedTracksdEdx(), MergerShadowAll.MergedTracksdEdx(), Merger.NMergedTracks() * sizeof(*Merger.MergedTracksdEdx()), outputStream, 0);
      }
      GPUMemCpy(RecoStep::TPCMerging, Merger.Clusters(), MergerShadowAll.Clusters(), Merger.NMergedTrackClusters() * sizeof(*Merger.Clusters()), outputStream, 0);
      GPUMemCpy(RecoStep::TPCMerging, Merger.ClusterAttachment(), MergerShadowAll.ClusterAttachment(), Merger.NMaxClusters() * sizeof(*Merger.ClusterAttachment()), outputStream, 0);
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
    runKernel<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(GetGridAuto(0, deviceType));
    mRec->ReturnVolatileDeviceMemory();
    SynchronizeEventAndRelease(mEvents->single, doGPU);

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
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NMergedTrackClusters(), Merger.NClusters());
  }
  return 0;
}
