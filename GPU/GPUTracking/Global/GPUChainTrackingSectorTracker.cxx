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

/// \file GPUChainTrackingSectorTracker.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUChainTrackingDebug.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include "GPUTPCClusterOccupancyMap.h"
#include "GPUDefParametersRuntime.h"
#include "GPUTPCExtrapolationTracking.h"
#include "GPUTPCCreateOccupancyMap.h"
#include "GPUTPCCreateTrackingData.h"
#include "GPUTPCNeighboursFinder.h"
#include "GPUTPCNeighboursCleaner.h"
#include "GPUTPCStartHitsFinder.h"
#include "GPUTPCStartHitsSorter.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUTPCTrackletSelector.h"
#include "GPUTPCSectorDebugSortKernels.h"
#include "utils/strtag.h"
#include <fstream>

using namespace o2::gpu;

uint32_t GPUChainTracking::StreamForSector(uint32_t sector) const
{
  return sector % mRec->NStreams();
}

int32_t GPUChainTracking::ExtrapolationTracking(uint32_t iSector, bool blocking)
{
  const uint32_t stream = StreamForSector(iSector);
  runKernel<GPUTPCExtrapolationTracking>({GetGridBlk(256, stream), {iSector}});
  TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, processors()->tpcTrackers[iSector].MemoryResCommon(), stream);
  if (blocking) {
    SynchronizeStream(stream);
  }
  return (0);
}

int32_t GPUChainTracking::RunTPCTrackingSectors()
{
  if (mRec->GPUStuck()) {
    GPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
    return (1);
  }

  const auto& threadContext = GetThreadContext();

  int32_t retVal = RunTPCTrackingSectors_internal();
  if (retVal) {
    SynchronizeGPU();
  }
  return (retVal != 0);
}

int32_t GPUChainTracking::RunTPCTrackingSectors_internal()
{
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Sector Tracker");
  }
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCSectorTracking;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]);
    if (doGPU) {
      processorsShadow()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]); // TODO: not needed I think, anyway copied in SetupGPUProcessor
    }
  }
  mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  GPUInfo("Event has %u TPC Clusters, %d TRD Tracklets", (uint32_t)mRec->MemoryScalers()->nTPCHits, mIOPtrs.nTRDTracklets);

  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    processors()->tpcTrackers[iSector].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSector], false); // Prepare custom allocation for 1st stack level
    mRec->AllocateRegisteredMemory(processors()->tpcTrackers[iSector].MemoryResSectorScratch());
  }
  mRec->PushNonPersistentMemory(qStr2Tag("TPCSLTRK"));
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSector], true);             // Now we allocate
    mRec->ResetRegisteredMemoryPointers(&processors()->tpcTrackers[iSector]); // TODO: The above call breaks the GPU ptrs to already allocated memory. This fixes them. Should actually be cleaned up at the source.
    processors()->tpcTrackers[iSector].SetupCommonMemory();
  }

  bool streamInit[GPUCA_MAX_STREAMS] = {false};
  int32_t streamInitAndOccMap = mRec->NStreams() - 1;

  if (doGPU) {
    // Copy Tracker Object to GPU Memory
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Tracker objects to GPU");
    }
    if (PrepareProfile()) {
      return 2;
    }

    WriteToConstantMemory(RecoStep::TPCSectorTracking, (char*)processors()->tpcTrackers - (char*)processors(), processorsShadow()->tpcTrackers, sizeof(GPUTPCTracker) * NSECTORS, streamInitAndOccMap, &mEvents->init);

    std::fill(streamInit, streamInit + mRec->NStreams(), false);
    streamInit[streamInitAndOccMap] = true;
  }

  if (param().rec.tpc.occupancyMapTimeBins || param().rec.tpc.sysClusErrorC12Norm) {
    AllocateRegisteredMemory(mInputsHost->mResourceOccupancyMap, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcOccupancyMap)]);
  }
  if (param().rec.tpc.occupancyMapTimeBins) {
    if (doGPU) {
      ReleaseEvent(mEvents->init);
    }
    uint32_t* ptr = doGPU ? mInputsShadow->mTPCClusterOccupancyMap : mInputsHost->mTPCClusterOccupancyMap;
    auto* ptrTmp = (GPUTPCClusterOccupancyMapBin*)mRec->AllocateVolatileMemory(GPUTPCClusterOccupancyMapBin::getTotalSize(param()), doGPU);
    runKernel<GPUMemClean16>(GetGridAutoStep(streamInitAndOccMap, RecoStep::TPCSectorTracking), ptrTmp, GPUTPCClusterOccupancyMapBin::getTotalSize(param()));
    runKernel<GPUTPCCreateOccupancyMap, GPUTPCCreateOccupancyMap::fill>(GetGridBlk(GPUCA_NSECTORS * GPUCA_ROW_COUNT, streamInitAndOccMap), ptrTmp);
    runKernel<GPUTPCCreateOccupancyMap, GPUTPCCreateOccupancyMap::fold>(GetGridBlk(GPUTPCClusterOccupancyMapBin::getNBins(param()), streamInitAndOccMap), ptrTmp, ptr + 2);
    mRec->ReturnVolatileMemory();
    mInputsHost->mTPCClusterOccupancyMap[1] = param().rec.tpc.occupancyMapTimeBins * 0x10000 + param().rec.tpc.occupancyMapTimeBinsAverage;
    if (doGPU) {
      GPUMemCpy(RecoStep::TPCSectorTracking, mInputsHost->mTPCClusterOccupancyMap + 2, mInputsShadow->mTPCClusterOccupancyMap + 2, sizeof(*ptr) * GPUTPCClusterOccupancyMapBin::getNBins(mRec->GetParam()), streamInitAndOccMap, false, &mEvents->init);
    } else {
      TransferMemoryResourceLinkToGPU(RecoStep::TPCSectorTracking, mInputsHost->mResourceOccupancyMap, streamInitAndOccMap, &mEvents->init);
    }
  }
  if (param().rec.tpc.occupancyMapTimeBins || param().rec.tpc.sysClusErrorC12Norm) {
    uint32_t& occupancyTotal = *mInputsHost->mTPCClusterOccupancyMap;
    occupancyTotal = CAMath::Float2UIntRn(mRec->MemoryScalers()->nTPCHits / (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasNHBFPerTF ? mIOPtrs.settingsTF->nHBFPerTF : 128));
    mRec->UpdateParamOccupancyMap(param().rec.tpc.occupancyMapTimeBins ? mInputsHost->mTPCClusterOccupancyMap + 2 : nullptr, doGPU && param().rec.tpc.occupancyMapTimeBins ? mInputsShadow->mTPCClusterOccupancyMap + 2 : nullptr, occupancyTotal, streamInitAndOccMap);
  }

  int32_t streamMap[NSECTORS];

  bool error = false;
  mRec->runParallelOuterLoop(doGPU, NSECTORS, [&](uint32_t iSector) {
    GPUTPCTracker& trk = processors()->tpcTrackers[iSector];
    GPUTPCTracker& trkShadow = doGPU ? processorsShadow()->tpcTrackers[iSector] : trk;
    int32_t useStream = StreamForSector(iSector);
    if (GetProcessingSettings().amdMI100SerializationWorkaround) {
      SynchronizeStream(useStream); // TODO: Remove this workaround once fixed on MI100
    }

    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Creating Sector Data (Sector %d)", iSector);
    }
    TransferMemoryResourcesToGPU(RecoStep::TPCSectorTracking, &trk, useStream);
    runKernel<GPUTPCCreateTrackingData>({doGPU ? GetGridBlk(GPUCA_ROW_COUNT, useStream) : GetGridAuto(0), {iSector}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init}}); // TODO: Check why GetGridAuto(0) is much fast on CPU
    streamInit[useStream] = true;
    if (GetProcessingSettings().deterministicGPUReconstruction) {
      runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::hitData>({GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSector}});
    }
    if (!doGPU && trk.CheckEmptySector() && GetProcessingSettings().debugLevel == 0) {
      return;
    }

    if (GetProcessingSettings().debugLevel >= 6) {
      if ((GetProcessingSettings().debugMask & 63)) {
        *mDebugFile << "\n\nReconstruction: Sector " << iSector << "/" << NSECTORS << std::endl;
      }
      if (GetProcessingSettings().debugMask & GPUChainTrackingDebugFlags::TPCSectorTrackingData) {
        if (doGPU) {
          TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &trk, -1, true);
        }
        trk.DumpTrackingData(*mDebugFile);
      }
    }

    runKernel<GPUMemClean16>(GetGridAutoStep(useStream, RecoStep::TPCSectorTracking), trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    runKernel<GPUTPCNeighboursFinder>({GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSector}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init}});
    streamInit[useStream] = true;

    if (GetProcessingSettings().keepDisplayMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinks()).Ptr(), mRec->Res(trk.MemoryResLinks()).Size());
      if (GetProcessingSettings().debugMask & GPUChainTrackingDebugFlags::TPCPreLinks) {
        trk.DumpLinks(*mDebugFile, 0);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>({GetGridBlk(GPUCA_ROW_COUNT - 2, useStream), {iSector}});
    DoDebugAndDump(RecoStep::TPCSectorTracking, GPUChainTrackingDebugFlags::TPCLinks, trk, &GPUTPCTracker::DumpLinks, *mDebugFile, 1);

    runKernel<GPUTPCStartHitsFinder>({GetGridBlk(GPUCA_ROW_COUNT - 6, useStream), {iSector}});
    if (mRec->getGPUParameters(doGPU).par_SORT_STARTHITS) {
      runKernel<GPUTPCStartHitsSorter>({GetGridAuto(useStream), {iSector}});
    }
    if (GetProcessingSettings().deterministicGPUReconstruction) {
      runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::startHits>({GetGrid(1, 1, useStream), {iSector}});
    }
    DoDebugAndDump(RecoStep::TPCSectorTracking, GPUChainTrackingDebugFlags::TPCStartHits, trk, &GPUTPCTracker::DumpStartHits, *mDebugFile);

    if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    runKernel<GPUTPCTrackletConstructor>({GetGridAuto(useStream), {iSector}});
    DoDebugAndDump(RecoStep::TPCSectorTracking, GPUChainTrackingDebugFlags::TPCTracklets, trk, &GPUTPCTracker::DumpTrackletHits, *mDebugFile);
    if (GetProcessingSettings().debugMask & GPUChainTrackingDebugFlags::TPCHitWeights && GetProcessingSettings().deterministicGPUReconstruction < 2) {
      trk.DumpHitWeights(*mDebugFile);
    }

    runKernel<GPUTPCTrackletSelector>({GetGridAuto(useStream), {iSector}});
    runKernel<GPUTPCExtrapolationTrackingCopyNumbers>({{1, -ThreadCount(), useStream}, {iSector}}, 1);
    if (GetProcessingSettings().deterministicGPUReconstruction) {
      runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::sectorTracks>({GetGrid(1, 1, useStream), {iSector}});
    }
    TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, trk.MemoryResCommon(), useStream, &mEvents->sector[iSector]);
    streamMap[iSector] = useStream;
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Sector %u, Number of tracks: %d", iSector, *trk.NTracks());
    }
    DoDebugAndDump(RecoStep::TPCSectorTracking, GPUChainTrackingDebugFlags::TPCSectorTracks, trk, &GPUTPCTracker::DumpTrackHits, *mDebugFile);
  });
  mRec->SetNActiveThreadsOuterLoop(1);
  if (error) {
    return (3);
  }

  if (doGPU || GetProcessingSettings().debugLevel >= 1) {
    if (param().rec.tpc.extrapolationTracking) {
      std::vector<bool> blocking(NSECTORS * mRec->NStreams());
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
        for (uint32_t iStream = 0; iStream < mRec->NStreams(); iStream++) {
          blocking[iSector * mRec->NStreams() + iStream] = StreamForSector(iSector) == iStream;
        }
      }
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
        uint32_t tmpSector = GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorOrder(iSector);
        uint32_t sectorLeft, sectorRight;
        GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorLeftRight(tmpSector, sectorLeft, sectorRight);
        if (doGPU && !blocking[tmpSector * mRec->NStreams() + StreamForSector(sectorLeft)]) {
          StreamWaitForEvents(StreamForSector(tmpSector), &mEvents->sector[sectorLeft]);
          blocking[tmpSector * mRec->NStreams() + StreamForSector(sectorLeft)] = true;
        }
        if (doGPU && !blocking[tmpSector * mRec->NStreams() + StreamForSector(sectorRight)]) {
          StreamWaitForEvents(StreamForSector(tmpSector), &mEvents->sector[sectorRight]);
          blocking[tmpSector * mRec->NStreams() + StreamForSector(sectorRight)] = true;
        }
        ExtrapolationTracking(tmpSector, false);
      }
    }
    if (doGPU) {
      ReleaseEvent(mEvents->init);
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
        ReleaseEvent(mEvents->sector[iSector]);
      }
    }
  } else {
    mRec->runParallelOuterLoop(doGPU, NSECTORS, [&](uint32_t iSector) {
      if (param().rec.tpc.extrapolationTracking) {
        ExtrapolationTracking(iSector, true);
      }
    });
    mRec->SetNActiveThreadsOuterLoop(1);
  }

  if (param().rec.tpc.extrapolationTracking && GetProcessingSettings().debugLevel >= 3) {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      GPUInfo("Sector %d - Tracks: Local %d Extrapolated %d - Hits: Local %d Extrapolated %d", iSector,
              processors()->tpcTrackers[iSector].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSector].CommonMemory()->nTracks, processors()->tpcTrackers[iSector].CommonMemory()->nLocalTrackHits, processors()->tpcTrackers[iSector].CommonMemory()->nTrackHits);
    }
  }

  if (DoProfile()) {
    return (1);
  }
  for (uint32_t i = 0; i < NSECTORS; i++) {
    mIOPtrs.nSectorTracks[i] = *processors()->tpcTrackers[i].NTracks();
    mIOPtrs.sectorTracks[i] = processors()->tpcTrackers[i].Tracks();
    mIOPtrs.nSectorClusters[i] = *processors()->tpcTrackers[i].NTrackHits();
    mIOPtrs.sectorClusters[i] = processors()->tpcTrackers[i].TrackHits();
    if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &processors()->tpcTrackers[i], -1, true);
    }
  }
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Sector Tracker finished");
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCSectorTracking, qStr2Tag("TPCSLTRK"));
  return 0;
}
