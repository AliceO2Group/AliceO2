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

/// \file GPUTPCGMMerger.cxx
/// \author Sergey Gorbunov, David Rohr

#define GPUCA_CADEBUG 0
#define GPUCA_MERGE_LOOPER_MC 0
// #define GPUCA_CADEBUG_ENABLED

#include "GPUCommonDef.h"

#if !defined(GPUCA_GPUCODE) && (defined(GPUCA_MERGER_BY_MC_LABEL) || defined(GPUCA_CADEBUG_ENABLED) || GPUCA_MERGE_LOOPER_MC)
#include "AliHLTTPCClusterMCData.h"
#include "GPUROOTDump.h"
#endif

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdio>
#include <cstring>
#include <cmath>
#include "GPUReconstruction.h"
#endif

#include "GPUTPCTracker.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUO2DataTypes.h"
#include "TPCFastTransformPOD.h"
#include "GPUTPCConvertImpl.h"
#include "GPUTPCGeometry.h"
#include "GPUDefParametersRuntime.h"

#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"
#include "GPUCommonConstants.h"

#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUTPCTrackLinearisation.h"

#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSectorTrack.h"
#include "GPUTPCGMBorderTrack.h"

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"
#ifndef GPUCA_GPUCODE
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace gputpcgmmergertypes;

namespace o2::gpu::internal
{
struct MergeLooperParam {
  float refz;
  float x;
  float y;
  uint32_t id;
};

struct MergeBorderTracks_compMax {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMax != b.fMax) ? (a.fMax < b.fMax) : (a.fId < b.fId), a.fMax < b.fMax);
  }
};
struct MergeBorderTracks_compMin {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMin != b.fMin) ? (a.fMin < b.fMin) : (a.fId < b.fId), a.fMin < b.fMin);
  }
};

struct GPUTPCGMMergerSortTracks_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracks_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    if (a.OK() != b.OK()) {
      return a.OK();
    }
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (a.NClusters() != b.NClusters()) {
        return a.NClusters() > b.NClusters();
      } if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return aa > bb;
    , // !GPUCA_DETERMINISTIC_CODE
      return a.NClusters() > b.NClusters();
    ) // clang-format on
  }
};

struct GPUTPCGMMergerSortTracksQPt_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracksQPt_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return a.GetParam().GetZ() > b.GetParam().GetZ();
    , // !GPUCA_DETERMINISTIC_CODE
      return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
    ) // clang-format on
  }
};

struct GPUTPCGMMergerMergeLoopers_comp {
  GPUd() bool operator()(const MergeLooperParam& a, const MergeLooperParam& b)
  {
    return GPUCA_DETERMINISTIC_CODE(CAMath::Abs(a.refz) != CAMath::Abs(b.refz) ? CAMath::Abs(a.refz) < CAMath::Abs(b.refz) : a.id < b.id, CAMath::Abs(a.refz) < CAMath::Abs(b.refz));
  }
};

} // namespace o2::gpu::internal

using namespace o2::gpu::internal;

#ifndef GPUCA_GPUCODE

#include "GPUQA.h"
#include "GPUMemorySizeScalers.h"

GPUTPCGMMerger::GPUTPCGMMerger()
{
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    mNextSectorInd[iSector] = iSector + 1;
    mPrevSectorInd[iSector] = iSector - 1;
  }
  int32_t mid = NSECTORS / 2 - 1;
  int32_t last = NSECTORS - 1;
  mNextSectorInd[mid] = 0;
  mPrevSectorInd[0] = mid;
  mNextSectorInd[last] = NSECTORS / 2;
  mPrevSectorInd[NSECTORS / 2] = last;
}

// DEBUG CODE
#if !defined(GPUCA_GPUCODE) && (defined(GPUCA_MERGER_BY_MC_LABEL) || defined(GPUCA_CADEBUG_ENABLED) || GPUCA_MERGE_LOOPER_MC)
#include "GPUQAHelper.h"

template <class T>
inline const auto* resolveMCLabels(const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* a, const AliHLTTPCClusterMCLabel* b)
{
  return a;
}
template <>
inline const auto* resolveMCLabels<AliHLTTPCClusterMCLabel>(const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* a, const AliHLTTPCClusterMCLabel* b)
{
  return b;
}

template <class T, class S>
int64_t GPUTPCGMMerger::GetTrackLabelA(const S& trk) const
{
  GPUTPCGMSectorTrack* sectorTrack = nullptr;
  int32_t nClusters = 0;
  if constexpr (std::is_same_v<S, GPUTPCGMBorderTrack&>) {
    sectorTrack = &mSectorTrackInfos[trk.TrackID()];
    nClusters = sectorTrack->OrigTrack()->NHits();
  } else {
    nClusters = trk.NClusters();
  }
  auto acc = GPUTPCTrkLbl<false, GPUTPCTrkLbl_ret>(resolveMCLabels<T>(GetConstantMem()->ioPtrs.clustersNative ? GetConstantMem()->ioPtrs.clustersNative->clustersMCTruth : nullptr, GetConstantMem()->ioPtrs.mcLabelsTPC), 0.5f);
  for (int32_t i = 0; i < nClusters; i++) {
    int32_t id;
    if constexpr (std::is_same_v<S, GPUTPCGMBorderTrack&>) {
      const GPUTPCTracker& tracker = GetConstantMem()->tpcTrackers[sectorTrack->Sector()];
      const GPUTPCHitId& ic = tracker.TrackHits()[sectorTrack->OrigTrack()->FirstHitID() + i];
      id = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex()) + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[sectorTrack->Sector()][0];
    } else {
      id = mClusters[trk.FirstClusterRef() + i].num;
    }
    acc.addLabel(id);
  }
  return acc.computeLabel().id;
}

template <class S>
int64_t GPUTPCGMMerger::GetTrackLabel(const S& trk) const
{
#ifndef GPUCA_RUN2
  if (GetConstantMem()->ioPtrs.clustersNative->clustersMCTruth) {
    return GetTrackLabelA<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>, S>(trk);
  } else
#endif
  {
    return GetTrackLabelA<AliHLTTPCClusterMCLabel, S>(trk);
  }
}

#endif
// END DEBUG CODE

void GPUTPCGMMerger::CheckCollectedTracks()
{
  uint32_t nErr = 0;
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    const GPUTPCGMMergedTrack& trk = mMergedTracks[i];
    if (trk.OK()) {
      if (trk.NClusters() == 0) {
        GPUError("FAILURE: Track marked ok but has 0 clusters");
        nErr++;
      }
      if (!trk.CCE() && !trk.MergedLooper()) {
        const GPUTPCGMMergedTrack* updTrk = &trk;
        while (updTrk->PrevSegment() >= 0) {
          auto next = &mMergedTracks[updTrk->PrevSegment()];
          if (!next->MergedLooper()) {
            GPUError("FAILURE: prev segment not marked as merged looper\n");
            nErr++;
          }
          if (next == &trk) {
            GPUError("FAILURE: segment cycle found\n");
            break;
          }
          updTrk = next;
        }
        if (updTrk->NClusters() == 0) {
          printf("FAILURE: segment leg has 0 clusters");
        }
      }
    }
  }

  if (nErr == 0) {
    GPUInfo("Merged Tracks OK");
  } else {
    throw std::runtime_error("Error during track merging");
  }
}

void GPUTPCGMMerger::CheckMergeGraph()
{
  uint32_t nErr = 0;
  std::vector<bool> trkUsed(SectorTrackInfoLocalTotal());
  for (int32_t i = 0; i < SectorTrackInfoLocalTotal(); i++) {
    trkUsed[i] = false;
  }

  for (int32_t itr = 0; itr < SectorTrackInfoLocalTotal(); itr++) {
    GPUTPCGMSectorTrack& track = mSectorTrackInfos[itr];
    if (track.PrevSegmentNeighbour() >= 0 && mSectorTrackInfos[track.PrevSegmentNeighbour()].NextSegmentNeighbour() != itr) {
      GPUError("FAILURE: Invalid reciprocal segment link: %d PrevSegmentNeighbour %d NextSegmentNeighbour %d", itr, track.PrevSegmentNeighbour(), mSectorTrackInfos[track.PrevSegmentNeighbour()].NextSegmentNeighbour());
      nErr++;
    }
    if (track.NextSegmentNeighbour() >= 0 && mSectorTrackInfos[track.NextSegmentNeighbour()].PrevSegmentNeighbour() != itr) {
      GPUError("FAILURE: Invalid reciprocal segment link: %d NextSegmentNeighbour %d PrevSegmentNeighbour %d", itr, track.NextSegmentNeighbour(), mSectorTrackInfos[track.NextSegmentNeighbour()].PrevSegmentNeighbour());
      nErr++;
    }
    if (track.PrevNeighbour() >= 0 && mSectorTrackInfos[track.PrevNeighbour()].NextNeighbour() != itr) {
      GPUError("FAILURE: Invalid reciprocal  link: %d PrevNeighbour %d NextNeighbour %d", itr, track.PrevNeighbour(), mSectorTrackInfos[track.PrevNeighbour()].NextNeighbour());
      nErr++;
    }
    if (track.NextNeighbour() >= 0 && mSectorTrackInfos[track.NextNeighbour()].PrevNeighbour() != itr) {
      GPUError("FAILURE: Invalid reciprocal  link: %d NextNeighbour %d PrevNeighbour %d", itr, track.NextNeighbour(), mSectorTrackInfos[track.NextNeighbour()].PrevNeighbour());
      nErr++;
    }
    if (track.PrevSegmentNeighbour() >= 0) {
      continue;
    }
    if (track.PrevNeighbour() >= 0) {
      continue;
    }
    GPUTPCGMSectorTrack *trbase = &track, *tr = &track;
    while (true) {
      int32_t iTrk = tr - mSectorTrackInfos;
      if (trkUsed[iTrk]) {
        GPUError("FAILURE: double use");
        nErr++;
        break;
      }
      trkUsed[iTrk] = true;

      int32_t jtr = tr->NextSegmentNeighbour();
      if (jtr >= 0) {
        tr = &(mSectorTrackInfos[jtr]);
        if (tr->PrevNeighbour() >= 0) {
          GPUError("FAILURE: Non-base segment has previous leg");
          nErr++;
        }
        continue;
      }
      jtr = trbase->NextNeighbour();
      if (jtr >= 0) {
        trbase = &(mSectorTrackInfos[jtr]);
        tr = trbase;
        if (tr->PrevSegmentNeighbour() >= 0) {
          GPUError("FAILURE: Neibhbour leg has previous segment neightbout");
          nErr++;
          break;
        }
        continue;
      }
      break;
    }
  }
  for (int32_t i = 0; i < SectorTrackInfoLocalTotal(); i++) {
    if (trkUsed[i] == false) {
      GPUError("FAILURE: trk missed");
      nErr++;
    }
  }
  if (nErr == 0) {
    GPUInfo("Merged Track Graph OK");
  } else {
    throw std::runtime_error("Invalid merge graph");
  }
}

void GPUTPCGMMerger::PrintMergeGraph(const GPUTPCGMSectorTrack* trk, std::ostream& out) const
{
  const GPUTPCGMSectorTrack* orgTrack = trk;
  while (trk->PrevSegmentNeighbour() >= 0) {
    trk = &mSectorTrackInfos[trk->PrevSegmentNeighbour()];
  }
  const GPUTPCGMSectorTrack* orgTower = trk;
  while (trk->PrevNeighbour() >= 0) {
    trk = &mSectorTrackInfos[trk->PrevNeighbour()];
  }

  int32_t nextId = trk - mSectorTrackInfos;
  out << "Graph of track " << (orgTrack - mSectorTrackInfos) << "\n";
  while (nextId >= 0) {
    trk = &mSectorTrackInfos[nextId];
    if (trk->PrevSegmentNeighbour() >= 0) {
      out << "TRACK TREE INVALID!!! " << trk->PrevSegmentNeighbour() << " --> " << nextId << "\n";
    }
    out << (trk == orgTower ? "--" : "  ");
    while (nextId >= 0) {
      GPUTPCGMSectorTrack* trk2 = &mSectorTrackInfos[nextId];
      if (trk != trk2 && (trk2->PrevNeighbour() >= 0 || trk2->NextNeighbour() >= 0)) {
        out << "   (TRACK TREE INVALID!!! " << trk2->PrevNeighbour() << " <-- " << nextId << " --> " << trk2->NextNeighbour() << ")   ";
      }
      char tmp[128];
      snprintf(tmp, 128, " %s%5d(%5.2f)", trk2 == orgTrack ? "!" : " ", nextId, trk2->QPt());
      out << tmp;
      nextId = trk2->NextSegmentNeighbour();
    }
    out << "\n";
    nextId = trk->NextNeighbour();
  }
}

void GPUTPCGMMerger::InitializeProcessor() {}

void* GPUTPCGMMerger::SetPointersMerger(void* mem)
{
  computePointerWithAlignment(mem, mSectorTrackInfos, mNTotalSectorTracks);
  computePointerWithAlignment(mem, mSectorTrackInfoIndex, NSECTORS * 2 + 1);
  if (mRec->GetProcessingSettings().deterministicGPUReconstruction) {
    computePointerWithAlignment(mem, mTmpSortMemory, std::max(mNTotalSectorTracks, mNMaxTracks * 2));
  }

  void* memBase = mem;
  computePointerWithAlignment(mem, mBorderMemory, 2 * mNTotalSectorTracks); // MergeBorders & Resolve
  computePointerWithAlignment(mem, mBorderRangeMemory, 2 * mNTotalSectorTracks);
  int32_t nTracks = 0;
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    const int32_t n = *mRec->GetConstantMem().tpcTrackers[iSector].NTracks();
    mBorder[iSector] = mBorderMemory + 2 * nTracks;
    mBorder[NSECTORS + iSector] = mBorderMemory + 2 * nTracks + n;
    mBorderRange[iSector] = mBorderRangeMemory + 2 * nTracks;
    nTracks += n;
  }
  computePointerWithAlignment(mem, mTrackLinks, mNTotalSectorTracks);
  computePointerWithAlignment(mem, mTrackCCRoots, mNTotalSectorTracks);
  void* memMax = mem;
  mem = memBase;
  computePointerWithAlignment(mem, mTrackIDs, GPUCA_NSECTORS * mNMaxSingleSectorTracks); // UnpackResetIds - RefitSectorTracks - UnpackSectorGlobal
  memMax = (void*)std::max((size_t)mem, (size_t)memMax);
  mem = memBase;
  computePointerWithAlignment(mem, mTrackSort, mNMaxTracks); // PrepareForFit0 - SortTracksQPt - PrepareForFit1 - PrepareForFit1 / Finalize0 - Finalize2
  computePointerWithAlignment(mem, mSharedCount, mNMaxClusters);
  memMax = (void*)std::max((size_t)mem, (size_t)memMax);
  mem = memBase;
  computePointerWithAlignment(mem, mLoopData, mNMaxTracks);      // GPUTPCGMMergerTrackFit - GPUTPCGMMergerFollowLoopers
  computePointerWithAlignment(mem, mRetryRefitIds, mNMaxTracks); // Reducing mNMaxTracks for mLoopData / mRetryRefitIds does not save memory, since the other parts are larger anyway
  memMax = (void*)std::max((size_t)mem, (size_t)memMax);
  mem = memBase;
  computePointerWithAlignment(mem, mLooperCandidates, mNMaxLooperMatches); // MergeLoopers 1-3
  memMax = (void*)std::max((size_t)mem, (size_t)memMax);
  return memMax;
}

void* GPUTPCGMMerger::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory);
  return mem;
}

void* GPUTPCGMMerger::SetPointersRefitScratch(void* mem)
{
  computePointerWithAlignment(mem, mTrackOrderAttach, mNMaxTracks);
  const bool mergerSortTracks = mRec->GetProcessingSettings().mergerSortTracks == -1 ? mRec->getGPUParameters(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCMerging).par_SORT_BEFORE_FIT : mRec->GetProcessingSettings().mergerSortTracks;
  if (mergerSortTracks) {
    computePointerWithAlignment(mem, mTrackOrderProcess, mNMaxTracks);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mMergedTracks, mNMaxTracks);
  if (mRec->GetParam().dodEdxEnabled) {
    computePointerWithAlignment(mem, mMergedTracksdEdx, mNMaxTracks);
    if (mRec->GetParam().rec.tpc.dEdxClusterRejectionFlagMask != mRec->GetParam().rec.tpc.dEdxClusterRejectionFlagMaskAlt) {
      computePointerWithAlignment(mem, mMergedTracksdEdxAlt, mNMaxTracks);
    }
  }
  computePointerWithAlignment(mem, mClusters, mNMaxMergedTrackClusters);
  computePointerWithAlignment(mem, mClusterAttachment, mNMaxClusters);
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputState(void* mem)
{
  if ((mRec->GetRecoSteps() & gpudatatypes::RecoStep::Refit) || mRec->GetProcessingSettings().outputSharedClusterMap) {
    computePointerWithAlignment(mem, mClusterStateExt, mNMaxClusters);
  } else {
    mClusterStateExt = nullptr;
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputO2(void* mem)
{
  computePointerWithAlignment(mem, mOutputTracksTPCO2, HostProcessor(this).NOutputTracksTPCO2());
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputO2Clus(void* mem)
{
  computePointerWithAlignment(mem, mOutputClusRefsTPCO2, HostProcessor(this).NOutputClusRefsTPCO2());
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputO2MC(void* mem)
{
  computePointerWithAlignment(mem, mOutputTracksTPCO2MC, HostProcessor(this).NOutputTracksTPCO2());
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputO2Scratch(void* mem)
{
  computePointerWithAlignment(mem, mTrackSortO2, mNMaxTracks);
  computePointerWithAlignment(mem, mClusRefTmp, mNMaxTracks);
  return mem;
}

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersMerger, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCMerger");
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersRefitScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCMergerRefitScratch");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutput, (mRec->GetProcessingSettings().createO2Output > 1 ? GPUMemoryResource::MEMORY_SCRATCH : GPUMemoryResource::MEMORY_OUTPUT) | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutput");
  mMemoryResOutputState = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputState, (mRec->GetProcessingSettings().outputSharedClusterMap ? GPUMemoryResource::MEMORY_OUTPUT : GPUMemoryResource::MEMORY_GPU) | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputState");
  if (mRec->GetProcessingSettings().createO2Output) {
    mMemoryResOutputO2Scratch = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputO2Scratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputO2Scratch");
    mMemoryResOutputO2 = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputO2, GPUMemoryResource::MEMORY_OUTPUT | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputO2");
    mMemoryResOutputO2Clus = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputO2Clus, GPUMemoryResource::MEMORY_OUTPUT | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputO2Clus");
    if (mRec->GetProcessingSettings().runMC) {
      mMemoryResOutputO2MC = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputO2MC, GPUMemoryResource::MEMORY_OUTPUT_FLAG | GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputO2MC");
    }
  }
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "TPCMergerMemory");
}

void GPUTPCGMMerger::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNTotalSectorTracks = 0;
  mNClusters = 0;
  mNMaxSingleSectorTracks = 0;
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    uint32_t ntrk = *mRec->GetConstantMem().tpcTrackers[iSector].NTracks();
    mNTotalSectorTracks += ntrk;
    mNClusters += *mRec->GetConstantMem().tpcTrackers[iSector].NTrackHits();
    if (mNMaxSingleSectorTracks < ntrk) {
      mNMaxSingleSectorTracks = ntrk;
    }
  }
  mNMaxMergedTrackClusters = mRec->MemoryScalers()->NTPCMergedTrackHits(mNClusters);
  if (CAMath::Abs(Param().polynomialField.GetNominalBz()) < (gpu_common_constants::kZeroFieldCut * gpu_common_constants::kCLight)) {
    mNMaxTracks = mRec->MemoryScalers()->getValue(mNTotalSectorTracks, mNTotalSectorTracks); // 0 magnetic field
  } else {
    mNMaxTracks = mRec->MemoryScalers()->NTPCMergedTracks(mNTotalSectorTracks);
  }
  if (io.clustersNative) {
    mNMaxClusters = io.clustersNative->nClustersTotal;
  } else if (mRec->GetRecoSteps() & gpudatatypes::RecoStep::TPCSectorTracking) {
    mNMaxClusters = 0;
    for (int32_t i = 0; i < NSECTORS; i++) {
      mNMaxClusters += mRec->GetConstantMem().tpcTrackers[i].NHitsTotal();
    }
  } else {
    mNMaxClusters = mNClusters;
  }
  mNMaxLooperMatches = mNMaxClusters / 4; // We have that much scratch memory anyway
}

int32_t GPUTPCGMMerger::CheckSectors()
{
  for (int32_t i = 0; i < NSECTORS; i++) {
    if (mRec->GetConstantMem().tpcTrackers[i].CommonMemory()->nLocalTracks > (int32_t)mNMaxSingleSectorTracks) {
      throw std::runtime_error("mNMaxSingleSectorTracks too small");
    }
  }
  if (!(mRec->GetRecoSteps() & gpudatatypes::RecoStep::TPCSectorTracking)) {
    throw std::runtime_error("Must run also sector tracking");
  }
  return 0;
}

#endif // GPUCA_GPUCODE

GPUd() void GPUTPCGMMerger::ClearTrackLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, bool output)
{
  const int32_t n = output ? mMemory->nMergedTracks : SectorTrackInfoLocalTotal();
  for (int32_t i = iBlock * nThreads + iThread; i < n; i += nThreads * nBlocks) {
    mTrackLinks[i] = -1;
  }
}

GPUd() int32_t GPUTPCGMMerger::RefitSectorTrack(GPUTPCGMSectorTrack& sectorTrack, const GPUTPCTrack* inTrack, float alpha, int32_t sector)
{
  GPUTPCGMPropagator prop;
  prop.SetMaterialTPC();
  prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  prop.SetSeedingErrors(true); // Larger errors for seeds, better since we don't start with good hypothesis
  prop.SetFitInProjections(true); // TODO: Was false, consider reenabling after fitInProjection is fixed
  prop.SetPolynomialField(&Param().polynomialField);
  GPUTPCGMTrackParam trk;
  trk.X() = inTrack->Param().GetX();
  trk.Y() = inTrack->Param().GetY();
  trk.Z() = inTrack->Param().GetZ();
  trk.SinPhi() = inTrack->Param().GetSinPhi();
  trk.DzDs() = inTrack->Param().GetDzDs();
  trk.QPt() = inTrack->Param().GetQPt();
  trk.TOffset() = Param().par.continuousTracking ? GetConstantMem()->calibObjects.fastTransform->convZOffsetToVertexTime(sector, inTrack->Param().GetZOffset(), Param().continuousMaxTimeBin) : 0;
  const auto tmp = sectorTrack.ClusterTN() > sectorTrack.ClusterT0() ? std::array<float, 2>{sectorTrack.ClusterTN(), sectorTrack.ClusterT0()} : std::array<float, 2>{sectorTrack.ClusterT0(), sectorTrack.ClusterTN()};
  trk.ShiftZ(this, sector, tmp[0], tmp[1], inTrack->Param().GetX()); // We do not store the inner / outer cluster X, so we just use the track X instead
  sectorTrack.SetX2(0.f);
  for (int32_t way = 0; way < 2; way++) {
    if (way) {
      prop.SetFitInProjections(true);
      prop.SetPropagateBzOnly(true);
    }
    trk.ResetCovariance();
    prop.SetTrack(&trk, alpha);
    int32_t start = way ? inTrack->NHits() - 1 : 0;
    int32_t end = way ? 0 : (inTrack->NHits() - 1);
    int32_t incr = way ? -1 : 1;
    for (int32_t i = start; i != end; i += incr) {
      float x, y, z;
      int32_t row, flags;
      const GPUTPCTracker& tracker = GetConstantMem()->tpcTrackers[sector];
      const GPUTPCHitId& ic = tracker.TrackHits()[inTrack->FirstHitID() + i];
      int32_t clusterIndex = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex());
      row = ic.RowIndex();
      const ClusterNative& cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[GetConstantMem()->ioPtrs.clustersNative->clusterOffset[sector][0] + clusterIndex];
      flags = cl.getFlags();
      GetConstantMem()->calibObjects.fastTransform->Transform(sector, row, cl.getPad(), cl.getTime(), x, y, z, trk.TOffset());
      if (prop.PropagateToXAlpha(x, alpha, way == 0)) {
        return way == 0;
      }
      trk.ConstrainSinPhi();
      if (prop.Update(y, z, row, Param(), flags & GPUTPCGMMergedTrackHit::clustererAndSharedFlags, 0, false, sector, -1.f, 0.f, 0.f)) { // TODO: Use correct time / avgCharge
        return way == 0;
      }
      trk.ConstrainSinPhi();
    }
    if (way) {
      sectorTrack.SetParam2(trk);
    } else {
      sectorTrack.Set(trk, inTrack, alpha, sector);
    }
  }
  return 0;
}

GPUd() void GPUTPCGMMerger::SetTrackClusterT(GPUTPCGMSectorTrack& track, int32_t iSector, const GPUTPCTrack* sectorTr)
{
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSector];
  const GPUTPCHitId& ic1 = trk.TrackHits()[sectorTr->FirstHitID()];
  const GPUTPCHitId& ic2 = trk.TrackHits()[sectorTr->FirstHitID() + sectorTr->NHits() - 1];
  int32_t clusterIndex1 = trk.Data().ClusterDataIndex(trk.Data().Row(ic1.RowIndex()), ic1.HitIndex());
  int32_t clusterIndex2 = trk.Data().ClusterDataIndex(trk.Data().Row(ic2.RowIndex()), ic2.HitIndex());
  const ClusterNative* cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[iSector][0];
  track.SetClusterT(cl[clusterIndex1].getTime(), cl[clusterIndex2].getTime());
}

GPUd() void GPUTPCGMMerger::UnpackSaveNumber(int32_t id)
{
  mSectorTrackInfoIndex[id] = mMemory->nUnpackedTracks;
}

GPUd() void GPUTPCGMMerger::UnpackSectorGlobal(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector)
{
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSector];
  float alpha = Param().Alpha(iSector);
  const GPUTPCTrack* sectorTr = mMemory->firstExtrapolatedTracks[iSector];
  uint32_t nLocalTracks = trk.CommonMemory()->nLocalTracks;
  uint32_t nTracks = *trk.NTracks();
  for (uint32_t itr = nLocalTracks + iBlock * nThreads + iThread; itr < nTracks; itr += nBlocks * nThreads) {
    sectorTr = &trk.Tracks()[itr];
    int32_t localId = mTrackIDs[((sectorTr->LocalTrackId() >> 24) & 0x3F) * mNMaxSingleSectorTracks + (sectorTr->LocalTrackId() & 0xFFFFFF)];
    if (localId == -1) {
      continue;
    }
    uint32_t myTrack = CAMath::AtomicAdd(&mMemory->nUnpackedTracks, 1u);
    GPUTPCGMSectorTrack& track = mSectorTrackInfos[myTrack];
    SetTrackClusterT(track, iSector, sectorTr);
    track.Set(this, sectorTr, alpha, iSector);
    track.SetGlobalSectorTrackCov();
    track.SetPrevNeighbour(-1);
    track.SetNextNeighbour(-1);
    track.SetNextSegmentNeighbour(-1);
    track.SetPrevSegmentNeighbour(-1);
    track.SetLocalTrackId(localId | (sectorTr->LocalTrackId() & 0x40000000));
  }
}

GPUd() void GPUTPCGMMerger::UnpackResetIds(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector)
{
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSector];
  uint32_t nLocalTracks = trk.CommonMemory()->nLocalTracks;
  for (uint32_t i = iBlock * nThreads + iThread; i < nLocalTracks; i += nBlocks * nThreads) {
    mTrackIDs[iSector * mNMaxSingleSectorTracks + i] = -1;
  }
}

GPUd() void GPUTPCGMMerger::RefitSectorTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector)
{
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSector];
  uint32_t nLocalTracks = trk.CommonMemory()->nLocalTracks;

  float alpha = Param().Alpha(iSector);
  const GPUTPCTrack* sectorTr = nullptr;

  for (uint32_t itr = iBlock * nThreads + iThread; itr < nLocalTracks; itr += nBlocks * nThreads) {
    sectorTr = &trk.Tracks()[itr];
    GPUTPCGMSectorTrack track;
    SetTrackClusterT(track, iSector, sectorTr);
    if (RefitSectorTrack(track, sectorTr, alpha, iSector)) {
      track.Set(this, sectorTr, alpha, iSector); // TODO: Why does the refit fail, it shouldn't, this workaround should be removed
      if (!track.FilterErrors(this, iSector, GPUCA_MAX_SIN_PHI, 0.1f)) {
        continue;
      }
    }

    CADEBUG(GPUInfo("INPUT Sector %d, Track %u, QPt %f DzDs %f", iSector, itr, track.QPt(), track.DzDs()));
    track.SetPrevNeighbour(-1);
    track.SetNextNeighbour(-1);
    track.SetNextSegmentNeighbour(-1);
    track.SetPrevSegmentNeighbour(-1);
    track.SetExtrapolatedTrackId(0, -1);
    track.SetExtrapolatedTrackId(1, -1);
    uint32_t myTrack = CAMath::AtomicAdd(&mMemory->nUnpackedTracks, 1u);
    mTrackIDs[iSector * mNMaxSingleSectorTracks + sectorTr->LocalTrackId()] = myTrack;
    mSectorTrackInfos[myTrack] = track;
  }
}

GPUd() void GPUTPCGMMerger::LinkExtrapolatedTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (int32_t itr = SectorTrackInfoGlobalFirst(0) + iBlock * nThreads + iThread; itr < SectorTrackInfoGlobalLast(NSECTORS - 1); itr += nThreads * nBlocks) {
    GPUTPCGMSectorTrack& extrapolatedTrack = mSectorTrackInfos[itr];
    GPUTPCGMSectorTrack& localTrack = mSectorTrackInfos[extrapolatedTrack.LocalTrackId() & 0xFFFFFF];
    int up = (extrapolatedTrack.LocalTrackId() & 0x40000000) ? 1 : 0;
    localTrack.SetExtrapolatedTrackId(up, itr);
  }
}

GPUd() void GPUTPCGMMerger::MergeSectorsPrepareStep2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(uint32_t) * nB, bool useOrigTrackParam)
{
  //* prepare sector tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = Param().bzCLight;

  float dAlpha = Param().dAlpha / 2;
  float x0 = 0;

  if (iBorder == 0) { // transport to the left edge of the sector and rotate horizontally
    dAlpha = dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 1) { // transport to the right edge of the sector and rotate horizontally
    dAlpha = -dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 2) { // transport to the middle of the sector and rotate vertically to the border on the left
    x0 = GPUTPCGeometry::Row2X(63);
  } else if (iBorder == 3) { // transport to the middle of the sector and rotate vertically to the border on the right
    dAlpha = -dAlpha;
    x0 = GPUTPCGeometry::Row2X(63);
  }

  const float maxSin = CAMath::Sin(60.f / 180.f * CAMath::Pi());
  float cosAlpha = CAMath::Cos(dAlpha);
  float sinAlpha = CAMath::Sin(dAlpha);

  GPUTPCGMSectorTrack trackTmp;
  for (int32_t itr = iBlock * nThreads + iThread; itr < SectorTrackInfoLocalTotal(); itr += nThreads * nBlocks) {
    const GPUTPCGMSectorTrack* track = &mSectorTrackInfos[itr];
    int32_t iSector = track->Sector();

    if (track->PrevSegmentNeighbour() >= 0 && track->Sector() == mSectorTrackInfos[track->PrevSegmentNeighbour()].Sector()) {
      continue;
    }
    if (useOrigTrackParam) { // TODO: Check how far this makes sense with sector track refit
      if (CAMath::Abs(track->QPt()) * Param().qptB5Scaler < Param().rec.tpc.mergerLooperQPtB5Limit) {
        continue;
      }
      const GPUTPCGMSectorTrack* trackMin = track;
      while (track->NextSegmentNeighbour() >= 0 && track->Sector() == mSectorTrackInfos[track->NextSegmentNeighbour()].Sector()) {
        track = &mSectorTrackInfos[track->NextSegmentNeighbour()];
        if (track->OrigTrack()->Param().X() < trackMin->OrigTrack()->Param().X()) {
          trackMin = track;
        }
      }
      trackTmp = *trackMin;
      track = &trackTmp;
      if (trackTmp.X2() != 0.f) {
        trackTmp.UseParam2();
      } else {
        trackTmp.Set(this, trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Sector());
      }
    } else {
      if (CAMath::Abs(track->QPt()) * Param().qptB5Scaler < Param().rec.tpc.mergerLooperSecondHorizontalQPtB5Limit) {
        if (iBorder == 0 && track->NextNeighbour() >= 0) {
          continue;
        }
        if (iBorder == 1 && track->PrevNeighbour() >= 0) {
          continue;
        }
      }
    }
    GPUTPCGMBorderTrack b;

    if (track->TransportToXAlpha(this, x0, sinAlpha, cosAlpha, fieldBz, b, maxSin)) {
      b.SetTrackID(itr);
      b.SetNClusters(track->NClusters());
      for (int32_t i = 0; i < 4; i++) {
        if (CAMath::Abs(b.Cov()[i]) >= 5.0f) {
          b.SetCov(i, 5.0f);
        }
      }
      if (CAMath::Abs(b.Cov()[4]) >= 0.5f) {
        b.SetCov(4, 0.5f);
      }
      uint32_t myTrack = CAMath::AtomicAdd(&nB[iSector], 1u);
      B[iSector][myTrack] = b;
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector1, const GPUTPCGMBorderTrack* B1, int32_t N1, int32_t iSector2, const GPUTPCGMBorderTrack* B2, int32_t N2, int32_t mergeMode)
{
  CADEBUG(GPUInfo("\nMERGING Sectors %d %d NTracks %d %d CROSS %d", iSector1, iSector2, N1, N2, mergeMode));
  GPUTPCGMBorderRange* range1 = mBorderRange[iSector1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSector2] + *GetConstantMem()->tpcTrackers[iSector2].NTracks();
  bool sameSector = (iSector1 == iSector2);
  for (int32_t itr = iBlock * nThreads + iThread; itr < N1; itr += nThreads * nBlocks) {
    const GPUTPCGMBorderTrack& b = B1[itr];
    float d = CAMath::Max(0.5f, 3.5f * CAMath::Sqrt(b.Cov()[1]));
    if (CAMath::Abs(b.Par()[4]) * Param().qptB5Scaler >= 20) {
      d *= 2;
    } else if (d > 3) {
      d = 3;
    }
    CADEBUG(printf("  Input Sector 1 %d Track %d: ", iSector1, itr); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf(" - D %8.3f\n", d));
    GPUTPCGMBorderRange range;
    range.fId = itr;
    range.fMin = b.Par()[1] + b.ZOffsetLinear() - d;
    range.fMax = b.Par()[1] + b.ZOffsetLinear() + d;
    range1[itr] = range;
    if (sameSector) {
      range2[itr] = range;
    }
  }
  if (!sameSector) {
    for (int32_t itr = iBlock * nThreads + iThread; itr < N2; itr += nThreads * nBlocks) {
      const GPUTPCGMBorderTrack& b = B2[itr];
      float d = CAMath::Max(0.5f, 3.5f * CAMath::Sqrt(b.Cov()[1]));
      if (CAMath::Abs(b.Par()[4]) * Param().qptB5Scaler >= 20) {
        d *= 2;
      } else if (d > 3) {
        d = 3;
      }
      CADEBUG(printf("  Input Sector 2 %d Track %d: ", iSector2, itr); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf(" - D %8.3f\n", d));
      GPUTPCGMBorderRange range;
      range.fId = itr;
      range.fMin = b.Par()[1] + b.ZOffsetLinear() - d;
      range.fMax = b.Par()[1] + b.ZOffsetLinear() + d;
      range2[itr] = range;
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector1, const GPUTPCGMBorderTrack* B1, int32_t N1, int32_t iSector2, const GPUTPCGMBorderTrack* B2, int32_t N2, int32_t mergeMode)
{
#if !defined(GPUCA_GPUCODE_COMPILEKERNELS)
  GPUTPCGMBorderRange* range1 = mBorderRange[iSector1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSector2] + *GetConstantMem()->tpcTrackers[iSector2].NTracks();

  if (iThread == 0) {
    if (iBlock == 0) {
      GPUCommonAlgorithm::sortDeviceDynamic(range1, range1 + N1, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return GPUCA_DETERMINISTIC_CODE((a.fMin != b.fMin) ? (a.fMin < b.fMin) : (a.fId < b.fId), a.fMin < b.fMin); });
    } else if (iBlock == 1) {
      GPUCommonAlgorithm::sortDeviceDynamic(range2, range2 + N2, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return GPUCA_DETERMINISTIC_CODE((a.fMax != b.fMax) ? (a.fMax < b.fMax) : (a.fId < b.fId), a.fMax < b.fMax); });
    }
  }
#else
  printf("This sorting variant is disabled for RTC");
#endif
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<3>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUTPCGMBorderRange* range, int32_t N, int32_t cmpMax)
{
#ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  if (iThread == 0 && iBlock == 0) {
    if (cmpMax) {
      GPUCommonAlgorithm::sortDeviceDynamic(range, range + N, MergeBorderTracks_compMax());
    } else {
      GPUCommonAlgorithm::sortDeviceDynamic(range, range + N, MergeBorderTracks_compMin());
    }
  }
#endif
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector1, const GPUTPCGMBorderTrack* B1, int32_t N1, int32_t iSector2, const GPUTPCGMBorderTrack* B2, int32_t N2, int32_t mergeMode)
{
  // int32_t statAll = 0, statMerged = 0;
  float factor2ys = Param().rec.tpc.trackMergerFactor2YS;
  float factor2zt = Param().rec.tpc.trackMergerFactor2ZT;
  float factor2k = Param().rec.tpc.trackMergerFactor2K;
  float factor2General = Param().rec.tpc.trackMergerFactor2General;

  factor2k = factor2General * factor2k;
  factor2ys = factor2General * factor2ys;
  factor2zt = factor2General * factor2zt;

  int32_t minNPartHits = Param().rec.tpc.trackMergerMinPartHits;
  int32_t minNTotalHits = Param().rec.tpc.trackMergerMinTotalHits;

  bool sameSector = (iSector1 == iSector2);

  GPUTPCGMBorderRange* range1 = mBorderRange[iSector1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSector2] + *GetConstantMem()->tpcTrackers[iSector2].NTracks();

  int32_t i2 = 0;
  for (int32_t i1 = iBlock * nThreads + iThread; i1 < N1; i1 += nThreads * nBlocks) {
    GPUTPCGMBorderRange r1 = range1[i1];
    while (i2 < N2 && range2[i2].fMax < r1.fMin) {
      i2++;
    }

    const GPUTPCGMBorderTrack& b1 = B1[r1.fId];
    if (b1.NClusters() < minNPartHits) {
      continue;
    }
    int32_t iBest2 = -1;
    int32_t lBest2 = 0;
    // statAll++;
    for (int32_t k2 = i2; k2 < N2; k2++) {
      GPUTPCGMBorderRange r2 = range2[k2];
      if (r2.fMin > r1.fMax) {
        break;
      }
      if (sameSector && (r1.fId >= r2.fId)) {
        continue;
      }
      // do check

      const GPUTPCGMBorderTrack& b2 = B2[r2.fId];
#if defined(GPUCA_MERGER_BY_MC_LABEL) && !defined(GPUCA_GPUCODE)
      int64_t label1 = GetTrackLabel(b1);
      int64_t label2 = GetTrackLabel(b2);
      if (label1 != label2 && label1 != -1) // DEBUG CODE, match by MC label
#endif
      {
        CADEBUG(if (GetConstantMem()->ioPtrs.mcLabelsTPC) {printf("Comparing track %3d to %3d: ", r1.fId, r2.fId); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b1.Par()[i]); } printf(" - "); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b1.Cov()[i]); } printf("\n%28s", ""); });
        CADEBUG(if (GetConstantMem()->ioPtrs.mcLabelsTPC) {for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b2.Par()[i]); } printf(" - "); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b2.Cov()[i]); } printf("   -   %5s   -   ", GetTrackLabel(b1) == GetTrackLabel(b2) ? "CLONE" : "FAKE"); });
        if (b2.NClusters() < lBest2) {
          CADEBUG2(continue, printf("!NCl1\n"));
        }
        if (mergeMode > 0) {
          // Merging CE tracks
          int32_t maxRowDiff = mergeMode == 2 ? 1 : 3; // TODO: check cut
          if (CAMath::Abs(b1.Row() - b2.Row()) > maxRowDiff) {
            CADEBUG2(continue, printf("!ROW\n"));
          }
          if (CAMath::Abs(b1.Par()[2] - b2.Par()[2]) > 0.5f || CAMath::Abs(b1.Par()[3] - b2.Par()[3]) > 0.5f) {
            CADEBUG2(continue, printf("!CE SinPhi/Tgl\n")); // Crude cut to avoid totally wrong matches, TODO: check cut
          }
        }

        GPUCA_DEBUG_STREAMER_CHECK(float weight = b1.Par()[4] * b1.Par()[4]; if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamMergeBorderTracksAll, b1.TrackID(), weight)) { MergedTrackStreamer(b1, b2, "merge_all_tracks", iSector1, iSector2, mergeMode, weight, o2::utils::DebugStreamer::getSamplingFrequency(o2::utils::StreamFlags::streamMergeBorderTracksAll)); });

        if (!b1.CheckChi2Y(b2, factor2ys)) {
          CADEBUG2(continue, printf("!Y\n"));
        }
        // if( !b1.CheckChi2Z(b2, factor2zt ) ) CADEBUG2(continue, printf("!NCl1\n"));
        if (!b1.CheckChi2QPt(b2, factor2k)) {
          CADEBUG2(continue, printf("!QPt\n"));
        }
        float fys = CAMath::Abs(b1.Par()[4]) * Param().qptB5Scaler < 20 ? factor2ys : (2.f * factor2ys);
        float fzt = CAMath::Abs(b1.Par()[4]) * Param().qptB5Scaler < 20 ? factor2zt : (2.f * factor2zt);
        if (!b1.CheckChi2YS(b2, fys)) {
          CADEBUG2(continue, printf("!YS\n"));
        }
        if (!b1.CheckChi2ZT(b2, fzt)) {
          CADEBUG2(continue, printf("!ZT\n"));
        }
        if (CAMath::Abs(b1.Par()[4]) * Param().qptB5Scaler < 20) {
          if (b2.NClusters() < minNPartHits) {
            CADEBUG2(continue, printf("!NCl2\n"));
          }
          if (b1.NClusters() + b2.NClusters() < minNTotalHits) {
            CADEBUG2(continue, printf("!NCl3\n"));
          }
        }
        CADEBUG(printf("OK: dZ %8.3f D1 %8.3f D2 %8.3f\n", CAMath::Abs(b1.Par()[1] - b2.Par()[1]), 3.5f * sqrt(b1.Cov()[1]), 3.5f * sqrt(b2.Cov()[1])));
      } // DEBUG CODE, match by MC label
      lBest2 = b2.NClusters();
      iBest2 = b2.TrackID();
    }

    if (iBest2 < 0) {
      continue;
    }
    GPUCA_DEBUG_STREAMER_CHECK(float weight = b1.Par()[4] * b1.Par()[4]; if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamMergeBorderTracksBest, b1.TrackID(), weight)) { MergedTrackStreamer(b1, MergedTrackStreamerFindBorderTrack(B2, N2, iBest2), "merge_best_track", iSector1, iSector2, mergeMode, weight, o2::utils::DebugStreamer::getSamplingFrequency(o2::utils::StreamFlags::streamMergeBorderTracksBest)); });

    // statMerged++;

    CADEBUG(GPUInfo("Found match %d %d", b1.TrackID(), iBest2));

    mTrackLinks[b1.TrackID()] = iBest2;
    if (mergeMode > 0) {
      GPUCA_DETERMINISTIC_CODE(CAMath::AtomicMax(&mTrackLinks[iBest2], b1.TrackID()), mTrackLinks[iBest2] = b1.TrackID());
    }
  }
  // GPUInfo("STAT: sectors %d, %d: all %d merged %d", iSector1, iSector2, statAll, statMerged);
}

GPUdii() void GPUTPCGMMerger::MergeBorderTracksSetup(int32_t& n1, int32_t& n2, GPUTPCGMBorderTrack*& b1, GPUTPCGMBorderTrack*& b2, int32_t& jSector, int32_t iSector, int8_t withinSector, int8_t mergeMode) const
{
  if (withinSector == 1) { // Merge tracks within the same sector
    jSector = iSector;
    n1 = n2 = mMemory->tmpCounter[iSector];
    b1 = b2 = mBorder[iSector];
  } else if (withinSector == -1) { // Merge tracks accross the central electrode
    jSector = (iSector + NSECTORS / 2);
    const int32_t offset = mergeMode == 2 ? NSECTORS : 0;
    n1 = mMemory->tmpCounter[iSector + offset];
    n2 = mMemory->tmpCounter[jSector + offset];
    b1 = mBorder[iSector + offset];
    b2 = mBorder[jSector + offset];
  } else { // Merge tracks of adjacent sectors
    jSector = mNextSectorInd[iSector];
    n1 = mMemory->tmpCounter[iSector];
    n2 = mMemory->tmpCounter[NSECTORS + jSector];
    b1 = mBorder[iSector];
    b2 = mBorder[NSECTORS + jSector];
  }
}

template <int32_t I>
GPUd() void GPUTPCGMMerger::MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector, int8_t withinSector, int8_t mergeMode)
{
  int32_t n1, n2;
  GPUTPCGMBorderTrack *b1, *b2;
  int32_t jSector;
  MergeBorderTracksSetup(n1, n2, b1, b2, jSector, iSector, withinSector, mergeMode);
  MergeBorderTracks<I>(nBlocks, nThreads, iBlock, iThread, iSector, b1, n1, jSector, b2, n2, mergeMode);
}

#if !defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_DEVICE) // FIXME: DR: WORKAROUND to avoid CUDA bug creating host symbols for device code.
template GPUdni() void GPUTPCGMMerger::MergeBorderTracks<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector, int8_t withinSector, int8_t mergeMode);
template GPUdni() void GPUTPCGMMerger::MergeBorderTracks<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector, int8_t withinSector, int8_t mergeMode);
template GPUdni() void GPUTPCGMMerger::MergeBorderTracks<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector, int8_t withinSector, int8_t mergeMode);
#endif

GPUd() void GPUTPCGMMerger::MergeWithinSectorsPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  float x0 = GPUTPCGeometry::Row2X(63);
  const float maxSin = CAMath::Sin(60.f / 180.f * CAMath::Pi());

  for (int32_t itr = iBlock * nThreads + iThread; itr < SectorTrackInfoLocalTotal(); itr += nThreads * nBlocks) {
    const GPUTPCGMSectorTrack& track = mSectorTrackInfos[itr];
    int32_t iSector = track.Sector();
    GPUTPCGMBorderTrack b;
    if (track.TransportToX(this, x0, Param().bzCLight, b, maxSin)) {
      b.SetTrackID(itr);
      CADEBUG(printf("WITHIN SECTOR %d Track %d - ", iSector, itr); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int32_t i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf("\n"));
      b.SetNClusters(track.NClusters());
      uint32_t myTrack = CAMath::AtomicAdd(&mMemory->tmpCounter[iSector], 1u);
      mBorder[iSector][myTrack] = b;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeSectorsPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t border0, int32_t border1, int8_t useOrigTrackParam)
{
  bool part2 = iBlock & 1;
  int32_t border = part2 ? border1 : border0;
  GPUAtomic(uint32_t)* n = mMemory->tmpCounter;
  GPUTPCGMBorderTrack** b = mBorder;
  if (part2) {
    n += NSECTORS;
    b += NSECTORS;
  }
  MergeSectorsPrepareStep2((nBlocks + !part2) >> 1, nThreads, iBlock >> 1, iThread, border, b, n, useOrigTrackParam);
}

GPUdi() void GPUTPCGMMerger::setBlockRange(int32_t elems, int32_t nBlocks, int32_t iBlock, int32_t& start, int32_t& end)
{
  start = (elems + nBlocks - 1) / nBlocks * iBlock;
  end = (elems + nBlocks - 1) / nBlocks * (iBlock + 1);
  end = CAMath::Min(elems, end);
}

GPUd() void GPUTPCGMMerger::hookEdge(int32_t u, int32_t v)
{
  if (v < 0) {
    return;
  }
  while (true) {
    u = mTrackCCRoots[u];
    v = mTrackCCRoots[v];
    if (u == v) {
      break;
    }
    int32_t h = CAMath::Max(u, v);
    int32_t l = CAMath::Min(u, v);

    int32_t old = CAMath::AtomicCAS(&mTrackCCRoots[h], h, l);
    if (old == h) {
      break;
    }

    u = mTrackCCRoots[h];
    v = l;
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsSetup(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  int32_t start, end;
  setBlockRange(SectorTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int32_t i = start + iThread; i < end; i += nThreads) {
    mTrackCCRoots[i] = i;
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsHookLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  // Compute connected components in parallel, step 1.
  // Source: Adaptive Work-Efficient Connected Components on the GPU, Sutton et al, 2016 (https://arxiv.org/pdf/1612.01178.pdf)
  int32_t start, end;
  setBlockRange(SectorTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int32_t itr = start + iThread; itr < end; itr += nThreads) {
    hookEdge(itr, mTrackLinks[itr]);
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsHookNeighbors(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  // Compute connected components in parallel, step 1 - Part 2.
  nBlocks = nBlocks / 4 * 4;
  if (iBlock >= nBlocks) {
    return;
  }

  int32_t start, end;
  setBlockRange(SectorTrackInfoLocalTotal(), nBlocks / 4, iBlock / 4, start, end);

  int32_t myNeighbor = iBlock % 4;

  for (int32_t itr = start + iThread; itr < end; itr += nThreads) {
    int32_t v = mSectorTrackInfos[itr].AnyNeighbour(myNeighbor);
    hookEdge(itr, v);
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsMultiJump(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  // Compute connected components in parallel, step 2.
  int32_t start, end;
  setBlockRange(SectorTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int32_t itr = start + iThread; itr < end; itr += nThreads) {
    int32_t root = itr;
    int32_t next = mTrackCCRoots[root];
    if (root == next) {
      continue;
    }
    do {
      root = next;
      next = mTrackCCRoots[next];
    } while (root != next);
    mTrackCCRoots[itr] = root;
  }
}

GPUd() void GPUTPCGMMerger::ResolveMergeSectors(GPUResolveSharedMemory& smem, int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int8_t useOrigTrackParam, int8_t mergeAll)
{
  if (!mergeAll) {
    /*int32_t neighborType = useOrigTrackParam ? 1 : 0;
    int32_t old1 = newTrack2.PrevNeighbour(0);
    int32_t old2 = newTrack1.NextNeighbour(0);
    if (old1 < 0 && old2 < 0) neighborType = 0;
    if (old1 == itr) continue;
    if (neighborType) old1 = newTrack2.PrevNeighbour(1);
    if ( old1 >= 0 )
    {
        GPUTPCGMSectorTrack &oldTrack1 = mSectorTrackInfos[old1];
        if ( oldTrack1.NClusters() < newTrack1.NClusters() ) {
            newTrack2.SetPrevNeighbour( -1, neighborType );
            oldTrack1.SetNextNeighbour( -1, neighborType );
        } else continue;
    }

    if (old2 == itr2) continue;
    if (neighborType) old2 = newTrack1.NextNeighbour(1);
    if ( old2 >= 0 )
    {
        GPUTPCGMSectorTrack &oldTrack2 = mSectorTrackInfos[old2];
        if ( oldTrack2.NClusters() < newTrack2.NClusters() )
        {
        oldTrack2.SetPrevNeighbour( -1, neighborType );
        } else continue;
    }
    newTrack1.SetNextNeighbour( itr2, neighborType );
    newTrack2.SetPrevNeighbour( itr, neighborType );*/
  }

  int32_t start, end;
  setBlockRange(SectorTrackInfoLocalTotal(), nBlocks, iBlock, start, end);

  for (int32_t baseIdx = 0; baseIdx < SectorTrackInfoLocalTotal(); baseIdx += nThreads) {
    int32_t itr = baseIdx + iThread;
    bool inRange = itr < SectorTrackInfoLocalTotal();

    int32_t itr2 = -1;
    if (inRange) {
      itr2 = mTrackLinks[itr];
    }

    bool resolveSector = (itr2 > -1);
    if (resolveSector) {
      int32_t root = mTrackCCRoots[itr];
      resolveSector &= (start <= root) && (root < end);
    }

    int16_t smemIdx = work_group_scan_inclusive_add(int16_t(resolveSector));

    if (resolveSector) {
      smem.iTrack1[smemIdx - 1] = itr;
      smem.iTrack2[smemIdx - 1] = itr2;
    }
    GPUbarrier();

    if (iThread < nThreads - 1) {
      continue;
    }

    const int32_t nSectors = smemIdx;

    for (int32_t i = 0; i < nSectors; i++) {
      itr = smem.iTrack1[i];
      itr2 = smem.iTrack2[i];

      GPUTPCGMSectorTrack* track1 = &mSectorTrackInfos[itr];
      GPUTPCGMSectorTrack* track2 = &mSectorTrackInfos[itr2];
      GPUTPCGMSectorTrack* track1Base = track1;
      GPUTPCGMSectorTrack* track2Base = track2;

      bool sameSegment = CAMath::Abs(track1->NClusters() > track2->NClusters() ? track1->QPt() : track2->QPt()) * Param().qptB5Scaler < 2 || track1->QPt() * track2->QPt() > 0;
      // GPUInfo("\nMerge %d with %d - same segment %d", itr, itr2, (int32_t) sameSegment);
      // PrintMergeGraph(track1, std::cout);
      // PrintMergeGraph(track2, std::cout);

      while (track2->PrevSegmentNeighbour() >= 0) {
        track2 = &mSectorTrackInfos[track2->PrevSegmentNeighbour()];
      }
      if (sameSegment) {
        if (track1 == track2) {
          continue;
        }
        while (track1->PrevSegmentNeighbour() >= 0) {
          track1 = &mSectorTrackInfos[track1->PrevSegmentNeighbour()];
          if (track1 == track2) {
            goto NextTrack;
          }
        }
        GPUCommonAlgorithm::swap(track1, track1Base);
        for (int32_t k = 0; k < 2; k++) {
          GPUTPCGMSectorTrack* tmp = track1Base;
          while (tmp->Neighbour(k) >= 0) {
            tmp = &mSectorTrackInfos[tmp->Neighbour(k)];
            if (tmp == track2) {
              goto NextTrack;
            }
          }
        }

        while (track1->NextSegmentNeighbour() >= 0) {
          track1 = &mSectorTrackInfos[track1->NextSegmentNeighbour()];
          if (track1 == track2) {
            goto NextTrack;
          }
        }
      } else {
        while (track1->PrevSegmentNeighbour() >= 0) {
          track1 = &mSectorTrackInfos[track1->PrevSegmentNeighbour()];
        }

        if (track1 == track2) {
          continue;
        }
        for (int32_t k = 0; k < 2; k++) {
          GPUTPCGMSectorTrack* tmp = track1;
          while (tmp->Neighbour(k) >= 0) {
            tmp = &mSectorTrackInfos[tmp->Neighbour(k)];
            if (tmp == track2) {
              goto NextTrack;
            }
          }
        }

        float z1min, z1max, z2min, z2max;
        z1min = track1->MinClusterT();
        z1max = track1->MaxClusterT();
        z2min = track2->MinClusterT();
        z2max = track2->MaxClusterT();
        if (track1 != track1Base) {
          z1min = CAMath::Min(z1min, track1Base->MinClusterT());
          z1max = CAMath::Max(z1max, track1Base->MaxClusterT());
        }
        if (track2 != track2Base) {
          z2min = CAMath::Min(z2min, track2Base->MinClusterT());
          z2max = CAMath::Max(z2max, track2Base->MaxClusterT());
        }
        bool goUp = z2max - z1min > z1max - z2min;

        if (track1->Neighbour(goUp) < 0 && track2->Neighbour(!goUp) < 0) {
          track1->SetNeighbor(track2 - mSectorTrackInfos, goUp);
          track2->SetNeighbor(track1 - mSectorTrackInfos, !goUp);
          // GPUInfo("Result (simple neighbor)");
          // PrintMergeGraph(track1, std::cout);
          continue;
        } else if (track1->Neighbour(goUp) < 0) {
          track2 = &mSectorTrackInfos[track2->Neighbour(!goUp)];
          GPUCommonAlgorithm::swap(track1, track2);
        } else if (track2->Neighbour(!goUp) < 0) {
          track1 = &mSectorTrackInfos[track1->Neighbour(goUp)];
        } else { // Both would work, but we use the simpler one
          track1 = &mSectorTrackInfos[track1->Neighbour(goUp)];
        }
        track1Base = track1;
      }

      track2Base = track2;
      if (!sameSegment) {
        while (track1->NextSegmentNeighbour() >= 0) {
          track1 = &mSectorTrackInfos[track1->NextSegmentNeighbour()];
        }
      }
      track1->SetNextSegmentNeighbour(track2 - mSectorTrackInfos);
      track2->SetPrevSegmentNeighbour(track1 - mSectorTrackInfos);
      // k = 0: Merge right side
      // k = 1: Merge left side
      for (int32_t k = 0; k < 2; k++) {
        track1 = track1Base;
        track2 = track2Base;
        while (track2->Neighbour(k) >= 0) {
          if (track1->Neighbour(k) >= 0) {
            GPUTPCGMSectorTrack* track1new = &mSectorTrackInfos[track1->Neighbour(k)];
            GPUTPCGMSectorTrack* track2new = &mSectorTrackInfos[track2->Neighbour(k)];
            track2->SetNeighbor(-1, k);
            track2new->SetNeighbor(-1, k ^ 1);
            track1 = track1new;
            while (track1->NextSegmentNeighbour() >= 0) {
              track1 = &mSectorTrackInfos[track1->NextSegmentNeighbour()];
            }
            track1->SetNextSegmentNeighbour(track2new - mSectorTrackInfos);
            track2new->SetPrevSegmentNeighbour(track1 - mSectorTrackInfos);
            track1 = track1new;
            track2 = track2new;
          } else {
            GPUTPCGMSectorTrack* track2new = &mSectorTrackInfos[track2->Neighbour(k)];
            track1->SetNeighbor(track2->Neighbour(k), k);
            track2->SetNeighbor(-1, k);
            track2new->SetNeighbor(track1 - mSectorTrackInfos, k ^ 1);
          }
        }
      }
      // GPUInfo("Result");
      // PrintMergeGraph(track1, std::cout);
    NextTrack:;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeCEFill(const GPUTPCGMSectorTrack* track, const GPUTPCGMMergedTrackHit& cls, int32_t itr)
{
  if (Param().rec.tpc.mergerCERowLimit > 0 && CAMath::Abs(track->QPt()) * Param().qptB5Scaler < 0.3f && (cls.row < Param().rec.tpc.mergerCERowLimit || cls.row >= GPUCA_NROWS - Param().rec.tpc.mergerCERowLimit)) {
    return;
  }

  float z = 0;
  {
    float x, y;
    auto& cln = mConstantMem->ioPtrs.clustersNative->clustersLinear[cls.num];
    GPUTPCConvertImpl::convert(*mConstantMem, cls.sector, cls.row, cln.getPad(), cln.getTime(), x, y, z);
  }

  if (!Param().par.continuousTracking && CAMath::Abs(z) > 10) {
    return;
  }
  int32_t sector = track->Sector();
  for (int32_t attempt = 0; attempt < 2; attempt++) {
    GPUTPCGMBorderTrack b;
    const float x0 = GPUTPCGeometry::Row2X(attempt == 0 ? 63 : cls.row);
    if (track->TransportToX(this, x0, Param().bzCLight, b, GPUCA_MAX_SIN_PHI_LOW)) {
      b.SetTrackID(itr);
      b.SetNClusters(mMergedTracks[itr].NClusters());
      if (CAMath::Abs(b.Cov()[4]) >= 0.5f) {
        b.SetCov(4, 0.5f); // TODO: Is this needed and better than the cut in BorderTrack?
      }
      if (track->CSide()) {
        b.SetPar(1, b.Par()[1] - 2 * (z - b.ZOffsetLinear()));
        b.SetZOffsetLinear(-b.ZOffsetLinear());
      }
      b.SetRow(cls.row);
      uint32_t id = sector + attempt * NSECTORS;
      uint32_t myTrack = CAMath::AtomicAdd(&mMemory->tmpCounter[id], 1u);
      mBorder[id][myTrack] = b;
      break;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeCE(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  const ClusterNative* cls = mConstantMem->ioPtrs.clustersNative->clustersLinear;
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nThreads * nBlocks) {
    if (mMergedTracks[i].CSide() == 0 && mTrackLinks[i] >= 0) {
      if (mTrackLinks[mTrackLinks[i]] != (int32_t)i) {
        continue;
      }
      GPUTPCGMMergedTrack* trk[2] = {&mMergedTracks[i], &mMergedTracks[mTrackLinks[i]]};

      if (!trk[1]->OK() || trk[1]->CCE()) {
        continue;
      }
      bool celooper = (trk[0]->GetParam().GetQPt() * Param().qptB5Scaler > 1 && trk[0]->GetParam().GetQPt() * trk[1]->GetParam().GetQPt() < 0);
      celooper |= trk[0]->PrevSegment() != -1 && trk[1]->PrevSegment() != -1;
      if (!celooper && trk[0]->GetParam().GetPar(3) * trk[1]->GetParam().GetPar(3) < 0) {
        continue;
      }

      bool needswap = false;
      if (trk[0]->PrevSegment() == -1 && trk[1]->PrevSegment() >= 0) {
        needswap = true;
      } else if (celooper) {
        const float z0max = -CAMath::Min(cls[mClusters[trk[0]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime());
        const float z1max = -CAMath::Min(cls[mClusters[trk[1]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime());
        if (z1max < z0max) {
          needswap = true;
        }
      } else {
        if (mClusters[trk[0]->FirstClusterRef()].row > mClusters[trk[1]->FirstClusterRef()].row) {
          needswap = true;
        }
      }
      if (needswap) {
        GPUCommonAlgorithm::swap(trk[0], trk[1]);
      }

      if (Param().par.continuousTracking) {
        GPUTPCGMMergedTrackHit* clsmax;
        const float tmax = CAMath::MaxWithRef(cls[mClusters[trk[0]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime(),
                                              cls[mClusters[trk[1]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime(),
                                              &mClusters[trk[0]->FirstClusterRef()], &mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1],
                                              &mClusters[trk[1]->FirstClusterRef()], &mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1], clsmax);
        const float offset = CAMath::Max(tmax - mConstantMem->calibObjects.fastTransform->getMaxDriftTime(clsmax->sector, clsmax->row, cls[clsmax->num].getPad()), 0.f);
        trk[1]->Param().Z() += mConstantMem->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(trk[1]->CSide() * NSECTORS / 2, trk[1]->Param().TOffset() - offset);
        trk[1]->Param().TOffset() = offset;
        if (celooper) {
          trk[0]->Param().Z() += mConstantMem->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(trk[0]->CSide() * NSECTORS / 2, trk[0]->Param().TOffset() - offset);
          trk[0]->Param().TOffset() = offset;
        }
      }

      if (celooper) { // TODO: Need propper handling, avoid falsely flagging the primary leg as looper
        trk[0]->SetMergedLooperConnected(true);
        trk[0]->SetCCE(true);
        trk[0]->SetLooper(true);
        trk[1]->SetMergedLooperConnected(true);
        trk[1]->SetCCE(true);
        trk[1]->SetLooper(true);
        continue;
      }

      uint32_t newRef = CAMath::AtomicAdd(&mMemory->nMergedTrackClusters, trk[0]->NClusters() + trk[1]->NClusters());
      if (newRef + trk[0]->NClusters() + trk[1]->NClusters() >= mNMaxMergedTrackClusters) {
        raiseError(GPUErrors::ERROR_MERGER_CE_HIT_OVERFLOW, newRef + trk[0]->NClusters() + trk[1]->NClusters(), mNMaxMergedTrackClusters);
        for (uint32_t k = newRef; k < mNMaxMergedTrackClusters; k++) {
          mClusters[k].num = 0;
          mClusters[k].state = 0;
        }
        CAMath::AtomicExch(&mMemory->nMergedTrackClusters, mNMaxMergedTrackClusters);
        return;
      }

      int32_t pos = newRef;
#pragma unroll
      for (int32_t k = 1; k >= 0; k--) {
        for (uint32_t j = 0; j != trk[k]->NClusters(); j++) {
          mClusters[pos++] = mClusters[trk[k]->FirstClusterRef() + j];
        }
      }
      trk[1]->SetFirstClusterRef(newRef);
      trk[1]->SetNClusters(trk[0]->NClusters() + trk[1]->NClusters());
      if (trk[1]->NClusters() > GPUCA_MERGER_MAX_TRACK_CLUSTERS) {
        trk[1]->SetFirstClusterRef(trk[1]->FirstClusterRef() + trk[1]->NClusters() - GPUCA_MERGER_MAX_TRACK_CLUSTERS);
        trk[1]->SetNClusters(GPUCA_MERGER_MAX_TRACK_CLUSTERS);
      }
      trk[1]->SetCCE(true);
      trk[0]->SetNClusters(0);
      trk[0]->SetOK(false);
    }
  }

  // for (int32_t i = 0;i < mMemory->nMergedTracks;i++) {if (mMergedTracks[i].CCE() == false) {mMergedTracks[i].SetNClusters(0);mMergedTracks[i].SetOK(false);}} //Remove all non-CE tracks
}

namespace o2::gpu::internal
{
namespace // anonymous
{
struct GPUTPCGMMerger_CompareClusterIds {
  const GPUTPCGMMerger::trackCluster* const mCmp;
  const bool revert;
  GPUd() GPUTPCGMMerger_CompareClusterIds(const GPUTPCGMMerger::trackCluster* cmp, bool r) : mCmp(cmp), revert(r) {}
  GPUd() bool operator()(const int16_t aa, const int16_t bb)
  {
    const GPUTPCGMMerger::trackCluster& a = mCmp[aa];
    const GPUTPCGMMerger::trackCluster& b = mCmp[bb];
    if (a.row != b.row) {
      return (a.row > b.row) ^ revert;
    }
    return GPUCA_DETERMINISTIC_CODE((a.id != b.id) ? (a.id > b.id) : (aa > bb), a.id > b.id);
  }
};
} // anonymous namespace
} // namespace o2::gpu::internal

GPUd() void GPUTPCGMMerger::CollectMergedTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  static constexpr int32_t kMaxParts = 16;
  static constexpr int32_t kMaxClusters = GPUCA_MERGER_MAX_TRACK_CLUSTERS;

  GPUTPCGMSectorTrack* trackParts[kMaxParts];

  int32_t itr = iBlock * nThreads + iThread;
  GPUTPCGMSectorTrack* trbase = nullptr;
  int32_t leg = 0;
  int32_t lastMergedSegment = -1;
  bool revertSegments = false;
  bool revertInSegment = false;
  while (true) {
    if (trbase && !Param().rec.tpc.dropLoopers) {
      int32_t jtr = trbase->NextNeighbour();
      if (jtr >= 0) {
        trbase = &(mSectorTrackInfos[jtr]);
        if (trbase->PrevSegmentNeighbour() >= 0) {
          trbase = nullptr;
        } else {
          if (Param().rec.enableCyclicGraphWorkarounds) {
            trbase->SetPrevSegmentNeighbour(1000000001);
          }
          leg += revertSegments ? 1 : -1;
        }
      } else {
        trbase = nullptr;
      }
    }

    if (trbase == nullptr) {
      while (itr < SectorTrackInfoLocalTotal()) {
        trbase = &mSectorTrackInfos[itr];
        if (trbase->PrevSegmentNeighbour() >= 0 || trbase->PrevNeighbour() >= 0) {
          itr += nThreads * nBlocks;
          continue;
        }
        break;
      }
      if (itr >= SectorTrackInfoLocalTotal()) {
        break;
      }
      revertSegments = false;
      revertInSegment = false;
      if (Param().rec.enableCyclicGraphWorkarounds) {
        trbase->SetPrevSegmentNeighbour(1000000000);
      }
      int32_t jtr = trbase->NextNeighbour();
      leg = 0;
      if (jtr >= 0) {
        int32_t lasttr = itr;
        while (jtr >= 0) { // --------------- count segments ---------------
          if (Param().rec.enableCyclicGraphWorkarounds && &mSectorTrackInfos[jtr] == trbase) {
            break; // Break cyclic graph
          }
          lasttr = jtr;
          leg++;
          jtr = mSectorTrackInfos[jtr].NextNeighbour();
        }

        float mainT = 1e9;
        revertSegments = true;
        for (uint32_t k = 0; k < 2; k++) { // --------------- check if first or last segment is primary ---------------
          int32_t ichk = k ? lasttr : itr;
          const GPUTPCGMSectorTrack* trchk = &mSectorTrackInfos[ichk];
          while (true) {
            float t = -trchk->MinClusterT();
            if (t < mainT) {
              if (k) {
                revertSegments = false;
                break;
              }
              mainT = t;
            }
            int32_t next = trchk->NextSegmentNeighbour();
            if (next < 0 || (Param().rec.enableCyclicGraphWorkarounds && next == ichk)) {
              break; // Breaks also cycles
            }
            trchk = &mSectorTrackInfos[next];
          }
        }
        if (revertSegments) {
          leg = 0;
        }

        { // --------------- find longest sector track of main segment ---------------
          int32_t length = 0;
          int32_t ichk = revertSegments ? itr : lasttr;
          const GPUTPCGMSectorTrack* trchk = &mSectorTrackInfos[ichk];
          const GPUTPCGMSectorTrack* longest = trchk;
          while (true) {
            if (trchk->OrigTrack()->NHits() > length) {
              longest = trchk;
              length = trchk->OrigTrack()->NHits();
            }
            int32_t next = trchk->NextSegmentNeighbour();
            if (next < 0 || (Param().rec.enableCyclicGraphWorkarounds && next == ichk)) {
              break; // Breaks also cycles
            }
            trchk = &mSectorTrackInfos[next];
          }
          revertInSegment = longest->ClusterT0() < longest->ClusterTN();
        }
      }
      lastMergedSegment = -1;
      itr += nThreads * nBlocks;
    }

    do {
      int32_t nParts = 0;
      int32_t nHits = 0;

      GPUTPCGMSectorTrack* tr = trbase;
      while (true) {
        if (nParts >= kMaxParts) {
          break;
        }
        if (nHits + tr->NClusters() > kMaxClusters) {
          break;
        }
        nHits += tr->NClusters();

        trackParts[nParts++] = tr;
        for (int32_t i = 0; i < 2; i++) {
          if (tr->ExtrapolatedTrackId(i) != -1) {
            if (nParts >= kMaxParts) {
              break;
            }
            if (nHits + mSectorTrackInfos[tr->ExtrapolatedTrackId(i)].NClusters() > kMaxClusters) {
              break;
            }
            trackParts[nParts++] = &mSectorTrackInfos[tr->ExtrapolatedTrackId(i)];
            nHits += mSectorTrackInfos[tr->ExtrapolatedTrackId(i)].NClusters();
          }
        }
        int32_t jtr = tr->NextSegmentNeighbour();
        if (jtr >= 0) {
          tr = &(mSectorTrackInfos[jtr]);
          if (Param().rec.enableCyclicGraphWorkarounds) {
            tr->SetPrevSegmentNeighbour(1000000002);
          }
          continue;
        }
        break;
      }

      // unpack and sort clusters
      if (nParts > 1 && (!revertInSegment ^ (leg & 1))) {
        GPUCommonAlgorithm::sort(trackParts, trackParts + nParts, [](const GPUTPCGMSectorTrack* a, const GPUTPCGMSectorTrack* b) {
          GPUCA_DETERMINISTIC_CODE( // clang-format off
            if (a->X() != b->X()) {
              return (a->X() > b->X());
            }
            if (a->Y() != b->Y()) {
              return (a->Y() > b->Y());
            }
            if (a->Z() != b->Z()) {
              return (a->Z() > b->Z());
            }
            return a->QPt() > b->QPt();
          , // !GPUCA_DETERMINISTIC_CODE
            return (a->X() > b->X());
          ) // clang-format on
        });
      }

      trackCluster trackClusters[kMaxClusters];
      nHits = 0;
      for (int32_t ipart = 0; ipart < nParts; ipart++) {
        const GPUTPCGMSectorTrack* t = trackParts[ipart];
        CADEBUG(printf("Collect Track %d Part %d QPt %f DzDs %f\n", mMemory->nMergedTracks, ipart, t->QPt(), t->DzDs()));
        int32_t nTrackHits = t->NClusters();
        trackCluster* c2 = trackClusters + nHits + nTrackHits - 1;
        for (int32_t i = 0; i < nTrackHits; i++, c2--) {
          const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[t->Sector()];
          const GPUTPCHitId& ic = trk.TrackHits()[t->OrigTrack()->FirstHitID() + i];
          uint32_t id = trk.Data().ClusterDataIndex(trk.Data().Row(ic.RowIndex()), ic.HitIndex()) + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[t->Sector()][0];
          *c2 = trackCluster{id, (uint8_t)ic.RowIndex(), t->Sector()};
        }
        nHits += nTrackHits;
      }
      if (nHits < GPUCA_TPC_MIN_HITS_B5(trbase->QPt() * Param().qptB5Scaler)) {
        break;
      }

      const bool mustReverse = revertInSegment ^ (leg & 1);
      bool ordered = !mustReverse;
      if (ordered) {
        for (int32_t i = 1; i < nHits; i++) {
          if ((trackClusters[i].row > trackClusters[i - 1].row) ^ mustReverse || trackClusters[i].id == trackClusters[i - 1].id) {
            ordered = false;
            break;
          }
        }
      }
      int32_t firstTrackIndex = 0;
      int32_t lastTrackIndex = nParts - 1;
      if (ordered == 0) {
        int32_t nTmpHits = 0;
        trackCluster trackClustersUnsorted[kMaxClusters];
        int16_t clusterIndices[kMaxClusters];
        for (int32_t i = 0; i < nHits; i++) {
          trackClustersUnsorted[i] = trackClusters[i];
          clusterIndices[i] = i;
        }

        GPUCommonAlgorithm::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIds(trackClusters, mustReverse));

        nTmpHits = 0;
        firstTrackIndex = lastTrackIndex = -1;
        for (int32_t i = 0; i < nParts; i++) {
          nTmpHits += trackParts[i]->NClusters();
          if (nTmpHits > clusterIndices[0] && firstTrackIndex == -1) {
            firstTrackIndex = i;
          }
          if (nTmpHits > clusterIndices[nHits - 1] && lastTrackIndex == -1) {
            lastTrackIndex = i;
          }
        }

        int32_t nFilteredHits = 0;
        int32_t indPrev = -1;
        for (int32_t i = 0; i < nHits; i++) {
          int32_t ind = clusterIndices[i];
          if (indPrev >= 0 && trackClustersUnsorted[ind].id == trackClustersUnsorted[indPrev].id) {
            continue;
          }
          indPrev = ind;
          trackClusters[nFilteredHits] = trackClustersUnsorted[ind];
          nFilteredHits++;
        }
        nHits = nFilteredHits;
      }

      const uint32_t iMergedTrackFirstCluster = CAMath::AtomicAdd(&mMemory->nMergedTrackClusters, (uint32_t)nHits);
      if (iMergedTrackFirstCluster + nHits > mNMaxMergedTrackClusters) {
        raiseError(GPUErrors::ERROR_MERGER_HIT_OVERFLOW, iMergedTrackFirstCluster, mNMaxMergedTrackClusters);
        CAMath::AtomicExch(&mMemory->nMergedTrackClusters, mNMaxMergedTrackClusters);
        break;
      }

      GPUTPCGMMergedTrackHit* const cl = mClusters + iMergedTrackFirstCluster;

      for (int32_t i = 0; i < nHits; i++) {
        uint8_t state;
        const ClusterNative& c = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[trackClusters[i].id];
        state = c.getFlags();
        cl[i].state = state & GPUTPCGMMergedTrackHit::clustererAndSharedFlags; // Only allow edge, deconvoluted, and shared flags
        cl[i].row = trackClusters[i].row;
        cl[i].num = trackClusters[i].id;
        cl[i].sector = trackClusters[i].sector;
      }

      uint32_t iOutputTrack = CAMath::AtomicAdd(&mMemory->nMergedTracks, 1u);
      if (iOutputTrack >= mNMaxTracks) {
        raiseError(GPUErrors::ERROR_MERGER_TRACK_OVERFLOW, iOutputTrack, mNMaxTracks);
        CAMath::AtomicExch(&mMemory->nMergedTracks, mNMaxTracks);
        break;
      }

      GPUTPCGMMergedTrack& mergedTrack = mMergedTracks[iOutputTrack];
      GPUTPCGMTrackParam& p1 = mergedTrack.Param();
      const GPUTPCGMSectorTrack& p2 = *trackParts[firstTrackIndex];
      mergedTrack.SetFlags(0);
      mergedTrack.SetOK(true);
      mergedTrack.SetLeg(leg);
      mergedTrack.SetLooper(leg > 0);
      mergedTrack.SetNClusters(nHits);
      mergedTrack.SetFirstClusterRef(iMergedTrackFirstCluster);
      mergedTrack.SetCSide(p2.CSide());
      mergedTrack.SetMergedLooperConnected(leg > 0);
      if (revertSegments) {
        mergedTrack.SetPrevSegment(-1);
        if (lastMergedSegment >= 0) {
          mMergedTracks[lastMergedSegment].SetPrevSegment(iOutputTrack);
        }
      } else {
        mergedTrack.SetPrevSegment(lastMergedSegment);
      }
      lastMergedSegment = iOutputTrack;

      GPUTPCGMBorderTrack b;
      const float toX = GPUTPCGeometry::Row2X(cl[0].row);
      if (p2.TransportToX<2>(this, toX, Param().bzCLight, b, GPUCA_MAX_SIN_PHI, false)) {
        p1.X() = toX;
        p1.Y() = b.Par()[0];
        p1.Z() = b.Par()[1];
        p1.SinPhi() = b.Par()[2];
      } else {
        p1.X() = p2.X();
        p1.Y() = p2.Y();
        p1.Z() = p2.Z();
        p1.SinPhi() = p2.SinPhi();
      }
      p1.TOffset() = p2.TOffset();
      p1.DzDs() = p2.DzDs();
      p1.QPt() = p2.QPt();
      mergedTrack.SetAlpha(p2.Alpha());
      if (CAMath::Abs(Param().polynomialField.GetNominalBz()) < (gpu_common_constants::kZeroFieldCut * gpu_common_constants::kCLight)) {
        p1.QPt() = 100.f / Param().rec.bz0Pt10MeV;
      }

      // if (nParts > 1) printf("Merged %d: QPt %f %d parts %d hits\n", mMemory->nMergedTracks, p1.QPt(), nParts, nHits);

      /*if (GPUQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(mMemory->nMergedTracks))
      {
        mergedTrack.SetOK(0);
        mergedTrack.SetNClusters(0);
      }
      if (mergedTrack.NClusters() && mergedTrack.OK()) */
      if (leg == 0 && Param().rec.tpc.mergeCE) {
        auto& cls = mConstantMem->ioPtrs.clustersNative->clustersLinear;
        bool CEside = cls[cl[0].num].getTime() < cls[cl[nHits - 1].num].getTime();
        MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], iOutputTrack);
      }
    } while (false);
  }
}

GPUd() void GPUTPCGMMerger::SortTracksPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nThreads * nBlocks) {
    mTrackOrderProcess[i] = i;
  }
}

GPUd() void GPUTPCGMMerger::PrepareForFit0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nBlocks * nThreads) {
    mTrackSort[i] = i;
  }
}

GPUd() void GPUTPCGMMerger::SortTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
#ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  if (iThread == 0 && iBlock == 0) {
    GPUCommonAlgorithm::sortDeviceDynamic(mTrackOrderProcess, mTrackOrderProcess + mMemory->nMergedTracks, GPUTPCGMMergerSortTracks_comp(mMergedTracks));
  }
#endif
}

GPUd() void GPUTPCGMMerger::SortTracksQPt(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
#ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  if (iThread == 0 && iBlock == 0) {
    GPUCommonAlgorithm::sortDeviceDynamic(mTrackSort, mTrackSort + mMemory->nMergedTracks, GPUTPCGMMergerSortTracksQPt_comp(mMergedTracks));
  }
#endif
}

GPUd() void GPUTPCGMMerger::PrepareForFit1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nBlocks * nThreads) {
    mTrackOrderAttach[mTrackSort[i]] = i;
    GPUTPCGMMergedTrack& trk = mMergedTracks[i];
    if (trk.OK()) {
      for (uint32_t j = 0; j < trk.NClusters(); j++) {
        uint32_t weight = attachAttached | attachGood;
        if (CAMath::Abs(trk.GetParam().GetQPt() * Param().qptB5Scaler) <= Param().rec.tpc.rejectQPtB5 && !trk.MergedLooper() && trk.Leg() == 0) {
          weight |= attachProtect;
        }
        CAMath::AtomicMax(&mClusterAttachment[mClusters[trk.FirstClusterRef() + j].num], weight);
        CAMath::AtomicAdd(&mSharedCount[mClusters[trk.FirstClusterRef() + j].num], 1u);
      }
      if (!trk.CCE() && !trk.MergedLooper()) {
        GPUTPCGMMergedTrack* updTrk = trk.GetFirstSegment(mMergedTracks, Param().rec.enableCyclicGraphWorkarounds);
        const auto &cl0 = mClusters[trk.FirstClusterRef()], &cln = mClusters[updTrk->FirstClusterRef() + updTrk->NClusters() - 1];
        const auto& GPUrestrict() cls = GetConstantMem()->ioPtrs.clustersNative->clustersLinear;
        float z0 = cls[cl0.num].getTime(), zn = cls[cln.num].getTime();
        const auto tmp = zn > z0 ? std::array<float, 3>{zn, z0, GPUTPCGeometry::Row2X(cln.row)} : std::array<float, 3>{z0, zn, GPUTPCGeometry::Row2X(cl0.row)};
        trk.Param().ShiftZ(this, cl0.sector, tmp[0], tmp[1], tmp[2]);
        updTrk = &trk;
        while (updTrk->PrevSegment() >= 0) {
          auto next = &mMergedTracks[updTrk->PrevSegment()];
          if (Param().rec.enableCyclicGraphWorkarounds && next == &trk) {
            break;
          }
          updTrk = next;
          updTrk->Param().TOffset() = trk.Param().TOffset();
        }
      }
    }
  }
}

GPUd() void GPUTPCGMMerger::PrepareForFit2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTrackClusters; i += nBlocks * nThreads) {
    if (mSharedCount[mClusters[i].num] > 1) {
      mClusters[i].state |= GPUTPCGMMergedTrackHit::flagShared;
    }
  }
  if (mClusterStateExt) {
    for (uint32_t i = iBlock * nThreads + iThread; i < mNMaxClusters; i += nBlocks * nThreads) {
      uint8_t state = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[i].getFlags();
      if (mSharedCount[i] > 1) {
        state |= GPUTPCGMMergedTrackHit::flagShared;
      }
      mClusterStateExt[i] = state;
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nThreads * nBlocks) {
    mTrackSort[mTrackOrderAttach[i]] = i;
  }
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTrackClusters; i += nThreads * nBlocks) {
    if (!(mClusterAttachment[mClusters[i].num] & attachProtect)) {
      mClusterAttachment[mClusters[i].num] = 0; // Reset adjacent attachment for attached clusters, set correctly below
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mMemory->nMergedTracks; i += nThreads * nBlocks) {
    const GPUTPCGMMergedTrack& trk = mMergedTracks[i];
    if (!trk.OK() || trk.NClusters() == 0) {
      continue;
    }
    for (uint32_t j = 0; j < trk.NClusters(); j++) {
      int32_t id = mClusters[trk.FirstClusterRef() + j].num;
      uint32_t weight = mTrackOrderAttach[i] | attachAttached;
      uint8_t clusterState = mClusters[trk.FirstClusterRef() + j].state;
      if (!(clusterState & GPUTPCGMMergedTrackHit::flagReject)) {
        weight |= attachGood;
      } else if (clusterState & GPUTPCGMMergedTrackHit::flagHighIncl) {
        weight |= attachHighIncl;
      }
      if (trk.Leg() == 0) {
        weight |= attachGoodLeg;
      }
      if (CAMath::Abs(trk.GetParam().GetQPt() * Param().qptB5Scaler) <= Param().rec.tpc.rejectQPtB5 && !trk.MergedLooper() && trk.Leg() == 0) {
        weight |= attachProtect;
      }
      CAMath::AtomicMax(&mClusterAttachment[id], weight);
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < mNMaxClusters; i += nThreads * nBlocks) {
    if (mClusterAttachment[i] != 0) {
      mClusterAttachment[i] = (mClusterAttachment[i] & attachFlagMask) | mTrackSort[mClusterAttachment[i] & attachTrackMask];
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeLoopersInit(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  const float lowPtThresh = Param().rec.tpc.rejectQPtB5 * 1.1f; // Might need to merge tracks above the threshold with parts below the rejection threshold
  for (uint32_t i = get_global_id(0); i < mMemory->nMergedTracks; i += get_global_size(0)) {
    const auto& trk = mMergedTracks[i];
    const auto& p = trk.GetParam();
    const float qptabs = CAMath::Abs(p.GetQPt());
    if (trk.OK() && trk.NClusters() && trk.Leg() == 0 && qptabs * Param().qptB5Scaler > 5.f && qptabs * Param().qptB5Scaler <= lowPtThresh) {
      const int32_t sector = mClusters[trk.FirstClusterRef() + trk.NClusters() - 1].sector;
      const float refz = p.GetZ() + (Param().par.continuousTracking ? GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(sector, p.GetTOffset(), Param().continuousMaxTimeBin) : 0) + (trk.CSide() ? -100 : 100);
      float sinA, cosA;
      CAMath::SinCos(trk.GetAlpha(), sinA, cosA);
      float gx = cosA * p.GetX() - sinA * p.GetY();
      float gy = cosA * p.GetY() + sinA * p.GetX();
      float bz = Param().polynomialField.GetFieldBz(gx, gy, p.GetZ());
      const float r1 = p.GetQPt() * bz;
      const float r = CAMath::Abs(r1) > 0.0001f ? (1.f / r1) : 10000;
      const float mx = p.GetX() + r * p.GetSinPhi();
      const float my = p.GetY() - r * CAMath::Sqrt(1 - p.GetSinPhi() * p.GetSinPhi());
      const float gmx = cosA * mx - sinA * my;
      const float gmy = cosA * my + sinA * mx;
      uint32_t myId = CAMath::AtomicAdd(&mMemory->nLooperMatchCandidates, 1u);
      if (myId >= mNMaxLooperMatches) {
        raiseError(GPUErrors::ERROR_LOOPER_MATCH_OVERFLOW, myId, mNMaxLooperMatches);
        CAMath::AtomicExch(&mMemory->nLooperMatchCandidates, mNMaxLooperMatches);
        return;
      }
      mLooperCandidates[myId] = MergeLooperParam{refz, gmx, gmy, i};

      /*printf("Track %u Sanity qpt %f snp %f bz %f\n", mMemory->nLooperMatchCandidates, p.GetQPt(), p.GetSinPhi(), bz);
      for (uint32_t k = 0;k < trk.NClusters();k++) {
        float xx, yy, zz;
        const ClusterNative& GPUrestrict() cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[mClusters[trk.FirstClusterRef() + k].num];
        GetConstantMem()->calibObjects.fastTransformHelper->Transform(mClusters[trk.FirstClusterRef() + k].sector, mClusters[trk.FirstClusterRef() + k].row, cl.getPad(), cl.getTime(), xx, yy, zz, p.GetTOffset());
        float sa2, ca2;
        CAMath::SinCos(Param().Alpha(mClusters[trk.FirstClusterRef() + k].sector), sa2, ca2);
        float cx = ca2 * xx - sa2 * yy;
        float cy = ca2 * yy + sa2 * xx;
        float dist = CAMath::Sqrt((cx - gmx) * (cx - gmx) + (cy - gmy) * (cy - gmy));
        printf("Hit %3d/%3d sector %d xy %f %f R %f\n", k, trk.NClusters(), (int32_t)mClusters[trk.FirstClusterRef() + k].sector, cx, cy, dist);
      }*/
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeLoopersSort(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
#ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  if (iThread == 0 && iBlock == 0) {
    GPUCommonAlgorithm::sortDeviceDynamic(mLooperCandidates, mLooperCandidates + mMemory->nLooperMatchCandidates, GPUTPCGMMergerMergeLoopers_comp());
  }
#endif
}

GPUd() void GPUTPCGMMerger::MergeLoopersMain(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread)
{
  const MergeLooperParam* candidates = mLooperCandidates;

#if GPUCA_MERGE_LOOPER_MC && !defined(GPUCA_GPUCODE)
  std::vector<int64_t> paramLabels(mMemory->nLooperMatchCandidates);
  for (uint32_t i = 0; i < mMemory->nLooperMatchCandidates; i++) {
    paramLabels[i] = GetTrackLabel(mMergedTracks[candidates[i].id]);
  }
  /*std::vector<bool> dropped(mMemory->nLooperMatchCandidates);
  std::vector<bool> droppedMC(mMemory->nLooperMatchCandidates);
  std::vector<int32_t> histMatch(101);
  std::vector<int32_t> histFail(101);*/
  if (!mRec->GetProcessingSettings().runQA) {
    throw std::runtime_error("Need QA enabled for the Merge Loopers MC QA");
  }
#endif

  for (uint32_t i = get_global_id(0); i < mMemory->nLooperMatchCandidates; i += get_global_size(0)) {
    for (uint32_t j = i + 1; j < mMemory->nLooperMatchCandidates; j++) {
      // int32_t bs = 0;
      assert(CAMath::Abs(candidates[i].refz) <= CAMath::Abs(candidates[j].refz));
      if (CAMath::Abs(candidates[j].refz) > CAMath::Abs(candidates[i].refz) + 100.f) {
        break;
      }
      const float d2xy = CAMath::Sum2(candidates[i].x - candidates[j].x, candidates[i].y - candidates[j].y);
      if (d2xy > 15.f) {
        // bs |= 1;
        continue;
      }

      const GPUTPCGMMergedTrack* trkI = &mMergedTracks[candidates[i].id];
      float refZI = candidates[i].refz;
      {
        const auto* tmp = trkI->GetFirstSegment(mMergedTracks, Param().rec.enableCyclicGraphWorkarounds);
        if (tmp != trkI && tmp->CSide() == trkI->CSide() && CAMath::Abs(tmp->GetParam().GetZ()) > CAMath::Abs(trkI->GetParam().GetZ())) {
          float tmpRefZ = refZI + tmp->GetParam().GetZ() - trkI->GetParam().GetZ();
          if (CAMath::Abs(tmpRefZ) < CAMath::Abs(candidates[j].refz) && CAMath::Abs(tmpRefZ) > CAMath::Abs(refZI)) {
            trkI = tmp;
            refZI = tmpRefZ;
          }
        }
      };
      const auto& trk1 = *trkI;
      const auto& trk2 = mMergedTracks[candidates[j].id];
      const auto& param1 = trk1.GetParam();
      const auto& param2 = trk2.GetParam();
      if (CAMath::Abs(param1.GetDzDs()) > 0.03f && CAMath::Abs(param2.GetDzDs()) > 0.03f && param1.GetDzDs() * param2.GetDzDs() * param1.GetQPt() * param2.GetQPt() < 0) {
        // bs |= 2;
        continue;
      }

      const float dznormalized = (CAMath::Abs(candidates[j].refz) - CAMath::Abs(refZI)) / (CAMath::TwoPi() * 0.5f * (CAMath::Abs(param1.GetDzDs()) + CAMath::Abs(param2.GetDzDs())) * 1.f / (0.5f * (CAMath::Abs(param1.GetQPt()) + CAMath::Abs(param2.GetQPt())) * CAMath::Abs(Param().polynomialField.GetNominalBz())));
      const float phasecorr = CAMath::Modf((CAMath::ASin(param1.GetSinPhi()) + trk1.GetAlpha() - CAMath::ASin(param2.GetSinPhi()) - trk2.GetAlpha()) / CAMath::TwoPi() + 5.5f, 1.f) - 0.5f;
      const float phasecorrdirection = (candidates[j].refz * param1.GetQPt() * param1.GetDzDs()) > 0 ? 1 : -1;
      const float dzcorr = dznormalized + phasecorr * phasecorrdirection;
      const bool sameside = !(trk1.CSide() ^ trk2.CSide());
      const float dzcorrlimit[4] = {sameside ? 0.018f : 0.012f, sameside ? 0.12f : 0.025f, 0.14f, 0.15f};
      const int32_t dzcorrcount = sameside ? 4 : 2;
      bool dzcorrok = false;
      float dznorm = 0.f;
      for (int32_t k = 0; k < dzcorrcount; k++) {
        const float d = CAMath::Abs(dzcorr - 0.5f * k);
        if (d <= dzcorrlimit[k]) {
          dzcorrok = true;
          dznorm = d / dzcorrlimit[k];
          break;
        }
      }
      if (!dzcorrok) {
        // bs |= 4;
        continue;
      }

      const float dtgl = param1.GetDzDs() - (param1.GetQPt() * param2.GetQPt() > 0 ? param2.GetDzDs() : -param2.GetDzDs());
      const float dqpt = (CAMath::Abs(param1.GetQPt()) - CAMath::Abs(param2.GetQPt())) / CAMath::Min(param1.GetQPt(), param2.GetQPt());
      float d = CAMath::Sum2(dtgl * (1.f / 0.03f), dqpt * (1.f / 0.04f)) + d2xy * (1.f / 4.f) + dznorm * (1.f / 0.3f);
      bool EQ = d < 6.f;
#if GPUCA_MERGE_LOOPER_MC && !defined(GPUCA_GPUCODE)
      const int64_t label1 = paramLabels[i];
      const int64_t label2 = paramLabels[j];
      bool labelEQ = label1 != -1 && label1 == label2;
      if (1 || EQ || labelEQ) {
        // printf("Matching track %d/%d %u-%u (%ld/%ld): dist %f side %d %d, tgl %f %f, qpt %f %f, x %f %f, y %f %f\n", (int32_t)EQ, (int32_t)labelEQ, i, j, label1, label2, d, (int32_t)mMergedTracks[candidates[i].id].CSide(), (int32_t)mMergedTracks[candidates[j].id].CSide(), candidates[i].tgl, candidates[j].tgl, candidates[i].qpt, candidates[j].qpt, candidates[i].x, candidates[j].x, candidates[i].y, candidates[j].y);
        static auto& tup = GPUROOTDump<TNtuple>::get("mergeloopers", "labeleq:sides:d2xy:tgl1:tgl2:qpt1:qpt2:dz:dzcorr:dtgl:dqpt:dznorm:bs");
        tup.Fill((float)labelEQ, (trk1.CSide() ? 1 : 0) | (trk2.CSide() ? 2 : 0), d2xy, param1.GetDzDs(), param2.GetDzDs(), param1.GetQPt(), param2.GetQPt(), CAMath::Abs(candidates[j].refz) - CAMath::Abs(refZI), dzcorr, dtgl, dqpt, dznorm, bs);
        static auto tup2 = GPUROOTDump<TNtuple>::getNew("mergeloopers2", "labeleq:refz1:refz2:tgl1:tgl2:qpt1:qpt2:snp1:snp2:a1:a2:dzn:phasecor:phasedir:dzcorr");
        tup2.Fill((float)labelEQ, refZI, candidates[j].refz, param1.GetDzDs(), param2.GetDzDs(), param1.GetQPt(), param2.GetQPt(), param1.GetSinPhi(), param2.GetSinPhi(), trk1.GetAlpha(), trk2.GetAlpha(), dznormalized, phasecorr, phasecorrdirection, dzcorr);
      }
      /*if (EQ) {
        dropped[j] = true;
      }
      if (labelEQ) {
        droppedMC[j] = true;
        histMatch[CAMath::Min<int32_t>(100, d * 10.f)]++;
      }
      if (d < 10.f && !labelEQ) {
        histFail[CAMath::Min<int32_t>(100, d * 10.f)]++;
    }*/
#endif
      if (EQ) {
        mMergedTracks[candidates[j].id].SetMergedLooperUnconnected(true);
        if (CAMath::Abs(param2.GetQPt() * Param().qptB5Scaler) >= Param().rec.tpc.rejectQPtB5) {
          mMergedTracks[candidates[i].id].SetMergedLooperUnconnected(true);
        }
      }
    }
  }
  /*#if GPUCA_MERGE_LOOPER_MC && !defined(GPUCA_GPUCODE)
  int32_t total = 0, totalmc = 0, good = 0, missed = 0, fake = 0;
  for (uint32_t i = 0; i < mMemory->nLooperMatchCandidates; i++) {
    total += dropped[i];
    totalmc += droppedMC[i];
    good += dropped[i] && droppedMC[i];
    missed += droppedMC[i] && !dropped[i];
    fake += dropped[i] && !droppedMC[i];
  }
  if (good) {
    printf("%20s: %8d\n", "Total", total);
    printf("%20s: %8d\n", "TotalMC", totalmc);
    printf("%20s: %8d (%8.3f%% %8.3f%%)\n", "Good", good, 100.f * good / total, 100.f * good / totalmc);
    printf("%20s: %8d (%8.3f%%)\n", "Missed", missed, 100.f * missed / totalmc);
    printf("%20s: %8d (%8.3f%%)\n", "Fake", fake, 100.f * fake / total);
  }
  printf("Match histo\n");
  for (uint32_t i = 0; i < histMatch.size(); i++) {
    printf("%8.3f: %3d\n", i / 10.f + 0.05f, histMatch[i]);
  }
  printf("Fake histo\n");
  for (uint32_t i = 0; i < histFail.size(); i++) {
    printf("%8.3f: %3d\n", i / 10.f + 0.05f, histFail[i]);
  }
#endif*/
}
