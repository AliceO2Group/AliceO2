// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMerger.cxx
/// \author Sergey Gorbunov, David Rohr

#define GPUCA_CADEBUG 0
#define GPUCA_MERGE_LOOPER_MC 0

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdio>
#include <cstring>
#include <cmath>
#include "GPUReconstruction.h"
#endif

#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUO2DataTypes.h"
#include "TPCFastTransform.h"
#include "GPUTPCConvertImpl.h"

#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"

#include "GPUTPCTrackParam.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUTPCTrackLinearisation.h"

#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUTPCGMBorderTrack.h"

#if !defined(GPUCA_GPUCODE) && (defined(GPUCA_MERGER_BY_MC_LABEL) || defined(GPUCA_CADEBUG_ENABLED) || GPUCA_MERGE_LOOPER_MC)
#include "AliHLTTPCClusterMCData.h"
#endif
#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"
#ifndef GPUCA_GPUCODE
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#endif
#else
#include "GPUO2FakeClasses.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace gputpcgmmergertypes;

static constexpr int kMaxParts = 400;
static constexpr int kMaxClusters = GPUCA_MERGER_MAX_TRACK_CLUSTERS;

//#define OFFLINE_FITTER

#if !defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPUCODE)
#undef OFFLINE_FITTER
#endif

#ifndef GPUCA_GPUCODE

#include "GPUQA.h"
#include "GPUMemorySizeScalers.h"

GPUTPCGMMerger::GPUTPCGMMerger()
  : mTrackLinks(nullptr), mNMaxSliceTracks(0), mNMaxTracks(0), mNMaxSingleSliceTracks(0), mNMaxOutputTrackClusters(0), mNMaxClusters(0), mMemoryResMemory(-1), mNClusters(0), mOutputTracks(nullptr), mSliceTrackInfos(nullptr), mSliceTrackInfoIndex(nullptr), mClusters(nullptr), mClustersXYZ(nullptr), mGlobalClusterIDs(nullptr), mClusterAttachment(nullptr), mOutputTracksTPCO2(nullptr), mOutputClusRefsTPCO2(nullptr), mOutputTracksTPCO2MC(nullptr), mTrackOrderAttach(nullptr), mTrackOrderProcess(nullptr), mTmpMem(nullptr), mBorderMemory(nullptr), mBorderRangeMemory(nullptr), mMemory(nullptr), mRetryRefitIds(nullptr), mLoopData(nullptr)
{
  //* constructor

  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    mNextSliceInd[iSlice] = iSlice + 1;
    mPrevSliceInd[iSlice] = iSlice - 1;
  }
  int mid = NSLICES / 2 - 1;
  int last = NSLICES - 1;
  mNextSliceInd[mid] = 0;
  mPrevSliceInd[0] = mid;
  mNextSliceInd[last] = NSLICES / 2;
  mPrevSliceInd[NSLICES / 2] = last;

  for (int i = 0; i < NSLICES; i++) {
    mkSlices[i] = nullptr;
  }
}

// DEBUG CODE
#if !defined(GPUCA_GPUCODE) && (defined(GPUCA_MERGER_BY_MC_LABEL) || defined(GPUCA_CADEBUG_ENABLED) || GPUCA_MERGE_LOOPER_MC)
#include "GPUQAHelper.h"

void GPUTPCGMMerger::CheckMergedTracks()
{
  std::vector<bool> trkUsed(SliceTrackInfoLocalTotal());
  for (int i = 0; i < SliceTrackInfoLocalTotal(); i++) {
    trkUsed[i] = false;
  }

  for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++) {
    GPUTPCGMSliceTrack& track = mSliceTrackInfos[itr];
    if (track.PrevSegmentNeighbour() >= 0) {
      continue;
    }
    if (track.PrevNeighbour() >= 0) {
      continue;
    }
    int leg = 0;
    GPUTPCGMSliceTrack *trbase = &track, *tr = &track;
    while (true) {
      int iTrk = tr - mSliceTrackInfos;
      if (trkUsed[iTrk]) {
        GPUError("FAILURE: double use");
      }
      trkUsed[iTrk] = true;

      int jtr = tr->NextSegmentNeighbour();
      if (jtr >= 0) {
        tr = &(mSliceTrackInfos[jtr]);
        continue;
      }
      jtr = trbase->NextNeighbour();
      if (jtr >= 0) {
        trbase = &(mSliceTrackInfos[jtr]);
        tr = trbase;
        if (tr->PrevSegmentNeighbour() >= 0) {
          break;
        }
        leg++;
        continue;
      }
      break;
    }
  }
  for (int i = 0; i < SliceTrackInfoLocalTotal(); i++) {
    if (trkUsed[i] == false) {
      GPUError("FAILURE: trk missed");
    }
  }
}

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
long int GPUTPCGMMerger::GetTrackLabelA(const S& trk)
{
  GPUTPCGMSliceTrack* sliceTrack = nullptr;
  int nClusters = 0;
  if constexpr (std::is_same<S, GPUTPCGMBorderTrack&>::value) {
    sliceTrack = &mSliceTrackInfos[trk.TrackID()];
    nClusters = sliceTrack->OrigTrack()->NHits();
  } else {
    nClusters = trk.NClusters();
  }
  auto acc = GPUTPCTrkLbl<false, GPUTPCTrkLbl_ret>(resolveMCLabels<T>(GetConstantMem()->ioPtrs.clustersNative ? GetConstantMem()->ioPtrs.clustersNative->clustersMCTruth : nullptr, GetConstantMem()->ioPtrs.mcLabelsTPC), 0.5f);
  for (int i = 0; i < nClusters; i++) {
    int id;
    if constexpr (std::is_same<S, GPUTPCGMBorderTrack&>::value) {
      if (Param().rec.mergerReadFromTrackerDirectly) {
        const GPUTPCTracker& tracker = GetConstantMem()->tpcTrackers[sliceTrack->Slice()];
        const GPUTPCHitId& ic = tracker.TrackHits()[sliceTrack->OrigTrack()->FirstHitID() + i];
        id = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex()) + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[sliceTrack->Slice()][0];
      } else {
        id = sliceTrack->OrigTrack()->OutTrackClusters()[i].GetId();
      }
    } else {
      id = mClusters[trk.FirstClusterRef() + i].num;
    }
    acc.addLabel(id);
  }
  return acc.computeLabel().id;
}

template <class S>
long int GPUTPCGMMerger::GetTrackLabel(const S& trk)
{
#ifdef GPUCA_TPC_GEOMETRY_O2
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

void GPUTPCGMMerger::PrintMergeGraph(const GPUTPCGMSliceTrack* trk, std::ostream& out)
{
  const GPUTPCGMSliceTrack* orgTrack = trk;
  while (trk->PrevSegmentNeighbour() >= 0) {
    trk = &mSliceTrackInfos[trk->PrevSegmentNeighbour()];
  }
  const GPUTPCGMSliceTrack* orgTower = trk;
  while (trk->PrevNeighbour() >= 0) {
    trk = &mSliceTrackInfos[trk->PrevNeighbour()];
  }

  int nextId = trk - mSliceTrackInfos;
  out << "Graph of track %d" << (orgTrack - mSliceTrackInfos) << "\n";
  while (nextId >= 0) {
    trk = &mSliceTrackInfos[nextId];
    if (trk->PrevSegmentNeighbour() >= 0) {
      out << "TRACK TREE INVALID!!! " << trk->PrevSegmentNeighbour() << " --> " << nextId << "\n";
    }
    out << (trk == orgTower ? "--" : "  ");
    while (nextId >= 0) {
      GPUTPCGMSliceTrack* trk2 = &mSliceTrackInfos[nextId];
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
  computePointerWithAlignment(mem, mSliceTrackInfos, mNMaxSliceTracks);
  computePointerWithAlignment(mem, mSliceTrackInfoIndex, NSLICES * 2 + 1);
  if (mRec->GetParam().rec.NonConsecutiveIDs) {
    computePointerWithAlignment(mem, mGlobalClusterIDs, mNMaxOutputTrackClusters);
  }
  computePointerWithAlignment(mem, mBorderMemory, 2 * mNMaxSliceTracks);
  computePointerWithAlignment(mem, mBorderRangeMemory, 2 * mNMaxSliceTracks);
  computePointerWithAlignment(mem, mTrackLinks, mNMaxSliceTracks);
  computePointerWithAlignment(mem, mTrackCCRoots, mNMaxSliceTracks);
  size_t tmpSize = CAMath::Max(mNMaxSingleSliceTracks, 1u) * NSLICES * sizeof(int);
  tmpSize = CAMath::Max(tmpSize, CAMath::nextMultipleOf<4>(mNMaxTracks) * sizeof(int) + mNMaxClusters * sizeof(unsigned int));
  tmpSize = CAMath::Max(tmpSize, 4 * mNMaxTracks * sizeof(int));
  tmpSize = CAMath::Max(tmpSize, mNMaxTracks * (sizeof(*mRetryRefitIds) + sizeof(*mLoopData)));
  computePointerWithAlignment(mem, mTmpMem, (tmpSize + sizeof(*mTmpMem) - 1) / sizeof(*mTmpMem));

  int nTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    const int n = mRec->GetParam().rec.mergerReadFromTrackerDirectly ? *mRec->GetConstantMem().tpcTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
    mBorder[iSlice] = mBorderMemory + 2 * nTracks;
    mBorder[NSLICES + iSlice] = mBorderMemory + 2 * nTracks + n;
    mBorderRange[iSlice] = mBorderRangeMemory + 2 * nTracks;
    nTracks += n;
  }
  mLoopData = (GPUTPCGMLoopData*)mTmpMem;
  mRetryRefitIds = (unsigned int*)(mLoopData + mNMaxTracks);
  return mem;
}

void* GPUTPCGMMerger::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory);
  return mem;
}

void* GPUTPCGMMerger::SetPointersRefitScratch(void* mem)
{
  if (mRec->GetProcessingSettings().fullMergerOnGPU) {
    mem = SetPointersRefitScratch2(mem);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersRefitScratch2(void* mem)
{
  computePointerWithAlignment(mem, mTrackOrderAttach, mNMaxTracks);
  if (mRec->GetProcessingSettings().mergerSortTracks) {
    computePointerWithAlignment(mem, mTrackOrderProcess, mNMaxTracks);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mOutputTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mClusters, mNMaxOutputTrackClusters);
  if (mRec->GetParam().par.earlyTpcTransform) {
    computePointerWithAlignment(mem, mClustersXYZ, mNMaxOutputTrackClusters);
  }
  computePointerWithAlignment(mem, mClusterAttachment, mNMaxClusters);
  if (!mRec->GetProcessingSettings().fullMergerOnGPU) {
    mem = SetPointersRefitScratch2(mem);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutputState(void* mem)
{
  if ((mRec->GetRecoSteps() & GPUDataTypes::RecoStep::Refit) || mRec->GetProcessingSettings().outputSharedClusterMap) {
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

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersMerger, (mRec->GetProcessingSettings().fullMergerOnGPU ? 0 : GPUMemoryResource::MEMORY_HOST) | GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCMerger");
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersRefitScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCMergerRefitScratch");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutput, (mRec->GetProcessingSettings().fullMergerOnGPU ? (mRec->GetProcessingSettings().createO2Output > 1 ? GPUMemoryResource::MEMORY_SCRATCH : GPUMemoryResource::MEMORY_OUTPUT) : GPUMemoryResource::MEMORY_INOUT) | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutput");
  mMemoryResOutputState = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutputState, (mRec->GetProcessingSettings().fullMergerOnGPU ? GPUMemoryResource::MEMORY_OUTPUT : GPUMemoryResource::MEMORY_HOST) | GPUMemoryResource::MEMORY_CUSTOM, "TPCMergerOutputState");
  if (mRec->GetProcessingSettings().createO2Output) {
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
  mNMaxSliceTracks = 0;
  mNClusters = 0;
  mNMaxSingleSliceTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    unsigned int ntrk = mRec->GetParam().rec.mergerReadFromTrackerDirectly ? *mRec->GetConstantMem().tpcTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
    mNMaxSliceTracks += ntrk;
    mNClusters += mRec->GetParam().rec.mergerReadFromTrackerDirectly ? *mRec->GetConstantMem().tpcTrackers[iSlice].NTrackHits() : mkSlices[iSlice]->NTrackClusters();
    if (mNMaxSingleSliceTracks < ntrk) {
      mNMaxSingleSliceTracks = ntrk;
    }
  }
  mNMaxOutputTrackClusters = mRec->MemoryScalers()->NTPCMergedTrackHits(mNClusters);
  mNMaxTracks = mRec->MemoryScalers()->NTPCMergedTracks(mNMaxSliceTracks);
  if (io.clustersNative) {
    mNMaxClusters = io.clustersNative->nClustersTotal;
  } else if (mRec->GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) {
    mNMaxClusters = 0;
    for (int i = 0; i < NSLICES; i++) {
      mNMaxClusters += mRec->GetConstantMem().tpcTrackers[i].NHitsTotal();
    }
  } else {
    mNMaxClusters = mNClusters;
  }
}

int GPUTPCGMMerger::CheckSlices()
{
  for (int i = 0; i < NSLICES; i++) {
    if ((Param().rec.mergerReadFromTrackerDirectly ? mRec->GetConstantMem().tpcTrackers[i].CommonMemory()->nLocalTracks : mkSlices[i]->NLocalTracks()) > mNMaxSingleSliceTracks) {
      throw std::runtime_error("mNMaxSingleSliceTracks too small");
    }
  }
  if (!(mRec->GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) && (!Param().rec.NonConsecutiveIDs || Param().rec.mergerReadFromTrackerDirectly)) {
    throw std::runtime_error("Must run also slice tracking if NonConsecutiveIDs = false or mergerReadFromTrackerDirectly");
  }
  return 0;
}

#endif // GPUCA_GPUCODE

GPUd() void GPUTPCGMMerger::ClearTrackLinks(int nBlocks, int nThreads, int iBlock, int iThread, bool nOutput)
{
  const int n = nOutput ? mMemory->nOutputTracks : SliceTrackInfoLocalTotal();
  for (int i = iBlock * nThreads + iThread; i < n; i += nThreads * nBlocks) {
    mTrackLinks[i] = -1;
  }
}

GPUd() int GPUTPCGMMerger::RefitSliceTrack(GPUTPCGMSliceTrack& sliceTrack, const GPUTPCTrack* inTrack, float alpha, int slice)
{
  GPUTPCGMPropagator prop;
  prop.SetMaterialTPC();
  prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  prop.SetToyMCEventsFlag(false);
  prop.SetSeedingErrors(true); // Larger errors for seeds, better since we don't start with good hypothesis
  prop.SetFitInProjections(false);
  prop.SetPolynomialField(&Param().polynomialField);
  GPUTPCGMTrackParam trk;
  trk.X() = inTrack->Param().GetX();
  trk.Y() = inTrack->Param().GetY();
  trk.Z() = inTrack->Param().GetZ();
  trk.SinPhi() = inTrack->Param().GetSinPhi();
  trk.DzDs() = inTrack->Param().GetDzDs();
  trk.QPt() = inTrack->Param().GetQPt();
  trk.TZOffset() = Param().par.earlyTpcTransform ? inTrack->Param().GetZOffset() : GetConstantMem()->calibObjects.fastTransform->convZOffsetToVertexTime(slice, inTrack->Param().GetZOffset(), Param().par.continuousMaxTimeBin);
  trk.ShiftZ(this, slice, sliceTrack.ClusterZT0(), sliceTrack.ClusterZTN());
  for (int way = 0; way < 2; way++) {
    if (way) {
      prop.SetFitInProjections(true);
      prop.SetPropagateBzOnly(true);
    }
    trk.ResetCovariance();
    prop.SetTrack(&trk, alpha);
    int start = way ? inTrack->NHits() - 1 : 0;
    int end = way ? 0 : (inTrack->NHits() - 1);
    int incr = way ? -1 : 1;
    for (int i = start; i != end; i += incr) {
      float x, y, z;
      int row, flags;
      if (Param().rec.mergerReadFromTrackerDirectly) {
        const GPUTPCTracker& tracker = GetConstantMem()->tpcTrackers[slice];
        const GPUTPCHitId& ic = tracker.TrackHits()[inTrack->FirstHitID() + i];
        int clusterIndex = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex());
        row = ic.RowIndex();
        const ClusterNative& cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[GetConstantMem()->ioPtrs.clustersNative->clusterOffset[slice][0] + clusterIndex];
        flags = cl.getFlags();
        if (Param().par.earlyTpcTransform) {
          x = tracker.Data().ClusterData()[clusterIndex].x;
          y = tracker.Data().ClusterData()[clusterIndex].y;
          z = tracker.Data().ClusterData()[clusterIndex].z - trk.TZOffset();
        } else {
          GetConstantMem()->calibObjects.fastTransform->Transform(slice, row, cl.getPad(), cl.getTime(), x, y, z, trk.TZOffset());
        }
      } else {
        const GPUTPCSliceOutCluster& clo = inTrack->OutTrackCluster(i);
        row = clo.GetRow();
        flags = clo.GetFlags();
        if (Param().par.earlyTpcTransform) {
          x = clo.GetX();
          y = clo.GetY();
          z = clo.GetZ() - trk.TZOffset();
        } else {
          const ClusterNative& cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clo.GetId()];
          GetConstantMem()->calibObjects.fastTransform->Transform(slice, row, cl.getPad(), cl.getTime(), x, y, z, trk.TZOffset());
        }
      }
      if (prop.PropagateToXAlpha(x, alpha, true)) {
        return way == 0;
      }
      trk.ConstrainSinPhi();
      if (prop.Update(y, z, row, Param(), flags & GPUTPCGMMergedTrackHit::clustererAndSharedFlags, 0, nullptr, false)) {
        return way == 0;
      }
      trk.ConstrainSinPhi();
    }
    if (way) {
      sliceTrack.SetParam2(trk);
    } else {
      sliceTrack.Set(trk, inTrack, alpha, slice);
      sliceTrack.SetX2(0.f);
    }
  }
  return 0;
}

GPUd() void GPUTPCGMMerger::SetTrackClusterZT(GPUTPCGMSliceTrack& track, int iSlice, const GPUTPCTrack* sliceTr)
{
  if (Param().rec.mergerReadFromTrackerDirectly) {
    const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSlice];
    const GPUTPCHitId& ic1 = trk.TrackHits()[sliceTr->FirstHitID()];
    const GPUTPCHitId& ic2 = trk.TrackHits()[sliceTr->FirstHitID() + sliceTr->NHits() - 1];
    int clusterIndex1 = trk.Data().ClusterDataIndex(trk.Data().Row(ic1.RowIndex()), ic1.HitIndex());
    int clusterIndex2 = trk.Data().ClusterDataIndex(trk.Data().Row(ic2.RowIndex()), ic2.HitIndex());
    if (Param().par.earlyTpcTransform) {
      track.SetClusterZT(trk.Data().ClusterData()[clusterIndex1].z, trk.Data().ClusterData()[clusterIndex2].z);
    } else {
      const ClusterNative* cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[iSlice][0];
      track.SetClusterZT(cl[clusterIndex1].getTime(), cl[clusterIndex2].getTime());
    }
  } else {
    if (Param().par.earlyTpcTransform) {
      track.SetClusterZT(sliceTr->OutTrackClusters()->GetZ(), (sliceTr->OutTrackClusters() + sliceTr->NHits() - 1)->GetZ());
    } else {
      const ClusterNative* cls = mConstantMem->ioPtrs.clustersNative->clustersLinear;
      track.SetClusterZT(cls[sliceTr->OutTrackClusters()->GetId()].getTime(), cls[(sliceTr->OutTrackClusters() + sliceTr->NHits() - 1)->GetId()].getTime());
    }
  }
}

GPUd() void GPUTPCGMMerger::UnpackSaveNumber(int id)
{
  mSliceTrackInfoIndex[id] = mMemory->nUnpackedTracks;
}

GPUd() void GPUTPCGMMerger::UnpackSliceGlobal(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice)
{
  int* TrackIds = (int*)mTmpMem;
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSlice];
  float alpha = Param().Alpha(iSlice);
  const GPUTPCTrack* sliceTr = mMemory->firstGlobalTracks[iSlice];
  unsigned int nLocalTracks = Param().rec.mergerReadFromTrackerDirectly ? trk.CommonMemory()->nLocalTracks : mkSlices[iSlice]->NLocalTracks();
  unsigned int nTracks = Param().rec.mergerReadFromTrackerDirectly ? *trk.NTracks() : mkSlices[iSlice]->NTracks();
  for (unsigned int itr = nLocalTracks + iBlock * nThreads + iThread; itr < nTracks; itr += nBlocks * nThreads) {
    if (Param().rec.mergerReadFromTrackerDirectly) {
      sliceTr = &trk.Tracks()[itr];
    } else if (itr > nLocalTracks) {
      sliceTr = sliceTr->GetNextTrack();
    }
    int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * mNMaxSingleSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
    if (localId == -1) {
      continue;
    }
    unsigned int myTrack = CAMath::AtomicAdd(&mMemory->nUnpackedTracks, 1u);
    GPUTPCGMSliceTrack& track = mSliceTrackInfos[myTrack];
    SetTrackClusterZT(track, iSlice, sliceTr);
    track.Set(this, sliceTr, alpha, iSlice);
    track.SetGlobalSectorTrackCov();
    track.SetPrevNeighbour(-1);
    track.SetNextNeighbour(-1);
    track.SetNextSegmentNeighbour(-1);
    track.SetPrevSegmentNeighbour(-1);
    track.SetLocalTrackId(localId);
  }
}

GPUd() void GPUTPCGMMerger::UnpackResetIds(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice)
{
  int* TrackIds = (int*)mTmpMem;
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSlice];
  unsigned int nLocalTracks = Param().rec.mergerReadFromTrackerDirectly ? trk.CommonMemory()->nLocalTracks : mkSlices[iSlice]->NLocalTracks();
  for (unsigned int i = iBlock * nThreads + iThread; i < nLocalTracks; i += nBlocks * nThreads) {
    TrackIds[iSlice * mNMaxSingleSliceTracks + i] = -1;
  }
}

GPUd() void GPUTPCGMMerger::RefitSliceTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice)
{
  int* TrackIds = (int*)mTmpMem;
  const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[iSlice];
  unsigned int nLocalTracks = Param().rec.mergerReadFromTrackerDirectly ? trk.CommonMemory()->nLocalTracks : mkSlices[iSlice]->NLocalTracks();

  float alpha = Param().Alpha(iSlice);
  const GPUTPCTrack* sliceTr = Param().rec.mergerReadFromTrackerDirectly ? nullptr : mkSlices[iSlice]->GetFirstTrack();

  for (unsigned int itr = iBlock * nThreads + iThread; itr < nLocalTracks; itr += nBlocks * nThreads) {
    if (Param().rec.mergerReadFromTrackerDirectly) {
      sliceTr = &trk.Tracks()[itr];
    } else if (itr) {
      sliceTr = sliceTr->GetNextTrack();
    }
    GPUTPCGMSliceTrack track;
    SetTrackClusterZT(track, iSlice, sliceTr);
    if (Param().rec.mergerCovSource == 0) {
      track.Set(this, sliceTr, alpha, iSlice);
      if (!track.FilterErrors(this, iSlice, GPUCA_MAX_SIN_PHI, 0.1f)) {
        continue;
      }
    } else if (Param().rec.mergerCovSource == 1) {
      track.Set(this, sliceTr, alpha, iSlice);
      track.CopyBaseTrackCov();
    } else if (Param().rec.mergerCovSource == 2) {
      if (RefitSliceTrack(track, sliceTr, alpha, iSlice)) {
        track.Set(this, sliceTr, alpha, iSlice); // TODO: Why does the refit fail, it shouldn't, this workaround should be removed
        if (!track.FilterErrors(this, iSlice, GPUCA_MAX_SIN_PHI, 0.1f)) {
          continue;
        }
      }
    }

    CADEBUG(GPUInfo("INPUT Slice %d, Track %u, QPt %f DzDs %f", iSlice, itr, track.QPt(), track.DzDs()));
    track.SetPrevNeighbour(-1);
    track.SetNextNeighbour(-1);
    track.SetNextSegmentNeighbour(-1);
    track.SetPrevSegmentNeighbour(-1);
    track.SetGlobalTrackId(0, -1);
    track.SetGlobalTrackId(1, -1);
    unsigned int myTrack = CAMath::AtomicAdd(&mMemory->nUnpackedTracks, 1u);
    TrackIds[iSlice * mNMaxSingleSliceTracks + sliceTr->LocalTrackId()] = myTrack;
    mSliceTrackInfos[myTrack] = track;
  }
  if (!Param().rec.mergerReadFromTrackerDirectly) {
    mMemory->firstGlobalTracks[iSlice] = nLocalTracks ? sliceTr->GetNextTrack() : mkSlices[iSlice]->GetFirstTrack();
  }
}

GPUd() void GPUTPCGMMerger::LinkGlobalTracks(int nBlocks, int nThreads, int iBlock, int iThread)
{
  for (int itr = SliceTrackInfoGlobalFirst(0) + iBlock * nThreads + iThread; itr < SliceTrackInfoGlobalLast(NSLICES - 1); itr += nThreads * nBlocks) {
    GPUTPCGMSliceTrack& globalTrack = mSliceTrackInfos[itr];
    GPUTPCGMSliceTrack& localTrack = mSliceTrackInfos[globalTrack.LocalTrackId()];
    localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, itr); // Todo: broken in parallel
  }
}

GPUd() void GPUTPCGMMerger::MakeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(unsigned int) * nB, bool useOrigTrackParam)
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = Param().par.ConstBz;

  float dAlpha = Param().par.DAlpha / 2;
  float x0 = 0;

  if (iBorder == 0) { // transport to the left edge of the sector and rotate horizontally
    dAlpha = dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 1) { // transport to the right edge of the sector and rotate horizontally
    dAlpha = -dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 2) { // transport to the middle of the sector and rotate vertically to the border on the left
    x0 = Param().tpcGeometry.Row2X(63);
  } else if (iBorder == 3) { // transport to the middle of the sector and rotate vertically to the border on the right
    dAlpha = -dAlpha;
    x0 = Param().tpcGeometry.Row2X(63);
  } else if (iBorder == 4) { // transport to the middle of the sßector, w/o rotation
    dAlpha = 0;
    x0 = Param().tpcGeometry.Row2X(63);
  }

  const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());
  float cosAlpha = CAMath::Cos(dAlpha);
  float sinAlpha = CAMath::Sin(dAlpha);

  GPUTPCGMSliceTrack trackTmp;
  for (int itr = iBlock * nThreads + iThread; itr < SliceTrackInfoLocalTotal(); itr += nThreads * nBlocks) {
    const GPUTPCGMSliceTrack* track = &mSliceTrackInfos[itr];
    int iSlice = track->Slice();

    if (track->PrevSegmentNeighbour() >= 0 && track->Slice() == mSliceTrackInfos[track->PrevSegmentNeighbour()].Slice()) {
      continue;
    }
    if (useOrigTrackParam) { // TODO: Check how far this makes sense with slice track refit
      if (CAMath::Abs(track->QPt()) < GPUCA_MERGER_LOOPER_QPT_LIMIT) {
        continue;
      }
      const GPUTPCGMSliceTrack* trackMin = track;
      while (track->NextSegmentNeighbour() >= 0 && track->Slice() == mSliceTrackInfos[track->NextSegmentNeighbour()].Slice()) {
        track = &mSliceTrackInfos[track->NextSegmentNeighbour()];
        if (track->OrigTrack()->Param().X() < trackMin->OrigTrack()->Param().X()) {
          trackMin = track;
        }
      }
      trackTmp = *trackMin;
      track = &trackTmp;
      if (Param().rec.mergerCovSource == 2 && trackTmp.X2() != 0.f) {
        trackTmp.UseParam2();
      } else {
        trackTmp.Set(this, trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Slice());
      }
    } else {
      if (CAMath::Abs(track->QPt()) < GPUCA_MERGER_HORIZONTAL_DOUBLE_QPT_LIMIT) {
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
      for (int i = 0; i < 4; i++) {
        if (CAMath::Abs(b.Cov()[i]) >= 5.0) {
          b.SetCov(i, 5.0);
        }
      }
      if (CAMath::Abs(b.Cov()[4]) >= 0.5) {
        b.SetCov(4, 0.5);
      }
      unsigned int myTrack = CAMath::AtomicAdd(&nB[iSlice], 1u);
      B[iSlice][myTrack] = b;
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<0>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice1, GPUTPCGMBorderTrack* B1, int N1, int iSlice2, GPUTPCGMBorderTrack* B2, int N2, int mergeMode)
{
  CADEBUG(GPUInfo("\nMERGING Slices %d %d NTracks %d %d CROSS %d", iSlice1, iSlice2, N1, N2, mergeMode));
  GPUTPCGMBorderRange* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());
  bool sameSlice = (iSlice1 == iSlice2);
  for (int itr = iBlock * nThreads + iThread; itr < N1; itr += nThreads * nBlocks) {
    GPUTPCGMBorderTrack& b = B1[itr];
    float d = CAMath::Max(0.5f, 3.5f * CAMath::Sqrt(b.Cov()[1]));
    if (CAMath::Abs(b.Par()[4]) >= 20) {
      d *= 2;
    } else if (d > 3) {
      d = 3;
    }
    CADEBUG(
      printf("  Input Slice 1 %d Track %d: ", iSlice1, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf(" - D %8.3f\n", d));
    GPUTPCGMBorderRange range;
    range.fId = itr;
    range.fMin = b.Par()[1] + b.ZOffsetLinear() - d;
    range.fMax = b.Par()[1] + b.ZOffsetLinear() + d;
    range1[itr] = range;
    if (sameSlice) {
      range2[itr] = range;
    }
  }
  if (!sameSlice) {
    for (int itr = iBlock * nThreads + iThread; itr < N2; itr += nThreads * nBlocks) {
      GPUTPCGMBorderTrack& b = B2[itr];
      float d = CAMath::Max(0.5f, 3.5f * CAMath::Sqrt(b.Cov()[1]));
      if (CAMath::Abs(b.Par()[4]) >= 20) {
        d *= 2;
      } else if (d > 3) {
        d = 3;
      }
      CADEBUG(
        printf("  Input Slice 2 %d Track %d: ", iSlice2, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf(" - D %8.3f\n", d));
      GPUTPCGMBorderRange range;
      range.fId = itr;
      range.fMin = b.Par()[1] + b.ZOffsetLinear() - d;
      range.fMax = b.Par()[1] + b.ZOffsetLinear() + d;
      range2[itr] = range;
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<1>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice1, GPUTPCGMBorderTrack* B1, int N1, int iSlice2, GPUTPCGMBorderTrack* B2, int N2, int mergeMode)
{
  GPUTPCGMBorderRange* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());

  if (iThread == 0) {
    if (iBlock == 0) {
      GPUCommonAlgorithm::sortDeviceDynamic(range1, range1 + N1, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return a.fMin < b.fMin; });
    } else if (iBlock == 1) {
      GPUCommonAlgorithm::sortDeviceDynamic(range2, range2 + N2, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return a.fMax < b.fMax; });
    }
  }
}

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GPUCA_GPUCODE_GENRTC) // Specialize MergeBorderTracks<3>
struct MergeBorderTracks_compMax {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return a.fMax < b.fMax;
  }
};
struct MergeBorderTracks_compMin {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return a.fMin < b.fMin;
  }
};

template <>
void GPUCA_KRNL_BACKEND_CLASS::runKernelBackendInternal<GPUTPCGMMergerMergeBorders, 3>(krnlSetup& _xyz, GPUTPCGMBorderRange* const& range, int const& N, int const& cmpMax)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<GPUTPCGMBorderRange> p(range);
  ThrustVolatileAsyncAllocator alloc(this);
  if (cmpMax) {
    thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(mInternals->Streams[_xyz.x.stream]), p, p + N, MergeBorderTracks_compMax());
  } else {
    thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(mInternals->Streams[_xyz.x.stream]), p, p + N, MergeBorderTracks_compMin());
  }
}
#endif // __CUDACC__ || __HIPCC__ - MergeBorderTracks<3>

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<3>(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCGMBorderRange* range, int N, int cmpMax)
{
  if (iThread == 0) {
    if (cmpMax) {
      GPUCommonAlgorithm::sortDeviceDynamic(range, range + N, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return a.fMax < b.fMax; });
    } else {
      GPUCommonAlgorithm::sortDeviceDynamic(range, range + N, [](const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b) { return a.fMin < b.fMin; });
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<2>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice1, GPUTPCGMBorderTrack* B1, int N1, int iSlice2, GPUTPCGMBorderTrack* B2, int N2, int mergeMode)
{
  int statAll = 0, statMerged = 0;
  float factor2ys = 1.5; // 1.5;//SG!!!
  float factor2zt = 1.5; // 1.5;//SG!!!
  float factor2k = 2.0;  // 2.2;

  factor2k = 3.5 * 3.5 * factor2k * factor2k;
  factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
  factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;

  int minNPartHits = 10; // SG!!!
  int minNTotalHits = 20;

  bool sameSlice = (iSlice1 == iSlice2);

  GPUTPCGMBorderRange* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderRange* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());

  int i2 = 0;
  for (int i1 = iBlock * nThreads + iThread; i1 < N1; i1 += nThreads * nBlocks) {
    GPUTPCGMBorderRange r1 = range1[i1];
    while (i2 < N2 && range2[i2].fMax < r1.fMin) {
      i2++;
    }

    GPUTPCGMBorderTrack& b1 = B1[r1.fId];
    if (b1.NClusters() < minNPartHits) {
      continue;
    }
    int iBest2 = -1;
    int lBest2 = 0;
    statAll++;
    for (int k2 = i2; k2 < N2; k2++) {
      GPUTPCGMBorderRange r2 = range2[k2];
      if (r2.fMin > r1.fMax) {
        break;
      }
      if (sameSlice && (r1.fId >= r2.fId)) {
        continue;
      }
      // do check

      GPUTPCGMBorderTrack& b2 = B2[r2.fId];
#if defined(GPUCA_MERGER_BY_MC_LABEL) && !defined(GPUCA_GPUCODE)
      long int label1 = GetTrackLabel(b1);
      long int label2 = GetTrackLabel(b2);
      if (label1 != label2 && label1 != -1) // DEBUG CODE, match by MC label
#endif
      {
        CADEBUG(
          if (GetConstantMem()->ioPtrs.mcLabelsTPC) {printf("Comparing track %3d to %3d: ", r1.fId, r2.fId); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Cov()[i]); } printf("\n%28s", ""); });
        CADEBUG(
          if (GetConstantMem()->ioPtrs.mcLabelsTPC) {for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Cov()[i]); } printf("   -   %5s   -   ", GetTrackLabel(b1) == GetTrackLabel(b2) ? "CLONE" : "FAKE"); });
        if (b2.NClusters() < lBest2) {
          CADEBUG2(continue, printf("!NCl1\n"));
        }
        if (mergeMode > 0) {
          // Merging CE tracks
          int maxRowDiff = mergeMode == 2 ? 1 : 3; // TODO: check cut
          if (CAMath::Abs(b1.Row() - b2.Row()) > maxRowDiff) {
            CADEBUG2(continue, printf("!ROW\n"));
          }
          if (CAMath::Abs(b1.Par()[2] - b2.Par()[2]) > 0.5 || CAMath::Abs(b1.Par()[3] - b2.Par()[3]) > 0.5) {
            CADEBUG2(continue, printf("!CE SinPhi/Tgl\n")); // Crude cut to avoid totally wrong matches, TODO: check cut
          }
        }
        if (!b1.CheckChi2Y(b2, factor2ys)) {
          CADEBUG2(continue, printf("!Y\n"));
        }
        // if( !b1.CheckChi2Z(b2, factor2zt ) ) CADEBUG2(continue, printf("!NCl1\n"));
        if (!b1.CheckChi2QPt(b2, factor2k)) {
          CADEBUG2(continue, printf("!QPt\n"));
        }
        float fys = CAMath::Abs(b1.Par()[4]) < 20 ? factor2ys : (2. * factor2ys);
        float fzt = CAMath::Abs(b1.Par()[4]) < 20 ? factor2zt : (2. * factor2zt);
        if (!b1.CheckChi2YS(b2, fys)) {
          CADEBUG2(continue, printf("!YS\n"));
        }
        if (!b1.CheckChi2ZT(b2, fzt)) {
          CADEBUG2(continue, printf("!ZT\n"));
        }
        if (CAMath::Abs(b1.Par()[4]) < 20) {
          if (b2.NClusters() < minNPartHits) {
            CADEBUG2(continue, printf("!NCl2\n"));
          }
          if (b1.NClusters() + b2.NClusters() < minNTotalHits) {
            CADEBUG2(continue, printf("!NCl3\n"));
          }
        }
        CADEBUG(printf("OK: dZ %8.3f D1 %8.3f D2 %8.3f\n", CAMath::Abs(b1.Par()[1] - b2.Par()[1]), 3.5 * sqrt(b1.Cov()[1]), 3.5 * sqrt(b2.Cov()[1])));
      } // DEBUG CODE, match by MC label
      lBest2 = b2.NClusters();
      iBest2 = b2.TrackID();
    }

    if (iBest2 < 0) {
      continue;
    }
    statMerged++;

    CADEBUG(GPUInfo("Found match %d %d", b1.TrackID(), iBest2));

    mTrackLinks[b1.TrackID()] = iBest2;
    if (mergeMode > 0) {
      mTrackLinks[iBest2] = b1.TrackID();
    }
  }
  // GPUInfo("STAT: slices %d, %d: all %d merged %d", iSlice1, iSlice2, statAll, statMerged);
}

GPUdii() void GPUTPCGMMerger::MergeBorderTracksSetup(int& n1, int& n2, GPUTPCGMBorderTrack*& b1, GPUTPCGMBorderTrack*& b2, int& jSlice, int iSlice, char withinSlice, char mergeMode)
{
  if (withinSlice == 1) {
    jSlice = iSlice;
    n1 = n2 = mMemory->tmpCounter[iSlice];
    b1 = b2 = mBorder[iSlice];
  } else if (withinSlice == -1) {
    jSlice = (iSlice + NSLICES / 2);
    const int offset = mergeMode == 2 ? NSLICES : 0;
    n1 = mMemory->tmpCounter[iSlice + offset];
    n2 = mMemory->tmpCounter[jSlice + offset];
    b1 = mBorder[iSlice + offset];
    b2 = mBorder[jSlice + offset];
  } else {
    jSlice = mNextSliceInd[iSlice];
    n1 = mMemory->tmpCounter[iSlice];
    n2 = mMemory->tmpCounter[NSLICES + jSlice];
    b1 = mBorder[iSlice];
    b2 = mBorder[NSLICES + jSlice];
  }
}

template <int I>
GPUd() void GPUTPCGMMerger::MergeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode)
{
  int n1, n2;
  GPUTPCGMBorderTrack *b1, *b2;
  int jSlice;
  MergeBorderTracksSetup(n1, n2, b1, b2, jSlice, iSlice, withinSlice, mergeMode);
  MergeBorderTracks<I>(nBlocks, nThreads, iBlock, iThread, iSlice, b1, n1, jSlice, b2, n2, mergeMode);
}

template GPUd() void GPUTPCGMMerger::MergeBorderTracks<0>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode);
template GPUd() void GPUTPCGMMerger::MergeBorderTracks<1>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode);
template GPUd() void GPUTPCGMMerger::MergeBorderTracks<2>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode);

GPUd() void GPUTPCGMMerger::MergeWithinSlicesPrepare(int nBlocks, int nThreads, int iBlock, int iThread)
{
  float x0 = Param().tpcGeometry.Row2X(63);
  const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());

  for (int itr = iBlock * nThreads + iThread; itr < SliceTrackInfoLocalTotal(); itr += nThreads * nBlocks) {
    GPUTPCGMSliceTrack& track = mSliceTrackInfos[itr];
    int iSlice = track.Slice();
    GPUTPCGMBorderTrack b;
    ;
    if (track.TransportToX(this, x0, Param().par.ConstBz, b, maxSin)) {
      b.SetTrackID(itr);
      CADEBUG(
        printf("WITHIN SLICE %d Track %d - ", iSlice, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf("\n"));
      b.SetNClusters(track.NClusters());
      unsigned int myTrack = CAMath::AtomicAdd(&mMemory->tmpCounter[iSlice], 1u);
      mBorder[iSlice][myTrack] = b;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeSlicesPrepare(int nBlocks, int nThreads, int iBlock, int iThread, int border0, int border1, char useOrigTrackParam)
{
  bool part2 = iBlock & 1;
  int border = part2 ? border1 : border0;
  GPUAtomic(unsigned int)* n = mMemory->tmpCounter;
  GPUTPCGMBorderTrack** b = mBorder;
  if (part2) {
    n += NSLICES;
    b += NSLICES;
  }
  MakeBorderTracks((nBlocks + 1) >> 1, nThreads, iBlock >> 1, iThread, border, b, n, useOrigTrackParam);
}

GPUdi() void GPUTPCGMMerger::setBlockRange(int elems, int nBlocks, int iBlock, int& start, int& end)
{
  start = (elems + nBlocks - 1) / nBlocks * iBlock;
  end = (elems + nBlocks - 1) / nBlocks * (iBlock + 1);
  end = CAMath::Min(elems, end);
}

GPUd() void GPUTPCGMMerger::hookEdge(int u, int v)
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
    int h = CAMath::Max(u, v);
    int l = CAMath::Min(u, v);

    int old = CAMath::AtomicCAS(&mTrackCCRoots[h], h, l);
    if (old == h) {
      break;
    }

    u = mTrackCCRoots[h];
    v = l;
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsSetup(int nBlocks, int nThreads, int iBlock, int iThread)
{
  int start, end;
  setBlockRange(SliceTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int i = start + iThread; i < end; i += nThreads) {
    mTrackCCRoots[i] = i;
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsHookLinks(int nBlocks, int nThreads, int iBlock, int iThread)
{
  // Compute connected components in parallel, step 1.
  // Source: Adaptive Work-Efficient Connected Components on the GPU, Sutton et al, 2016 (https://arxiv.org/pdf/1612.01178.pdf)
  int start, end;
  setBlockRange(SliceTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int itr = start + iThread; itr < end; itr += nThreads) {
    hookEdge(itr, mTrackLinks[itr]);
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsHookNeighbors(int nBlocks, int nThreads, int iBlock, int iThread)
{
  // Compute connected components in parallel, step 1 - Part 2.
  nBlocks = nBlocks / 4 * 4;
  if (iBlock >= nBlocks) {
    return;
  }

  int start, end;
  setBlockRange(SliceTrackInfoLocalTotal(), nBlocks / 4, iBlock / 4, start, end);

  int myNeighbor = iBlock % 4;

  for (int itr = start + iThread; itr < end; itr += nThreads) {
    int v = mSliceTrackInfos[itr].AnyNeighbour(myNeighbor);
    hookEdge(itr, v);
  }
}

GPUd() void GPUTPCGMMerger::ResolveFindConnectedComponentsMultiJump(int nBlocks, int nThreads, int iBlock, int iThread)
{
  // Compute connected components in parallel, step 2.
  int start, end;
  setBlockRange(SliceTrackInfoLocalTotal(), nBlocks, iBlock, start, end);
  for (int itr = start + iThread; itr < end; itr += nThreads) {
    int root = itr;
    int next = mTrackCCRoots[root];
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

GPUd() void GPUTPCGMMerger::ResolveMergeSlices(GPUResolveSharedMemory& smem, int nBlocks, int nThreads, int iBlock, int iThread, char useOrigTrackParam, char mergeAll)
{
  if (!mergeAll) {
    /*int neighborType = useOrigTrackParam ? 1 : 0;
    int old1 = newTrack2.PrevNeighbour(0);
    int old2 = newTrack1.NextNeighbour(0);
    if (old1 < 0 && old2 < 0) neighborType = 0;
    if (old1 == itr) continue;
    if (neighborType) old1 = newTrack2.PrevNeighbour(1);
    if ( old1 >= 0 )
    {
        GPUTPCGMSliceTrack &oldTrack1 = mSliceTrackInfos[old1];
        if ( oldTrack1.NClusters() < newTrack1.NClusters() ) {
            newTrack2.SetPrevNeighbour( -1, neighborType );
            oldTrack1.SetNextNeighbour( -1, neighborType );
        } else continue;
    }

    if (old2 == itr2) continue;
    if (neighborType) old2 = newTrack1.NextNeighbour(1);
    if ( old2 >= 0 )
    {
        GPUTPCGMSliceTrack &oldTrack2 = mSliceTrackInfos[old2];
        if ( oldTrack2.NClusters() < newTrack2.NClusters() )
        {
        oldTrack2.SetPrevNeighbour( -1, neighborType );
        } else continue;
    }
    newTrack1.SetNextNeighbour( itr2, neighborType );
    newTrack2.SetPrevNeighbour( itr, neighborType );*/
  }

  int start, end;
  setBlockRange(SliceTrackInfoLocalTotal(), nBlocks, iBlock, start, end);

  for (int baseIdx = 0; baseIdx < SliceTrackInfoLocalTotal(); baseIdx += nThreads) {
    int itr = baseIdx + iThread;
    bool inRange = itr < SliceTrackInfoLocalTotal();

    int itr2 = -1;
    if (inRange) {
      itr2 = mTrackLinks[itr];
    }

    bool resolveSlice = (itr2 > -1);
    if (resolveSlice) {
      int root = mTrackCCRoots[itr];
      resolveSlice &= (start <= root) && (root < end);
    }

    short smemIdx = work_group_scan_inclusive_add(short(resolveSlice));

    if (resolveSlice) {
      smem.iTrack1[smemIdx - 1] = itr;
      smem.iTrack2[smemIdx - 1] = itr2;
    }
    GPUbarrier();

    if (iThread < nThreads - 1) {
      continue;
    }

    const int nSlices = smemIdx;

    for (int i = 0; i < nSlices; i++) {
      itr = smem.iTrack1[i];
      itr2 = smem.iTrack2[i];

      GPUTPCGMSliceTrack* track1 = &mSliceTrackInfos[itr];
      GPUTPCGMSliceTrack* track2 = &mSliceTrackInfos[itr2];
      GPUTPCGMSliceTrack* track1Base = track1;
      GPUTPCGMSliceTrack* track2Base = track2;

      bool sameSegment = CAMath::Abs(track1->NClusters() > track2->NClusters() ? track1->QPt() : track2->QPt()) < 2 || track1->QPt() * track2->QPt() > 0;
      // GPUInfo("\nMerge %d with %d - same segment %d", itr, itr2, (int) sameSegment);
      // PrintMergeGraph(track1, std::cout);
      // PrintMergeGraph(track2, std::cout);

      while (track2->PrevSegmentNeighbour() >= 0) {
        track2 = &mSliceTrackInfos[track2->PrevSegmentNeighbour()];
      }
      if (sameSegment) {
        if (track1 == track2) {
          continue;
        }
        while (track1->PrevSegmentNeighbour() >= 0) {
          track1 = &mSliceTrackInfos[track1->PrevSegmentNeighbour()];
          if (track1 == track2) {
            goto NextTrack;
          }
        }
        GPUCommonAlgorithm::swap(track1, track1Base);
        for (int k = 0; k < 2; k++) {
          GPUTPCGMSliceTrack* tmp = track1Base;
          while (tmp->Neighbour(k) >= 0) {
            tmp = &mSliceTrackInfos[tmp->Neighbour(k)];
            if (tmp == track2) {
              goto NextTrack;
            }
          }
        }

        while (track1->NextSegmentNeighbour() >= 0) {
          track1 = &mSliceTrackInfos[track1->NextSegmentNeighbour()];
          if (track1 == track2) {
            goto NextTrack;
          }
        }
      } else {
        while (track1->PrevSegmentNeighbour() >= 0) {
          track1 = &mSliceTrackInfos[track1->PrevSegmentNeighbour()];
        }

        if (track1 == track2) {
          continue;
        }
        for (int k = 0; k < 2; k++) {
          GPUTPCGMSliceTrack* tmp = track1;
          while (tmp->Neighbour(k) >= 0) {
            tmp = &mSliceTrackInfos[tmp->Neighbour(k)];
            if (tmp == track2) {
              goto NextTrack;
            }
          }
        }

        float z1min, z1max, z2min, z2max;
        z1min = track1->MinClusterZT();
        z1max = track1->MaxClusterZT();
        z2min = track2->MinClusterZT();
        z2max = track2->MaxClusterZT();
        if (track1 != track1Base) {
          z1min = CAMath::Min(z1min, track1Base->MinClusterZT());
          z1max = CAMath::Max(z1max, track1Base->MaxClusterZT());
        }
        if (track2 != track2Base) {
          z2min = CAMath::Min(z2min, track2Base->MinClusterZT());
          z2max = CAMath::Max(z2max, track2Base->MaxClusterZT());
        }
        bool goUp = z2max - z1min > z1max - z2min;

        if (track1->Neighbour(goUp) < 0 && track2->Neighbour(!goUp) < 0) {
          track1->SetNeighbor(track2 - mSliceTrackInfos, goUp);
          track2->SetNeighbor(track1 - mSliceTrackInfos, !goUp);
          // GPUInfo("Result (simple neighbor)");
          // PrintMergeGraph(track1, std::cout);
          continue;
        } else if (track1->Neighbour(goUp) < 0) {
          track2 = &mSliceTrackInfos[track2->Neighbour(!goUp)];
          GPUCommonAlgorithm::swap(track1, track2);
        } else if (track2->Neighbour(!goUp) < 0) {
          track1 = &mSliceTrackInfos[track1->Neighbour(goUp)];
        } else { // Both would work, but we use the simpler one
          track1 = &mSliceTrackInfos[track1->Neighbour(goUp)];
        }
        track1Base = track1;
      }

      track2Base = track2;
      if (!sameSegment) {
        while (track1->NextSegmentNeighbour() >= 0) {
          track1 = &mSliceTrackInfos[track1->NextSegmentNeighbour()];
        }
      }
      track1->SetNextSegmentNeighbour(track2 - mSliceTrackInfos);
      track2->SetPrevSegmentNeighbour(track1 - mSliceTrackInfos);
      // k = 0: Merge right side
      // k = 1: Merge left side
      for (int k = 0; k < 2; k++) {
        track1 = track1Base;
        track2 = track2Base;
        while (track2->Neighbour(k) >= 0) {
          if (track1->Neighbour(k) >= 0) {
            GPUTPCGMSliceTrack* track1new = &mSliceTrackInfos[track1->Neighbour(k)];
            GPUTPCGMSliceTrack* track2new = &mSliceTrackInfos[track2->Neighbour(k)];
            track2->SetNeighbor(-1, k);
            track2new->SetNeighbor(-1, k ^ 1);
            track1 = track1new;
            while (track1->NextSegmentNeighbour() >= 0) {
              track1 = &mSliceTrackInfos[track1->NextSegmentNeighbour()];
            }
            track1->SetNextSegmentNeighbour(track2new - mSliceTrackInfos);
            track2new->SetPrevSegmentNeighbour(track1 - mSliceTrackInfos);
            track1 = track1new;
            track2 = track2new;
          } else {
            GPUTPCGMSliceTrack* track2new = &mSliceTrackInfos[track2->Neighbour(k)];
            track1->SetNeighbor(track2->Neighbour(k), k);
            track2->SetNeighbor(-1, k);
            track2new->SetNeighbor(track1 - mSliceTrackInfos, k ^ 1);
          }
        }
      }
      // GPUInfo("Result");
      // PrintMergeGraph(track1, std::cout);
    NextTrack:;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeCEFill(const GPUTPCGMSliceTrack* track, const GPUTPCGMMergedTrackHit& cls, const GPUTPCGMMergedTrackHitXYZ* clsXYZ, int itr)
{
  if (Param().rec.NonConsecutiveIDs) {
    return;
  }

#ifdef GPUCA_MERGER_CE_ROWLIMIT
  if (CAMath::Abs(track->QPt()) < 0.3 && (cls.row < GPUCA_MERGER_CE_ROWLIMIT || cls.row >= GPUCA_ROW_COUNT - GPUCA_MERGER_CE_ROWLIMIT)) {
    return;
  }
#endif

  float z = 0;
  if (Param().par.earlyTpcTransform) {
    z = clsXYZ->z;
  } else {
    float x, y;
    auto& cln = mConstantMem->ioPtrs.clustersNative->clustersLinear[cls.num];
    GPUTPCConvertImpl::convert(*mConstantMem, cls.slice, cls.row, cln.getPad(), cln.getTime(), x, y, z);
  }

  if (!Param().par.ContinuousTracking && CAMath::Abs(z) > 10) {
    return;
  }
  int slice = track->Slice();
  for (int attempt = 0; attempt < 2; attempt++) {
    GPUTPCGMBorderTrack b;
    const float x0 = Param().tpcGeometry.Row2X(attempt == 0 ? 63 : cls.row);
    if (track->TransportToX(this, x0, Param().par.ConstBz, b, GPUCA_MAX_SIN_PHI_LOW)) {
      b.SetTrackID(itr);
      b.SetNClusters(mOutputTracks[itr].NClusters());
      if (CAMath::Abs(b.Cov()[4]) >= 0.5) {
        b.SetCov(4, 0.5); // TODO: Is this needed and better than the cut in BorderTrack?
      }
      if (track->CSide()) {
        b.SetPar(1, b.Par()[1] - 2 * (z - b.ZOffsetLinear()));
        b.SetZOffsetLinear(-b.ZOffsetLinear());
      }
      b.SetRow(cls.row);
      unsigned int id = slice + attempt * NSLICES;
      unsigned int myTrack = CAMath::AtomicAdd(&mMemory->tmpCounter[id], 1u);
      mBorder[id][myTrack] = b;
      break;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeCE(int nBlocks, int nThreads, int iBlock, int iThread)
{
  const ClusterNative* cls = Param().par.earlyTpcTransform ? nullptr : mConstantMem->ioPtrs.clustersNative->clustersLinear;
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nThreads * nBlocks) {
    if (mOutputTracks[i].CSide() == 0 && mTrackLinks[i] >= 0) {
      if (mTrackLinks[mTrackLinks[i]] != (int)i) {
        continue;
      }
      GPUTPCGMMergedTrack* trk[2] = {&mOutputTracks[i], &mOutputTracks[mTrackLinks[i]]};

      if (!trk[1]->OK() || trk[1]->CCE()) {
        continue;
      }
      bool looper = trk[0]->Looper() || trk[1]->Looper() || (trk[0]->GetParam().GetQPt() > 1 && trk[0]->GetParam().GetQPt() * trk[1]->GetParam().GetQPt() < 0);
      if (!looper && trk[0]->GetParam().GetPar(3) * trk[1]->GetParam().GetPar(3) < 0) {
        continue;
      }

      unsigned int newRef = CAMath::AtomicAdd(&mMemory->nOutputTrackClusters, trk[0]->NClusters() + trk[1]->NClusters());
      if (newRef + trk[0]->NClusters() + trk[1]->NClusters() >= mNMaxOutputTrackClusters) {
        raiseError(GPUErrors::ERROR_MERGER_CE_HIT_OVERFLOW, newRef + trk[0]->NClusters() + trk[1]->NClusters(), mNMaxOutputTrackClusters);
        for (unsigned int k = newRef; k < mNMaxOutputTrackClusters; k++) {
          mClusters[k].num = 0;
          mClusters[k].state = 0;
        }
        CAMath::AtomicExch(&mMemory->nOutputTrackClusters, mNMaxOutputTrackClusters);
        return;
      }

      bool needswap = false;
      if (looper) {
        float z0max, z1max;
        if (Param().par.earlyTpcTransform) {
          z0max = CAMath::Max(CAMath::Abs(mClustersXYZ[trk[0]->FirstClusterRef()].z), CAMath::Abs(mClustersXYZ[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z));
          z1max = CAMath::Max(CAMath::Abs(mClustersXYZ[trk[1]->FirstClusterRef()].z), CAMath::Abs(mClustersXYZ[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z));
        } else {
          z0max = -CAMath::Min(cls[mClusters[trk[0]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime());
          z1max = -CAMath::Min(cls[mClusters[trk[1]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime());
        }
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

      bool reverse[2] = {false, false};
      if (looper) {
        if (Param().par.earlyTpcTransform) {
          reverse[0] = (mClustersXYZ[trk[0]->FirstClusterRef()].z > mClustersXYZ[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) ^ (trk[0]->CSide() > 0);
          reverse[1] = (mClustersXYZ[trk[1]->FirstClusterRef()].z < mClustersXYZ[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) ^ (trk[1]->CSide() > 0);
        } else {
          reverse[0] = cls[mClusters[trk[0]->FirstClusterRef()].num].getTime() < cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime();
          reverse[1] = cls[mClusters[trk[1]->FirstClusterRef()].num].getTime() > cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime();
        }
      }

      if (Param().par.ContinuousTracking) {
        if (Param().par.earlyTpcTransform) {
          const float z0 = trk[0]->CSide() ? CAMath::Max(mClustersXYZ[trk[0]->FirstClusterRef()].z, mClustersXYZ[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) : CAMath::Min(mClustersXYZ[trk[0]->FirstClusterRef()].z, mClustersXYZ[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z);
          const float z1 = trk[1]->CSide() ? CAMath::Max(mClustersXYZ[trk[1]->FirstClusterRef()].z, mClustersXYZ[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) : CAMath::Min(mClustersXYZ[trk[1]->FirstClusterRef()].z, mClustersXYZ[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z);
          const float offset = CAMath::Abs(z1) > CAMath::Abs(z0) ? -z0 : z1;
          trk[1]->Param().Z() += trk[1]->Param().TZOffset() - offset;
          trk[1]->Param().TZOffset() = offset;
        } else {
          GPUTPCGMMergedTrackHit* clsmax;
          const float tmax = CAMath::MaxWithRef(cls[mClusters[trk[0]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime(),
                                                cls[mClusters[trk[1]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime(),
                                                &mClusters[trk[0]->FirstClusterRef()], &mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1],
                                                &mClusters[trk[1]->FirstClusterRef()], &mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1], clsmax);
          const float offset = CAMath::Max(tmax - mConstantMem->calibObjects.fastTransform->getMaxDriftTime(clsmax->slice, clsmax->row, cls[clsmax->num].getPad()), 0.f);
          trk[1]->Param().Z() += mConstantMem->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(trk[1]->CSide() * NSLICES / 2, trk[1]->Param().TZOffset() - offset);
          trk[1]->Param().TZOffset() = offset;
        }
      }

      int pos = newRef;
      for (int k = 1; k >= 0; k--) {
        if (reverse[k]) {
          for (int j = trk[k]->NClusters() - 1; j >= 0; j--) {
            if (Param().par.earlyTpcTransform) {
              mClustersXYZ[pos] = mClustersXYZ[trk[k]->FirstClusterRef() + j];
            }
            mClusters[pos++] = mClusters[trk[k]->FirstClusterRef() + j];
          }
        } else {
          for (unsigned int j = 0; j < trk[k]->NClusters(); j++) {
            if (Param().par.earlyTpcTransform) {
              mClustersXYZ[pos] = mClustersXYZ[trk[k]->FirstClusterRef() + j];
            }
            mClusters[pos++] = mClusters[trk[k]->FirstClusterRef() + j];
          }
        }
      }
      trk[1]->SetFirstClusterRef(newRef);
      trk[1]->SetNClusters(trk[0]->NClusters() + trk[1]->NClusters());
      if (trk[1]->NClusters() > GPUCA_MERGER_MAX_TRACK_CLUSTERS) {
        trk[1]->SetFirstClusterRef(trk[1]->FirstClusterRef() + trk[1]->NClusters() - GPUCA_MERGER_MAX_TRACK_CLUSTERS);
        trk[1]->SetNClusters(GPUCA_MERGER_MAX_TRACK_CLUSTERS);
      }
      trk[1]->SetCCE(true);
      if (looper) {
        trk[1]->SetLooper(true);
      }
      trk[1]->SetLegs(trk[1]->Legs() + trk[0]->Legs());
      trk[0]->SetNClusters(0);
      trk[0]->SetOK(false);
    }
  }

  // for (int i = 0;i < mMemory->nOutputTracks;i++) {if (mOutputTracks[i].CCE() == false) {mOutputTracks[i].SetNClusters(0);mOutputTracks[i].SetOK(false);}} //Remove all non-CE tracks
}

struct GPUTPCGMMerger_CompareClusterIdsLooper {
  struct clcomparestruct {
    unsigned char leg;
  };

  const unsigned char leg;
  const bool outwards;
  const GPUTPCGMMerger::trackCluster* const cmp1;
  const clcomparestruct* const cmp2;
  GPUd() GPUTPCGMMerger_CompareClusterIdsLooper(unsigned char l, bool o, const GPUTPCGMMerger::trackCluster* c1, const clcomparestruct* c2) : leg(l), outwards(o), cmp1(c1), cmp2(c2) {}
  GPUd() bool operator()(const short aa, const short bb)
  {
    const clcomparestruct& a = cmp2[aa];
    const clcomparestruct& b = cmp2[bb];
    const GPUTPCGMMerger::trackCluster& a1 = cmp1[aa];
    const GPUTPCGMMerger::trackCluster& b1 = cmp1[bb];
    if (a.leg != b.leg) {
      return ((leg > 0) ^ (a.leg > b.leg));
    }
    if (a1.row != b1.row) {
      return ((a1.row > b1.row) ^ ((a.leg - leg) & 1) ^ outwards);
    }
    return a1.id > b1.id;
  }
};

struct GPUTPCGMMerger_CompareClusterIds {
  const GPUTPCGMMerger::trackCluster* const mCmp;
  GPUd() GPUTPCGMMerger_CompareClusterIds(const GPUTPCGMMerger::trackCluster* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const short aa, const short bb)
  {
    const GPUTPCGMMerger::trackCluster& a = mCmp[aa];
    const GPUTPCGMMerger::trackCluster& b = mCmp[bb];
    if (a.row != b.row) {
      return (a.row > b.row);
    }
    return (a.id > b.id);
  }
};

GPUd() void GPUTPCGMMerger::CollectMergedTracks(int nBlocks, int nThreads, int iBlock, int iThread)
{

  GPUTPCGMSliceTrack* trackParts[kMaxParts];

  for (int itr = iBlock * nThreads + iThread; itr < SliceTrackInfoLocalTotal(); itr += nThreads * nBlocks) {

    GPUTPCGMSliceTrack& track = mSliceTrackInfos[itr];

    if (track.PrevSegmentNeighbour() >= 0) {
      continue;
    }
    if (track.PrevNeighbour() >= 0) {
      continue;
    }
    int nParts = 0;
    int nHits = 0;
    int leg = 0;
    GPUTPCGMSliceTrack *trbase = &track, *tr = &track;
    tr->SetPrevSegmentNeighbour(1000000000);
    while (true) {
      if (nParts >= kMaxParts) {
        break;
      }
      if (nHits + tr->NClusters() > kMaxClusters) {
        break;
      }
      nHits += tr->NClusters();

      tr->SetLeg(leg);
      trackParts[nParts++] = tr;
      for (int i = 0; i < 2; i++) {
        if (tr->GlobalTrackId(i) != -1) {
          if (nParts >= kMaxParts) {
            break;
          }
          if (nHits + mSliceTrackInfos[tr->GlobalTrackId(i)].NClusters() > kMaxClusters) {
            break;
          }
          trackParts[nParts] = &mSliceTrackInfos[tr->GlobalTrackId(i)];
          trackParts[nParts++]->SetLeg(leg);
          nHits += mSliceTrackInfos[tr->GlobalTrackId(i)].NClusters();
        }
      }
      int jtr = tr->NextSegmentNeighbour();
      if (jtr >= 0) {
        tr = &(mSliceTrackInfos[jtr]);
        tr->SetPrevSegmentNeighbour(1000000002);
        continue;
      }
      jtr = trbase->NextNeighbour();
      if (jtr >= 0) {
        trbase = &(mSliceTrackInfos[jtr]);
        tr = trbase;
        if (tr->PrevSegmentNeighbour() >= 0) {
          break;
        }
        tr->SetPrevSegmentNeighbour(1000000001);
        leg++;
        continue;
      }
      break;
    }

    // unpack and sort clusters
    if (nParts > 1 && leg == 0) {
      GPUCommonAlgorithm::sort(trackParts, trackParts + nParts, [](const GPUTPCGMSliceTrack* a, const GPUTPCGMSliceTrack* b) { return (a->X() > b->X()); });
    }

    if (Param().rec.dropLoopers && leg > 0) {
      nParts = 1;
      leg = 0;
    }

    trackCluster trackClusters[kMaxClusters];
    nHits = 0;
    for (int ipart = 0; ipart < nParts; ipart++) {
      const GPUTPCGMSliceTrack* t = trackParts[ipart];
      CADEBUG(printf("Collect Track %d Part %d QPt %f DzDs %f\n", mMemory->nOutputTracks, ipart, t->QPt(), t->DzDs()));
      int nTrackHits = t->NClusters();
      trackCluster* c2 = trackClusters + nHits + nTrackHits - 1;
      for (int i = 0; i < nTrackHits; i++, c2--) {
        if (Param().rec.mergerReadFromTrackerDirectly) {
          const GPUTPCTracker& trk = GetConstantMem()->tpcTrackers[t->Slice()];
          const GPUTPCHitId& ic = trk.TrackHits()[t->OrigTrack()->FirstHitID() + i];
          unsigned int id = trk.Data().ClusterDataIndex(trk.Data().Row(ic.RowIndex()), ic.HitIndex()) + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[t->Slice()][0];
          *c2 = trackCluster{id, (unsigned char)ic.RowIndex(), t->Slice(), t->Leg()};
        } else {
          const GPUTPCSliceOutCluster& c = t->OrigTrack()->OutTrackClusters()[i];
          unsigned int id = Param().rec.NonConsecutiveIDs ? ((unsigned int)((unsigned int*)&c - (unsigned int*)mkSlices[t->Slice()]->GetFirstTrack())) : c.GetId();
          *c2 = trackCluster{id, c.GetRow(), t->Slice(), t->Leg()};
        }
      }
      nHits += nTrackHits;
    }
    if (nHits < GPUCA_TRACKLET_SELECTOR_MIN_HITS(track.QPt())) {
      continue;
    }

    int ordered = leg == 0;
    if (ordered) {
      for (int i = 1; i < nHits; i++) {
        if (trackClusters[i].row > trackClusters[i - 1].row || trackClusters[i].id == trackClusters[i - 1].id) {
          ordered = 0;
          break;
        }
      }
    }
    int firstTrackIndex = 0;
    int lastTrackIndex = nParts - 1;
    if (ordered == 0) {
      int nTmpHits = 0;
      trackCluster trackClustersUnsorted[kMaxClusters];
      short clusterIndices[kMaxClusters];
      for (int i = 0; i < nHits; i++) {
        trackClustersUnsorted[i] = trackClusters[i];
        clusterIndices[i] = i;
      }

      if (leg > 0) {
        // Find QPt and DzDs for the segment closest to the vertex, if low/mid Pt
        float baseZT = 1e9;
        unsigned char baseLeg = 0;
        for (int i = 0; i < nParts; i++) {
          if (trackParts[i]->Leg() == 0 || trackParts[i]->Leg() == leg) {
            float zt;
            if (Param().par.earlyTpcTransform) {
              zt = CAMath::Min(CAMath::Abs(trackParts[i]->ClusterZT0()), CAMath::Abs(trackParts[i]->ClusterZTN()));
            } else {
              zt = -trackParts[i]->MinClusterZT(); // Negative time ~ smallest z, to behave the same way // TODO: Check all these min / max ZT
            }
            if (zt < baseZT) {
              baseZT = zt;
              baseLeg = trackParts[i]->Leg();
            }
          }
        }
        int iLongest = 1e9;
        int length = 0;
        for (int i = (baseLeg ? (nParts - 1) : 0); baseLeg ? (i >= 0) : (i < nParts); baseLeg ? i-- : i++) {
          if (trackParts[i]->Leg() != baseLeg) {
            break;
          }
          if (trackParts[i]->OrigTrack()->NHits() > length) {
            iLongest = i;
            length = trackParts[i]->OrigTrack()->NHits();
          }
        }
        bool outwards;
        if (Param().par.earlyTpcTransform) {
          outwards = (trackParts[iLongest]->ClusterZT0() > trackParts[iLongest]->ClusterZTN()) ^ trackParts[iLongest]->CSide();
        } else {
          outwards = trackParts[iLongest]->ClusterZT0() < trackParts[iLongest]->ClusterZTN();
        }
        GPUTPCGMMerger_CompareClusterIdsLooper::clcomparestruct clusterSort[kMaxClusters];
        for (int iPart = 0; iPart < nParts; iPart++) {
          const GPUTPCGMSliceTrack* t = trackParts[iPart];
          int nTrackHits = t->NClusters();
          for (int j = 0; j < nTrackHits; j++) {
            int i = nTmpHits + j;
            clusterSort[i].leg = t->Leg();
          }
          nTmpHits += nTrackHits;
        }

        GPUCommonAlgorithm::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIdsLooper(baseLeg, outwards, trackClusters, clusterSort));
      } else {
        GPUCommonAlgorithm::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIds(trackClusters));
      }
      nTmpHits = 0;
      firstTrackIndex = lastTrackIndex = -1;
      for (int i = 0; i < nParts; i++) {
        nTmpHits += trackParts[i]->NClusters();
        if (nTmpHits > clusterIndices[0] && firstTrackIndex == -1) {
          firstTrackIndex = i;
        }
        if (nTmpHits > clusterIndices[nHits - 1] && lastTrackIndex == -1) {
          lastTrackIndex = i;
        }
      }

      int nFilteredHits = 0;
      int indPrev = -1;
      for (int i = 0; i < nHits; i++) {
        int ind = clusterIndices[i];
        if (indPrev >= 0 && trackClustersUnsorted[ind].id == trackClustersUnsorted[indPrev].id) {
          continue;
        }
        indPrev = ind;
        trackClusters[nFilteredHits] = trackClustersUnsorted[ind];
        nFilteredHits++;
      }
      nHits = nFilteredHits;
    }

    int iOutTrackFirstCluster = CAMath::AtomicAdd(&mMemory->nOutputTrackClusters, (unsigned int)nHits);

    GPUTPCGMMergedTrackHit* cl = mClusters + iOutTrackFirstCluster;
    GPUTPCGMMergedTrackHitXYZ* clXYZ = mClustersXYZ + iOutTrackFirstCluster;

    for (int i = 0; i < nHits; i++) {
      unsigned char state;
      if (Param().rec.NonConsecutiveIDs) {
        const GPUTPCSliceOutCluster* c = (const GPUTPCSliceOutCluster*)((const int*)mkSlices[trackClusters[i].slice]->GetFirstTrack() + trackClusters[i].id);
        clXYZ[i].x = c->GetX();
        clXYZ[i].y = c->GetY();
        clXYZ[i].z = c->GetZ();
        clXYZ[i].amp = c->GetAmp();
        trackClusters[i].id = c->GetId();
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
        cl[i] XYZ.pad = c->mPad;
        cl[i] XYZ.time = c->mTime;
#endif
        state = c->GetFlags();
      } else if (Param().par.earlyTpcTransform) {
        const GPUTPCClusterData& c = GetConstantMem()->tpcTrackers[trackClusters[i].slice].ClusterData()[trackClusters[i].id - GetConstantMem()->tpcTrackers[trackClusters[i].slice].Data().ClusterIdOffset()];
        clXYZ[i].x = c.x;
        clXYZ[i].y = c.y;
        clXYZ[i].z = c.z;
        clXYZ[i].amp = c.amp;
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
        clXYZ[i].pad = c.mPad;
        clXYZ[i].time = c.mTime;
#endif
        state = c.flags;
      } else {
        const ClusterNative& c = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[trackClusters[i].id];
        state = c.getFlags();
      }
#ifdef GPUCA_ALIROOT_LIB
      cl[i].x = clXYZ[i].x;
      cl[i].y = clXYZ[i].y;
      cl[i].z = clXYZ[i].z;
      cl[i].amp = clXYZ[i].amp;
#endif
      cl[i].state = state & GPUTPCGMMergedTrackHit::clustererAndSharedFlags; // Only allow edge, deconvoluted, and shared flags
      cl[i].row = trackClusters[i].row;
      if (!Param().rec.NonConsecutiveIDs) // We already have global consecutive numbers from the slice tracker, and we need to keep them for late cluster attachment
      {
        cl[i].num = trackClusters[i].id;
      } else { // Produce consecutive numbers for shared cluster flagging
        cl[i].num = iOutTrackFirstCluster + i;
        mGlobalClusterIDs[cl[i].num] = trackClusters[i].id;
      }
      cl[i].slice = trackClusters[i].slice;
      cl[i].leg = trackClusters[i].leg;
    } // nHits

    int iOutputTrack = CAMath::AtomicAdd(&mMemory->nOutputTracks, 1u);

    GPUTPCGMMergedTrack& mergedTrack = mOutputTracks[iOutputTrack];

    mergedTrack.SetFlags(0);
    mergedTrack.SetOK(1);
    mergedTrack.SetLooper(leg > 0);
    mergedTrack.SetLegs(leg);
    mergedTrack.SetNClusters(nHits);
    mergedTrack.SetFirstClusterRef(iOutTrackFirstCluster);
    GPUTPCGMTrackParam& p1 = mergedTrack.Param();
    const GPUTPCGMSliceTrack& p2 = *trackParts[firstTrackIndex];
    mergedTrack.SetCSide(p2.CSide());

    GPUTPCGMBorderTrack b;
    const float toX = Param().par.earlyTpcTransform ? clXYZ[0].x : Param().tpcGeometry.Row2X(cl[0].row);
    if (p2.TransportToX(this, toX, Param().par.ConstBz, b, GPUCA_MAX_SIN_PHI, false)) {
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
    p1.TZOffset() = p2.TZOffset();
    p1.DzDs() = p2.DzDs();
    p1.QPt() = p2.QPt();
    mergedTrack.SetAlpha(p2.Alpha());
    const double kCLight = 0.000299792458;
    if (CAMath::Abs(Param().polynomialField.GetNominalBz()) < (0.01 * kCLight)) {
      p1.QPt() = 0.01f * Param().rec.bz0Pt;
    }

    // if (nParts > 1) printf("Merged %d: QPt %f %d parts %d hits\n", mMemory->nOutputTracks, p1.QPt(), nParts, nHits);

    /*if (GPUQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(mMemory->nOutputTracks))
    {
      mergedTrack.SetOK(0);
      mergedTrack.SetNClusters(0);
    }
    if (mergedTrack.NClusters() && mergedTrack.OK()) */
    {
      bool CEside;
      if (Param().par.earlyTpcTransform) {
        CEside = (mergedTrack.CSide() != 0) ^ (clXYZ[0].z > clXYZ[nHits - 1].z);
      } else {
        auto& cls = mConstantMem->ioPtrs.clustersNative->clustersLinear;
        CEside = cls[cl[0].num].getTime() < cls[cl[nHits - 1].num].getTime();
      }
      MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], &clXYZ[CEside ? (nHits - 1) : 0], iOutputTrack);
    }
  } // itr
}

GPUd() void GPUTPCGMMerger::SortTracksPrepare(int nBlocks, int nThreads, int iBlock, int iThread)
{
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nThreads * nBlocks) {
    mTrackOrderProcess[i] = i;
  }
}

GPUd() void GPUTPCGMMerger::PrepareClustersForFit0(int nBlocks, int nThreads, int iBlock, int iThread)
{
  unsigned int* trackSort = (unsigned int*)mTmpMem;
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nBlocks * nThreads) {
    trackSort[i] = i;
  }
}

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GPUCA_GPUCODE_GENRTC) // Specialize GPUTPCGMMergerSortTracks and GPUTPCGMMergerSortTracksQPt
struct GPUTPCGMMergerSortTracks_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracks_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int aa, const int bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    if (a.Legs() != b.Legs()) {
      return a.Legs() > b.Legs();
    }
    return a.NClusters() > b.NClusters();
  }
};

template <>
void GPUCA_KRNL_BACKEND_CLASS::runKernelBackendInternal<GPUTPCGMMergerSortTracks, 0>(krnlSetup& _xyz)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<unsigned int> trackSort((unsigned int*)mProcessorsShadow->tpcMerger.TrackOrderProcess());
  ThrustVolatileAsyncAllocator alloc(this);
  thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(mInternals->Streams[_xyz.x.stream]), trackSort, trackSort + processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracks_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
}

struct GPUTPCGMMergerSortTracksQPt_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracksQPt_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int aa, const int bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    return (CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt()));
  }
};

template <>
void GPUCA_KRNL_BACKEND_CLASS::runKernelBackendInternal<GPUTPCGMMergerSortTracksQPt, 0>(krnlSetup& _xyz)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<unsigned int> trackSort((unsigned int*)mProcessorsShadow->tpcMerger.TmpMem());
  ThrustVolatileAsyncAllocator alloc(this);
  thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(mInternals->Streams[_xyz.x.stream]), trackSort, trackSort + processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracksQPt_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
}
#endif // __CUDACC__ || __HIPCC__ - Specialize GPUTPCGMMergerSortTracks and GPUTPCGMMergerSortTracksQPt

GPUd() void GPUTPCGMMerger::SortTracks(int nBlocks, int nThreads, int iBlock, int iThread)
{
  if (iThread || iBlock) {
    return;
  }
  // Have to duplicate sort comparison: Thrust cannot use the Lambda but OpenCL cannot use the object
  auto comp = [cmp = mOutputTracks](const int aa, const int bb) {
    const GPUTPCGMMergedTrack& GPUrestrict() a = cmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = cmp[bb];
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    if (a.Legs() != b.Legs()) {
      return a.Legs() > b.Legs();
    }
    return a.NClusters() > b.NClusters();
  };

  GPUCommonAlgorithm::sortDeviceDynamic(mTrackOrderProcess, mTrackOrderProcess + mMemory->nOutputTracks, comp);
}

GPUd() void GPUTPCGMMerger::SortTracksQPt(int nBlocks, int nThreads, int iBlock, int iThread)
{
  if (iThread || iBlock) {
    return;
  }
  unsigned int* trackSort = (unsigned int*)mTmpMem;
  // Have to duplicate sort comparison: Thrust cannot use the Lambda but OpenCL cannot use the object
  auto comp = [cmp = mOutputTracks](const int aa, const int bb) {
    const GPUTPCGMMergedTrack& GPUrestrict() a = cmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = cmp[bb];
    return (CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt()));
  };

  GPUCommonAlgorithm::sortDeviceDynamic(trackSort, trackSort + mMemory->nOutputTracks, comp);
}

GPUd() void GPUTPCGMMerger::PrepareClustersForFit1(int nBlocks, int nThreads, int iBlock, int iThread)
{
  unsigned int* trackSort = (unsigned int*)mTmpMem;
  GPUAtomic(unsigned int)* sharedCount = (GPUAtomic(unsigned int)*)(trackSort + CAMath::nextMultipleOf<4>(mMemory->nOutputTracks));
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nBlocks * nThreads) {
    mTrackOrderAttach[trackSort[i]] = i;
    const GPUTPCGMMergedTrack& trk = mOutputTracks[i];
    if (trk.OK()) {
      for (unsigned int j = 0; j < trk.NClusters(); j++) {
        mClusterAttachment[mClusters[trk.FirstClusterRef() + j].num] = attachAttached | attachGood;
        CAMath::AtomicAdd(&sharedCount[mClusters[trk.FirstClusterRef() + j].num], 1u);
      }
    }
  }
}

GPUd() void GPUTPCGMMerger::PrepareClustersForFit2(int nBlocks, int nThreads, int iBlock, int iThread)
{
  unsigned int* sharedCount = (unsigned int*)mTmpMem + CAMath::nextMultipleOf<4>(mMemory->nOutputTracks);
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTrackClusters; i += nBlocks * nThreads) {
    if (sharedCount[mClusters[i].num] > 1) {
      mClusters[i].state |= GPUTPCGMMergedTrackHit::flagShared;
    }
  }
  if (mClusterStateExt) {
    for (unsigned int i = iBlock * nThreads + iThread; i < mNMaxClusters; i += nBlocks * nThreads) {
      unsigned char state = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[i].getFlags();
      if (sharedCount[i] > 1) {
        state |= GPUTPCGMMergedTrackHit::flagShared;
      }
      mClusterStateExt[i] = state;
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize0(int nBlocks, int nThreads, int iBlock, int iThread)
{
  if (Param().rec.NonConsecutiveIDs) {
    for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTrackClusters; i += nThreads * nBlocks) {
      mClusters[i].num = mGlobalClusterIDs[i];
    }
  } else {
    int* trkOrderReverse = (int*)mTmpMem;
    for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nThreads * nBlocks) {
      trkOrderReverse[mTrackOrderAttach[i]] = i;
    }
    for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTrackClusters; i += nThreads * nBlocks) {
      mClusterAttachment[mClusters[i].num] = 0; // Reset adjacent attachment for attached clusters, set correctly below
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize1(int nBlocks, int nThreads, int iBlock, int iThread)
{
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nThreads * nBlocks) {
    const GPUTPCGMMergedTrack& trk = mOutputTracks[i];
    if (!trk.OK() || trk.NClusters() == 0) {
      continue;
    }
    char goodLeg = mClusters[trk.FirstClusterRef() + trk.NClusters() - 1].leg;
    for (unsigned int j = 0; j < trk.NClusters(); j++) {
      int id = mClusters[trk.FirstClusterRef() + j].num;
      unsigned int weight = mTrackOrderAttach[i] | attachAttached;
      unsigned char clusterState = mClusters[trk.FirstClusterRef() + j].state;
      if (!(clusterState & GPUTPCGMMergedTrackHit::flagReject)) {
        weight |= attachGood;
      } else if (clusterState & GPUTPCGMMergedTrackHit::flagNotFit) {
        weight |= attachHighIncl;
      }
      if (mClusters[trk.FirstClusterRef() + j].leg == goodLeg) {
        weight |= attachGoodLeg;
      }
      CAMath::AtomicMax(&mClusterAttachment[id], weight);
    }
  }
}

GPUd() void GPUTPCGMMerger::Finalize2(int nBlocks, int nThreads, int iBlock, int iThread)
{
  int* trkOrderReverse = (int*)mTmpMem;
  for (unsigned int i = iBlock * nThreads + iThread; i < mNMaxClusters; i += nThreads * nBlocks) {
    if (mClusterAttachment[i] != 0) {
      mClusterAttachment[i] = (mClusterAttachment[i] & attachFlagMask) | trkOrderReverse[mClusterAttachment[i] & attachTrackMask];
    }
  }
}

struct MergeLooperParam {
  float absz;
  float tgl;
  float qpt;
  float x;
  float y;
  unsigned int id;
};

GPUd() void GPUTPCGMMerger::MergeLoopers(int nBlocks, int nThreads, int iBlock, int iThread)
{
  if (iThread || iBlock) {
    return;
  }
#ifndef GPUCA_GPUCODE
  std::vector<MergeLooperParam> params;
  const float lowPtThresh = Param().rec.tpcRejectQPt * 1.1f; // Might need to merge tracks above the threshold with parts below the threshold
  for (unsigned int i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[i];
    const auto& p = trk.GetParam();
    const float qptabs = CAMath::Abs(p.GetQPt());
    if (trk.NClusters() && qptabs > 5.f && qptabs <= lowPtThresh) {
      const int slice = mClusters[trk.FirstClusterRef() + trk.NClusters() - 1].slice;
      const float z = p.GetZ() + (Param().par.earlyTpcTransform ? p.GetTZOffset() : GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(slice, p.GetTZOffset(), Param().par.continuousMaxTimeBin));
      float sinA, cosA;
      CAMath::SinCos(trk.GetAlpha(), sinA, cosA);
      float gx = cosA * p.GetX() - sinA * p.GetY();
      float gy = cosA * p.GetY() + sinA * p.GetX();
      float bz = Param().polynomialField.GetFieldBz(gx, gy, p.GetZ());
      const float r1 = p.GetQPt() * bz;
      const float r = CAMath::Abs(r1) > 0.0001 ? (1.f / r1) : 10000;
      const float mx = p.GetX() + r * p.GetSinPhi();
      const float my = p.GetY() - r * CAMath::Sqrt(1 - p.GetSinPhi() * p.GetSinPhi());
      const float gmx = cosA * mx - sinA * my;
      const float gmy = cosA * my + sinA * mx;
      params.emplace_back(MergeLooperParam{CAMath::Abs(z), CAMath::Abs(p.GetDzDs()), p.GetDzDs() > 0 ? p.GetQPt() : -p.GetQPt(), gmx, gmy, i});

      /*printf("Track %d Sanity qpt %f snp %f bz %f\n", (int)params.size(), p.GetQPt(), p.GetSinPhi(), bz);
      for (unsigned int k = 0;k < trk.NClusters();k++) {
        float xx, yy, zz;
        if (Param().par.earlyTpcTransform) {
          const float zOffset = (mClusters[trk.FirstClusterRef() + k].slice < 18) == (mClusters[trk.FirstClusterRef() + 0].slice < 18) ? p.GetTZOffset() : -p.GetTZOffset();
          xx = mClustersXYZ[trk.FirstClusterRef() + k].x;
          yy = mClustersXYZ[trk.FirstClusterRef() + k].y;
          zz = mClustersXYZ[trk.FirstClusterRef() + k].z - zOffset;
        } else {
          const ClusterNative& GPUrestrict() cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[mClusters[trk.FirstClusterRef() + k].num];
          GetConstantMem()->calibObjects.fastTransform->Transform(mClusters[trk.FirstClusterRef() + k].slice, mClusters[trk.FirstClusterRef() + k].row, cl.getPad(), cl.getTime(), xx, yy, zz, p.GetTZOffset());
        }
        float sa2, ca2;
        CAMath::SinCos(Param().Alpha(mClusters[trk.FirstClusterRef() + k].slice), sa2, ca2);
        float cx = ca2 * xx - sa2 * yy;
        float cy = ca2 * yy + sa2 * xx;
        float dist = sqrtf((cx - gmx) * (cx - gmx) + (cy - gmy) * (cy - gmy));
        printf("Hit %3d/%3d slice %d xy %f %f R %f\n", k, trk.NClusters(), (int)mClusters[trk.FirstClusterRef() + k].slice, cx, cy, dist);
      }*/
    }
  }
  std::sort(params.begin(), params.end(), [](const MergeLooperParam& a, const MergeLooperParam& b) { return a.absz < b.absz; });
#if GPUCA_MERGE_LOOPER_MC
  std::vector<long int> paramLabels(params.size());
  for (unsigned int i = 0; i < params.size(); i++) {
    paramLabels[i] = GetTrackLabel(mOutputTracks[params[i].id]);
  }
  std::vector<bool> dropped(params.size());
  std::vector<bool> droppedMC(params.size());
  std::vector<int> histMatch(101);
  std::vector<int> histFail(101);
#endif

  for (unsigned int i = 0; i < params.size(); i++) {
    for (unsigned int j = i + 1; j < params.size(); j++) {
      if (params[j].absz > params[i].absz + 100.f) {
        break;
      }
      float dqpt = CAMath::Min(CAMath::Abs(params[i].tgl), CAMath::Abs(params[j].tgl)) < 0.05f ? (CAMath::Abs(params[i].qpt) - CAMath::Abs(params[i].qpt)) : (params[i].qpt - params[j].qpt);
      float d = CAMath::Sum2((params[i].x - params[j].x) * (1.f / 5.f), (params[i].y - params[j].y) * (1.f / 5.f), (params[i].tgl - params[j].tgl) * (1.f / 0.15f), dqpt / CAMath::Min(params[i].qpt, params[j].qpt) * (1.f / 0.15f));
      //bool EQ = CAMath::Abs(params[i].x - params[j].x) < 10.f && CAMath::Abs(params[i].y - params[j].y) < 10.f && CAMath::Abs(params[i].tgl - params[j].tgl) < 0.15f && CAMath::Abs((params[i].qpt - params[j].qpt) / CAMath::Min(params[i].qpt, params[j].qpt)) < 0.15f;
      bool EQ = d < 1.5f;
#if GPUCA_MERGE_LOOPER_MC
      const long int label1 = paramLabels[i];
      const long int label2 = paramLabels[j];
      bool labelEQ = label1 != -1 && label1 == label2;
      if (EQ || labelEQ) {
        printf("Matching track %d/%d %u-%u (%ld/%ld): dist %f side %d %d, tgl %f %f, qpt %f %f, x %f %f, y %f %f\n", (int)EQ, (int)labelEQ, i, j, label1, label2, d, (int)mOutputTracks[params[i].id].CSide(), (int)mOutputTracks[params[j].id].CSide(), params[i].tgl, params[j].tgl, params[i].qpt, params[j].qpt, params[i].x, params[j].x, params[i].y, params[j].y);
      }
      if (EQ) {
        dropped[j] = true;
      }
      if (labelEQ) {
        droppedMC[j] = true;
        histMatch[CAMath::Min<int>(100, d * 10.f)]++;
      }
      if (d < 10.f && !labelEQ) {
        histFail[CAMath::Min<int>(100, d * 10.f)]++;
      }
#endif
      if (EQ) {
        mOutputTracks[params[j].id].SetMergedLooper(true);
        if (CAMath::Abs(params[j].qpt) >= Param().rec.tpcRejectQPt) {
          mOutputTracks[params[i].id].SetMergedLooper(true);
        }
      }
    }
  }
#if GPUCA_MERGE_LOOPER_MC
  int total = 0, totalmc = 0, good = 0, missed = 0, fake = 0;
  for (unsigned int i = 0; i < params.size(); i++) {
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
  for (unsigned int i = 0; i < histMatch.size(); i++) {
    printf("%8.3f: %3d\n", i / 10.f + 0.05f, histMatch[i]);
  }
  printf("Fake histo\n");
  for (unsigned int i = 0; i < histFail.size(); i++) {
    printf("%8.3f: %3d\n", i / 10.f + 0.05f, histFail[i]);
  }
#endif
#endif
}
