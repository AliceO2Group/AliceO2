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

#ifndef __OPENCL__
#include <cstdio>
#include <cstring>
#include <cmath>
#endif

#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUReconstruction.h"
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

#ifdef GPUCA_CADEBUG_ENABLED
#include "AliHLTTPCClusterMCData.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace GPUTPCGMMergerTypes;

static constexpr int kMaxParts = 400;
static constexpr int kMaxClusters = GPUCA_MERGER_MAX_TRACK_CLUSTERS;

//#define OFFLINE_FITTER

#if !defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPUCODE)
#undef OFFLINE_FITTER
#endif

#ifndef GPUCA_GPUCODE

#include "GPUChainTracking.h"
#include "GPUQA.h"

GPUTPCGMMerger::GPUTPCGMMerger()
  : mTrackLinks(nullptr), mNMaxSliceTracks(0), mNMaxTracks(0), mNMaxSingleSliceTracks(0), mNMaxOutputTrackClusters(0), mNMaxClusters(0), mMemoryResMemory(-1), mNClusters(0), mOutputTracks(nullptr), mSliceTrackInfos(nullptr), mSliceTrackInfoIndex(nullptr), mClusters(nullptr), mGlobalClusterIDs(nullptr), mClusterAttachment(nullptr), mTrackOrderAttach(nullptr), mTrackOrderProcess(nullptr), mTmpMem(nullptr), mTmpCounter(nullptr), mBorderMemory(nullptr), mBorderRangeMemory(nullptr), mMemory(nullptr), mRetryRefitIds(nullptr), mLoopData(nullptr)
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
#if defined(GPUCA_MERGER_BY_MC_LABEL) || defined(GPUCA_CADEBUG_ENABLED)
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
    tr->SetPrevSegmentNeighbour(1000000000);
    while (true) {
      int iTrk = tr - mSliceTrackInfos;
      if (trkUsed[iTrk]) {
        GPUError("FAILURE: double use");
      }
      trkUsed[iTrk] = true;

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
  }
  for (int i = 0; i < SliceTrackInfoLocalTotal(); i++) {
    if (trkUsed[i] == false) {
      GPUError("FAILURE: trk missed");
    }
  }
}

int GPUTPCGMMerger::GetTrackLabel(const GPUTPCGMBorderTrack& trk)
{
  GPUTPCGMSliceTrack* track = &mSliceTrackInfos[trk.TrackID()];
  int nClusters = track->OrigTrack()->NHits();
  std::vector<int> labels;
  for (int i = 0; i < nClusters; i++) {
    int id;
    if (Param().rec.mergerReadFromTrackerDirectly) {
      const GPUTPCTracker& tracker = GetConstantMem()->tpcTrackers[track->Slice()];
      const GPUTPCHitId& ic = tracker.TrackHits()[track->OrigTrack()->FirstHitID() + i];
      id = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex()) + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[track->Slice()][0];
    } else {
      id = track->OrigTrack()->OutTrackClusters()[i].GetId();
    }
    for (int j = 0; j < 3; j++) {
      int label = GetConstantMem()->ioPtrs.mcLabelsTPC[id].fClusterID[j].fMCID;
      if (label >= 0) {
        labels.push_back(label);
      }
    }
  }
  if (labels.size() == 0) {
    return (-1);
  }
  labels.push_back(-1);
  std::sort(labels.begin(), labels.end());
  int bestLabel = -1, bestLabelCount = 0;
  int curLabel = labels[0], curCount = 1;
  for (unsigned int i = 1; i < labels.size(); i++) {
    if (labels[i] == curLabel) {
      curCount++;
    } else {
      if (curCount > bestLabelCount) {
        bestLabelCount = curCount;
        bestLabel = curLabel;
      }
      curLabel = labels[i];
      curCount = 1;
    }
  }
  return bestLabel;
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
  size_t tmpSize = CAMath::Max(CAMath::Max<unsigned int>(mNMaxSingleSliceTracks, 1) * NSLICES * sizeof(int), CAMath::nextMultipleOf<4>(mNMaxTracks) * sizeof(int) + mNMaxClusters * sizeof(unsigned int));
  computePointerWithAlignment(mem, mTmpMem, (tmpSize + sizeof(*mTmpMem) - 1) / sizeof(*mTmpMem));
  computePointerWithAlignment(mem, mTmpCounter, 2 * NSLICES);

  int nTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    const int n = mRec->GetParam().rec.mergerReadFromTrackerDirectly ? *mRec->GetConstantMem().tpcTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
    mBorder[iSlice] = mBorderMemory + 2 * nTracks;
    mBorder[NSLICES + iSlice] = mBorderMemory + 2 * nTracks + n;
    mBorderRange[iSlice] = mBorderRangeMemory + 2 * nTracks;
    nTracks += n;
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory);
  return mem;
}

void* GPUTPCGMMerger::SetPointersRefitScratch(void* mem)
{
  computePointerWithAlignment(mem, mRetryRefitIds, mNMaxTracks);
  computePointerWithAlignment(mem, mLoopData, mNMaxTracks);
  if (mRec->GetDeviceProcessingSettings().fullMergerOnGPU) {
    mem = SetPointersRefitScratch2(mem);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersRefitScratch2(void* mem)
{
  computePointerWithAlignment(mem, mTrackOrderAttach, mNMaxTracks);
  if (mRec->GetDeviceProcessingSettings().mergerSortTracks) {
    computePointerWithAlignment(mem, mTrackOrderProcess, mNMaxTracks);
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mOutputTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mClusters, mNMaxOutputTrackClusters);
  computePointerWithAlignment(mem, mClusterAttachment, mNMaxClusters);
  if (!mRec->GetDeviceProcessingSettings().fullMergerOnGPU) {
    mem = SetPointersRefitScratch2(mem);
  }

  return mem;
}

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersMerger, mRec->GetDeviceProcessingSettings().fullMergerOnGPU ? GPUMemoryResource::MEMORY_SCRATCH : (GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_HOST), "TPCMerger");
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersRefitScratch, GPUMemoryResource::MEMORY_SCRATCH, "TPCMergerRefitScratch");
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersOutput, mRec->GetDeviceProcessingSettings().fullMergerOnGPU ? GPUMemoryResource::MEMORY_OUTPUT : GPUMemoryResource::MEMORY_INOUT, "TPCMergerOutput");
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
  mNMaxOutputTrackClusters = mNClusters * 1.1f + 1000;
  mNMaxTracks = mNMaxSliceTracks;
  mNMaxClusters = 0;
  if (io.clustersNative) {
    mNMaxClusters = io.clustersNative->nClustersTotal;
  } else if (mRec->GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) {
    for (int i = 0; i < NSLICES; i++) {
      mNMaxClusters += mRec->GetConstantMem().tpcTrackers[i].NHitsTotal();
    }
  } else {
    mNMaxClusters = mNClusters;
  }
}

void GPUTPCGMMerger::SetSliceData(int index, const GPUTPCSliceOutput* sliceData) { mkSlices[index] = sliceData; }

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
  static constexpr float kRho = 1.025e-3f;  // 0.9e-3;
  static constexpr float kRadLen = 29.532f; // 28.94;
  GPUTPCGMPropagator prop;
  prop.SetMaterial(kRadLen, kRho);
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
  trk.TZOffset() = Param().earlyTpcTransform ? inTrack->Param().GetZOffset() : GetConstantMem()->calibObjects.fastTransform->convZOffsetToVertexTime(slice, inTrack->Param().GetZOffset(), Param().continuousMaxTimeBin);
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
        if (Param().earlyTpcTransform) {
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
        if (Param().earlyTpcTransform) {
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
    if (Param().earlyTpcTransform) {
      track.SetClusterZT(trk.Data().ClusterData()[clusterIndex1].z, trk.Data().ClusterData()[clusterIndex2].z);
    } else {
      const ClusterNative* cl = GetConstantMem()->ioPtrs.clustersNative->clustersLinear + GetConstantMem()->ioPtrs.clustersNative->clusterOffset[iSlice][0];
      track.SetClusterZT(cl[clusterIndex1].getTime(), cl[clusterIndex2].getTime());
    }
  } else {
    if (Param().earlyTpcTransform) {
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

GPUd() void GPUTPCGMMerger::MakeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(unsigned int) * nB, bool useOrigTrackParam)
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = Param().ConstBz;

  float dAlpha = Param().DAlpha / 2;
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
  GPUTPCGMBorderTrack::Range* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderTrack::Range* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());
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
    range1[itr].fId = itr;
    range1[itr].fMin = b.Par()[1] + b.ZOffsetLinear() - d;
    range1[itr].fMax = b.Par()[1] + b.ZOffsetLinear() + d;
    if (sameSlice) {
      for (int i = 0; i < N1; i++) {
        range2[i] = range1[i];
      }
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
      range2[itr].fId = itr;
      range2[itr].fMin = b.Par()[1] + b.ZOffsetLinear() - d;
      range2[itr].fMax = b.Par()[1] + b.ZOffsetLinear() + d;
    }
  }
}

template <>
GPUd() void GPUTPCGMMerger::MergeBorderTracks<1>(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice1, GPUTPCGMBorderTrack* B1, int N1, int iSlice2, GPUTPCGMBorderTrack* B2, int N2, int mergeMode)
{
  GPUTPCGMBorderTrack::Range* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderTrack::Range* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());

  if (iThread == 0) {
    if (iBlock == 1) {
      GPUCommonAlgorithm::sortDeviceDynamic(range1, range1 + N1, [](const GPUTPCGMBorderTrack::Range& a, const GPUTPCGMBorderTrack::Range& b) { return a.fMin < b.fMin; });
    } else if (iBlock == 0) {
      GPUCommonAlgorithm::sortDeviceDynamic(range2, range2 + N2, [](const GPUTPCGMBorderTrack::Range& a, const GPUTPCGMBorderTrack::Range& b) { return a.fMax < b.fMax; });
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

  GPUTPCGMBorderTrack::Range* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderTrack::Range* range2 = mBorderRange[iSlice2] + (Param().rec.mergerReadFromTrackerDirectly ? *GetConstantMem()->tpcTrackers[iSlice2].NTracks() : mkSlices[iSlice2]->NTracks());

  int i2 = 0;
  for (int i1 = iBlock * nThreads + iThread; i1 < N1; i1 += nThreads * nBlocks) {
    GPUTPCGMBorderTrack::Range r1 = range1[i1];
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
      GPUTPCGMBorderTrack::Range r2 = range2[k2];
      if (r2.fMin > r1.fMax) {
        break;
      }
      if (sameSlice && (r1.fId >= r2.fId)) {
        continue;
      }
      // do check

      GPUTPCGMBorderTrack& b2 = B2[r2.fId];
#ifdef GPUCA_MERGER_BY_MC_LABEL
      if (GetTrackLabel(b1) != GetTrackLabel(b2)) // DEBUG CODE, match by MC label
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

template <int I>
GPUd() void GPUTPCGMMerger::MergeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode)
{
  int n1, n2;
  GPUTPCGMBorderTrack *b1, *b2;
  int jSlice;
  if (withinSlice == 1) {
    jSlice = iSlice;
    n1 = n2 = mTmpCounter[iSlice];
    b1 = b2 = mBorder[iSlice];
  } else if (withinSlice == -1) {
    jSlice = (iSlice + NSLICES / 2);
    const int offset = mergeMode == 2 ? NSLICES : 0;
    n1 = mTmpCounter[iSlice + offset];
    n2 = mTmpCounter[jSlice + offset];
    b1 = mBorder[iSlice + offset];
    b2 = mBorder[jSlice + offset];
  } else {
    jSlice = mNextSliceInd[iSlice];
    n1 = mTmpCounter[iSlice];
    n2 = mTmpCounter[NSLICES + jSlice];
    b1 = mBorder[iSlice];
    b2 = mBorder[NSLICES + jSlice];
  }
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
    if (track.TransportToX(this, x0, Param().ConstBz, b, maxSin)) {
      b.SetTrackID(itr);
      CADEBUG(
        printf("WITHIN SLICE %d Track %d - ", iSlice, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf("\n"));
      b.SetNClusters(track.NClusters());
      unsigned int myTrack = CAMath::AtomicAdd(&mTmpCounter[iSlice], 1u);
      mBorder[iSlice][myTrack] = b;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeSlicesPrepare(int nBlocks, int nThreads, int iBlock, int iThread, int border0, int border1, char useOrigTrackParam)
{
  bool part2 = iBlock & 1;
  int border = part2 ? border1 : border0;
  GPUAtomic(unsigned int)* n = mTmpCounter;
  GPUTPCGMBorderTrack** b = mBorder;
  if (part2) {
    n += NSLICES;
    b += NSLICES;
  }
  MakeBorderTracks((nBlocks + 1) >> 1, nThreads, iBlock >> 1, iThread, border, b, n, useOrigTrackParam);
}

GPUd() void GPUTPCGMMerger::ResolveMergeSlices(int nBlocks, int nThreads, int iBlock, int iThread, char useOrigTrackParam, char mergeAll)
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

  for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++) {
    int itr2 = mTrackLinks[itr];
    if (itr2 < 0) {
      continue;
    }
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

GPUd() void GPUTPCGMMerger::MergeCEFill(const GPUTPCGMSliceTrack* track, const GPUTPCGMMergedTrackHit& cls, int itr)
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
  if (Param().earlyTpcTransform) {
    z = cls.z;
  } else {
    float x, y;
    auto& cln = mConstantMem->ioPtrs.clustersNative->clustersLinear[cls.num];
    GPUTPCConvertImpl::convert(*mConstantMem, cls.slice, cls.row, cln.getPad(), cln.getTime(), x, y, z);
  }

  if (!Param().ContinuousTracking && CAMath::Abs(z) > 10) {
    return;
  }
  int slice = track->Slice();
  for (int attempt = 0; attempt < 2; attempt++) {
    GPUTPCGMBorderTrack b;
    const float x0 = Param().tpcGeometry.Row2X(attempt == 0 ? 63 : cls.row);
    if (track->TransportToX(this, x0, Param().ConstBz, b, GPUCA_MAX_SIN_PHI_LOW)) {
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
      unsigned int myTrack = CAMath::AtomicAdd(&mTmpCounter[id], 1u);
      mBorder[id][myTrack] = b;
      break;
    }
  }
}

GPUd() void GPUTPCGMMerger::MergeCE(int nBlocks, int nThreads, int iBlock, int iThread)
{
  const ClusterNative* cls = Param().earlyTpcTransform ? nullptr : mConstantMem->ioPtrs.clustersNative->clustersLinear;
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
#ifndef GPUCA_GPUCODE
        printf("Insufficient cluster memory for merging CE tracks (OutputClusters %d, max clusters %u)\n", mMemory->nOutputTrackClusters, mNMaxOutputTrackClusters);
#else
        // TODO: proper overflow handling
#endif
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
        if (Param().earlyTpcTransform) {
          z0max = CAMath::Max(CAMath::Abs(mClusters[trk[0]->FirstClusterRef()].z), CAMath::Abs(mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z));
          z1max = CAMath::Max(CAMath::Abs(mClusters[trk[1]->FirstClusterRef()].z), CAMath::Abs(mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z));
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
        if (Param().earlyTpcTransform) {
          reverse[0] = (mClusters[trk[0]->FirstClusterRef()].z > mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) ^ (trk[0]->CSide() > 0);
          reverse[1] = (mClusters[trk[1]->FirstClusterRef()].z < mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) ^ (trk[1]->CSide() > 0);
        } else {
          reverse[0] = cls[mClusters[trk[0]->FirstClusterRef()].num].getTime() < cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime();
          reverse[1] = cls[mClusters[trk[1]->FirstClusterRef()].num].getTime() > cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime();
        }
      }

      if (Param().ContinuousTracking) {
        if (Param().earlyTpcTransform) {
          const float z0 = trk[0]->CSide() ? CAMath::Max(mClusters[trk[0]->FirstClusterRef()].z, mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) : CAMath::Min(mClusters[trk[0]->FirstClusterRef()].z, mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z);
          const float z1 = trk[1]->CSide() ? CAMath::Max(mClusters[trk[1]->FirstClusterRef()].z, mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) : CAMath::Min(mClusters[trk[1]->FirstClusterRef()].z, mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z);
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
            mClusters[pos++] = mClusters[trk[k]->FirstClusterRef() + j];
          }
        } else {
          for (unsigned int j = 0; j < trk[k]->NClusters(); j++) {
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

GPUd() void GPUTPCGMMerger::LinkGlobalTracks(int nBlocks, int nThreads, int iBlock, int iThread)
{
  for (int itr = SliceTrackInfoGlobalFirst(0) + iBlock * nThreads + iThread; itr < SliceTrackInfoGlobalLast(NSLICES - 1); itr += nThreads * nBlocks) {
    GPUTPCGMSliceTrack& globalTrack = mSliceTrackInfos[itr];
    GPUTPCGMSliceTrack& localTrack = mSliceTrackInfos[globalTrack.LocalTrackId()];
    localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, itr);
  }
}

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
            if (Param().earlyTpcTransform) {
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
        if (Param().earlyTpcTransform) {
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

    for (int i = 0; i < nHits; i++) {
      unsigned char state;
      if (Param().rec.NonConsecutiveIDs) {
        const GPUTPCSliceOutCluster* c = (const GPUTPCSliceOutCluster*)((const int*)mkSlices[trackClusters[i].slice]->GetFirstTrack() + trackClusters[i].id);
        cl[i].x = c->GetX();
        cl[i].y = c->GetY();
        cl[i].z = c->GetZ();
        cl[i].amp = c->GetAmp();
        trackClusters[i].id = c->GetId();
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
        cl[i].pad = c->mPad;
        cl[i].time = c->mTime;
#endif
        state = c->GetFlags();
      } else if (Param().earlyTpcTransform) {
        const GPUTPCClusterData& c = GetConstantMem()->tpcTrackers[trackClusters[i].slice].ClusterData()[trackClusters[i].id - GetConstantMem()->tpcTrackers[trackClusters[i].slice].Data().ClusterIdOffset()];
        cl[i].x = c.x;
        cl[i].y = c.y;
        cl[i].z = c.z;
        cl[i].amp = c.amp;
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
        cl[i].pad = c.mPad;
        cl[i].time = c.mTime;
#endif
        state = c.flags;
      } else {
        const ClusterNative& c = GetConstantMem()->ioPtrs.clustersNative->clustersLinear[trackClusters[i].id];
        state = c.getFlags();
      }
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
    const float toX = Param().earlyTpcTransform ? cl[0].x : Param().tpcGeometry.Row2X(cl[0].row);
    if (p2.TransportToX(this, toX, Param().ConstBz, b, GPUCA_MAX_SIN_PHI, false)) {
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
      if (Param().earlyTpcTransform) {
        CEside = (mergedTrack.CSide() != 0) ^ (cl[0].z > cl[nHits - 1].z);
      } else {
        auto& cls = mConstantMem->ioPtrs.clustersNative->clustersLinear;
        CEside = cls[cl[0].num].getTime() < cls[cl[nHits - 1].num].getTime();
      }
      MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], iOutputTrack);
    }
  } // itr
}

GPUd() void GPUTPCGMMerger::SortTracksPrepare(int nBlocks, int nThreads, int iBlock, int iThread)
{
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTracks; i += nThreads * nBlocks) {
    const GPUTPCGMMergedTrack& trk = mOutputTracks[i];
    if (trk.CCE() || trk.Legs()) {
      CAMath::AtomicAdd(&mMemory->nSlowTracks, 1u);
    }
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

#ifdef __CUDACC__
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
  GPUDebugTiming timer(mDeviceProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<unsigned int> trackSort((unsigned int*)mProcessorsShadow->tpcMerger.TrackOrderProcess());
  thrust::sort(thrust::cuda::par.on(mInternals->Streams[_xyz.x.stream]), trackSort, trackSort + processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracks_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
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
  GPUDebugTiming timer(mDeviceProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<unsigned int> trackSort((unsigned int*)mProcessorsShadow->tpcMerger.TmpMem());
  thrust::sort(thrust::cuda::par.on(mInternals->Streams[_xyz.x.stream]), trackSort, trackSort + processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracksQPt_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
}
#endif

GPUd() void GPUTPCGMMerger::SortTracks(int nBlocks, int nThreads, int iBlock, int iThread)
{
  auto comp = [cmp = mOutputTracks](const int aa, const int bb) { // Have to duplicate sort comparison: Thrust cannot use the Lambda but OpenCL cannot use the object
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
  unsigned int* trackSort = (unsigned int*)mTmpMem;
  auto comp = [cmp = mOutputTracks](const int aa, const int bb) { // Have to duplicate sort comparison: Thrust cannot use the Lambda but OpenCL cannot use the object
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
  }
  for (unsigned int i = iBlock * nThreads + iThread; i < mMemory->nOutputTrackClusters; i += nBlocks * nThreads) {
    mClusterAttachment[mClusters[i].num] = attachAttached | attachGood;
    CAMath::AtomicAdd(&sharedCount[mClusters[i].num], 1u);
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
