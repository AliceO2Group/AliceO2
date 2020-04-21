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

#include <cstdio>
#include <cstring>
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUO2DataTypes.h"
#include "TPCFastTransform.h"
#include "GPUTPCConvertImpl.h"
#include "GPUQA.h"

#include "GPUCommonMath.h"

#include "GPUTPCTrackParam.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUTPCTrackLinearisation.h"

#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

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

GPUTPCGMMerger::GPUTPCGMMerger()
  : mTrackLinks(nullptr), mNMaxSliceTracks(0), mNMaxTracks(0), mNMaxSingleSliceTracks(0), mNMaxOutputTrackClusters(0), mNMaxClusters(0), mMemoryResRefit(-1), mMaxID(0), mNClusters(0), mNOutputTracks(0), mNOutputTrackClusters(0), mOutputTracks(nullptr), mSliceTrackInfos(nullptr), mClusters(nullptr), mGlobalClusterIDs(nullptr), mClusterAttachment(nullptr), mTrackOrderAttach(nullptr), mTrackOrderProcess(nullptr), mTmpMem(nullptr), mBorderMemory(nullptr), mBorderRangeMemory(nullptr), mMemory(nullptr), mRetryRefitIds(nullptr), mLoopData(nullptr), mSliceTrackers(nullptr), mChainTracking(nullptr)
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
    for (int j = 0; j < 2; j++) {
      mBorderCETracks[j][i] = 0;
    }
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
      const GPUTPCTracker& tracker = mSliceTrackers[track->Slice()];
      const GPUTPCHitId& ic = tracker.TrackHits()[track->OrigTrack()->FirstHitID() + i];
      id = tracker.Data().ClusterDataIndex(tracker.Data().Row(ic.RowIndex()), ic.HitIndex()) + mChainTracking->mIOPtrs.clustersNative->clusterOffset[track->Slice()][0];
    } else {
      id = track->OrigTrack()->OutTrackClusters()[i].GetId();
    }
    for (int j = 0; j < 3; j++) {
      int label = mChainTracking->mIOPtrs.mcLabelsTPC[id].fClusterID[j].fMCID;
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

void GPUTPCGMMerger::InitializeProcessor() { mSliceTrackers = mChainTracking->GetTPCSliceTrackers(); }

void* GPUTPCGMMerger::SetPointersHostOnly(void* mem)
{
  computePointerWithAlignment(mem, mSliceTrackInfos, mNMaxSliceTracks);
  if (mRec->GetParam().rec.NonConsecutiveIDs) {
    computePointerWithAlignment(mem, mGlobalClusterIDs, mNMaxOutputTrackClusters);
  }
  computePointerWithAlignment(mem, mBorderMemory, mNMaxSliceTracks);
  computePointerWithAlignment(mem, mBorderRangeMemory, 2 * mNMaxSliceTracks);
  computePointerWithAlignment(mem, mTrackLinks, mNMaxSliceTracks);
  size_t tmpSize = CAMath::Max(mNMaxSingleSliceTracks * NSLICES * sizeof(int), mNMaxTracks * sizeof(int) + mNMaxClusters * sizeof(char));
  computePointerWithAlignment(mem, mTmpMem, tmpSize);

  int nTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    mBorder[iSlice] = mBorderMemory + nTracks;
    mBorderRange[iSlice] = mBorderRangeMemory + 2 * nTracks;
    nTracks += Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersGPURefit(void* mem)
{
  computePointerWithAlignment(mem, mMemory);
  computePointerWithAlignment(mem, mRetryRefitIds, mNMaxTracks);
  computePointerWithAlignment(mem, mLoopData, mNMaxTracks);
  computePointerWithAlignment(mem, mOutputTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mClusters, mNMaxOutputTrackClusters);
  computePointerWithAlignment(mem, mTrackOrderAttach, mNMaxTracks);
  if (mRec->GetDeviceProcessingSettings().mergerSortTracks) {
    computePointerWithAlignment(mem, mTrackOrderProcess, mNMaxTracks);
  }
  computePointerWithAlignment(mem, mClusterAttachment, mNMaxClusters);

  return mem;
}

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersHostOnly, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_HOST, "TPCMergerHost");
  mMemoryResRefit = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersGPURefit, GPUMemoryResource::MEMORY_INOUT, "TPCMergerRefit");
}

void GPUTPCGMMerger::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxSliceTracks = 0;
  mNClusters = 0;
  mNMaxSingleSliceTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    unsigned int ntrk = Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
    mNMaxSliceTracks += ntrk;
    mNClusters += Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[iSlice].NTrackHits() : mkSlices[iSlice]->NTrackClusters();
    if (mNMaxSingleSliceTracks < ntrk) {
      mNMaxSingleSliceTracks = ntrk;
    }
  }
  mNMaxOutputTrackClusters = mNClusters * 1.1f + 1000;
  mNMaxTracks = mNMaxSliceTracks;
  mNMaxClusters = 0;
  if (io.clustersNative) {
    mNMaxClusters = io.clustersNative->nClustersTotal;
  } else if (mSliceTrackers) {
    for (int i = 0; i < NSLICES; i++) {
      mNMaxClusters += mSliceTrackers[i].NHitsTotal();
    }
  } else {
    mNMaxClusters = mNClusters;
  }
}

void GPUTPCGMMerger::SetSliceData(int index, const GPUTPCSliceOutput* sliceData) { mkSlices[index] = sliceData; }

void GPUTPCGMMerger::ClearTrackLinks(int n)
{
  for (int i = 0; i < n; i++) {
    mTrackLinks[i] = -1;
  }
}

int GPUTPCGMMerger::RefitSliceTrack(GPUTPCGMSliceTrack& sliceTrack, const GPUTPCTrack* inTrack, float alpha, int slice)
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
  trk.ResetCovariance();
  prop.SetTrack(&trk, alpha);
  for (int i = 0; i < inTrack->NHits(); i++) {
    float x, y, z;
    int row, flags;
    if (Param().rec.mergerReadFromTrackerDirectly) {
      const GPUTPCTracker& tracker = mSliceTrackers[slice];
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
      return 1;
    }
    trk.ConstrainSinPhi();
    if (prop.Update(y, z, row, Param(), flags & GPUTPCGMMergedTrackHit::clustererAndSharedFlags, 0, nullptr, false)) {
      return 1;
    }
    trk.ConstrainSinPhi();
  }
  sliceTrack.Set(trk, inTrack, alpha, slice);
  return 0;
}

void GPUTPCGMMerger::SetTrackClusterZT(GPUTPCGMSliceTrack& track, int iSlice, const GPUTPCTrack* sliceTr)
{
  if (Param().rec.mergerReadFromTrackerDirectly) {
    const GPUTPCTracker& trk = mSliceTrackers[iSlice];
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

void GPUTPCGMMerger::UnpackSlices()
{
  //* unpack the cluster information from the slice tracks and initialize track info array
  int nTracksCurrent = 0;

  for (int i = 0; i < NSLICES; i++) {
    if ((Param().rec.mergerReadFromTrackerDirectly ? mSliceTrackers[i].CommonMemory()->nLocalTracks : mkSlices[i]->NLocalTracks()) > mNMaxSingleSliceTracks) {
      throw std::runtime_error("mNMaxSingleSliceTracks too small");
    }
  }
  if (mSliceTrackers == nullptr && (!Param().rec.NonConsecutiveIDs || Param().rec.mergerReadFromTrackerDirectly)) {
    throw std::runtime_error("Must provide slice tracker pointer if NonConsecutiveIDs = false or mergerReadFromTrackerDirectly");
  }

  int* TrackIds = (int*)mTmpMem;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    const GPUTPCTracker& trk = mSliceTrackers[iSlice];
    unsigned int nLocalTracks = Param().rec.mergerReadFromTrackerDirectly ? trk.CommonMemory()->nLocalTracks : mkSlices[iSlice]->NLocalTracks();
    for (unsigned int i = 0; i < nLocalTracks; i++) {
      TrackIds[iSlice * mNMaxSingleSliceTracks + i] = -1;
    }
    mSliceTrackInfoIndex[iSlice] = nTracksCurrent;

    float alpha = Param().Alpha(iSlice);
    const GPUTPCTrack* sliceTr = Param().rec.mergerReadFromTrackerDirectly ? nullptr : mkSlices[iSlice]->GetFirstTrack();

    for (unsigned int itr = 0; itr < nLocalTracks; itr++) {
      if (Param().rec.mergerReadFromTrackerDirectly) {
        sliceTr = &trk.Tracks()[itr];
      } else if (itr) {
        sliceTr = sliceTr->GetNextTrack();
      }
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[nTracksCurrent];
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
      TrackIds[iSlice * mNMaxSingleSliceTracks + sliceTr->LocalTrackId()] = nTracksCurrent;
      nTracksCurrent++;
    }
    if (!Param().rec.mergerReadFromTrackerDirectly && nLocalTracks) {
      mMemory->firstGlobalTracks[iSlice] = sliceTr->GetNextTrack();
    }
  }
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    mSliceTrackInfoIndex[NSLICES + iSlice] = nTracksCurrent;
    const GPUTPCTracker& trk = mSliceTrackers[iSlice];

    float alpha = Param().Alpha(iSlice);
    const GPUTPCTrack* sliceTr = mMemory->firstGlobalTracks[iSlice];
    unsigned int nLocalTracks = Param().rec.mergerReadFromTrackerDirectly ? trk.CommonMemory()->nLocalTracks : mkSlices[iSlice]->NLocalTracks();
    unsigned int nTracks = Param().rec.mergerReadFromTrackerDirectly ? *trk.NTracks() : mkSlices[iSlice]->NTracks();
    for (unsigned int itr = nLocalTracks; itr < nTracks; itr++) {
      if (Param().rec.mergerReadFromTrackerDirectly) {
        sliceTr = &trk.Tracks()[itr];
      } else if (itr > nLocalTracks) {
        sliceTr = sliceTr->GetNextTrack();
      }
      int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * mNMaxSingleSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
      if (localId == -1) {
        continue;
      }
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[nTracksCurrent];
      SetTrackClusterZT(track, iSlice, sliceTr);
      track.Set(this, sliceTr, alpha, iSlice);
      track.SetGlobalSectorTrackCov();
      track.SetPrevNeighbour(-1);
      track.SetNextNeighbour(-1);
      track.SetNextSegmentNeighbour(-1);
      track.SetPrevSegmentNeighbour(-1);
      track.SetLocalTrackId(localId);
      nTracksCurrent++;
    }
  }
  mSliceTrackInfoIndex[2 * NSLICES] = nTracksCurrent;
}

void GPUTPCGMMerger::MakeBorderTracks(int iSlice, int iBorder, GPUTPCGMBorderTrack B[], int& nB, bool useOrigTrackParam)
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = Param().ConstBz;

  nB = 0;

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
  for (int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++) {
    const GPUTPCGMSliceTrack* track = &mSliceTrackInfos[itr];

    if (track->PrevSegmentNeighbour() >= 0 && track->Slice() == mSliceTrackInfos[track->PrevSegmentNeighbour()].Slice()) {
      continue;
    }
    if (useOrigTrackParam) { // TODO: Check how far this makes sense with slice track refit
      if (fabsf(track->QPt()) < GPUCA_MERGER_LOOPER_QPT_LIMIT) {
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
      trackTmp.Set(this, trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Slice());
    } else {
      if (fabsf(track->QPt()) < GPUCA_MERGER_HORIZONTAL_DOUBLE_QPT_LIMIT) {
        if (iBorder == 0 && track->NextNeighbour() >= 0) {
          continue;
        }
        if (iBorder == 1 && track->PrevNeighbour() >= 0) {
          continue;
        }
      }
    }
    GPUTPCGMBorderTrack& b = B[nB];

    if (track->TransportToXAlpha(this, x0, sinAlpha, cosAlpha, fieldBz, b, maxSin)) {
      b.SetTrackID(itr);
      b.SetNClusters(track->NClusters());
      for (int i = 0; i < 4; i++) {
        if (fabsf(b.Cov()[i]) >= 5.0) {
          b.SetCov(i, 5.0);
        }
      }
      if (fabsf(b.Cov()[4]) >= 0.5) {
        b.SetCov(4, 0.5);
      }
      nB++;
    }
  }
}

void GPUTPCGMMerger::MergeBorderTracks(int iSlice1, GPUTPCGMBorderTrack B1[], int N1, int iSlice2, GPUTPCGMBorderTrack B2[], int N2, int mergeMode)
{
  //* merge two sets of tracks
  if (N1 == 0 || N2 == 0) {
    return;
  }

  CADEBUG(GPUInfo("\nMERGING Slices %d %d NTracks %d %d CROSS %d", iSlice1, iSlice2, N1, N2, mergeMode));
  int statAll = 0, statMerged = 0;
  float factor2ys = 1.5; // 1.5;//SG!!!
  float factor2zt = 1.5; // 1.5;//SG!!!
  float factor2k = 2.0;  // 2.2;

  factor2k = 3.5 * 3.5 * factor2k * factor2k;
  factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
  factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;

  int minNPartHits = 10; // SG!!!
  int minNTotalHits = 20;

  GPUTPCGMBorderTrack::Range* range1 = mBorderRange[iSlice1];
  GPUTPCGMBorderTrack::Range* range2 = mBorderRange[iSlice2] + N2;

  bool sameSlice = (iSlice1 == iSlice2);
  {
    for (int itr = 0; itr < N1; itr++) {
      GPUTPCGMBorderTrack& b = B1[itr];
      float d = CAMath::Max(0.5f, 3.5f * sqrtf(b.Cov()[1]));
      if (fabsf(b.Par()[4]) >= 20) {
        d *= 2;
      } else if (d > 3) {
        d = 3;
      }
      CADEBUG(
        printf("  Input Slice 1 %d Track %d: ", iSlice1, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf(" - D %8.3f\n", d));
      range1[itr].fId = itr;
      range1[itr].fMin = b.Par()[1] + b.ZOffsetLinear() - d;
      range1[itr].fMax = b.Par()[1] + b.ZOffsetLinear() + d;
    }
    std::sort(range1, range1 + N1, GPUTPCGMBorderTrack::Range::CompMin);
    if (sameSlice) {
      for (int i = 0; i < N1; i++) {
        range2[i] = range1[i];
      }
      std::sort(range2, range2 + N1, GPUTPCGMBorderTrack::Range::CompMax);
      N2 = N1;
      B2 = B1;
    } else {
      for (int itr = 0; itr < N2; itr++) {
        GPUTPCGMBorderTrack& b = B2[itr];
        float d = CAMath::Max(0.5f, 3.5f * sqrtf(b.Cov()[1]));
        if (fabsf(b.Par()[4]) >= 20) {
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
      std::sort(range2, range2 + N2, GPUTPCGMBorderTrack::Range::CompMax);
    }
  }

  int i2 = 0;
  for (int i1 = 0; i1 < N1; i1++) {
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
          if (mChainTracking->mIOPtrs.mcLabelsTPC) {printf("Comparing track %3d to %3d: ", r1.fId, r2.fId); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Cov()[i]); } printf("\n%28s", ""); });
        CADEBUG(
          if (mChainTracking->mIOPtrs.mcLabelsTPC) {for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Cov()[i]); } printf("   -   %5s   -   ", GetTrackLabel(b1) == GetTrackLabel(b2) ? "CLONE" : "FAKE"); });
        if (b2.NClusters() < lBest2) {
          CADEBUG2(continue, printf("!NCl1\n"));
        }
        if (mergeMode > 0) {
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
        float fys = fabsf(b1.Par()[4]) < 20 ? factor2ys : (2. * factor2ys);
        float fzt = fabsf(b1.Par()[4]) < 20 ? factor2zt : (2. * factor2zt);
        if (!b1.CheckChi2YS(b2, fys)) {
          CADEBUG2(continue, printf("!YS\n"));
        }
        if (!b1.CheckChi2ZT(b2, fzt)) {
          CADEBUG2(continue, printf("!ZT\n"));
        }
        if (fabsf(b1.Par()[4]) < 20) {
          if (b2.NClusters() < minNPartHits) {
            CADEBUG2(continue, printf("!NCl2\n"));
          }
          if (b1.NClusters() + b2.NClusters() < minNTotalHits) {
            CADEBUG2(continue, printf("!NCl3\n"));
          }
        }
        CADEBUG(printf("OK: dZ %8.3f D1 %8.3f D2 %8.3f\n", fabsf(b1.Par()[1] - b2.Par()[1]), 3.5 * sqrt(b1.Cov()[1]), 3.5 * sqrt(b2.Cov()[1])));
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
  }
  // GPUInfo("STAT: slices %d, %d: all %d merged %d", iSlice1, iSlice2, statAll, statMerged);
}

void GPUTPCGMMerger::MergeWithingSlices()
{
  float x0 = Param().tpcGeometry.Row2X(63);
  const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());

  ClearTrackLinks(SliceTrackInfoLocalTotal());
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    int nBord = 0;
    for (int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++) {
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[itr];
      GPUTPCGMBorderTrack& b = mBorder[iSlice][nBord];
      if (track.TransportToX(this, x0, Param().ConstBz, b, maxSin)) {
        b.SetTrackID(itr);
        CADEBUG(
          printf("WITHIN SLICE %d Track %d - ", iSlice, itr); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b.Cov()[i]); } printf("\n"));
        b.SetNClusters(track.NClusters());
        nBord++;
      }
    }

    MergeBorderTracks(iSlice, mBorder[iSlice], nBord, iSlice, mBorder[iSlice], nBord);
  }

  ResolveMergeSlices(false, true);
}

void GPUTPCGMMerger::MergeSlices()
{
  MergeSlicesStep(2, 3, false);
  MergeSlicesStep(0, 1, false);
  MergeSlicesStep(0, 1, true);
}

void GPUTPCGMMerger::MergeSlicesStep(int border0, int border1, bool useOrigTrackParam)
{
  ClearTrackLinks(SliceTrackInfoLocalTotal());
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    int jSlice = mNextSliceInd[iSlice];
    GPUTPCGMBorderTrack *bCurr = mBorder[iSlice], *bNext = mBorder[jSlice];
    int nCurr = 0, nNext = 0;
    MakeBorderTracks(iSlice, border0, bCurr, nCurr, useOrigTrackParam);
    MakeBorderTracks(jSlice, border1, bNext, nNext, useOrigTrackParam);
    MergeBorderTracks(iSlice, bCurr, nCurr, jSlice, bNext, nNext, useOrigTrackParam ? -1 : 0);
  }
  ResolveMergeSlices(useOrigTrackParam, false);
}

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

void GPUTPCGMMerger::ResolveMergeSlices(bool useOrigTrackParam, bool mergeAll)
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

    bool sameSegment = fabsf(track1->NClusters() > track2->NClusters() ? track1->QPt() : track2->QPt()) < 2 || track1->QPt() * track2->QPt() > 0;
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
      std::swap(track1, track1Base);
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
        std::swap(track1, track2);
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

void GPUTPCGMMerger::MergeCEInit()
{
  for (int k = 0; k < 2; k++) {
    for (int i = 0; i < NSLICES; i++) {
      mBorderCETracks[k][i] = 0;
    }
  }
}

void GPUTPCGMMerger::MergeCEFill(const GPUTPCGMSliceTrack* track, const GPUTPCGMMergedTrackHit& cls, int itr)
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

  if (!Param().ContinuousTracking && fabsf(z) > 10) {
    return;
  }
  int slice = track->Slice();
  for (int attempt = 0; attempt < 2; attempt++) {
    unsigned int nTracks = Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[slice].NTracks() : mkSlices[slice]->NTracks();
    GPUTPCGMBorderTrack& b = attempt == 0 ? mBorder[slice][mBorderCETracks[0][slice]] : mBorder[slice][nTracks - 1 - mBorderCETracks[1][slice]];
    const float x0 = Param().tpcGeometry.Row2X(attempt == 0 ? 63 : cls.row);
    if (track->TransportToX(this, x0, Param().ConstBz, b, GPUCA_MAX_SIN_PHI_LOW)) {
      b.SetTrackID(itr);
      b.SetNClusters(mOutputTracks[itr].NClusters());
      if (fabsf(b.Cov()[4]) >= 0.5) {
        b.SetCov(4, 0.5); // TODO: Is this needed and better than the cut in BorderTrack?
      }
      if (track->CSide()) {
        b.SetPar(1, b.Par()[1] - 2 * (z - b.ZOffsetLinear()));
        b.SetZOffsetLinear(-b.ZOffsetLinear());
      }
      b.SetRow(cls.row);
      mBorderCETracks[attempt][slice]++;
      break;
    }
  }
}

void GPUTPCGMMerger::MergeCE()
{
  ClearTrackLinks(mNOutputTracks);
  const ClusterNative* cls = Param().earlyTpcTransform ? nullptr : mConstantMem->ioPtrs.clustersNative->clustersLinear;
  for (int iSlice = 0; iSlice < NSLICES / 2; iSlice++) {
    int jSlice = iSlice + NSLICES / 2;
    unsigned int nTracksI = Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[iSlice].NTracks() : mkSlices[iSlice]->NTracks();
    unsigned int nTracksJ = Param().rec.mergerReadFromTrackerDirectly ? *mSliceTrackers[jSlice].NTracks() : mkSlices[jSlice]->NTracks();
    MergeBorderTracks(iSlice, mBorder[iSlice], mBorderCETracks[0][iSlice], jSlice, mBorder[jSlice], mBorderCETracks[0][jSlice], 1);
    MergeBorderTracks(iSlice, mBorder[iSlice] + nTracksI - mBorderCETracks[1][iSlice], mBorderCETracks[1][iSlice], jSlice, mBorder[jSlice] + nTracksJ - mBorderCETracks[1][jSlice], mBorderCETracks[1][jSlice], 2);
  }
  for (int i = 0; i < mNOutputTracks; i++) {
    if (mTrackLinks[i] >= 0) {
      GPUTPCGMMergedTrack* trk[2] = {&mOutputTracks[i], &mOutputTracks[mTrackLinks[i]]};

      if (!trk[1]->OK() || trk[1]->CCE()) {
        continue;
      }

      if (mNOutputTrackClusters + trk[0]->NClusters() + trk[1]->NClusters() >= mNMaxOutputTrackClusters) {
        printf("Insufficient cluster memory for merging CE tracks (OutputClusters %d, max clusters %u)\n", mNOutputTrackClusters, mNMaxOutputTrackClusters);
        return;
      }

      bool looper = trk[0]->Looper() || trk[1]->Looper() || (trk[0]->GetParam().GetQPt() > 1 && trk[0]->GetParam().GetQPt() * trk[1]->GetParam().GetQPt() < 0);
      if (!looper && trk[0]->GetParam().GetPar(3) * trk[1]->GetParam().GetPar(3) < 0) {
        continue;
      }
      bool needswap = false;
      if (looper) {
        float z0max, z1max;
        if (Param().earlyTpcTransform) {
          z0max = CAMath::Max(fabsf(mClusters[trk[0]->FirstClusterRef()].z), fabsf(mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z));
          z1max = CAMath::Max(fabsf(mClusters[trk[1]->FirstClusterRef()].z), fabsf(mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z));
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
        std::swap(trk[0], trk[1]);
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
          const float offset = fabsf(z1) > fabsf(z0) ? -z0 : z1;
          trk[1]->Param().Z() += trk[1]->Param().TZOffset() - offset;
          trk[1]->Param().TZOffset() = offset;
        } else {
          const float t0 = CAMath::Max(cls[mClusters[trk[0]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].num].getTime());
          const float t1 = CAMath::Max(cls[mClusters[trk[1]->FirstClusterRef()].num].getTime(), cls[mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].num].getTime());
          const float offset = CAMath::Max(CAMath::Max(t0, t1) - 250.f / mConstantMem->calibObjects.fastTransform->getVDrift(), 0.f); // TODO: Replace me by correct drift time
          trk[1]->Param().Z() += mConstantMem->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(trk[1]->CSide() * NSLICES / 2, trk[1]->Param().TZOffset() - offset);
          trk[1]->Param().TZOffset() = offset;
        }
      }

      int newRef = mNOutputTrackClusters;
      for (int k = 1; k >= 0; k--) {
        if (reverse[k]) {
          for (int j = trk[k]->NClusters() - 1; j >= 0; j--) {
            mClusters[mNOutputTrackClusters++] = mClusters[trk[k]->FirstClusterRef() + j];
          }
        } else {
          for (unsigned int j = 0; j < trk[k]->NClusters(); j++) {
            mClusters[mNOutputTrackClusters++] = mClusters[trk[k]->FirstClusterRef() + j];
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

  // for (int i = 0;i < mNOutputTracks;i++) {if (mOutputTracks[i].CCE() == false) {mOutputTracks[i].SetNClusters(0);mOutputTracks[i].SetOK(false);}} //Remove all non-CE tracks
}

struct GPUTPCGMMerger_CompareClusterIdsLooper {
  struct clcomparestruct {
    unsigned char leg;
  };

  const unsigned char leg;
  const bool outwards;
  const GPUTPCGMMerger::trackCluster* const cmp1;
  const clcomparestruct* const cmp2;
  GPUTPCGMMerger_CompareClusterIdsLooper(unsigned char l, bool o, const GPUTPCGMMerger::trackCluster* c1, const clcomparestruct* c2) : leg(l), outwards(o), cmp1(c1), cmp2(c2) {}
  bool operator()(const short aa, const short bb)
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
  GPUTPCGMMerger_CompareClusterIds(const GPUTPCGMMerger::trackCluster* cmp) : mCmp(cmp) {}
  bool operator()(const short aa, const short bb)
  {
    const GPUTPCGMMerger::trackCluster& a = mCmp[aa];
    const GPUTPCGMMerger::trackCluster& b = mCmp[bb];
    if (a.row != b.row) {
      return (a.row > b.row);
    }
    return (a.id > b.id);
  }
};

struct GPUTPCGMMerger_CompareTracksAttachWeight {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUTPCGMMerger_CompareTracksAttachWeight(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  bool operator()(const int aa, const int bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    return (fabsf(a.GetParam().GetQPt()) > fabsf(b.GetParam().GetQPt()));
  }
};

struct GPUTPCGMMerger_CompareTracksProcess {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUTPCGMMerger_CompareTracksProcess(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  bool operator()(const int aa, const int bb)
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

bool GPUTPCGMMerger_CompareParts(const GPUTPCGMSliceTrack* a, const GPUTPCGMSliceTrack* b) { return (a->X() > b->X()); }

void GPUTPCGMMerger::CollectMergedTracks()
{
  // Resolve connections for global tracks first
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    for (int itr = SliceTrackInfoGlobalFirst(iSlice); itr < SliceTrackInfoGlobalLast(iSlice); itr++) {
      GPUTPCGMSliceTrack& globalTrack = mSliceTrackInfos[itr];
      GPUTPCGMSliceTrack& localTrack = mSliceTrackInfos[globalTrack.LocalTrackId()];
      localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, itr);
    }
  }

  // CheckMergedTracks();

  // Now collect the merged tracks
  mNOutputTracks = 0;
  int nOutTrackClusters = 0;

  GPUTPCGMSliceTrack* trackParts[kMaxParts];

  for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++) {
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
      std::sort(trackParts, trackParts + nParts, GPUTPCGMMerger_CompareParts);
    }

    if (Param().rec.dropLoopers && leg > 0) {
      nParts = 1;
      leg = 0;
    }

    trackCluster trackClusters[kMaxClusters];
    nHits = 0;
    for (int ipart = 0; ipart < nParts; ipart++) {
      const GPUTPCGMSliceTrack* t = trackParts[ipart];
      CADEBUG(printf("Collect Track %d Part %d QPt %f DzDs %f\n", mNOutputTracks, ipart, t->QPt(), t->DzDs()));
      int nTrackHits = t->NClusters();
      trackCluster* c2 = trackClusters + nHits + nTrackHits - 1;
      for (int i = 0; i < nTrackHits; i++, c2--) {
        if (Param().rec.mergerReadFromTrackerDirectly) {
          const GPUTPCTracker& trk = mSliceTrackers[t->Slice()];
          const GPUTPCHitId& ic = trk.TrackHits()[t->OrigTrack()->FirstHitID() + i];
          unsigned int id = trk.Data().ClusterDataIndex(trk.Data().Row(ic.RowIndex()), ic.HitIndex()) + mChainTracking->mIOPtrs.clustersNative->clusterOffset[t->Slice()][0];
          *c2 = trackCluster{id, (unsigned char)ic.RowIndex(), t->Slice(), t->Leg()};
        } else {
          const GPUTPCSliceOutCluster c = t->OrigTrack()->OutTrackClusters()[i];
          unsigned int id = Param().rec.NonConsecutiveIDs ? ((int)((int*)&c - (int*)mkSlices[t->Slice()]->GetFirstTrack())) : c.GetId();
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

        std::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIdsLooper(baseLeg, outwards, trackClusters, clusterSort));
      } else {
        std::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIds(trackClusters));
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

    GPUTPCGMMergedTrackHit* cl = mClusters + nOutTrackClusters;
    for (int i = 0; i < nHits; i++) {
      unsigned char state;
      if (Param().rec.NonConsecutiveIDs) {
        const GPUTPCSliceOutCluster* c = (const GPUTPCSliceOutCluster*)((int*)mkSlices[trackClusters[i].slice]->GetFirstTrack() + trackClusters[i].id);
        cl[i].x = c->GetX();
        cl[i].y = c->GetY();
        cl[i].z = c->GetZ();
        cl[i].amp = c->GetAmp();
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
        cl[i].pad = c->mPad;
        cl[i].time = c->mTime;
#endif
        state = c->GetFlags();
      } else if (Param().earlyTpcTransform) {
        int index = trackClusters[i].id - mChainTracking->mIOPtrs.clustersNative->clusterOffset[trackClusters[i].slice][0];
        const GPUTPCClusterData& c = mSliceTrackers[trackClusters[i].slice].ClusterData()[index];
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
        const ClusterNative& c = mChainTracking->mIOPtrs.clustersNative->clustersLinear[trackClusters[i].id];
        state = c.getFlags();
      }
      cl[i].state = state & GPUTPCGMMergedTrackHit::clustererAndSharedFlags; // Only allow edge, deconvoluted, and shared flags
      cl[i].row = trackClusters[i].row;
      if (!Param().rec.NonConsecutiveIDs) // We already have global consecutive numbers from the slice tracker, and we need to keep them for late cluster attachment
      {
        cl[i].num = trackClusters[i].id;
      } else { // Produce consecutive numbers for shared cluster flagging
        cl[i].num = nOutTrackClusters + i;
        mGlobalClusterIDs[cl[i].num] = trackClusters[i].id;
      }
      cl[i].slice = trackClusters[i].slice;
      cl[i].leg = trackClusters[i].leg;
    }

    GPUTPCGMMergedTrack& mergedTrack = mOutputTracks[mNOutputTracks];
    mergedTrack.SetFlags(0);
    mergedTrack.SetOK(1);
    mergedTrack.SetLooper(leg > 0);
    mergedTrack.SetLegs(leg);
    mergedTrack.SetNClusters(nHits);
    mergedTrack.SetFirstClusterRef(nOutTrackClusters);
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

    // if (nParts > 1) printf("Merged %d: QPt %f %d parts %d hits\n", mNOutputTracks, p1.QPt(), nParts, nHits);

    /*if (GPUQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(mNOutputTracks))
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
      MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], mNOutputTracks);
    }
    mNOutputTracks++;
    nOutTrackClusters += nHits;
  }
  mNOutputTrackClusters = nOutTrackClusters;
}

void GPUTPCGMMerger::PrepareClustersForFit()
{
  unsigned int maxId = 0;
  maxId = Param().rec.NonConsecutiveIDs ? mNOutputTrackClusters : mNMaxClusters;
  if (maxId > mNMaxClusters) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  mMaxID = maxId;

  unsigned int* trackSort = (unsigned int*)mTmpMem;
  unsigned char* sharedCount = (unsigned char*)(trackSort + mNOutputTracks);

  if (mRec->GetDeviceProcessingSettings().mergerSortTracks) {
    mNSlowTracks = 0;
    for (int i = 0; i < mNOutputTracks; i++) {
      const GPUTPCGMMergedTrack& trk = mOutputTracks[i];
      if (trk.CCE() || trk.Legs()) {
        mNSlowTracks++;
      }
      mTrackOrderProcess[i] = i;
    }
    std::sort(mTrackOrderProcess, mTrackOrderProcess + mNOutputTracks, GPUTPCGMMerger_CompareTracksProcess(mOutputTracks));
  }

  if (!Param().rec.NonConsecutiveIDs) {
    for (int i = 0; i < mNOutputTracks; i++) {
      trackSort[i] = i;
    }
    std::sort(trackSort, trackSort + mNOutputTracks, GPUTPCGMMerger_CompareTracksAttachWeight(mOutputTracks));
    memset(mClusterAttachment, 0, maxId * sizeof(mClusterAttachment[0]));
    for (int i = 0; i < mNOutputTracks; i++) {
      mTrackOrderAttach[trackSort[i]] = i;
    }
    for (int i = 0; i < mNOutputTrackClusters; i++) {
      mClusterAttachment[mClusters[i].num] = attachAttached | attachGood;
    }
    for (unsigned int k = 0; k < maxId; k++) {
      sharedCount[k] = 0;
    }
    for (int k = 0; k < mNOutputTrackClusters; k++) {
      sharedCount[mClusters[k].num] = (sharedCount[mClusters[k].num] << 1) | 1;
    }
    for (int k = 0; k < mNOutputTrackClusters; k++) {
      if (sharedCount[mClusters[k].num] > 1) {
        mClusters[k].state |= GPUTPCGMMergedTrackHit::flagShared;
      }
    }
  }
}

void GPUTPCGMMerger::Finalize()
{
  if (Param().rec.NonConsecutiveIDs) {
    for (int i = 0; i < mNOutputTrackClusters; i++) {
      mClusters[i].num = mGlobalClusterIDs[i];
    }
  } else {
    int* trkOrderReverse = (int*)mTmpMem;
    for (int i = 0; i < mNOutputTracks; i++) {
      trkOrderReverse[mTrackOrderAttach[i]] = i;
    }
    for (int i = 0; i < mNOutputTrackClusters; i++) {
      mClusterAttachment[mClusters[i].num] = 0; // Reset adjacent attachment for attached clusters, set correctly below
    }
    for (int i = 0; i < mNOutputTracks; i++) {
      const GPUTPCGMMergedTrack& trk = mOutputTracks[i];
      if (!trk.OK() || trk.NClusters() == 0) {
        continue;
      }
      char goodLeg = mClusters[trk.FirstClusterRef() + trk.NClusters() - 1].leg;
      for (unsigned int j = 0; j < trk.NClusters(); j++) {
        int id = mClusters[trk.FirstClusterRef() + j].num;
        int weight = mTrackOrderAttach[i] | attachAttached;
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
    for (int i = 0; i < mMaxID; i++) {
      if (mClusterAttachment[i] != 0) {
        mClusterAttachment[i] = (mClusterAttachment[i] & attachFlagMask) | trkOrderReverse[mClusterAttachment[i] & attachTrackMask];
      }
    }
  }
}
