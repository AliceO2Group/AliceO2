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
#include "GPUTPCSliceOutTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
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

static constexpr int kMaxParts = 400;
static constexpr int kMaxClusters = 1000;

//#define OFFLINE_FITTER

#if !defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPUCODE)
#undef OFFLINE_FITTER
#endif

GPUTPCGMMerger::GPUTPCGMMerger()
  : mTrackLinks(nullptr), mNMaxSliceTracks(0), mNMaxTracks(0), mNMaxSingleSliceTracks(0), mNMaxOutputTrackClusters(0), mNMaxClusters(0), mMemoryResMerger(-1), mMemoryResRefit(-1), mMaxID(0), mNClusters(0), mNOutputTracks(0), mNOutputTrackClusters(0), mOutputTracks(nullptr), mSliceTrackInfos(nullptr), mClusters(nullptr), mGlobalClusterIDs(nullptr), mClusterAttachment(nullptr), mTrackOrder(nullptr), mTmpMem(nullptr), mBorderMemory(nullptr), mBorderRangeMemory(nullptr), mSliceTrackers(nullptr), mChainTracking(nullptr)
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

int GPUTPCGMMerger::GetTrackLabel(GPUTPCGMBorderTrack& trk)
{
  GPUTPCGMSliceTrack* track = &mSliceTrackInfos[trk.TrackID()];
  const GPUTPCSliceOutCluster* clusters = track->OrigTrack()->Clusters();
  int nClusters = track->OrigTrack()->NClusters();
  std::vector<int> labels;
  for (int i = 0; i < nClusters; i++) {
    for (int j = 0; j < 3; j++) {
      int label = mChainTracking->mIOPtrs.mcLabelsTPC[clusters[i].GetId()].mClusterID[j].fMCID;
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
  if (mCAParam->rec.NonConsecutiveIDs) {
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
    nTracks += mkSlices[iSlice]->NTracks();
  }
  return mem;
}

void* GPUTPCGMMerger::SetPointersGPURefit(void* mem)
{
  computePointerWithAlignment(mem, mOutputTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mClusters, mNMaxOutputTrackClusters);
  computePointerWithAlignment(mem, mTrackOrder, mNMaxTracks);
  computePointerWithAlignment(mem, mClusterAttachment, mNMaxClusters);

  return mem;
}

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResMerger = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersHostOnly, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_HOST, "TPCMergerHost");
  mMemoryResRefit = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersGPURefit, GPUMemoryResource::MEMORY_INOUT, "TPCMergerRefit");
}

void GPUTPCGMMerger::SetMaxData()
{
  mNMaxSliceTracks = 0;
  mNClusters = 0;
  mNMaxSingleSliceTracks = 0;
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    if (!mkSlices[iSlice]) {
      continue;
    }
    mNMaxSliceTracks += mkSlices[iSlice]->NTracks();
    mNClusters += mkSlices[iSlice]->NTrackClusters();
    if (mNMaxSingleSliceTracks < mkSlices[iSlice]->NTracks()) {
      mNMaxSingleSliceTracks = mkSlices[iSlice]->NTracks();
    }
  }
  mNMaxOutputTrackClusters = mNClusters * 1.1f + 1000;
  mNMaxTracks = mNMaxSliceTracks;
  mNMaxClusters = 0;
  if (mSliceTrackers) {
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

void GPUTPCGMMerger::UnpackSlices()
{
  //* unpack the cluster information from the slice tracks and initialize track info array

  int nTracksCurrent = 0;

  const GPUTPCSliceOutTrack* firstGlobalTracks[NSLICES];

  unsigned int maxSliceTracks = 0;
  for (int i = 0; i < NSLICES; i++) {
    firstGlobalTracks[i] = nullptr;
    if (mkSlices[i]->NLocalTracks() > maxSliceTracks) {
      maxSliceTracks = mkSlices[i]->NLocalTracks();
    }
  }

  if (maxSliceTracks > mNMaxSingleSliceTracks) {
    throw std::runtime_error("mNMaxSingleSliceTracks too small");
  }

  int* TrackIds = (int*)mTmpMem;
  for (unsigned int i = 0; i < maxSliceTracks * NSLICES; i++) {
    TrackIds[i] = -1;
  }

  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    mSliceTrackInfoIndex[iSlice] = nTracksCurrent;

    float alpha = mCAParam->Alpha(iSlice);
    const GPUTPCSliceOutput& slice = *(mkSlices[iSlice]);
    const GPUTPCSliceOutTrack* sliceTr = slice.GetFirstTrack();

    for (unsigned int itr = 0; itr < slice.NLocalTracks(); itr++, sliceTr = sliceTr->GetNextTrack()) {
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[nTracksCurrent];
      track.Set(sliceTr, alpha, iSlice);
      if (!track.FilterErrors(*mCAParam, GPUCA_MAX_SIN_PHI, 0.1f)) {
        continue;
      }
      CADEBUG(GPUInfo("INPUT Slice %d, Track %u, QPt %f DzDs %f", iSlice, itr, track.QPt(), track.DzDs()));
      track.SetPrevNeighbour(-1);
      track.SetNextNeighbour(-1);
      track.SetNextSegmentNeighbour(-1);
      track.SetPrevSegmentNeighbour(-1);
      track.SetGlobalTrackId(0, -1);
      track.SetGlobalTrackId(1, -1);
      TrackIds[iSlice * maxSliceTracks + sliceTr->LocalTrackId()] = nTracksCurrent;
      nTracksCurrent++;
    }
    firstGlobalTracks[iSlice] = sliceTr;
  }
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    mSliceTrackInfoIndex[NSLICES + iSlice] = nTracksCurrent;

    float alpha = mCAParam->Alpha(iSlice);
    const GPUTPCSliceOutput& slice = *(mkSlices[iSlice]);
    const GPUTPCSliceOutTrack* sliceTr = firstGlobalTracks[iSlice];
    for (unsigned int itr = slice.NLocalTracks(); itr < slice.NTracks(); itr++, sliceTr = sliceTr->GetNextTrack()) {
      int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * maxSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
      if (localId == -1) {
        continue;
      }
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[nTracksCurrent];
      track.Set(sliceTr, alpha, iSlice);
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

void GPUTPCGMMerger::MakeBorderTracks(int iSlice, int iBorder, GPUTPCGMBorderTrack B[], int& nB, bool fromOrig)
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = mCAParam->ConstBz;

  nB = 0;

  float dAlpha = mCAParam->DAlpha / 2;
  float x0 = 0;

  if (iBorder == 0) { // transport to the left edge of the sector and rotate horizontally
    dAlpha = dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 1) { // transport to the right edge of the sector and rotate horizontally
    dAlpha = -dAlpha - CAMath::Pi() / 2;
  } else if (iBorder == 2) { // transport to the middle of the sector and rotate vertically to the border on the left
    x0 = mCAParam->tpcGeometry.Row2X(63);
  } else if (iBorder == 3) { // transport to the middle of the sector and rotate vertically to the border on the right
    dAlpha = -dAlpha;
    x0 = mCAParam->tpcGeometry.Row2X(63);
  } else if (iBorder == 4) { // transport to the middle of the sßector, w/o rotation
    dAlpha = 0;
    x0 = mCAParam->tpcGeometry.Row2X(63);
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
    if (fromOrig) {
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
      trackTmp.Set(trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Slice());
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

    if (track->TransportToXAlpha(x0, sinAlpha, cosAlpha, fieldBz, b, maxSin)) {
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

void GPUTPCGMMerger::MergeBorderTracks(int iSlice1, GPUTPCGMBorderTrack B1[], int N1, int iSlice2, GPUTPCGMBorderTrack B2[], int N2, int crossCE)
{
  //* merge two sets of tracks
  if (N1 == 0 || N2 == 0) {
    return;
  }

  CADEBUG(GPUInfo("\nMERGING Slices %d %d NTracks %d %d CROSS %d", iSlice1, iSlice2, N1, N2, crossCE));
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
      range1[itr].fMin = b.Par()[1] + b.ZOffset() - d;
      range1[itr].fMax = b.Par()[1] + b.ZOffset() + d;
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
        range2[itr].fMin = b.Par()[1] + b.ZOffset() - d;
        range2[itr].fMax = b.Par()[1] + b.ZOffset() + d;
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
          printf("Comparing track %3d to %3d: ", r1.fId, r2.fId); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b1.Cov()[i]); } printf("\n%28s", ""));
        CADEBUG(
          for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Par()[i]); } printf(" - "); for (int i = 0; i < 5; i++) { printf("%8.3f ", b2.Cov()[i]); } printf("   -   %5s   -   ", GetTrackLabel(b1) == GetTrackLabel(b2) ? "CLONE" : "FAKE"));
        if (b2.NClusters() < lBest2) {
          CADEBUG2(continue, printf("!NCl1\n"));
        }
        if (crossCE >= 2 && abs(b1.Row() - b2.Row()) > 1) {
          CADEBUG2(continue, printf("!ROW\n"));
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
  float x0 = mCAParam->tpcGeometry.Row2X(63);
  const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());

  ClearTrackLinks(SliceTrackInfoLocalTotal());
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    int nBord = 0;
    for (int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++) {
      GPUTPCGMSliceTrack& track = mSliceTrackInfos[itr];
      GPUTPCGMBorderTrack& b = mBorder[iSlice][nBord];
      if (track.TransportToX(x0, mCAParam->ConstBz, b, maxSin)) {
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

void GPUTPCGMMerger::MergeSlicesStep(int border0, int border1, bool fromOrig)
{
  ClearTrackLinks(SliceTrackInfoLocalTotal());
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    int jSlice = mNextSliceInd[iSlice];
    GPUTPCGMBorderTrack *bCurr = mBorder[iSlice], *bNext = mBorder[jSlice];
    int nCurr = 0, nNext = 0;
    MakeBorderTracks(iSlice, border0, bCurr, nCurr, fromOrig);
    MakeBorderTracks(jSlice, border1, bNext, nNext, fromOrig);
    MergeBorderTracks(iSlice, bCurr, nCurr, jSlice, bNext, nNext, fromOrig ? -1 : 0);
  }
  ResolveMergeSlices(fromOrig, false);
}

void GPUTPCGMMerger::PrintMergeGraph(GPUTPCGMSliceTrack* trk)
{
  GPUTPCGMSliceTrack* orgTrack = trk;
  while (trk->PrevSegmentNeighbour() >= 0) {
    trk = &mSliceTrackInfos[trk->PrevSegmentNeighbour()];
  }
  GPUTPCGMSliceTrack* orgTower = trk;
  while (trk->PrevNeighbour() >= 0) {
    trk = &mSliceTrackInfos[trk->PrevNeighbour()];
  }

  int nextId = trk - mSliceTrackInfos;
  GPUInfo("Graph of track %d", (int)(orgTrack - mSliceTrackInfos));
  while (nextId >= 0) {
    trk = &mSliceTrackInfos[nextId];
    if (trk->PrevSegmentNeighbour() >= 0) {
      GPUError("TRACK TREE INVALID!!! %d --> %d", trk->PrevSegmentNeighbour(), nextId);
    }
    printf(trk == orgTower ? "--" : "  ");
    while (nextId >= 0) {
      GPUTPCGMSliceTrack* trk2 = &mSliceTrackInfos[nextId];
      if (trk != trk2 && (trk2->PrevNeighbour() >= 0 || trk2->NextNeighbour() >= 0)) {
        printf("   (TRACK TREE INVALID!!! %d <-- %d --> %d)   ", trk2->PrevNeighbour(), nextId, trk2->NextNeighbour());
      }
      printf(" %s%5d(%5.2f)", trk2 == orgTrack ? "!" : " ", nextId, trk2->QPt());
      nextId = trk2->NextSegmentNeighbour();
    }
    printf("\n");
    nextId = trk->NextNeighbour();
  }
}

void GPUTPCGMMerger::ResolveMergeSlices(bool fromOrig, bool mergeAll)
{
  if (!mergeAll) {
    /*int neighborType = fromOrig ? 1 : 0;

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
    // PrintMergeGraph(track1);
    // PrintMergeGraph(track2);

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

      float z1min = track1->MinClusterZ(), z1max = track1->MaxClusterZ();
      float z2min = track2->MinClusterZ(), z2max = track2->MaxClusterZ();
      if (track1 != track1Base) {
        z1min = CAMath::Min(z1min, track1Base->MinClusterZ());
        z1max = CAMath::Max(z1max, track1Base->MaxClusterZ());
      }
      if (track2 != track2Base) {
        z2min = CAMath::Min(z2min, track2Base->MinClusterZ());
        z2max = CAMath::Max(z2max, track2Base->MaxClusterZ());
      }

      bool goUp = z2max - z1min > z1max - z2min;

      if (track1->Neighbour(goUp) < 0 && track2->Neighbour(!goUp) < 0) {
        track1->SetNeighbor(track2 - mSliceTrackInfos, goUp);
        track2->SetNeighbor(track1 - mSliceTrackInfos, !goUp);
        // GPUInfo("Result (simple neighbor)");
        // PrintMergeGraph(track1);
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
    // PrintMergeGraph(track1);
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
  if (mCAParam->rec.NonConsecutiveIDs) {
    return;
  }

#ifdef GPUCA_MERGER_CE_ROWLIMIT
  if (cls.row < GPUCA_MERGER_CE_ROWLIMIT || cls.row >= GPUCA_ROW_COUNT - MERGE_CE_ROWLIMIT) {
    return;
  }

#endif
  if (!mCAParam->ContinuousTracking && fabsf(cls.z) > 10) {
    return;
  }
  int slice = track->Slice();
  for (int attempt = 0; attempt < 2; attempt++) {
    GPUTPCGMBorderTrack& b = attempt == 0 ? mBorder[slice][mBorderCETracks[0][slice]] : mBorder[slice][mkSlices[slice]->NTracks() - 1 - mBorderCETracks[1][slice]];
    const float x0 = attempt == 0 ? mCAParam->tpcGeometry.Row2X(63) : cls.x;
    if (track->TransportToX(x0, mCAParam->ConstBz, b, GPUCA_MAX_SIN_PHI_LOW)) {
      b.SetTrackID(itr);
      b.SetNClusters(mOutputTracks[itr].NClusters());
      if (fabsf(b.Cov()[4]) >= 0.5) {
        b.SetCov(4, 0.5); // TODO: Is this needed and better than the cut in BorderTrack?
      }
      if (track->CSide()) {
        b.SetPar(1, b.Par()[1] - 2 * (cls.z - b.ZOffset()));
        b.SetZOffset(-b.ZOffset());
      }
      if (attempt) {
        b.SetRow(cls.row);
      }
      mBorderCETracks[attempt][slice]++;
      break;
    }
  }
}

void GPUTPCGMMerger::MergeCE()
{
  ClearTrackLinks(mNOutputTracks);
  for (int iSlice = 0; iSlice < NSLICES / 2; iSlice++) {
    int jSlice = iSlice + NSLICES / 2;
    MergeBorderTracks(iSlice, mBorder[iSlice], mBorderCETracks[0][iSlice], jSlice, mBorder[jSlice], mBorderCETracks[0][jSlice], 1);
    MergeBorderTracks(iSlice, mBorder[iSlice] + mkSlices[iSlice]->NTracks() - mBorderCETracks[1][iSlice], mBorderCETracks[1][iSlice], jSlice, mBorder[jSlice] + mkSlices[jSlice]->NTracks() - mBorderCETracks[1][jSlice], mBorderCETracks[1][jSlice], 2);
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
      bool needswap = false;
      if (looper) {
        const float z0max = CAMath::Max(fabsf(mClusters[trk[0]->FirstClusterRef()].z), fabsf(mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z));
        const float z1max = CAMath::Max(fabsf(mClusters[trk[1]->FirstClusterRef()].z), fabsf(mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z));
        if (z1max < z0max) {
          needswap = true;
        }
      } else {
        if (mClusters[trk[0]->FirstClusterRef()].x > mClusters[trk[1]->FirstClusterRef()].x) {
          needswap = true;
        }
      }
      if (needswap) {
        std::swap(trk[0], trk[1]);
      }

      bool reverse[2] = {false, false};
      if (looper) {
        reverse[0] = (mClusters[trk[0]->FirstClusterRef()].z > mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) ^ (trk[0]->CSide() > 0);
        reverse[1] = (mClusters[trk[1]->FirstClusterRef()].z < mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) ^ (trk[1]->CSide() > 0);
      }

      if (mCAParam->ContinuousTracking) {
        const float z0 = trk[0]->CSide() ? CAMath::Max(mClusters[trk[0]->FirstClusterRef()].z, mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z) : CAMath::Min(mClusters[trk[0]->FirstClusterRef()].z, mClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].z);
        const float z1 = trk[1]->CSide() ? CAMath::Max(mClusters[trk[1]->FirstClusterRef()].z, mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z) : CAMath::Min(mClusters[trk[1]->FirstClusterRef()].z, mClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].z);
        float offset = fabsf(z1) > fabsf(z0) ? -z0 : z1;
        trk[1]->Param().Z() += trk[1]->Param().ZOffset() - offset;
        trk[1]->Param().ZOffset() = offset;
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
      trk[1]->SetCCE(true);
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
  const GPUTPCSliceOutCluster* const cmp1;
  const clcomparestruct* const cmp2;
  GPUTPCGMMerger_CompareClusterIdsLooper(unsigned char l, bool o, const GPUTPCSliceOutCluster* c1, const clcomparestruct* c2) : leg(l), outwards(o), cmp1(c1), cmp2(c2) {}
  bool operator()(const int aa, const int bb)
  {
    const clcomparestruct& a = cmp2[aa];
    const clcomparestruct& b = cmp2[bb];
    const GPUTPCSliceOutCluster& a1 = cmp1[aa];
    const GPUTPCSliceOutCluster& b1 = cmp1[bb];
    if (a.leg != b.leg) {
      return ((leg > 0) ^ (a.leg > b.leg));
    }
    if (a1.GetX() != b1.GetX()) {
      return ((a1.GetX() > b1.GetX()) ^ ((a.leg - leg) & 1) ^ outwards);
    }
    return a1.GetId() > b1.GetId();
  }
};

struct GPUTPCGMMerger_CompareClusterIds {
  const GPUTPCSliceOutCluster* const mCmp;
  GPUTPCGMMerger_CompareClusterIds(const GPUTPCSliceOutCluster* cmp) : mCmp(cmp) {}
  bool operator()(const int aa, const int bb)
  {
    const GPUTPCSliceOutCluster& a = mCmp[aa];
    const GPUTPCSliceOutCluster& b = mCmp[bb];
    if (a.GetX() != b.GetX()) {
      return (a.GetX() > b.GetX());
    }
    return (a.GetId() > b.GetId());
  }
};

struct GPUTPCGMMerger_CompareTracks {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUTPCGMMerger_CompareTracks(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  bool operator()(const int aa, const int bb)
  {
    const GPUTPCGMMergedTrack& a = mCmp[aa];
    const GPUTPCGMMergedTrack& b = mCmp[bb];
    return (fabsf(a.GetParam().GetQPt()) > fabsf(b.GetParam().GetQPt()));
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

    GPUTPCSliceOutCluster trackClusters[kMaxClusters];
    uchar2 clA[kMaxClusters];
    nHits = 0;
    for (int ipart = 0; ipart < nParts; ipart++) {
      const GPUTPCGMSliceTrack* t = trackParts[ipart];
      CADEBUG(printf("Collect Track %d Part %d QPt %f DzDs %f\n", mNOutputTracks, ipart, t->QPt(), t->DzDs()));
      int nTrackHits = t->NClusters();
      const GPUTPCSliceOutCluster* c = t->OrigTrack()->Clusters();
      GPUTPCSliceOutCluster* c2 = trackClusters + nHits + nTrackHits - 1;
      for (int i = 0; i < nTrackHits; i++, c++, c2--) {
        *c2 = *c;
        clA[nHits].x = t->Slice();
        clA[nHits++].y = t->Leg();
      }
    }
    if (nHits < GPUCA_TRACKLET_SELECTOR_MIN_HITS(track.QPt())) {
      continue;
    }

    int ordered = leg == 0;
    if (ordered) {
      for (int i = 1; i < nHits; i++) {
        if (trackClusters[i].GetX() > trackClusters[i - 1].GetX() || trackClusters[i].GetId() == trackClusters[i - 1].GetId()) {
          ordered = 0;
          break;
        }
      }
    }
    int firstTrackIndex = 0;
    int lastTrackIndex = nParts - 1;
    if (ordered == 0) {
      int nTmpHits = 0;
      GPUTPCSliceOutCluster trackClustersUnsorted[kMaxClusters];
      uchar2 clAUnsorted[kMaxClusters];
      int clusterIndices[kMaxClusters];
      for (int i = 0; i < nHits; i++) {
        trackClustersUnsorted[i] = trackClusters[i];
        clAUnsorted[i] = clA[i];
        clusterIndices[i] = i;
      }

      if (leg > 0) {
        // Find QPt and DzDs for the segment closest to the vertex, if low/mid Pt
        float baseZ = 1e9;
        unsigned char baseLeg = 0;
        const float factor = trackParts[0]->CSide() ? -1.f : 1.f;
        for (int i = 0; i < nParts; i++) {
          if (trackParts[i]->Leg() == 0 || trackParts[i]->Leg() == leg) {
            float z = CAMath::Min(trackParts[i]->OrigTrack()->Clusters()[0].GetZ() * factor, trackParts[i]->OrigTrack()->Clusters()[trackParts[i]->OrigTrack()->NClusters() - 1].GetZ() * factor);
            if (z < baseZ) {
              baseZ = z;
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
          if (trackParts[i]->OrigTrack()->NClusters() > length) {
            iLongest = i;
            length = trackParts[i]->OrigTrack()->NClusters();
          }
        }
        bool outwards = (trackParts[iLongest]->OrigTrack()->Clusters()[0].GetZ() > trackParts[iLongest]->OrigTrack()->Clusters()[trackParts[iLongest]->OrigTrack()->NClusters() - 1].GetZ()) ^ trackParts[iLongest]->CSide();

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
        if (indPrev >= 0 && trackClustersUnsorted[ind].GetId() == trackClustersUnsorted[indPrev].GetId()) {
          continue;
        }
        indPrev = ind;
        trackClusters[nFilteredHits] = trackClustersUnsorted[ind];
        clA[nFilteredHits] = clAUnsorted[ind];
        nFilteredHits++;
      }
      nHits = nFilteredHits;
    }

    GPUTPCGMMergedTrackHit* cl = mClusters + nOutTrackClusters;
    int* clid = mGlobalClusterIDs + nOutTrackClusters;
    for (int i = 0; i < nHits; i++) {
      cl[i].x = trackClusters[i].GetX();
      cl[i].y = trackClusters[i].GetY();
      cl[i].z = trackClusters[i].GetZ();
      cl[i].row = trackClusters[i].GetRow();
      if (!mCAParam->rec.NonConsecutiveIDs) // We already have global consecutive numbers from the slice tracker, and we need to keep them for late cluster attachment
      {
        cl[i].num = trackClusters[i].GetId();
      } else { // Produce consecutive numbers for shared cluster flagging
        cl[i].num = nOutTrackClusters + i;
        clid[i] = trackClusters[i].GetId();
      }
      cl[i].amp = trackClusters[i].GetAmp();
      cl[i].state = trackClusters[i].GetFlags() & GPUTPCGMMergedTrackHit::hwcmFlags; // Only allow edge and deconvoluted flags
      cl[i].slice = clA[i].x;
      cl[i].leg = clA[i].y;
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
      cl[i].pad = trackClusters[i].mPad;
      cl[i].time = trackClusters[i].mTime;
#endif
    }

    GPUTPCGMMergedTrack& mergedTrack = mOutputTracks[mNOutputTracks];
    mergedTrack.SetFlags(0);
    mergedTrack.SetOK(1);
    mergedTrack.SetLooper(leg > 0);
    mergedTrack.SetNClusters(nHits);
    mergedTrack.SetFirstClusterRef(nOutTrackClusters);
    GPUTPCGMTrackParam& p1 = mergedTrack.Param();
    const GPUTPCGMSliceTrack& p2 = *trackParts[firstTrackIndex];
    mergedTrack.SetCSide(p2.CSide());

    GPUTPCGMBorderTrack b;
    if (p2.TransportToX(cl[0].x, mCAParam->ConstBz, b, GPUCA_MAX_SIN_PHI, false)) {
      p1.X() = cl[0].x;
      p1.Y() = b.Par()[0];
      p1.Z() = b.Par()[1];
      p1.SinPhi() = b.Par()[2];
    } else {
      p1.X() = p2.X();
      p1.Y() = p2.Y();
      p1.Z() = p2.Z();
      p1.SinPhi() = p2.SinPhi();
    }
    p1.ZOffset() = p2.ZOffset();
    p1.DzDs() = p2.DzDs();
    p1.QPt() = p2.QPt();
    mergedTrack.SetAlpha(p2.Alpha());

    // if (nParts > 1) printf("Merged %d: QPt %f %d parts %d hits\n", mNOutputTracks, p1.QPt(), nParts, nHits);

    /*if (GPUQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(mNOutputTracks))
                {
                    mergedTrack.SetOK(0);
                    mergedTrack.SetNClusters(0);
                }*/

    bool CEside = (mergedTrack.CSide() != 0) ^ (cl[0].z > cl[nHits - 1].z);
    if (mergedTrack.NClusters() && mergedTrack.OK()) {
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
  maxId = mCAParam->rec.NonConsecutiveIDs ? mNOutputTrackClusters : mNMaxClusters;
  if (maxId > mNMaxClusters) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  mMaxID = maxId;

  unsigned int* trackSort = (unsigned int*)mTmpMem;
  unsigned char* sharedCount = (unsigned char*)(trackSort + mNOutputTracks);

  if (!mCAParam->rec.NonConsecutiveIDs) {
    for (int i = 0; i < mNOutputTracks; i++) {
      trackSort[i] = i;
    }
    std::sort(trackSort, trackSort + mNOutputTracks, GPUTPCGMMerger_CompareTracks(mOutputTracks));
    memset(mClusterAttachment, 0, maxId * sizeof(mClusterAttachment[0]));
    for (int i = 0; i < mNOutputTracks; i++) {
      mTrackOrder[trackSort[i]] = i;
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

int GPUTPCGMMerger::CheckSlices()
{
  for (int i = 0; i < NSLICES; i++) {
    if (mkSlices[i] == nullptr) {
      printf("Slice %d missing\n", i);
      return 1;
    }
  }
  return 0;
}

void GPUTPCGMMerger::Finalize()
{
  if (mCAParam->rec.NonConsecutiveIDs) {
    for (int i = 0; i < mNOutputTrackClusters; i++) {
      mClusters[i].num = mGlobalClusterIDs[i];
    }
  } else {
    int* trkOrderReverse = (int*)mTmpMem;
    for (int i = 0; i < mNOutputTracks; i++) {
      trkOrderReverse[mTrackOrder[i]] = i;
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
        int weight = mTrackOrder[i] | attachAttached;
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
  mTrackOrder = nullptr;
}
