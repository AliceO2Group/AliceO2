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

/// \file GPUTPCTrackletSelector.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCTrackletSelector.h"
#include "GPUTPCTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTracklet.h"
#include "GPUCommonMath.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCTrackletSelector::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, processorType& GPUrestrict() tracker)
{
  // select best tracklets and kill clones

  if (iThread == 0) {
    s.mNTracklets = *tracker.NTracklets();
    s.mNThreadsTotal = nThreads * nBlocks;
    s.mItr0 = nThreads * iBlock;
  }
  GPUbarrier();

  GPUTPCHitId trackHits[GPUCA_ROW_COUNT - GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE];
  const float maxSharedFrac = tracker.Param().rec.tpc.trackletMaxSharedFraction;

  for (int32_t itr = s.mItr0 + iThread; itr < s.mNTracklets; itr += s.mNThreadsTotal) {
    GPUbarrierWarp();

    GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) & GPUrestrict() tracklet = tracker.Tracklets()[itr];

    int32_t firstRow = tracklet.FirstRow();
    int32_t lastRow = tracklet.LastRow();

    const int32_t w = tracklet.HitWeight();

    int32_t irow = firstRow;

    int32_t gap = 0;
    int32_t nShared = 0;
    int32_t nHits = 0;
    const int32_t minHits = tracker.Param().rec.tpc.minNClustersTrackSeed == -1 ? GPUCA_TRACKLET_SELECTOR_MIN_HITS_B5(tracklet.Param().QPt() * tracker.Param().qptB5Scaler) : tracker.Param().rec.tpc.minNClustersTrackSeed;
    const int32_t sharingMinNorm = minHits * tracker.Param().rec.tpc.trackletMinSharedNormFactor;
    float maxShared = maxSharedFrac * sharingMinNorm;

    GPUCA_UNROLL(, U(1))
    for (irow = firstRow; irow <= lastRow && lastRow - irow + nHits >= minHits; irow++) {
      calink ih = tracker.TrackletRowHits()[tracklet.FirstHit() + (irow - firstRow)];
      if (ih != CALINK_DEAD_CHANNEL) {
        gap++;
      }
      if (ih != CALINK_INVAL && ih != CALINK_DEAD_CHANNEL) {
        GPUglobalref() const MEM_GLOBAL(GPUTPCRow)& row = tracker.Row(irow);
        bool own = (tracker.HitWeight(row, ih) <= w);
        bool sharedOK = nShared <= (nHits < sharingMinNorm ? maxShared : nHits * maxSharedFrac);
        if (own || sharedOK) { // SG!!!
          gap = 0;
#if GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
          if (nHits < GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE) {
            s.mHits[nHits][iThread].Set(irow, ih);
          } else
#endif // GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
          {
            trackHits[nHits - GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE].Set(irow, ih);
          }
          nHits++;
          if (!own) {
            nShared++;
          }
        }
      }

      if (gap > tracker.Param().rec.tpc.trackFollowingMaxRowGap || irow == lastRow) { // store
        if (nHits >= minHits) {
          uint32_t nFirstTrackHit = CAMath::AtomicAdd(tracker.NTrackHits(), (uint32_t)nHits);
          if (nFirstTrackHit + nHits > tracker.NMaxTrackHits()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_HIT_OVERFLOW, tracker.ISlice(), nFirstTrackHit + nHits, tracker.NMaxTrackHits());
            CAMath::AtomicExch(tracker.NTrackHits(), tracker.NMaxTrackHits());
            return;
          }
          uint32_t itrout = CAMath::AtomicAdd(tracker.NTracks(), 1u);
          if (itrout >= tracker.NMaxTracks()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_OVERFLOW, tracker.ISlice(), itrout, tracker.NMaxTracks());
            CAMath::AtomicExch(tracker.NTracks(), tracker.NMaxTracks());
            return;
          }
          tracker.Tracks()[itrout].SetLocalTrackId(itrout);
          tracker.Tracks()[itrout].SetParam(tracklet.Param());
          tracker.Tracks()[itrout].SetFirstHitID(nFirstTrackHit);
          tracker.Tracks()[itrout].SetNHits(nHits);
          for (int32_t jh = 0; jh < nHits; jh++) {
#if GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
            if (jh < GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE) {
              tracker.TrackHits()[nFirstTrackHit + jh] = s.mHits[jh][iThread];
            } else
#endif // GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
            {
              tracker.TrackHits()[nFirstTrackHit + jh] = trackHits[jh - GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE];
            }
          }
        }
        nHits = 0;
        gap = 0;
        nShared = 0;
      }
    }
  }
}
