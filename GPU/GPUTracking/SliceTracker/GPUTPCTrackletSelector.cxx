// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
GPUdii() void GPUTPCTrackletSelector::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() s, processorType& GPUrestrict() tracker)
{
  // select best tracklets and kill clones

  if (iThread == 0) {
    s.mNTracklets = *tracker.NTracklets();
    s.mNThreadsTotal = nThreads * nBlocks;
    s.mItr0 = nThreads * iBlock;
  }
  GPUbarrier();

  GPUTPCHitId trackHits[GPUCA_ROW_COUNT - GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE];

  for (int itr = s.mItr0 + iThread; itr < s.mNTracklets; itr += s.mNThreadsTotal) {
    while (tracker.Tracklets()[itr].NHits() == 0) {
      itr += s.mNThreadsTotal;
      if (itr >= s.mNTracklets) {
        return;
      }
    }

    GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) & GPUrestrict() tracklet = tracker.Tracklets()[itr];
    const int kMaxRowGap = 4;
    const float kMaxShared = .1f;

    int firstRow = tracklet.FirstRow();
    int lastRow = tracklet.LastRow();

    const int w = tracklet.HitWeight();

    int irow = firstRow;

    int gap = 0;
    int nShared = 0;
    int nHits = 0;
    const int minHits = tracker.Param().rec.MinNTrackClusters == -1 ? GPUCA_TRACKLET_SELECTOR_MIN_HITS(tracklet.Param().QPt()) : tracker.Param().rec.MinNTrackClusters;

    for (irow = firstRow; irow <= lastRow && lastRow - irow + nHits >= minHits; irow++) {
      gap++;
      calink ih = tracker.TrackletRowHits()[tracklet.FirstHit() + (irow - firstRow)];
      if (ih != CALINK_INVAL) {
        GPUglobalref() const MEM_GLOBAL(GPUTPCRow)& row = tracker.Row(irow);
        bool own = (tracker.HitWeight(row, ih) <= w);
        bool sharedOK = ((nShared < nHits * kMaxShared));
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

      if (gap > kMaxRowGap || irow == lastRow) { // store
        if (nHits >= minHits) {
          unsigned int itrout = CAMath::AtomicAdd(tracker.NTracks(), 1u);
          unsigned int nFirstTrackHit = CAMath::AtomicAdd(tracker.NTrackHits(), (unsigned int)nHits);
          if (itrout >= tracker.NMaxTracks() || nFirstTrackHit + nHits > tracker.NMaxTrackHits()) {
            tracker.CommonMemory()->kernelError = (itrout >= tracker.NMaxTracks()) ? GPUCA_ERROR_TRACK_OVERFLOW : GPUCA_ERROR_TRACK_HIT_OVERFLOW;
            CAMath::AtomicExch(tracker.NTracks(), 0u);
            if (nFirstTrackHit + nHits > tracker.NMaxTrackHits()) {
              CAMath::AtomicExch(tracker.NTrackHits(), tracker.NMaxTrackHits());
            }
            return;
          }
          tracker.Tracks()[itrout].SetLocalTrackId(itrout);
          tracker.Tracks()[itrout].SetParam(tracklet.Param());
          tracker.Tracks()[itrout].SetFirstHitID(nFirstTrackHit);
          tracker.Tracks()[itrout].SetNHits(nHits);
          for (int jh = 0; jh < nHits; jh++) {
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
