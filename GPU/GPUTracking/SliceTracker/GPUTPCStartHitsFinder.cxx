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

/// \file GPUTPCStartHitsFinder.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCStartHitsFinder.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCStartHitsFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, processorType& GPUrestrict() tracker)
{
  // find start hits for tracklets
  if (iThread == 0) {
    s.mIRow = iBlock + 1;
    s.mNRowStartHits = 0;
    if (s.mIRow <= GPUCA_ROW_COUNT - 4) {
      s.mNHits = tracker.mData.mRows[s.mIRow].mNHits;
    } else {
      s.mNHits = -1;
    }
  }
  GPUbarrier();
  GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.mData.mRows[s.mIRow];
  GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowUp = tracker.mData.mRows[s.mIRow + 2];
  for (int ih = iThread; ih < s.mNHits; ih += nThreads) {
    long int lHitNumberOffset = row.mHitNumberOffset;
    unsigned int linkUpData = tracker.mData.mLinkUpData[lHitNumberOffset + ih];

    if (tracker.mData.mLinkDownData[lHitNumberOffset + ih] == CALINK_INVAL && linkUpData != CALINK_INVAL && tracker.mData.mLinkUpData[rowUp.mHitNumberOffset + linkUpData] != CALINK_INVAL) {
#ifdef GPUCA_SORT_STARTHITS
      GPUglobalref() GPUTPCHitId* const GPUrestrict() startHits = tracker.mTrackletTmpStartHits + s.mIRow * tracker.mNMaxRowStartHits;
      unsigned int nextRowStartHits = CAMath::AtomicAddShared(&s.mNRowStartHits, 1u);
      if (nextRowStartHits >= tracker.mNMaxRowStartHits) {
        tracker.raiseError(GPUErrors::ERROR_ROWSTARTHIT_OVERFLOW, tracker.ISlice() * 1000 + s.mIRow, nextRowStartHits, tracker.mNMaxRowStartHits);
        CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, tracker.mNMaxRowStartHits);
        break;
      }
#else
      GPUglobalref() GPUTPCHitId* const GPUrestrict() startHits = tracker.mTrackletStartHits;
      unsigned int nextRowStartHits = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, 1u);
      if (nextRowStartHits >= tracker.mNMaxStartHits) {
        tracker.raiseError(GPUErrors::ERROR_STARTHIT_OVERFLOW, tracker.ISlice() * 1000 + s.mIRow, nextRowStartHits, tracker.mNMaxStartHits);
        CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, tracker.mNMaxStartHits);
        break;
      }
#endif
      startHits[nextRowStartHits].Set(s.mIRow, ih);
    }
  }
  GPUbarrier();

#ifdef GPUCA_SORT_STARTHITS
  if (iThread == 0) {
    unsigned int nOffset = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, s.mNRowStartHits);
    tracker.mRowStartHitCountOffset[s.mIRow] = s.mNRowStartHits;
    if (nOffset + s.mNRowStartHits > tracker.mNMaxStartHits) {
      tracker.raiseError(GPUErrors::ERROR_STARTHIT_OVERFLOW, tracker.ISlice() * 1000 + s.mIRow, nOffset + s.mNRowStartHits, tracker.mNMaxStartHits);
      CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, tracker.mNMaxStartHits);
    }
  }
#endif
}
