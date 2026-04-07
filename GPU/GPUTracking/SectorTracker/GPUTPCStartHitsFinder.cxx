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

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCStartHitsFinder::Thread<0>(int32_t /*nBlocks*/, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  // find start hits for tracklets
  if (iThread == 0) {
    s.mIRow = iBlock + 1;
    s.mNRowStartHits = 0;
    if (s.mIRow <= GPUCA_NROWS - 4) {
      s.mNHits = tracker.mData.mRows[s.mIRow].mNHits;
    } else {
      s.mNHits = -1;
    }
  }
  GPUbarrier();
  GPUglobalref() const GPUTPCRow& GPUrestrict() row = tracker.mData.mRows[s.mIRow];
  GPUglobalref() const GPUTPCRow& GPUrestrict() rowUp = tracker.mData.mRows[s.mIRow + 2];
  for (int32_t ih = iThread; ih < s.mNHits; ih += nThreads) {
    int64_t lHitNumberOffset = row.mHitNumberOffset;
    uint32_t linkUpData = tracker.mData.mLinkUpData[lHitNumberOffset + ih];

    if (tracker.mData.mLinkDownData[lHitNumberOffset + ih] == CALINK_INVAL && linkUpData != CALINK_INVAL && tracker.mData.mLinkUpData[rowUp.mHitNumberOffset + linkUpData] != CALINK_INVAL) {
      GPUglobalref() GPUTPCHitId* GPUrestrict() startHits;
      uint32_t nextRowStartHits;
      if constexpr (GPUCA_PAR_SORT_STARTHITS > 0) {
        startHits = tracker.mTrackletTmpStartHits + s.mIRow * tracker.mNMaxRowStartHits;
        nextRowStartHits = CAMath::AtomicAddShared(&s.mNRowStartHits, 1u);
        if (nextRowStartHits >= tracker.mNMaxRowStartHits) {
          tracker.raiseError(GPUErrors::ERROR_ROWSTARTHIT_OVERFLOW, tracker.ISector() * 1000 + s.mIRow, nextRowStartHits, tracker.mNMaxRowStartHits);
          CAMath::AtomicExchShared(&s.mNRowStartHits, tracker.mNMaxRowStartHits);
          break;
        }
      } else {
        startHits = tracker.mTrackletStartHits;
        nextRowStartHits = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, 1u);
        if (nextRowStartHits >= tracker.mNMaxStartHits) {
          tracker.raiseError(GPUErrors::ERROR_STARTHIT_OVERFLOW, tracker.ISector() * 1000 + s.mIRow, nextRowStartHits, tracker.mNMaxStartHits);
          CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, tracker.mNMaxStartHits);
          break;
        }
      }
      startHits[nextRowStartHits].Set(s.mIRow, ih);
    }
  }
  GPUbarrier();

  if constexpr (GPUCA_PAR_SORT_STARTHITS > 0) {
    if (iThread == 0) {
      uint32_t nOffset = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, s.mNRowStartHits);
      tracker.mRowStartHitCountOffset[s.mIRow] = s.mNRowStartHits;
      if (nOffset + s.mNRowStartHits > tracker.mNMaxStartHits) {
        tracker.raiseError(GPUErrors::ERROR_STARTHIT_OVERFLOW, tracker.ISector() * 1000 + s.mIRow, nOffset + s.mNRowStartHits, tracker.mNMaxStartHits);
        CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, tracker.mNMaxStartHits);
      }
    }
  }
}
