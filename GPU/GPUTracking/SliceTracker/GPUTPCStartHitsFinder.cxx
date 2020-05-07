// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
GPUdii() void GPUTPCStartHitsFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() s, processorType& GPUrestrict() trackerX)
{
  // find start hits for tracklets
  HIPGPUconstantref() processorType& GPUrestrict() tracker = (HIPGPUconstantref() processorType&)trackerX;

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
      CONSTEXPR int errCode = GPUCA_ERROR_ROWSTARTHIT_OVERFLOW;
      if (nextRowStartHits >= tracker.mNMaxRowStartHits)
#else
      GPUglobalref() GPUTPCHitId* const GPUrestrict() startHits = tracker.mTrackletStartHits;
      unsigned int nextRowStartHits = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, 1u);
      CONSTEXPR int errCode = GPUCA_ERROR_STARTHIT_OVERFLOW;
      if (nextRowStartHits >= tracker.mNMaxStartHits)
#endif
      {
        trackerX.CommonMemory()->kernelError = errCode;
        CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, 0u);
        break;
      }
      startHits[nextRowStartHits].Set(s.mIRow, ih);
    }
  }
  GPUbarrier();

#ifdef GPUCA_SORT_STARTHITS
  if (iThread == 0) {
    unsigned int nOffset = CAMath::AtomicAdd(&tracker.mCommonMem->nStartHits, s.mNRowStartHits);
    tracker.mRowStartHitCountOffset[s.mIRow] = s.mNRowStartHits;
    if (nOffset + s.mNRowStartHits > tracker.mNMaxStartHits) {
      trackerX.CommonMemory()->kernelError = GPUCA_ERROR_STARTHIT_OVERFLOW;
      CAMath::AtomicExch(&tracker.mCommonMem->nStartHits, 0u);
    }
  }
#endif
}
