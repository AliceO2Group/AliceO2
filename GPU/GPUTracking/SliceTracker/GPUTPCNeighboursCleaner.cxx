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

/// \file GPUTPCNeighboursCleaner.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCNeighboursCleaner.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"
using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCNeighboursCleaner::Thread<0>(int32_t /*nBlocks*/, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, processorType& GPUrestrict() tracker)
{
  // *
  // * kill link to the neighbour if the neighbour is not pointed to the cluster
  // *

  if (iThread == 0) {
    s.mIRow = iBlock + 2;
    if (s.mIRow <= GPUCA_ROW_COUNT - 3) {
      s.mIRowUp = s.mIRow + 2;
      s.mIRowDn = s.mIRow - 2;
      s.mNHits = tracker.Row(s.mIRow).NHits();
    }
  }
  GPUbarrier();

  if (s.mIRow <= GPUCA_ROW_COUNT - 3) {
#ifdef GPUCA_GPUCODE
    int32_t Up = s.mIRowUp;
    int32_t Dn = s.mIRowDn;
    GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.Row(s.mIRow);
    GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowUp = tracker.Row(Up);
    GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowDn = tracker.Row(Dn);
#else
    const GPUTPCRow& GPUrestrict() row = tracker.Row(s.mIRow);
    const GPUTPCRow& GPUrestrict() rowUp = tracker.Row(s.mIRowUp);
    const GPUTPCRow& GPUrestrict() rowDn = tracker.Row(s.mIRowDn);
#endif

    // - look at up link, if it's valid but the down link in the row above doesn't link to us remove
    //   the link
    // - look at down link, if it's valid but the up link in the row below doesn't link to us remove
    //   the link
    for (int32_t ih = iThread; ih < s.mNHits; ih += nThreads) {
      calink up = tracker.HitLinkUpData(row, ih);
      if (up != CALINK_INVAL) {
        calink upDn = tracker.HitLinkDownData(rowUp, up);
        if ((upDn != (calink)ih)) {
          tracker.SetHitLinkUpData(row, ih, CALINK_INVAL);
        }
      }
      calink dn = tracker.HitLinkDownData(row, ih);
      if (dn != CALINK_INVAL) {
        calink dnUp = tracker.HitLinkUpData(rowDn, dn);
        if (dnUp != (calink)ih) {
          tracker.SetHitLinkDownData(row, ih, CALINK_INVAL);
        }
      }
    }
  }
}
