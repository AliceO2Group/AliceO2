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

/// \file GPUTPCNeighboursFinder.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCNEIGHBOURSFINDER_H
#define GPUTPCNEIGHBOURSFINDER_H

#include "GPUTPCDef.h"
#include "GPUTPCRow.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursFinder
 *
 */
class GPUTPCNeighboursFinder : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory {
    int32_t mNHits;  // n hits
    float mUpDx;     // x distance to the next row
    float mDnDx;     // x distance to the previous row
    float mUpTx;     // normalized x distance to the next row
    float mDnTx;     // normalized x distance to the previous row
    int32_t mIRow;   // row number
    int32_t mIRowUp; // next row number
    int32_t mIRowDn; // previous row number
    static_assert(GPUCA_MAXN >= GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP);
    float mA1[GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
    float mA2[GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
    calink mB[GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
    GPUTPCRow mRow, mRowUp, mRowDown;
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);
};
} // namespace o2::gpu

#endif // GPUTPCNEIGHBOURSFINDER_H
