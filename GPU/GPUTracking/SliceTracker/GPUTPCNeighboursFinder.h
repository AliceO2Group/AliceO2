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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursFinder
 *
 */
class GPUTPCNeighboursFinder : public GPUKernelTemplate
{
 public:
  MEM_CLASS_PRE()
  struct GPUSharedMemory {
    int mNHits;  // n hits
    float mUpDx; // x distance to the next row
    float mDnDx; // x distance to the previous row
    float mUpTx; // normalized x distance to the next row
    float mDnTx; // normalized x distance to the previous row
    int mIRow;   // row number
    int mIRowUp; // next row number
    int mIRowDn; // previous row number
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
    float mA1[GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
    float mA2[GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
    calink mB[GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNeighboursFinder)];
#endif
    MEM_LG(GPUTPCRow)
    mRow, mRowUp, mRowDown;
  };

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  MEM_TEMPLATE()
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCNEIGHBOURSFINDER_H
