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

/// \file GPUTPCTrackletSelector.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLETSELECTOR_H
#define GPUTPCTRACKLETSELECTOR_H

#include "GPUTPCDef.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCTrackletSelector
 *
 */
class GPUTPCTrackletSelector : public GPUKernelTemplate
{
 public:
  MEM_CLASS_PRE()
  struct GPUSharedMemory {
    int mItr0;          // index of the first track in the block
    int mNThreadsTotal; // total n threads
    int mNTracklets;    // n of tracklets
    int mReserved;      // for alignment reasons
#if GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
    GPUTPCHitId mHits[GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCTrackletSelector)];
#endif // GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
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

#endif // GPUTPCTRACKLETSELECTOR_H
