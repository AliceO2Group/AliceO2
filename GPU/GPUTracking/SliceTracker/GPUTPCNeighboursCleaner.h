// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNeighboursCleaner.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCNEIGHBOURSCLEANER_H
#define GPUTPCNEIGHBOURSCLEANER_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursCleaner
 *
 */
class GPUTPCNeighboursCleaner
{
 public:
  MEM_CLASS_PRE()
  class GPUTPCSharedMemory
  {
    friend class GPUTPCNeighboursCleaner;

   public:
#if !defined(GPUCA_GPUCODE)
    GPUTPCSharedMemory() : mIRow(0), mIRowUp(0), mIRowDn(0), mNHits(0)
    {
    }
    GPUTPCSharedMemory(const GPUTPCSharedMemory& /*dummy*/) : mIRow(0), mIRowUp(0), mIRowDn(0), mNHits(0) {}
    GPUTPCSharedMemory& operator=(const GPUTPCSharedMemory& /*dummy*/) { return *this; }
#endif //! GPUCA_GPUCODE

   protected:
    int mIRow;   // current row index
    int mIRowUp; // current row index
    int mIRowDn; // current row index
    int mNHits;  // number of hits
  };

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  MEM_TEMPLATE()
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = 0>
  GPUd() static void Thread(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & smem, processorType& tracker);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCNEIGHBOURSCLEANER_H
