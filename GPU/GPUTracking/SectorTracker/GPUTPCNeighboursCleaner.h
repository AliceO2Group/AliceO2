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

/// \file GPUTPCNeighboursCleaner.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCNEIGHBOURSCLEANER_H
#define GPUTPCNEIGHBOURSCLEANER_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursCleaner
 *
 */
class GPUTPCNeighboursCleaner : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory {
    uint32_t mIRow;   // current row index
    uint32_t mIRowUp; // current row index
    uint32_t mIRowDn; // current row index
    uint32_t mNHits;  // number of hits
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t /*nBlocks*/, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);
};
} // namespace o2::gpu

#endif // GPUTPCNEIGHBOURSCLEANER_H
