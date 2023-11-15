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

/// \file GPUTPCGlobalTracking.h
/// \author David Rohr

#ifndef GPUTPCGLOBALTRACKING_H
#define GPUTPCGLOBALTRACKING_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
class GPUTPCTracker;

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
class GPUTPCGlobalTracking : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory {
    CA_SHARED_STORAGE(MEM_LG(GPUTPCRow) mRows[GPUCA_ROW_COUNT]);
  };

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker);

  GPUd() static int GlobalTrackingSliceOrder(int iSlice);
  GPUd() static void GlobalTrackingSliceLeftRight(unsigned int iSlice, unsigned int& left, unsigned int& right);

 private:
  GPUd() static int PerformGlobalTrackingRun(GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, const GPUTPCTracker& sliceSource, int iTrack, int rowIndex, float angle, int direction);
  GPUd() static void PerformGlobalTracking(int nBlocks, int nThreads, int iBlock, int iThread, const GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, GPUTPCTracker& sliceTarget, bool right);
};
#endif

class GPUTPCGlobalTrackingCopyNumbers : public GPUKernelTemplate
{
 public:
  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  MEM_TEMPLATE()
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker, int n);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLETCONSTRUCTOR_H
