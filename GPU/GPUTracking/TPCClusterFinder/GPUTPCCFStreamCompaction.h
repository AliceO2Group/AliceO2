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

/// \file StreamCompaction.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_STREAM_COMPACTION_H
#define O2_GPU_STREAM_COMPACTION_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"

namespace o2::gpu
{

class GPUTPCCFStreamCompaction : public GPUKernelTemplate
{

 public:
  enum K : int32_t {
    scanStart = 0,
    scanUp = 1,
    scanTop = 2,
    scanDown = 3,
    compactDigits = 4,
  };

#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)
  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int32_t, GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE> {
  };
  static_assert(GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE == GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFStreamCompaction_scanStart));
  static_assert(GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE == GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFStreamCompaction_scanUp));
  static_assert(GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE == GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFStreamCompaction_scanTop));
  static_assert(GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE == GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFStreamCompaction_scanDown));
  static_assert(GPUCA_PAR_CF_SCAN_WORKGROUP_SIZE == GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFStreamCompaction_compactDigits));
#endif

  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }

  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep()
  {
    return gpudatatypes::RecoStep::TPCClusterFinding;
  }

  template <int32_t iKernel = GPUKernelTemplate::defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

 private:
  static GPUd() int32_t CompactionElems(processorType& clusterer, int32_t stage);
};

} // namespace o2::gpu

#endif
