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

/// \file GPUGeneralKernels.h
/// \author David Rohr

#ifndef GPUGENERALKERNELS_H
#define GPUGENERALKERNELS_H

#include "GPUDef.h"
#include "GPUDataTypesIO.h"
#include "GPUDataTypesConfig.h"

#if defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_COMPILEKERNELS) && !defined(GPUCA_GPUCODE_HOSTONLY)
#if defined(__CUDACC__)
#include <cub/cub.cuh>
#elif defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
#endif
#endif

#if defined(__HIPCC__)
#define GPUCA_CUB_NAMESPACE hipcub
#else
#define GPUCA_CUB_NAMESPACE cub
#endif

namespace o2::gpu
{
struct GPUConstantMem;

class GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           step0 = 0,
           step1 = 1,
           step2 = 2,
           step3 = 3,
           step4 = 4,
           step5 = 5 };

  struct GPUSharedMemory {
  };

  template <class T, int32_t I>
  struct GPUSharedMemoryWarpScan64 {
    // Provides the shared memory resources for warp wide CUB collectives
#if (defined(__CUDACC__) || defined(__HIPCC__)) && defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_HOSTONLY)
    typedef GPUCA_CUB_NAMESPACE::WarpScan<T> WarpScan;
    union {
      typename WarpScan::TempStorage cubWarpTmpMem;
    };
#endif
  };

  template <class T, int32_t I>
  struct GPUSharedMemoryScan64 {
    // Provides the shared memory resources for CUB collectives
#if (defined(__CUDACC__) || defined(__HIPCC__)) && defined(GPUCA_GPUCODE) && !defined(GPUCA_GPUCODE_HOSTONLY)
    typedef GPUCA_CUB_NAMESPACE::BlockScan<T, I> BlockScan;
    typedef GPUCA_CUB_NAMESPACE::BlockReduce<T, I> BlockReduce;
    typedef GPUCA_CUB_NAMESPACE::WarpScan<T> WarpScan;
    union {
      typename BlockScan::TempStorage cubTmpMem;
      typename BlockReduce::TempStorage cubReduceTmpMem;
      typename WarpScan::TempStorage cubWarpTmpMem;
      int32_t tmpBroadcast;
      int32_t warpPredicateSum[I / GPUCA_WARP_SIZE];
    };
#endif
  };

  typedef GPUconstantref() GPUConstantMem processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::NoRecoStep; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return &processors;
  }
  template <int32_t iKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, Args... args)
  {
  }
};

// Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class GPUMemClean16 : public GPUKernelTemplate
{
 public:
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::NoRecoStep; }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUglobalref() void* ptr, uint64_t size);
};

// Fill with incrementing sequnce of integers
class GPUitoa : public GPUKernelTemplate
{
 public:
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::NoRecoStep; }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUglobalref() int32_t* ptr, uint64_t size);
};

} // namespace o2::gpu

#undef GPUCA_CUB_NAMESPACE

#endif
