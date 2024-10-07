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

/// \file GPUTPCCompressionKernels.h
/// \author David Rohr

#ifndef GPUTPCCONMPRESSIONKERNELS_H
#define GPUTPCCONMPRESSIONKERNELS_H

#include "GPUGeneralKernels.h"

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{
class GPUTPCCompressionKernels : public GPUKernelTemplate
{
 public:
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCCompression; }

  enum K : int32_t {
    step0attached = 0,
    step1unattached = 1,
  };

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int32_t, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached)> {
    GPUAtomic(uint32_t) nCount;
    uint32_t lastIndex;
    uint32_t sortBuffer[GPUCA_TPC_COMP_CHUNK_SIZE];
  };

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  template <int32_t I>
  class GPUTPCCompressionKernels_Compare
  {
   public:
    GPUhdi() GPUTPCCompressionKernels_Compare(const o2::tpc::ClusterNative* p) : mClsPtr(p) {}
    GPUd() bool operator()(uint32_t a, uint32_t b) const;

   protected:
    const o2::tpc::ClusterNative* mClsPtr;
  };
};

class GPUTPCCompressionGatherKernels : public GPUKernelTemplate
{

 public:
  enum K : int32_t {
    unbuffered,
    buffered32,
    buffered64,
    buffered128,
    multiBlock
  };

  using Vec16 = uint16_t;
  using Vec32 = uint32_t;
  using Vec64 = uint64_t;
  using Vec128 = uint4;

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<uint32_t, GPUCA_GET_THREAD_COUNT(GPUCA_LB_COMPRESSION_GATHER)> {
    union {
      uint32_t warpOffset[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)];
      Vec32 buf32[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      Vec64 buf64[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      Vec128 buf128[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      struct {
        uint32_t sizes[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
        uint32_t srcOffsets[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      } unbuffered;
    };

    template <typename V>
    GPUdi() V* getBuffer(int32_t iWarp);
  };

  template <typename Scalar, typename BaseVector>
  union CpyVector {
    enum {
      Size = sizeof(BaseVector) / sizeof(Scalar),
    };
    BaseVector all;
    Scalar elems[Size];
  };

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  template <typename T, typename S>
  GPUdi() static bool isAlignedTo(const S* ptr);

  template <typename T>
  GPUdi() static void compressorMemcpy(GPUgeneric() T* dst, GPUgeneric() const T* src, uint32_t size, int32_t nThreads, int32_t iThread);

  template <typename Scalar, typename Vector>
  GPUdi() static void compressorMemcpyVectorised(Scalar* dst, const Scalar* src, uint32_t size, int32_t nThreads, int32_t iThread);

  template <typename T>
  GPUdi() static void compressorMemcpyBasic(T* dst, const T* src, uint32_t size, int32_t nThreads, int32_t iThread, int32_t nBlocks = 1, int32_t iBlock = 0);

  template <typename V, typename T, typename S>
  GPUdi() static void compressorMemcpyBuffered(V* buf, T* dst, const T* src, const S* nums, const uint32_t* srcOffets, uint32_t nEntries, int32_t nLanes, int32_t iLane, int32_t diff = 0, size_t scaleBase1024 = 1024);

  template <typename T>
  GPUdi() static uint32_t calculateWarpOffsets(GPUSharedMemory& smem, T* nums, uint32_t start, uint32_t end, int32_t nWarps, int32_t iWarp, int32_t nLanes, int32_t iLane);

  template <typename V>
  GPUdii() static void gatherBuffered(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  GPUdii() static void gatherMulti(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
