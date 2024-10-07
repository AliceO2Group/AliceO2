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

  enum K : int {
    step0attached = 0,
    step1unattached = 1,
  };

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached)> {
    GPUAtomic(unsigned int) nCount;
    unsigned int lastIndex;
    unsigned int sortBuffer[GPUCA_TPC_COMP_CHUNK_SIZE];
  };

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  template <int I>
  class GPUTPCCompressionKernels_Compare
  {
   public:
    GPUhdi() GPUTPCCompressionKernels_Compare(const o2::tpc::ClusterNative* p) : mClsPtr(p) {}
    GPUd() bool operator()(unsigned int a, unsigned int b) const;

   protected:
    const o2::tpc::ClusterNative* mClsPtr;
  };
};

class GPUTPCCompressionGatherKernels : public GPUKernelTemplate
{

 public:
  enum K : int {
    unbuffered,
    buffered32,
    buffered64,
    buffered128,
    multiBlock
  };

  using Vec16 = unsigned short;
  using Vec32 = unsigned int;
  using Vec64 = unsigned long;
  using Vec128 = uint4;

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<unsigned int, GPUCA_GET_THREAD_COUNT(GPUCA_LB_COMPRESSION_GATHER)> {
    union {
      unsigned int warpOffset[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)];
      Vec32 buf32[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      Vec64 buf64[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      Vec128 buf128[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      struct {
        unsigned int sizes[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
        unsigned int srcOffsets[GPUCA_GET_WARP_COUNT(GPUCA_LB_COMPRESSION_GATHER)][GPUCA_WARP_SIZE];
      } unbuffered;
    };

    template <typename V>
    GPUdi() V* getBuffer(int iWarp);
  };

  template <typename Scalar, typename BaseVector>
  union CpyVector {
    enum {
      Size = sizeof(BaseVector) / sizeof(Scalar),
    };
    BaseVector all;
    Scalar elems[Size];
  };

  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  template <typename T, typename S>
  GPUdi() static bool isAlignedTo(const S* ptr);

  template <typename T>
  GPUdi() static void compressorMemcpy(GPUgeneric() T* dst, GPUgeneric() const T* src, unsigned int size, int nThreads, int iThread);

  template <typename Scalar, typename Vector>
  GPUdi() static void compressorMemcpyVectorised(Scalar* dst, const Scalar* src, unsigned int size, int nThreads, int iThread);

  template <typename T>
  GPUdi() static void compressorMemcpyBasic(T* dst, const T* src, unsigned int size, int nThreads, int iThread, int nBlocks = 1, int iBlock = 0);

  template <typename V, typename T, typename S>
  GPUdi() static void compressorMemcpyBuffered(V* buf, T* dst, const T* src, const S* nums, const unsigned int* srcOffets, unsigned int nEntries, int nLanes, int iLane, int diff = 0, size_t scaleBase1024 = 1024);

  template <typename T>
  GPUdi() static unsigned int calculateWarpOffsets(GPUSharedMemory& smem, T* nums, unsigned int start, unsigned int end, int nWarps, int iWarp, int nLanes, int iLane);

  template <typename V>
  GPUdii() static void gatherBuffered(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  GPUdii() static void gatherMulti(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
