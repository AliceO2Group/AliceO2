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

/// \file CfUtils.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CF_UTILS_H
#define O2_GPU_CF_UTILS_H

#include "clusterFinderDefs.h"
#include "GPUCommonAlgorithm.h"
#include "Array2D.h"
#include "CfConsts.h"
#include "GPUTPCClusterFinderKernels.h"

namespace GPUCA_NAMESPACE::gpu
{

class CfUtils
{

 public:
  static GPUdi() bool isAtEdge(const ChargePos& pos, tpccf::GlobalPad padsPerRow)
  {
    return (pos.pad() < 2 || pos.pad() >= padsPerRow - 2);
  }

  static GPUdi() bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << cfconsts::OuterToInner[outerIdx]);
  }

  static GPUdi() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << cfconsts::OuterToInnerInv[outerIdx]);
  }

  static GPUdi() bool isPeak(uchar peak) { return peak & 0x01; }

  static GPUdi() bool isAboveThreshold(uchar peak) { return peak >> 1; }

  static GPUdi() int warpPredicateScan(int pred, int* sum)
  {
#ifdef __HIPCC__
    int iLane = hipThreadIdx_x % warpSize;
    uint64_t waveMask = __ballot(pred);
    uint64_t lowerWarpMask = (1ull << iLane) - 1ull;
    int myOffset = __popcll(waveMask & lowerWarpMask);
    *sum = __popcll(waveMask);
    return myOffset;
#elif defined(__CUDACC__)
    int iLane = threadIdx.x % warpSize;
    uint32_t waveMask = __ballot_sync(0xFFFFFFFF, pred);
    uint32_t lowerWarpMask = (1u << iLane) - 1u;
    int myOffset = __popc(waveMask & lowerWarpMask);
    *sum = __popc(waveMask);
    return myOffset;
#else // CPU / OpenCL fallback
    int myOffset = warp_scan_inclusive_add(pred ? 1 : 0);
    *sum = warp_broadcast(myOffset, GPUCA_WARP_SIZE - 1);
    myOffset--;
    return myOffset;
#endif
  }

  template <size_t BlockSize, typename SharedMemory>
  static GPUdi() int blockPredicateScan(SharedMemory& smem, int pred, int* sum = nullptr)
  {
#if defined(__HIPCC__) || defined(__CUDACC__)
    int iThread =
#ifdef __HIPCC__
      hipThreadIdx_x;
#else
      threadIdx.x;
#endif

    int iWarp = iThread / warpSize;
    int nWarps = BlockSize / warpSize;

    int warpSum;
    int laneOffset = warpPredicateScan(pred, &warpSum);

    if (iThread % warpSize == 0) {
      smem.warpPredicateSum[iWarp] = warpSum;
    }
    __syncthreads();

    int warpOffset = 0;

    if (sum == nullptr) {
      for (int w = 0; w < iWarp; w++) {
        int s = smem.warpPredicateSum[w];
        warpOffset += s;
      }
    } else {
      *sum = 0;
      for (int w = 0; w < nWarps; w++) {
        int s = smem.warpPredicateSum[w];
        if (w < iWarp) {
          warpOffset += s;
        }
        *sum += s;
      }
    }

    return warpOffset + laneOffset;
#else // CPU / OpenCL fallback
    int lpos = work_group_scan_inclusive_add(pred ? 1 : 0);
    if (sum != nullptr) {
      *sum = work_group_broadcast(lpos, BlockSize - 1);
    }
    lpos--;
    return lpos;
#endif
  }

  template <size_t BlockSize, typename SharedMemory>
  static GPUdi() int blockPredicateSum(SharedMemory& smem, int pred)
  {
#if defined(__HIPCC__) || defined(__CUDACC__)
    int iThread =
#ifdef __HIPCC__
      hipThreadIdx_x;
#else
      threadIdx.x;
#endif

    int iWarp = iThread / warpSize;
    int nWarps = BlockSize / warpSize;

    int warpSum =
#ifdef __HIPCC__
      __popcll(__ballot(pred));
#else
      __popc(__ballot_sync(0xFFFFFFFF, pred));
#endif

    if (iThread % warpSize == 0) {
      smem.warpPredicateSum[iWarp] = warpSum;
    }
    __syncthreads();

    int sum = 0;
    for (int w = 0; w < nWarps; w++) {
      sum += smem.warpPredicateSum[w];
    }

    return sum;
#else // CPU / OpenCL fallback
    return work_group_reduce_add(pred ? 1 : 0);
#endif
  }

  template <size_t SCRATCH_PAD_WORK_GROUP_SIZE, typename SharedMemory>
  static GPUdi() ushort partition(SharedMemory& smem, ushort ll, bool pred, ushort partSize, ushort* newPartSize)
  {
    bool participates = ll < partSize;

    int part;
    int lpos = blockPredicateScan<SCRATCH_PAD_WORK_GROUP_SIZE>(smem, int(!pred && participates), &part);

    ushort pos = (participates && !pred) ? lpos : part;

    *newPartSize = part;
    return pos;
  }

  template <typename T>
  static GPUdi() void blockLoad(
    const Array2D<T>& map,
    uint wgSize,
    uint elems,
    ushort ll,
    uint offset,
    uint N,
    GPUconstexprref() const tpccf::Delta2* neighbors,
    const ChargePos* posBcast,
    GPUgeneric() T* buf)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort x = ll % N;
    ushort y = ll / N;
    tpccf::Delta2 d = neighbors[x + offset];

    for (unsigned int i = y; i < wgSize; i += (elems / N)) {
      ChargePos readFrom = posBcast[i];
      uint writeTo = N * i + x;
      buf[writeTo] = map[readFrom.delta(d)];
    }
    GPUbarrier();
#else
    if (ll >= wgSize) {
      return;
    }

    ChargePos readFrom = posBcast[ll];

    GPUbarrier();

    for (unsigned int i = 0; i < N; i++) {
      tpccf::Delta2 d = neighbors[i + offset];

      uint writeTo = N * ll + i;
      buf[writeTo] = map[readFrom.delta(d)];
    }

    GPUbarrier();
#endif
  }

  template <typename T, bool Inv = false>
  static GPUdi() void condBlockLoad(
    const Array2D<T>& map,
    ushort wgSize,
    ushort elems,
    ushort ll,
    ushort offset,
    ushort N,
    GPUconstexprref() const tpccf::Delta2* neighbors,
    const ChargePos* posBcast,
    const uchar* aboveThreshold,
    GPUgeneric() T* buf)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort y = ll / N;
    ushort x = ll % N;
    tpccf::Delta2 d = neighbors[x + offset];
    for (unsigned int i = y; i < wgSize; i += (elems / N)) {
      ChargePos readFrom = posBcast[i];
      uchar above = aboveThreshold[i];
      uint writeTo = N * i + x;
      T v(0);
      bool cond = (Inv) ? innerAboveThresholdInv(above, x + offset)
                        : innerAboveThreshold(above, x + offset);
      if (cond) {
        v = map[readFrom.delta(d)];
      }
      buf[writeTo] = v;
    }
    GPUbarrier();
#else
    if (ll >= wgSize) {
      return;
    }

    ChargePos readFrom = posBcast[ll];
    uchar above = aboveThreshold[ll];
    GPUbarrier();

    for (unsigned int i = 0; i < N; i++) {
      tpccf::Delta2 d = neighbors[i + offset];

      uint writeTo = N * ll + i;
      T v(0);
      bool cond = (Inv) ? innerAboveThresholdInv(above, i + offset)
                        : innerAboveThreshold(above, i + offset);
      if (cond) {
        v = map[readFrom.delta(d)];
      }
      buf[writeTo] = v;
    }

    GPUbarrier();
#endif
  }
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
