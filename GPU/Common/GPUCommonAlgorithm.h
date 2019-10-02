// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonAlgorithm.h
/// \author David Rohr

#ifndef GPUCOMMONALGORITHM_H
#define GPUCOMMONALGORITHM_H

#include "GPUCommonDef.h"

#if !defined(GPUCA_GPUCODE_DEVICE)
#include <algorithm>
#endif

// ----------------------------- SORTING -----------------------------

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUCommonAlgorithm
{
 public:
  template <class T>
  GPUd() static void sort(T* begin, T* end);
  template <class T>
  GPUd() static void sortInBlock(T* begin, T* end);
  template <class T, class S>
  GPUd() static void sort(T* begin, T* end, const S& comp);
  template <class T, class S>
  GPUd() static void sortInBlock(T* begin, T* end, const S& comp);

 private:
  // Quicksort implementation
  template <class T>
  GPUd() static void SortSwap(GPUgeneric() T* v1, GPUgeneric() T* v2);
  template <class T>
  GPUd() static T* QuicksortPartition(T* left, T* right);
  template <class T>
  GPUd() static void Quicksort(T* left, T* right);

  // Quicksort impkementation with comparison object
  template <class T, class S>
  GPUd() static T* QuicksortPartition(T* left, T* right, const S& comp);
  template <class T, class S>
  GPUd() static void Quicksort(T* left, T* right, const S& comp);

  // Insertionsort implementation
  template <class T>
  GPUd() static void Insertionsort(T* left, T* right);

  // Insertionsort implementation with comparison object
  template <class T, class S>
  GPUd() static void Insertionsort(T* left, T* right, const S& comp);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

namespace GPUCA_NAMESPACE
{
namespace gpu
{
template <class T>
GPUdi() void GPUCommonAlgorithm::SortSwap(GPUgeneric() T* v1, GPUgeneric() T* v2)
{
  T tmp = *v1;
  *v1 = *v2;
  *v2 = tmp;
}

template <class T>
GPUdi() T* GPUCommonAlgorithm::QuicksortPartition(T* left, T* right)
{
  T* mid = left + ((right - left) / 2);
  T pivot = *mid;
  SortSwap(mid, left);
  T* i = left + 1;
  T* j = right;
  while (i <= j) {
    while (i <= j && *i <= pivot) {
      i++;
    }
    while (i <= j && *j > pivot) {
      j--;
    }
    if (i < j) {
      SortSwap(i, j);
    }
  }
  SortSwap(i - 1, left);
  return i - 1;
}

template <class T, class S>
GPUdi() T* GPUCommonAlgorithm::QuicksortPartition(T* left, T* right, const S& comp)
{
  T* mid = left + ((right - left) / 2);
  SortSwap(mid, right);
  T* pivot = right;
  T* i = left;
  T* j = right - 1;
  while (i <= j) {
    while (i <= j && !comp(*j, *pivot)) {
      j--;
    }
    while (i <= j && comp(*i, *pivot)) {
      i++;
    }
    if (i < j) {
      SortSwap(i, j);
    }
  }
  SortSwap(j + 1, right);
  return j + 1;
}

template <class T>
GPUdi() void GPUCommonAlgorithm::Quicksort(T* left, T* right)
{
  if (left >= right) {
    return;
  }
  if (right - left <= 4) {
    Insertionsort(left, right);
    return;
  }
  T* part = QuicksortPartition(left, right);

  Quicksort(left, part - 1);
  Quicksort(part + 1, right);
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::Quicksort(T* left, T* right, const S& comp)
{
  if (left >= right) {
    return;
  }
  if (right - left <= 4) {
    Insertionsort(left, right, comp);
    return;
  }
  T* part = QuicksortPartition(left, right, comp);

  Quicksort(left, part - 1, comp);
  Quicksort(part + 1, right, comp);
}

template <class T>
GPUdi() void GPUCommonAlgorithm::Insertionsort(T* left, T* right)
{
  if (left >= right) {
    return;
  }
  while (left < right) {
    T* min = left;
    for (T* test = left + 1; test <= right; test++) {
      if (*test < *min) {
        min = test;
      }
    }
    if (min != left) {
      SortSwap(left, min);
    }
    left++;
  }
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::Insertionsort(T* left, T* right, const S& comp)
{
  if (left >= right) {
    return;
  }
  while (left < right) {
    T* min = left;
    for (T* test = left + 1; test <= right; test++) {
      if (comp(*test, *min)) {
        min = test;
      }
    }
    if (min != left) {
      SortSwap(left, min);
    }
    left++;
  }
}

typedef GPUCommonAlgorithm CAAlgo;

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#ifdef __CUDACC__
#include "GPUCommonAlgorithmCUDA.cuh"

#else

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end)
{
#ifndef GPUCA_GPUCODE_DEVICE
  std::sort(begin, end);
#elif defined(__CUDACC__)
  Quicksort(begin, end - 1);
#else
  Insertionsort(begin, end - 1);
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end, const S& comp)
{
#ifndef GPUCA_GPUCODE_DEVICE
  std::sort(begin, end, comp);
#elif defined(__CUDACC__)
  Quicksort(begin, end - 1, comp);
#else
  Insertionsort(begin, end - 1, comp);
#endif
}

template <class T>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end)
{
#ifndef GPUCA_GPUCODE_DEVICE
  GPUCommonAlgorithm::sort(begin, end);
#else
  if (get_local_id(0) == 0) {
    GPUCommonAlgorithm::sort(begin, end);
  }
  GPUbarrier();
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp)
{
#ifndef GPUCA_GPUCODE_DEVICE
  GPUCommonAlgorithm::sort(begin, end, comp);
#else
  if (get_local_id(0) == 0) {
    GPUCommonAlgorithm::sort(begin, end, comp);
  }
  GPUbarrier();
#endif
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // ifdef __CUDACC__

// ----------------------------- WORK GROUP FUNCTIONS -----------------------------

#ifdef __OPENCL__
// Nothing to do, work_group functions available
#elif defined(__CUDACC__)
#include <cub/cub.cuh>

#define work_group_scan_inclusive_add(v) work_group_scan_inclusive_add_FUNC(v, smem)
template <class T, class S>
GPUdi() T work_group_scan_inclusive_add_FUNC(T v, S& smem)
{
  S::BlockScan(smem.cubTmpMem).InclusiveSum(v, v);
  __syncthreads();
  return v;
}

#define work_group_broadcast(v, i) work_group_broadcast_FUNC(v, i, smem)
template <class T, class S>
GPUdi() T work_group_broadcast_FUNC(T v, int i, S& smem)
{
  if (threadIdx.x == i) {
    smem.tmpBroadcast = v;
  }
  __syncthreads();
  T retVal = smem.tmpBroadcast;
  __syncthreads();
  return retVal;
}
#elif defined(__HIPCC__) // BUG: THESE ARE WRONG, BUT CANNOT USE HIPCUB YET!
//#include <hipcub/hipcub.hpp> // BUG: Disabled, until hipcub c++17 issue is solved

template <class T>
GPUdi() T work_group_scan_inclusive_add(T v)
{
  return v;
}
template <class T>
GPUdi() T work_group_broadcast(T v, int i)
{
  return v;
}
#else
template <class T>
GPUdi() T work_group_scan_inclusive_add(T v)
{
  return v;
}
template <class T>
GPUdi() T work_group_broadcast(T v, int i)
{
  return v;
}
#endif

#endif
