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

#if !defined(GPUCA_GPUCODE)
//&& (!defined __cplusplus || __cplusplus < 201402L) // This would enable to custom search also on the CPU if available by the compiler, but it is not always faster, so we stick to std::sort
#include <algorithm>
#define GPUCA_ALGORITHM_STD
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
  template <class T>
  GPUd() static void sortDeviceDynamic(T* begin, T* end);
  template <class T, class S>
  GPUd() static void sort(T* begin, T* end, const S& comp);
  template <class T, class S>
  GPUd() static void sortInBlock(T* begin, T* end, const S& comp);
  template <class T, class S>
  GPUd() static void sortDeviceDynamic(T* begin, T* end, const S& comp);
  template <class T>
  GPUd() static void swap(T& a, T& b);

 private:
  // Quicksort implementation
  template <typename I>
  GPUd() static void QuickSort(I f, I l) noexcept;

  // Quicksort implementation
  template <typename I, typename Cmp>
  GPUd() static void QuickSort(I f, I l, Cmp cmp) noexcept;

  // Insertionsort implementation
  template <typename I, typename Cmp>
  GPUd() static void InsertionSort(I f, I l, Cmp cmp) noexcept;

  // Helper for Quicksort implementation
  template <typename I, typename Cmp>
  GPUd() static I MedianOf3Select(I f, I l, Cmp cmp) noexcept;

  // Helper for Quicksort implementation
  template <typename I, typename T, typename Cmp>
  GPUd() static I UnguardedPartition(I f, I l, T piv, Cmp cmp) noexcept;

  // Helper
  template <typename I>
  GPUd() static void IterSwap(I a, I b) noexcept;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#ifndef GPUCA_ALGORITHM_STD
template <typename I>
GPUdi() void GPUCommonAlgorithm::IterSwap(I a, I b) noexcept
{
  auto tmp = *a;
  *a = *b;
  *b = tmp;
}

template <typename I, typename Cmp>
GPUdi() void GPUCommonAlgorithm::InsertionSort(I f, I l, Cmp cmp) noexcept
{
  auto it0{f};
  while (it0 != l) {
    auto tmp{*it0};

    auto it1{it0};
    while (it1 != f && cmp(tmp, it1[-1])) {
      it1[0] = it1[-1];
      --it1;
    }
    it1[0] = tmp;

    ++it0;
  }
}

template <typename I, typename Cmp>
GPUdi() I GPUCommonAlgorithm::MedianOf3Select(I f, I l, Cmp cmp) noexcept
{
  auto m = f + (l - f) / 2;

  --l;

  if (cmp(*f, *m)) {
    if (cmp(*m, *l)) {
      return m;
    } else if (cmp(*f, *l)) {
      return l;
    } else {
      return f;
    }
  } else if (cmp(*f, *l)) {
    return f;
  } else if (cmp(*m, *l)) {
    return l;
  } else {
    return m;
  }
}

template <typename I, typename T, typename Cmp>
GPUdi() I GPUCommonAlgorithm::UnguardedPartition(I f, I l, T piv, Cmp cmp) noexcept
{
  do {
    while (cmp(*f, piv)) {
      ++f;
    }
    --l;
    while (cmp(piv, *l)) {
      --l;
    }

    if (l <= f) {
      return f;
    }
    IterSwap(f, l);
    ++f;
  } while (true);
}

template <typename I, typename Cmp>
GPUdi() void GPUCommonAlgorithm::QuickSort(I f, I l, Cmp cmp) noexcept
{
  if (f == l) {
    return;
  }
  using IndexType = unsigned short;

  struct pair {
    IndexType first;
    IndexType second;
  };

  struct Stack {
    pair data[11];
    unsigned char n{0};

    GPUd() void emplace(IndexType x, IndexType y)
    {
      data[n++] = {x, y};
    }
    GPUd() bool empty() const { return n == 0; }
    GPUd() pair& top() { return data[n - 1]; }
    GPUd() void pop() { --n; }
  };

  Stack s;
  s.emplace(0, l - f);
  while (!s.empty()) {
    const auto it0 = f + s.top().first;
    const auto it1 = f + s.top().second;
    s.pop();

    const auto piv = *MedianOf3Select(it0, it1, cmp);
    const auto pp = UnguardedPartition(it0, it1, piv, cmp);

    constexpr auto cutoff = 50u;
    const auto lsz = pp - it0;
    const auto rsz = it1 - pp;
    if (lsz < rsz) {
      if (rsz > cutoff) {
        s.emplace(pp - f, it1 - f);
      }
      if (lsz > cutoff) {
        s.emplace(it0 - f, pp - f);
      }
    } else {
      if (lsz > cutoff) {
        s.emplace(it0 - f, pp - f);
      }
      if (rsz > cutoff) {
        s.emplace(pp - f, it1 - f);
      }
    }
  }
  InsertionSort(f, l, cmp);
}

template <typename I>
GPUdi() void GPUCommonAlgorithm::QuickSort(I f, I l) noexcept
{
  QuickSort(f, l, [](auto&& x, auto&& y) { return x < y; });
}
#endif

typedef GPUCommonAlgorithm CAAlgo;

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#if ((defined(__CUDACC__) && !defined(__clang__)) || defined(__HIPCC__))

#include "GPUCommonAlgorithmThrust.h"

#else

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
GPUdi() void GPUCommonAlgorithm::sortDeviceDynamic(T* begin, T* end)
{
#ifndef GPUCA_GPUCODE
  GPUCommonAlgorithm::sort(begin, end);
#else
  GPUCommonAlgorithm::sortDeviceDynamic(begin, end, [](auto&& x, auto&& y) { return x < y; });
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortDeviceDynamic(T* begin, T* end, const S& comp)
{
#ifndef GPUCA_GPUCODE
  GPUCommonAlgorithm::sort(begin, end, comp);
#else
  GPUCommonAlgorithm::sortInBlock(begin, end, comp);
#endif
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // THRUST
// sort and sortInBlock below are not taken from Thrust, since our implementations are faster

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end)
{
#ifdef GPUCA_ALGORITHM_STD
  std::sort(begin, end);
#else
  QuickSort(begin, end, [](auto&& x, auto&& y) { return x < y; });
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end, const S& comp)
{
#ifdef GPUCA_ALGORITHM_STD
  std::sort(begin, end, comp);
#else
  QuickSort(begin, end, comp);
#endif
}

template <class T>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end)
{
#ifndef GPUCA_GPUCODE
  GPUCommonAlgorithm::sort(begin, end);
#else
  GPUCommonAlgorithm::sortInBlock(begin, end, [](auto&& x, auto&& y) { return x < y; });
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp)
{
#ifndef GPUCA_GPUCODE
  GPUCommonAlgorithm::sort(begin, end, comp);
#else
  int n = end - begin;
  for (int i = 0; i < n; i++) {
    for (int tIdx = get_local_id(0); tIdx < n; tIdx += get_local_size(0)) {
      int offset = i % 2;
      int curPos = 2 * tIdx + offset;
      int nextPos = curPos + 1;

      if (nextPos < n) {
        if (!comp(begin[curPos], begin[nextPos])) {
          IterSwap(&begin[curPos], &begin[nextPos]);
        }
      }
    }
    GPUbarrier();
  }
#endif
}

#ifdef GPUCA_GPUCODE
template <class T>
GPUdi() void GPUCommonAlgorithm::swap(T& a, T& b)
{
  auto tmp = a;
  a = b;
  b = tmp;
}
#else
template <class T>
GPUdi() void GPUCommonAlgorithm::swap(T& a, T& b)
{
  std::swap(a, b);
}
#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

// ----------------------------- WORK GROUP FUNCTIONS -----------------------------

#ifdef __OPENCL__
// Nothing to do, work_group functions available

#elif (defined(__CUDACC__) || defined(__HIPCC__))
// CUDA and HIP work the same way using cub, need just different header

#if !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_HOSTONLY)
#if defined(__CUDACC__)
#include <cub/cub.cuh>
#elif defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
#endif
#endif

#define work_group_scan_inclusive_add(v) work_group_scan_inclusive_add_FUNC(v, smem)
template <class T, class S>
GPUdi() T work_group_scan_inclusive_add_FUNC(T v, S& smem)
{
  typename S::BlockScan(smem.cubTmpMem).InclusiveSum(v, v);
  __syncthreads();
  return v;
}

#define work_group_broadcast(v, i) work_group_broadcast_FUNC(v, i, smem)
template <class T, class S>
GPUdi() T work_group_broadcast_FUNC(T v, int i, S& smem)
{
  if ((int)threadIdx.x == i) {
    smem.tmpBroadcast = v;
  }
  __syncthreads();
  T retVal = smem.tmpBroadcast;
  __syncthreads();
  return retVal;
}

#define work_group_reduce_add(v) work_group_reduce_add_FUNC(v, smem)
template <class T, class S>
GPUdi() T work_group_reduce_add_FUNC(T v, S& smem)
{
  v = typename S::BlockReduce(smem.cubReduceTmpMem).Sum(v);
  __syncthreads();
  v = work_group_broadcast(v, 0);
  return v;
}

#define warp_scan_inclusive_add(v) warp_scan_inclusive_add_FUNC(v, smem)
template <class T, class S>
GPUdi() T warp_scan_inclusive_add_FUNC(T v, S& smem)
{
  typename S::WarpScan(smem.cubWarpTmpMem).InclusiveSum(v, v);
  return v;
}

#else
// Trivial implementation for the CPU

template <class T>
GPUdi() T work_group_scan_inclusive_add(T v)
{
  return v;
}

template <class T>
GPUdi() T work_group_reduce_add(T v)
{
  return v;
}

template <class T>
GPUdi() T work_group_broadcast(T v, int i)
{
  return v;
}

template <class T>
GPUdi() T warp_scan_inclusive_add(T v)
{
  return v;
}

#endif

#endif
