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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUCommonAlgorithm
{
 public:
  template <class T>
  GPUd() static void sort(T* begin, T* end);

 private:
  template <class T>
  GPUd() static void SortSwap(T* v1, T* v2);
  template <class T>
  GPUd() static T* QuicksortPartition(T* left, T* right);
  template <class T>
  GPUd() static void Quicksort(T* left, T* right);

  template <class T>
  GPUd() static void Insertionsort(T* left, T* right);
};

template <class T>
GPUdi() void GPUCommonAlgorithm::SortSwap(T* v1, T* v2)
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

typedef GPUCommonAlgorithm CAAlgo;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
