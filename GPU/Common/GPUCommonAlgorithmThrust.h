// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonAlgorithmThrust.h
/// \author Michael Lettrich

#ifndef GPUCOMMONALGORITHMTHRUST_H
#define GPUCOMMONALGORITHMTHRUST_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop

#include "GPUCommonDef.h"

using namespace std;

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#if defined(__HIPCC__)
/*template<typename I, typename Cmp>
void shell_sort(I f, I l, Cmp cmp) {
    if (f == l) return;
    constexpr unsigned int gaps[]{701, 301, 132, 57, 23, 10, 4, 1};
    const auto n{l - f};
    for (auto &&gap : gaps) {
        for (auto i = gap; i < n; ++i) {
            auto tmp{f[i]};

            auto j{i};
            while (j >= gap && cmp(tmp, f[j - gap])) {
                f[j] = f[j - gap];
                j -= gap;
            }

            f[j] = tmp;
        }
    }
}*/
//using namespace std;

#endif

template <class T>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end)
{
#if defined(__HIPCC__)
  if (get_local_id(0) == 0) {
    GPUCommonAlgorithm::QuickSort(begin, end);
  }
#else
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(thrust::seq, thrustBegin, thrustEnd);
#endif
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end, const S& comp)
{
#if defined(__HIPCC__)
  if (get_local_id(0) == 0) {
    GPUCommonAlgorithm::QuickSort(begin, end, comp);
  }
#else
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(thrust::seq, thrustBegin, thrustEnd, comp);
#endif
}

template <class T>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end)
{
  if (get_local_id(0) == 0) {
    thrust::device_ptr<T> thrustBegin(begin);
    thrust::device_ptr<T> thrustEnd(end);
#if defined(__CUDACC__)
    thrust::sort(thrust::cuda::par, thrustBegin, thrustEnd);
#elif defined(__HIPCC__)
    GPUCommonAlgorithm::QuickSort(begin, end);
#endif
  }
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp)
{
  if (get_local_id(0) == 0) {
    thrust::device_ptr<T> thrustBegin(begin);
    thrust::device_ptr<T> thrustEnd(end);
#if defined(__CUDACC__)
    thrust::sort(thrust::cuda::par, thrustBegin, thrustEnd, comp);
#elif defined(__HIPCC__)
    GPUCommonAlgorithm::QuickSort(begin, end, comp);
#endif
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
