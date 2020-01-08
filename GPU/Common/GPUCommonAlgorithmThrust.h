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

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end)
{
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(thrust::seq, thrustBegin, thrustEnd);
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end, const S& comp)
{
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(thrust::seq, thrustBegin, thrustEnd, comp);
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
    thrust::sort(thrust::hip::par, thrustBegin, thrustEnd);
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
    thrust::sort(thrust::hip::par, thrustBegin, thrustEnd, comp);
#endif
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
