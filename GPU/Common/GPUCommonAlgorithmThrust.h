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

#if !defined(GPUCA_GPUCODE_GENRTC) && !defined(GPUCA_GPUCODE_HOSTONLY)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop
#endif

#include "GPUCommonDef.h"

#ifdef __CUDACC__
#define GPUCA_THRUST_NAMESPACE thrust::cuda
#else
#define GPUCA_THRUST_NAMESPACE thrust::hip
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// - Our quicksort and bubble sort implementations are faster
/*
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
    sortDeviceDynamic(begin, end);
  }
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp)
{
  if (get_local_id(0) == 0) {
    sortDeviceDynamic(begin, end, comp);
  }
}

*/

template <class T>
GPUdi() void GPUCommonAlgorithm::sortDeviceDynamic(T* begin, T* end)
{
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(GPUCA_THRUST_NAMESPACE::par, thrustBegin, thrustEnd);
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortDeviceDynamic(T* begin, T* end, const S& comp)
{
  thrust::device_ptr<T> thrustBegin(begin);
  thrust::device_ptr<T> thrustEnd(end);
  thrust::sort(GPUCA_THRUST_NAMESPACE::par, thrustBegin, thrustEnd, comp);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
