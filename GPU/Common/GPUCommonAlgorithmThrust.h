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

/// \file GPUCommonAlgorithmThrust.h
/// \author Michael Lettrich

#ifndef GPUCOMMONALGORITHMTHRUST_H
#define GPUCOMMONALGORITHMTHRUST_H

#ifndef GPUCA_GPUCODE_COMPILEKERNELS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop

#include "GPUCommonDef.h"
#include "GPUCommonHelpers.h"

#ifndef __HIPCC__ // CUDA
#include <cub/cub.cuh>
#else // HIP
#include <hipcub/hipcub.hpp>
#endif
#endif // GPUCA_GPUCODE_COMPILEKERNELS

#ifndef __HIPCC__ // CUDA
#define GPUCA_THRUST_NAMESPACE thrust::cuda
#define GPUCA_CUB_NAMESPACE cub
#else // HIP
#define GPUCA_THRUST_NAMESPACE thrust::hip
#define GPUCA_CUB_NAMESPACE hipcub
#endif

namespace o2::gpu
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
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end GPUCA_THREAD_INFO_DECL) // TODO: Try cub::BlockMergeSort
{
  if (get_local_id(0) == 0) {
    sortDeviceDynamic(begin, end);
  }
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp GPUCA_THREAD_INFO_DECL)
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

#ifndef GPUCA_GPUCODE_COMPILEKERNELS
template <class T, class S>
GPUhi() void GPUCommonAlgorithm::sortOnDevice(auto* rec, int32_t stream, T* begin, size_t N, const S& comp)
{
  thrust::device_ptr<T> p(begin);
#if 0 // Use Thrust
  auto alloc = rec->getThrustVolatileDeviceAllocator();
  thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(rec->mInternals->Streams[stream]), p, p + N, comp);
#else // Use CUB
  size_t tempSize = 0;
  void* tempMem = nullptr;
  GPUChkErrS(GPUCA_CUB_NAMESPACE::DeviceMergeSort::SortKeys(tempMem, tempSize, begin, N, comp, rec->mInternals->Streams[stream]));
  tempMem = rec->AllocateVolatileDeviceMemory(tempSize);
  GPUChkErrS(GPUCA_CUB_NAMESPACE::DeviceMergeSort::SortKeys(tempMem, tempSize, begin, N, comp, rec->mInternals->Streams[stream]));
#endif
}
#endif // #ifndef GPUCA_GPUCODE_COMPILEKERNELS

} // namespace o2::gpu

#undef GPUCA_THRUST_NAMESPACE
#undef GPUCA_CUB_NAMESPACE

#endif
