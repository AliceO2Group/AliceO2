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

/// \file HIPThrustHelpers.h
/// \author David Rohr

#ifndef GPU_HIPTHRUSTHELPERS_H
#define GPU_HIPTHRUSTHELPERS_H

#include "GPULogging.h"
#include <vector>
#include <memory>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class ThrustVolatileAsyncAllocator
{
 public:
  typedef char value_type;

  ThrustVolatileAsyncAllocator(GPUReconstruction* r) : mRec(r) {}
  char* allocate(std::ptrdiff_t n) { return (char*)mRec->AllocateVolatileDeviceMemory(n); }

  void deallocate(char* ptr, size_t) {}

 private:
  GPUReconstruction* mRec;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#ifndef __HIPCC__
// Override synchronize call at end of thrust algorithm running on stream, just don't run hipStreamSynchronize
namespace thrust
{
namespace hip_cub
{

typedef thrust::hip_cub::execution_policy<typeof(thrust::hip::par(*(GPUCA_NAMESPACE::gpu::ThrustVolatileAsyncAllocator*)nullptr).on(*(hipStream_t*)nullptr))> thrustStreamPolicy;
template <>
__host__ __device__ inline hipError_t synchronize<thrustStreamPolicy>(thrustStreamPolicy& policy)
{
#ifndef GPUCA_GPUCODE_DEVICE
  // Do not synchronize!
  return hipSuccess;
#else
  return synchronize_stream(derived_cast(policy));
#endif
}

} // namespace hip_cub
} // namespace thrust
#endif // __HIPCC__

#endif // GPU_HIPTHRUSTHELPERS_H
