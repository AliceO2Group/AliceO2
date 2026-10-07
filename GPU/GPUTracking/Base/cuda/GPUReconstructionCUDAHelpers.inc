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

/// \file GPUReconstructionCUDAHelpers.inc
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCUDAHELPERS_INC_H
#define GPURECONSTRUCTIONCUDAHELPERS_INC_H

#include "GPUCommonHelpers.h"
#include "GPUReconstruction.h"

namespace o2::gpu::internal
{
int32_t __attribute__((weak)) GPUReconstructionCUDAChkErr(const int64_t retVal, const char* file, int32_t line)
{
  if (retVal != cudaSuccess) {
    GPUError("CUDA Error: %ld / %s (%s:%d)", retVal, cudaGetErrorString((cudaError_t)retVal), file, line);
  }
  return retVal != cudaSuccess;
}
} // namespace o2::gpu::internal

namespace o2::gpu
{
char* __attribute__((weak)) ThrustVolatileAllocator::allocate(std::ptrdiff_t n) { return mAlloc(n); }
void __attribute__((weak)) ThrustVolatileAllocator::deallocate(char* ptr, size_t) {}
} // namespace o2::gpu

#endif
