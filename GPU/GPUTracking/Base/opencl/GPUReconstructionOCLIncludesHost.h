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

/// \file GPUReconstructionOCLIncludesHost.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCLINCLUDESHOST_H
#define GPURECONSTRUCTIONOCLINCLUDESHOST_H

#define GPUCA_GPUTYPE_OPENCL
#define __OPENCL_HOST__

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <vector>
#include <string>
#include <memory>

typedef cl_half half;

#include "GPULogging.h"

#include "GPUReconstructionOCL.h"
#include "GPUReconstructionIncludes.h"
#include "GPUCommonHelpers.h"

using namespace o2::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

namespace o2::gpu
{
struct GPUReconstructionOCLInternals {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue[constants::GPU_MAX_STREAMS];
  cl_mem mem_gpu;
  cl_mem mem_constant;
  cl_mem mem_host;
  cl_program program;

  std::vector<cl_kernel> kernels;
};
} // namespace o2::gpu

template <typename T, typename... Args>
inline int64_t GPUReconstructionOCL::OCLsetKernelParameters_helper(cl_kernel& kernel, int32_t i, const T& firstParameter, const Args&... restOfParameters)
{
  int64_t retVal = clSetKernelArg(kernel, i, sizeof(T), &firstParameter);
  if (retVal) {
    return retVal;
  }
  if constexpr (sizeof...(restOfParameters) > 0) {
    return OCLsetKernelParameters_helper(kernel, i + 1, restOfParameters...);
  }
  return 0;
}

template <typename... Args>
inline int64_t GPUReconstructionOCL::OCLsetKernelParameters(cl_kernel& kernel, const Args&... args)
{
  return OCLsetKernelParameters_helper(kernel, 0, args...);
}

inline int64_t GPUReconstructionOCL::clExecuteKernelA(cl_command_queue queue, cl_kernel krnl, size_t local_size, size_t global_size, cl_event* pEvent, cl_event* wait, cl_int nWaitEvents)
{
  return clEnqueueNDRangeKernel(queue, krnl, 1, nullptr, &global_size, &local_size, wait == nullptr ? 0 : nWaitEvents, wait, pEvent);
}

#endif // GPURECONSTRUCTIONOCLINCLUDESHOST_H
