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

/// \file GPUReconstructionOCLInternals.h
/// \author David Rohr, Sergey Gorbunov

// All OpenCL-header related stuff goes here, so we can run CING over GPUReconstructionOCL

#ifndef GPUTPCGPUTRACKEROPENCLINTERNALS_H
#define GPUTPCGPUTRACKEROPENCLINTERNALS_H

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <vector>
#include <string>
#include <memory>
#include "GPULogging.h"

namespace GPUCA_NAMESPACE::gpu
{

static const char* opencl_error_string(int errorcode)
{
  switch (errorcode) {
    case CL_SUCCESS:
      return "Success!";
    case CL_DEVICE_NOT_FOUND:
      return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Program build failure";
    case CL_MAP_FAILURE:
      return "Map failure";
    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return "Invalid device type";
    case CL_INVALID_PLATFORM:
      return "Invalid platform";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_CONTEXT:
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list";
    case CL_INVALID_EVENT:
      return "Invalid event";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return "Invalid mip-map level";
    default:
      return "Unknown Errorcode";
  }
}

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

static inline long int OCLsetKernelParameters_helper(cl_kernel& k, int i)
{
  return 0;
}

template <typename T, typename... Args>
static inline long int OCLsetKernelParameters_helper(cl_kernel& kernel, int i, const T& firstParameter, const Args&... restOfParameters)
{
  long int retVal = clSetKernelArg(kernel, i, sizeof(T), &firstParameter);
  if (retVal) {
    return retVal;
  }
  return OCLsetKernelParameters_helper(kernel, i + 1, restOfParameters...);
}

template <typename... Args>
static inline long int OCLsetKernelParameters(cl_kernel& kernel, const Args&... args)
{
  return OCLsetKernelParameters_helper(kernel, 0, args...);
}

static inline long int clExecuteKernelA(cl_command_queue queue, cl_kernel krnl, size_t local_size, size_t global_size, cl_event* pEvent, cl_event* wait = nullptr, cl_int nWaitEvents = 1)
{
  return clEnqueueNDRangeKernel(queue, krnl, 1, nullptr, &global_size, &local_size, wait == nullptr ? 0 : nWaitEvents, wait, pEvent);
}

struct GPUReconstructionOCLInternals {
  cl_platform_id platform;
  cl_device_id device;
  std::unique_ptr<cl_platform_id[]> platforms;
  std::unique_ptr<cl_device_id[]> devices;
  cl_context context;
  cl_command_queue command_queue[GPUCA_MAX_STREAMS];
  cl_mem mem_gpu;
  cl_mem mem_constant;
  cl_mem mem_host;
  cl_program program;

  std::vector<std::pair<cl_kernel, std::string>> kernels;
};

template <typename K, typename... Args>
int GPUReconstructionOCL::runKernelBackendCommon(krnlSetup& _xyz, K& k, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  auto& z = _xyz.z;
  if (y.num <= 1) {
    GPUFailedMsg(OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.start, args...));
  } else {
    GPUFailedMsg(OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.start, y.num, args...));
  }

  cl_event ev;
  cl_event* evr;
  bool tmpEvent = false;
  if (z.ev == nullptr && mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0) {
    evr = &ev;
    tmpEvent = true;
  } else {
    evr = (cl_event*)z.ev;
  }
  GPUFailedMsg(clExecuteKernelA(mInternals->command_queue[x.stream], k, x.nThreads, x.nThreads * x.nBlocks, evr, (cl_event*)z.evList, z.nEvents));
  if (mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0) {
    cl_ulong time_start, time_end;
    GPUFailedMsg(clWaitForEvents(1, evr));
    GPUFailedMsg(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
    GPUFailedMsg(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
    _xyz.t = (time_end - time_start) * 1.e-9;
    if (tmpEvent) {
      GPUFailedMsg(clReleaseEvent(ev));
    }
  }
  return 0;
}

template <class T, int I>
int GPUReconstructionOCL::AddKernel(bool multi)
{
  std::string name(GetKernelName<T, I>());
  if (multi) {
    name += "_multi";
  }
  std::string kname("krnl_" + name);

  cl_int ocl_error;
  cl_kernel krnl = clCreateKernel(mInternals->program, kname.c_str(), &ocl_error);
  if (GPUFailedMsgI(ocl_error)) {
    GPUError("Error creating OPENCL Kernel: %s", name.c_str());
    return 1;
  }
  mInternals->kernels.emplace_back(krnl, name);
  return 0;
}

template <class T, int I>
inline unsigned int GPUReconstructionOCL::FindKernel(int num)
{
  std::string name(GetKernelName<T, I>());
  if (num > 1) {
    name += "_multi";
  }

  for (unsigned int k = 0; k < mInternals->kernels.size(); k++) {
    if (mInternals->kernels[k].second == name) {
      return (k);
    }
  }
  GPUError("Could not find OpenCL kernel %s", name.c_str());
  throw ::std::runtime_error("Requested unsupported OpenCL kernel");
}

static_assert(std::is_convertible<cl_event, void*>::value, "OpenCL event type incompatible to deviceEvent");
} // namespace GPUCA_NAMESPACE::gpu

#endif
