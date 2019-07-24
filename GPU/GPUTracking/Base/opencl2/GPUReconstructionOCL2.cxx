// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL2.cxx
/// \author David Rohr

#define GPUCA_GPUTYPE_RADEON

#include "GPUReconstructionOCL2.h"
#include "GPUReconstructionOCL2Internals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

extern "C" char _makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl[];
extern "C" unsigned int _makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl_size;

GPUReconstruction* GPUReconstruction_Create_OCL2(const GPUSettingsProcessing& cfg) { return new GPUReconstructionOCL2(cfg); }

GPUReconstructionOCL2Backend::GPUReconstructionOCL2Backend(const GPUSettingsProcessing& cfg) : GPUReconstructionOCL(cfg)
{
}

template <class T, int I, typename... Args>
int GPUReconstructionOCL2Backend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
  cl_kernel k = getKernelObject<cl_kernel, T, I>(y.num);
  if (y.num == -1) {
    if (OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, args...)) {
      return 1;
    }
  } else if (y.num == 0) {
    if (OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.start, args...)) {
      return 1;
    }
  } else {
    if (OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.start, y.num, args...)) {
      return 1;
    }
  }
  return clExecuteKernelA(mInternals->command_queue[x.stream], k, x.nThreads, x.nThreads * x.nBlocks, (cl_event*)z.ev, (cl_event*)z.evList, z.nEvents);
}

template <class S, class T, int I>
S& GPUReconstructionOCL2Backend::getKernelObject(int num)
{
  static int krnl = FindKernel<T, I>(num);
  return mInternals->kernels[krnl].first;
}

int GPUReconstructionOCL2Backend::GetOCLPrograms()
{
  size_t program_sizes[1] = { _makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl_size };
  char* program_binaries[1] = { _makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl };
  cl_int return_status[1];
  cl_int ocl_error;
  mInternals->program = clCreateProgramWithBinary(mInternals->context, 1, &mInternals->device, program_sizes, (const unsigned char**)program_binaries, return_status, &ocl_error);
  if (GPUFailedMsgI(ocl_error)) {
    GPUError("Error creating OpenCL program from binary");
    return 1;
  }
  if (GPUFailedMsgI(return_status[0])) {
    GPUError("Error creating OpenCL program from binary (device status)");
    return 1;
  }

  if (GPUFailedMsgI(clBuildProgram(mInternals->program, 1, &mInternals->device, "", NULL, NULL))) {
    cl_build_status status;
    if (GPUFailedMsgI(clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr)) == 0 && status == CL_BUILD_ERROR) {
      size_t log_size;
      clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::unique_ptr<char[]> build_log(new char[log_size + 1]);
      clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_LOG, log_size, build_log.get(), nullptr);
      build_log[log_size] = 0;
      GPUError("Build Log:\n\n%s\n", build_log.get());
    }
    return 1;
  }
  return 0;
}

bool GPUReconstructionOCL2Backend::CheckPlatform(unsigned int i)
{
  char platform_version[64], platform_vendor[64];
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
  if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0 && strstr(platform_version, "OpenCL 2.1") != nullptr) {
    if (mDeviceProcessingSettings.debugLevel >= 2) {
      GPUInfo("AMD ROCm OpenCL Platform found");
    }
    return true;
  }
  return false;
}
