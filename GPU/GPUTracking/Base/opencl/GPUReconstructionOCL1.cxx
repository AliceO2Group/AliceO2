// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL1.cxx
/// \author David Rohr

#define GPUCA_GPUTYPE_RADEON

#include "GPUReconstructionOCL1.h"
#include "GPUReconstructionOCL1Internals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

#include "../makefiles/opencl_obtain_program.h"
extern "C" char _makefile_opencl_program_Base_opencl_common_GPUReconstructionOCL_cl[];

GPUReconstruction* GPUReconstruction_Create_OCL(const GPUSettingsProcessing& cfg) { return new GPUReconstructionOCL1(cfg); }

GPUReconstructionOCL1Backend::GPUReconstructionOCL1Backend(const GPUSettingsProcessing& cfg) : GPUReconstructionOCL(cfg)
{
}

template <class T, int I, typename... Args>
int GPUReconstructionOCL1Backend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
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
S& GPUReconstructionOCL1Backend::getKernelObject(int num)
{
  static int krnl = FindKernel<T, I>(num);
  return mInternals->kernels[krnl].first;
}

int GPUReconstructionOCL1Backend::GetOCLPrograms()
{
  cl_uint count;
  if (GPUFailedMsgI(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count))) {
    GPUError("Error getting OPENCL Device Count");
    return (1);
  }

  if (_makefiles_opencl_obtain_program_helper(mInternals->context, count, mInternals->devices.get(), &mInternals->program, _makefile_opencl_program_Base_opencl_common_GPUReconstructionOCL_cl)) {
    clReleaseContext(mInternals->context);
    GPUError("Could not obtain OpenCL progarm");
    return 1;
  }
  return 0;
}

bool GPUReconstructionOCL1Backend::CheckPlatform(unsigned int i)
{
  char platform_version[64], platform_vendor[64];
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
  if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0 && strstr(platform_version, "OpenCL 2.0 AMD-APP (") != nullptr) {
    float ver = 0;
    sscanf(platform_version, "OpenCL 2.0 AMD-APP (%f)", &ver);
    if (ver < 2000.f) {
      if (mDeviceProcessingSettings.debugLevel >= 2) {
        GPUInfo("AMD APP OpenCL Platform found");
      }
      return true;
    }
  }
  return false;
}
