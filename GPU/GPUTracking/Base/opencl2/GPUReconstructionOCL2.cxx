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

/// \file GPUReconstructionOCL2.cxx
/// \author David Rohr

#define GPUCA_GPUTYPE_OPENCL
#define __OPENCL_HOST__

#include "GPUReconstructionOCL2.h"
#include "GPUReconstructionOCL2Internals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionOCL2Code_src);
#ifdef OPENCL2_ENABLED_SPIRV
QGET_LD_BINARY_SYMBOLS(GPUReconstructionOCL2Code_spirv);
#endif

GPUReconstruction* GPUReconstruction_Create_OCL2(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionOCL2(cfg); }

GPUReconstructionOCL2Backend::GPUReconstructionOCL2Backend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionOCL(cfg)
{
}

template <class T, int I, typename... Args>
int GPUReconstructionOCL2Backend::runKernelBackend(krnlSetup& _xyz, const Args&... args)
{
  cl_kernel k = _xyz.y.num > 1 ? getKernelObject<cl_kernel, T, I, true>() : getKernelObject<cl_kernel, T, I, false>();
  return runKernelBackendCommon(_xyz, k, args...);
}

template <class S, class T, int I, bool MULTI>
S& GPUReconstructionOCL2Backend::getKernelObject()
{
  static unsigned int krnl = FindKernel<T, I>(MULTI ? 2 : 1);
  return mInternals->kernels[krnl].first;
}

int GPUReconstructionOCL2Backend::GetOCLPrograms()
{
  char platform_version[256] = {};
  GPUFailedMsg(clGetPlatformInfo(mInternals->platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr));
  float ver = 0;
  sscanf(platform_version, "OpenCL %f", &ver);

  cl_int ocl_error;

  const char* ocl_flags = GPUCA_M_STR(OCL_FLAGS);

#ifdef OPENCL2_ENABLED_SPIRV // clang-format off
  if (ver >= 2.2) {
    GPUInfo("Reading OpenCL program from SPIR-V IL (Platform version %f)", ver);
    mInternals->program = clCreateProgramWithIL(mInternals->context, _binary_GPUReconstructionOCL2Code_spirv_start, _binary_GPUReconstructionOCL2Code_spirv_len, &ocl_error);
    ocl_flags = "";
  } else
#endif // clang-format on
  {
    GPUInfo("Compiling OpenCL program from sources (Platform version %f, %s)", ver);
    size_t program_sizes[1] = {_binary_GPUReconstructionOCL2Code_src_len};
    char* programs_sources[1] = {_binary_GPUReconstructionOCL2Code_src_start};
    mInternals->program = clCreateProgramWithSource(mInternals->context, (cl_uint)1, (const char**)&programs_sources, program_sizes, &ocl_error);
  }

  if (GPUFailedMsgI(ocl_error)) {
    GPUError("Error creating OpenCL program from binary");
    return 1;
  }

  if (GPUFailedMsgI(clBuildProgram(mInternals->program, 1, &mInternals->device, ocl_flags, nullptr, nullptr))) {
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

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward) \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>(false)) {                     \
    return 1;                                                                 \
  }
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward) \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>(true)) {                     \
    return 1;                                                                \
  }
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

  return 0;
}

bool GPUReconstructionOCL2Backend::CheckPlatform(unsigned int i)
{
  char platform_version[64] = {}, platform_vendor[64] = {};
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
  float ver1 = 0;
  sscanf(platform_version, "OpenCL %f", &ver1);
  if (ver1 >= 2.2f) {
    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("OpenCL 2.2 capable platform found");
    }
    return true;
  }

  if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0 && ver1 >= 2.0f) {
    float ver2 = 0;
    const char* pos = strchr(platform_version, '(');
    if (pos) {
      sscanf(pos, "(%f)", &ver2);
    }
    if ((ver1 >= 2.f && ver2 >= 2000.f) || ver1 >= 2.1f) {
      if (mProcessingSettings.debugLevel >= 2) {
        GPUInfo("AMD ROCm OpenCL Platform found");
      }
      return true;
    }
  }
  return false;
}
