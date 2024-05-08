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

/// \file GPUReconstructionOCL1.cxx
/// \author David Rohr

#define GPUCA_GPUTYPE_OPENCL
#define __OPENCL_HOST__

#include "GPUReconstructionOCL1.h"
#include "GPUReconstructionOCL1Internals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

#include "utils/opencl_obtain_program.h"
#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionOCL1Code_bin);

GPUReconstruction* GPUReconstruction_Create_OCL(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionOCL1(cfg); }

GPUReconstructionOCL1Backend::GPUReconstructionOCL1Backend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionOCL(cfg)
{
}

template <class T, int I, typename... Args>
int GPUReconstructionOCL1Backend::runKernelBackend(const krnlSetupArgs<T, I, Args...>& args)
{
  cl_kernel k = args.s.y.num > 1 ? getKernelObject<cl_kernel, T, I, true>() : getKernelObject<cl_kernel, T, I, false>();
  return std::apply([this, &args, &k](auto&... vals) { return runKernelBackendInternal(args.s, k, vals...); }, args.v);
}

template <class S, class T, int I, bool MULTI>
S& GPUReconstructionOCL1Backend::getKernelObject()
{
  static unsigned int krnl = FindKernel<T, I>(MULTI ? 2 : 1);
  return mInternals->kernels[krnl].first;
}

int GPUReconstructionOCL1Backend::GetOCLPrograms()
{
  cl_uint count;
  if (GPUFailedMsgI(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count))) {
    GPUError("Error getting OPENCL Device Count");
    return (1);
  }

  if (_makefiles_opencl_obtain_program_helper(mInternals->context, count, mInternals->devices.get(), &mInternals->program, _binary_GPUReconstructionOCL1Code_bin_start)) {
    clReleaseContext(mInternals->context);
    GPUError("Could not obtain OpenCL progarm");
    return 1;
  }

#define GPUCA_OPENCL1
#define GPUCA_KRNL(...) \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, __VA_ARGS__)
#define GPUCA_KRNL_LOAD_single(x_class, ...)              \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>(false)) { \
    return 1;                                             \
  }
#define GPUCA_KRNL_LOAD_multi(x_class, ...)              \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>(true)) { \
    return 1;                                            \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#undef GPUCA_OPENCL1
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

  return 0;
}

bool GPUReconstructionOCL1Backend::CheckPlatform(unsigned int i)
{
  char platform_version[64] = {}, platform_vendor[64] = {};
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
  clGetPlatformInfo(mInternals->platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
  if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0 && strstr(platform_version, "OpenCL 2.0 AMD-APP (") != nullptr) {
    float ver = 0;
    sscanf(platform_version, "OpenCL 2.0 AMD-APP (%f)", &ver);
    if (ver < 2000.f) {
      if (mProcessingSettings.debugLevel >= 2) {
        GPUInfo("AMD APP OpenCL Platform found");
      }
      return true;
    }
  }
  return false;
}
