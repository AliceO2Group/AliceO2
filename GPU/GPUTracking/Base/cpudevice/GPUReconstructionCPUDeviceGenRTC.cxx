// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCPUDeviceGenRTC.cxx
/// \author David Rohr

#include "GPUReconstructionCPUDevice.h"
#include "GPULogging.h"

#include <stdexcept>
#include <string>

#ifdef GPUCA_CPUDEVICE_RTC

#include "GPUReconstructionCPUDeviceInternals.h"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUReconstructionKernelMacros.h"
#include "GPUReconstructionRTCHelpers.inc"
#include "utils/qGetLdBinarySymbols.h"

#include <dlfcn.h>

QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_src);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_command);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_command_no_fast_math);

using namespace o2::gpu;
namespace rtc = o2::gpu::internal::rtc;

static constexpr const char* RTC_SRC_EXTENSION = ".cxx";
static constexpr const char* RTC_BIN_EXTENSION = ".so";
static constexpr const char* RTC_MAP_EXTENSION = ".map";

static bool kernelWithoutRTC(const std::string& name) { return name == "GPUTPCCFCheckPadBaseline"; } // Device code requires more than 1 thread per block, use the host-compiled kernel

int32_t GPUReconstructionCPUDevice::genRTC(std::string& filename, uint32_t& nCompile)
{
  const auto& proc = GetProcessingSettings();
  const std::string_view rtcSource(_binary_GPUReconstructionCPUDevicertc_src_start, _binary_GPUReconstructionCPUDevicertc_src_len);
  const std::string rtcparam = rtc::paramCode(proc, param());
  rtc::tmpFilename(filename, "/tmp/o2cagpu_rtc_cpudevice_");

  std::vector<std::string> kernels;
  getRTCKernelCalls(kernels);
  for (uint32_t i = 0; i < kernels.size(); i++) {
    if (kernelWithoutRTC(GetKernelName(i))) {
      kernels[i].clear();
    }
  }
  const std::string kernelsall = rtc::joinKernels(kernels);

  const std::string baseCommand = rtc::baseCommand(proc, std::string_view(_binary_GPUReconstructionCPUDevicertc_command_start, _binary_GPUReconstructionCPUDevicertc_command_len)) + " " + proc.rtctech.overrideArchitecture;
  const std::string noFastMath(_binary_GPUReconstructionCPUDevicertc_command_no_fast_math_start, _binary_GPUReconstructionCPUDevicertc_command_no_fast_math_len);

  rtc::loadLaunchBounds(proc, *mParDevice);
  const std::string launchBounds = rtc::launchBoundsCode(proc, *mParDevice, 0, mWarpSize);
  if (proc.rtctech.printLaunchBounds >= 2) {
    return 1;
  }

  nCompile = proc.rtc.compilePerKernel ? kernels.size() : 1;
  const std::string compilerVersions = getBackendVersions();
  rtc::Cache cache(proc, "rtc.cpudevice.cache", {rtcSource, rtcparam, baseCommand, kernelsall, launchBounds, compilerVersions});
  if (!cache.load(filename, RTC_BIN_EXTENSION, nCompile)) {
    rtc::compile(proc, nCompile, "CPUDevice", [&](uint32_t i) {
      const std::string fileBase = filename + "_" + std::to_string(i);
      if (proc.debugLevel >= 3) {
        printf("Compiling %s\n", (fileBase + RTC_SRC_EXTENSION).c_str());
      }
      const bool deterministic = rtc::deterministic(proc, GetKernelName(i));
      rtc::writeSource(fileBase + RTC_SRC_EXTENSION, deterministic, rtcparam, launchBounds, rtcSource, proc.rtc.compilePerKernel ? kernels[i] : kernelsall);

      // Export only the kernel entry points, everything else stays local to the RTC library to avoid symbol collisions
      FILE* fp = fopen((fileBase + RTC_MAP_EXTENSION).c_str(), "w+b");
      if (fp == nullptr || fprintf(fp, "{ global: krnl_*; local: *; };\n") < 0) {
        throw std::runtime_error("Error writing version script");
      }
      fclose(fp);

      const std::string command = baseCommand + (deterministic ? " " + noFastMath : "") + " -Wl,--version-script=" + fileBase + RTC_MAP_EXTENSION + " -x c++ " + fileBase + RTC_SRC_EXTENSION + " -o " + fileBase + RTC_BIN_EXTENSION;
      rtc::runCommand(proc, command, fileBase + RTC_SRC_EXTENSION, "Error during CPUDevice RTC compilation");
    });
    cache.store(filename, RTC_BIN_EXTENSION, nCompile);
  }
  return 0;
}

void GPUReconstructionCPUDevice::genAndLoadRTC()
{
  std::string filename = "";
  uint32_t nCompile = 0;
  if (genRTC(filename, nCompile)) {
    throw std::runtime_error("Runtime compilation failed");
  }
  for (uint32_t i = 0; i < nCompile; i++) {
    const std::string fileBase = filename + "_" + std::to_string(i);
    if (GetProcessingSettings().rtctech.runTest != 2) {
      void* lib = dlopen((fileBase + RTC_BIN_EXTENSION).c_str(), RTLD_NOW | RTLD_LOCAL);
      if (lib == nullptr) {
        throw std::runtime_error(std::string("Error loading CPUDevice RTC library: ") + dlerror());
      }
      mInternals->rtcLibraries.emplace_back(lib);
    }
    if (!GetProcessingSettings().rtctech.keepTempFiles) {
      remove((fileBase + RTC_SRC_EXTENSION).c_str());
      remove((fileBase + RTC_BIN_EXTENSION).c_str());
      remove((fileBase + RTC_MAP_EXTENSION).c_str());
    }
  }
  if (GetProcessingSettings().rtctech.runTest == 2) {
    return;
  }

  const bool perKernel = GetProcessingSettings().rtc.compilePerKernel;
  mInternals->rtcKernels.resize(GetNKernels());
  uint32_t j = 0;
#define GPUCA_KRNL(x_class, ...)                                                                                                               \
  if (GetKernelNum<GPUCA_M_KRNL_TEMPLATE(x_class)>() != j) {                                                                                   \
    GPUFatal("kernel numbers out of sync");                                                                                                    \
  }                                                                                                                                            \
  mInternals->rtcKernels[j] = dlsym(mInternals->rtcLibraries[perKernel ? j : 0], GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)))); \
  if (mInternals->rtcKernels[j] == nullptr && !kernelWithoutRTC(GetKernelName(j))) {                                                           \
    GPUFatal("Error loading RTC kernel %s", GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));                                      \
  }                                                                                                                                            \
  j++;
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  if (GetProcessingSettings().debugLevel >= 1) {
    GPUInfo("Loaded %u CPUDevice RTC kernels from %u libraries", j, (uint32_t)mInternals->rtcLibraries.size());
  }
}

#else // GPUCA_CPUDEVICE_RTC

using namespace o2::gpu;

int32_t GPUReconstructionCPUDevice::genRTC(std::string& filename, uint32_t& nCompile)
{
  throw std::runtime_error("CPUDevice backend compiled without RTC support (requires Linux)");
}

void GPUReconstructionCPUDevice::genAndLoadRTC()
{
  throw std::runtime_error("CPUDevice backend compiled without RTC support (requires Linux)");
}

#endif // GPUCA_CPUDEVICE_RTC
