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

/// \file GPUReconstructionCUDAGenRTC.cu
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionRTCHelpers.inc"
#include "GPUKernelsWith1Warp.inc"
#include <algorithm>

using namespace o2::gpu;
namespace rtc = o2::gpu::internal::rtc;

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_src);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_command);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_command_arch);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_command_no_fast_math);

int32_t GPUReconstructionCUDA::genRTC(std::string& filename, uint32_t& nCompile)
{
  const auto& proc = GetProcessingSettings();
  const std::string_view rtcSource(_binary_GPUReconstructionCUDArtc_src_start, _binary_GPUReconstructionCUDArtc_src_len);
  const std::string rtcparam = rtc::paramCode(proc, param());
  rtc::tmpFilename(filename, "/tmp/o2cagpu_rtc_");

  std::vector<std::string> kernels;
  getRTCKernelCalls(kernels);
  const std::string kernelsall = rtc::joinKernels(kernels);

  const std::string baseCommand = rtc::baseCommand(proc, std::string_view(_binary_GPUReconstructionCUDArtc_command_start, _binary_GPUReconstructionCUDArtc_command_len)) + " " +
                                  (proc.rtctech.overrideArchitecture != "" ? proc.rtctech.overrideArchitecture : std::string(_binary_GPUReconstructionCUDArtc_command_arch_start, _binary_GPUReconstructionCUDArtc_command_arch_len));
  const std::string noFastMath(_binary_GPUReconstructionCUDArtc_command_no_fast_math_start, _binary_GPUReconstructionCUDArtc_command_no_fast_math_len);

  rtc::loadLaunchBounds(proc, *mParDevice);
  if constexpr (std::string_view("CUDA") == "HIP") { // Check if we are RTC-compiling for HIP
    if (proc.hipOverrideAMDEUSperCU > 0) {
      mParDevice->par_AMD_EUS_PER_CU = proc.hipOverrideAMDEUSperCU;
    } else if (mParDevice->par_AMD_EUS_PER_CU <= 0) {
      GPUFatal("AMD_EUS_PER_CU not set in the parameters provided for the AMD GPU, you can override this via --PROChipOverrideAMDEUSperCU [n]");
    }
  }
  for (uint32_t i = 0; i < GetNKernels(); i++) {
    if (std::find(gpuKernelsWith1Warp.begin(), gpuKernelsWith1Warp.end(), GetKernelName(i)) != gpuKernelsWith1Warp.end()) {
      mParDevice->par_LB_maxThreads[i] = mWarpSize;
    }
  }
  const std::string launchBounds = rtc::launchBoundsCode(proc, *mParDevice, mParDevice->par_AMD_EUS_PER_CU ? (mParDevice->par_AMD_EUS_PER_CU * mWarpSize) : 0, mWarpSize);
  if (proc.rtctech.printLaunchBounds >= 2) {
    return 1;
  }

  nCompile = proc.rtc.compilePerKernel ? kernels.size() : 1;
  const std::string compilerVersions = getBackendVersions();
  rtc::Cache cache(proc, "rtc.cuda.cache", {rtcSource, rtcparam, baseCommand, kernelsall, launchBounds, compilerVersions});
  if (!cache.load(filename, mRtcBinExtension, nCompile)) {
    rtc::compile(proc, nCompile, "CUDA", [&](uint32_t i) {
      const std::string fileBase = filename + "_" + std::to_string(i);
      if (proc.debugLevel >= 3) {
        printf("Compiling %s\n", (fileBase + mRtcSrcExtension).c_str());
      }
      const bool deterministic = rtc::deterministic(proc, GetKernelName(i));
      const std::string kernel = std::string("extern \"C\" {") + (proc.rtc.compilePerKernel ? kernels[i] : kernelsall) + "}";
      rtc::writeSource(fileBase + mRtcSrcExtension, deterministic, rtcparam, launchBounds, rtcSource, kernel);
      const std::string command = baseCommand + (deterministic ? " " + noFastMath : "") + " -c " + fileBase + mRtcSrcExtension + " -o " + fileBase + mRtcBinExtension;
      rtc::runCommand(proc, command, fileBase + mRtcSrcExtension, "Error during CUDA compilation");
    });
    cache.store(filename, mRtcBinExtension, nCompile);
  }
  return 0;
}
