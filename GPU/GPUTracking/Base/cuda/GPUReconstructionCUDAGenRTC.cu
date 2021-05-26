// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDAGenRTC.cu
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY
#include <omp.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "GPUParamRTC.h"
#include "GPUDefMacros.h"
#include <unistd.h>
#ifdef GPUCA_HAVE_O2HEADERS
#include "Framework/SHA1.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_ALIROOT_LIB
extern "C" char _curtc_GPUReconstructionCUDArtc_cu_src[];
extern "C" unsigned int _curtc_GPUReconstructionCUDArtc_cu_src_size;
extern "C" char _curtc_GPUReconstructionCUDArtc_cu_command[];
#endif

int GPUReconstructionCUDA::genRTC()
{
#ifndef GPUCA_ALIROOT_LIB
  std::string rtcparam = GPUParamRTC::generateRTCCode(param(), mProcessingSettings.rtc.optConstexpr);
  std::string filename = "/tmp/o2cagpu_rtc_";
  filename += std::to_string(getpid());
  filename += "_";
  filename += std::to_string(rand());

  std::vector<std::string> kernels;
  std::string kernelsall;
#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward) kernels.emplace_back(GPUCA_M_STR(GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward)));
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward) kernels.emplace_back(GPUCA_M_STR(GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward)));
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi
  for (unsigned int i = 0; i < kernels.size(); i++) {
    kernelsall += kernels[i];
  }

#ifdef GPUCA_HAVE_O2HEADERS
  char shasource[21], shaparam[21], shacmd[21], shakernels[21];
  if (mProcessingSettings.rtc.cacheOutput) {
    o2::framework::internal::SHA1(shasource, _curtc_GPUReconstructionCUDArtc_cu_src, _curtc_GPUReconstructionCUDArtc_cu_src_size);
    o2::framework::internal::SHA1(shaparam, rtcparam.c_str(), rtcparam.size());
    o2::framework::internal::SHA1(shacmd, _curtc_GPUReconstructionCUDArtc_cu_command, strlen(_curtc_GPUReconstructionCUDArtc_cu_command));
    o2::framework::internal::SHA1(shakernels, kernelsall.c_str(), kernelsall.size());
  }
#endif

  unsigned int nCompile = mProcessingSettings.rtc.compilePerKernel ? kernels.size() : 1;
  bool cacheLoaded = false;
  if (mProcessingSettings.rtc.cacheOutput) {
#ifndef GPUCA_HAVE_O2HEADERS
    throw std::runtime_error("Cannot use RTC cache without O2 headers");
#else
    FILE* fp = fopen("rtc.cuda.cache", "rb");
    char sharead[20];
    if (fp) {
      size_t len;
      while (true) {
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (memcmp(sharead, shasource, 20)) {
          GPUInfo("Cache file content outdated (source)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (memcmp(sharead, shaparam, 20)) {
          GPUInfo("Cache file content outdated (param)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (memcmp(sharead, shacmd, 20)) {
          GPUInfo("Cache file content outdated (commandline)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (memcmp(sharead, shakernels, 20)) {
          GPUInfo("Cache file content outdated (kernel definitions)");
          break;
        }
        GPUSettingsProcessingRTC cachedSettings;
        if (fread(&cachedSettings, sizeof(cachedSettings), 1, fp) != 1) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (memcmp(&cachedSettings, &mProcessingSettings.rtc, sizeof(cachedSettings))) {
          GPUInfo("Cache file content outdated (rtc parameters)");
          break;
        }
        std::vector<char> buffer;
        for (unsigned int i = 0; i < nCompile; i++) {
          if (fread(&len, sizeof(len), 1, fp) != 1) {
            throw std::runtime_error("Cache file corrupt");
          }
          buffer.resize(len);
          if (fread(buffer.data(), 1, len, fp) != len) {
            throw std::runtime_error("Cache file corrupt");
          }
          FILE* fp2 = fopen((filename + "_" + std::to_string(i) + ".o").c_str(), "w+b");
          if (fp2 == nullptr) {
            throw std::runtime_error("Cannot open tmp file");
          }
          if (fwrite(buffer.data(), 1, len, fp2) != len) {
            throw std::runtime_error("Error writing file");
          }
          fclose(fp2);
        }
        GPUInfo("Using RTC cache file");
        cacheLoaded = true;
        break;
      };
      fclose(fp);
    }
#endif
  }
  if (!cacheLoaded) {
    if (mProcessingSettings.debugLevel >= 0) {
      GPUInfo("Starting CUDA RTC Compilation");
    }
    HighResTimer rtcTimer;
    rtcTimer.ResetStart();
#pragma omp parallel for
    for (unsigned int i = 0; i < nCompile; i++) {
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Compiling %s\n", (filename + "_" + std::to_string(i) + ".cu").c_str());
      }
      FILE* fp = fopen((filename + "_" + std::to_string(i) + ".cu").c_str(), "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Error opening file");
      }

      std::string kernel = "extern \"C\" {";
      kernel += mProcessingSettings.rtc.compilePerKernel ? kernels[i] : kernelsall;
      kernel += "}";

      if (fwrite(rtcparam.c_str(), 1, rtcparam.size(), fp) != rtcparam.size() ||
          fwrite(_curtc_GPUReconstructionCUDArtc_cu_src, 1, _curtc_GPUReconstructionCUDArtc_cu_src_size, fp) != _curtc_GPUReconstructionCUDArtc_cu_src_size ||
          fwrite(kernel.c_str(), 1, kernel.size(), fp) != kernel.size()) {
        throw std::runtime_error("Error writing file");
      }
      fclose(fp);
      std::string command = _curtc_GPUReconstructionCUDArtc_cu_command;
      command += " -cubin -c " + filename + "_" + std::to_string(i) + ".cu -o " + filename + "_" + std::to_string(i) + ".o";
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Running command %s\n", command.c_str());
      }
      if (system(command.c_str())) {
        throw std::runtime_error("Error during CUDA compilation");
      }
    }
    if (mProcessingSettings.debugLevel >= 0) {
      GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
    }
    if (mProcessingSettings.rtc.cacheOutput) {
      FILE* fp = fopen("rtc.cuda.cache", "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Cannot open cache file for writing");
      }
      GPUInfo("Storing RTC compilation result in cache file");

      if (fwrite(shasource, 1, 20, fp) != 20 ||
          fwrite(shaparam, 1, 20, fp) != 20 ||
          fwrite(shacmd, 1, 20, fp) != 20 ||
          fwrite(shakernels, 1, 20, fp) != 20 ||
          fwrite(&mProcessingSettings.rtc, sizeof(mProcessingSettings.rtc), 1, fp) != 1) {
        throw std::runtime_error("Error writing cache file");
      }

      std::vector<char> buffer;
      for (unsigned int i = 0; i < nCompile; i++) {
        FILE* fp2 = fopen((filename + "_" + std::to_string(i) + ".o").c_str(), "rb");
        if (fp2 == nullptr) {
          throw std::runtime_error("Cannot open cuda module file");
        }
        fseek(fp2, 0, SEEK_END);
        size_t size = ftell(fp2);
        buffer.resize(size);
        fseek(fp2, 0, SEEK_SET);
        if (fread(buffer.data(), 1, size, fp2) != size) {
          throw std::runtime_error("Error reading cuda module file");
        }
        fclose(fp2);

        if (fwrite(&size, sizeof(size), 1, fp) != 1 ||
            fwrite(buffer.data(), 1, size, fp) != size) {
          throw std::runtime_error("Error writing cache file");
        }
      }
      fclose(fp);
    }
  }

  for (unsigned int i = 0; i < nCompile; i++) {
    mInternals->rtcModules.emplace_back(std::make_unique<CUmodule>());
    GPUFailedMsg(cuModuleLoad(mInternals->rtcModules.back().get(), (filename + "_" + std::to_string(i) + ".o").c_str()));
    remove((filename + "_" + std::to_string(i) + ".cu").c_str());
    remove((filename + "_" + std::to_string(i) + ".o").c_str());
  }

  int j = 0;
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)                          \
  mInternals->getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->rtcFunctions.size()); \
  mInternals->rtcFunctions.emplace_back(new CUfunction);                                               \
  GPUFailedMsg(cuModuleGetFunction(mInternals->rtcFunctions.back().get(), *mInternals->rtcModules[mProcessingSettings.rtc.compilePerKernel ? j++ : 0], GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)))));
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)                          \
  mInternals->getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(mInternals->rtcFunctions.size()); \
  mInternals->rtcFunctions.emplace_back(new CUfunction);                                              \
  GPUFailedMsg(cuModuleGetFunction(mInternals->rtcFunctions.back().get(), *mInternals->rtcModules[mProcessingSettings.rtc.compilePerKernel ? j++ : 0], GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi))));
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

#endif
  return 0;
}
