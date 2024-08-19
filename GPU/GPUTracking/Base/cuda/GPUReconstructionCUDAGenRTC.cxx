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
#ifdef WITH_OPENMP
#include <omp.h>
#endif
#include "GPUReconstructionCUDA.h"
#include "GPUParamRTC.h"
#include "GPUDefMacros.h"
#include <unistd.h>
#ifdef GPUCA_HAVE_O2HEADERS
#include "Framework/SHA1.h"
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <filesystem>

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_ALIROOT_LIB
#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_src);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_command);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCUDArtc_command_arch);
#endif

int GPUReconstructionCUDA::genRTC(std::string& filename, unsigned int& nCompile)
{
#ifndef GPUCA_ALIROOT_LIB
  std::string rtcparam = GPUParamRTC::generateRTCCode(param(), mProcessingSettings.rtc.optConstexpr);
  if (filename == "") {
    filename = "/tmp/o2cagpu_rtc_";
  }
  filename += std::to_string(getpid());
  filename += "_";
  filename += std::to_string(rand());

  std::vector<std::string> kernels;
  getRTCKernelCalls(kernels);
  std::string kernelsall;
  for (unsigned int i = 0; i < kernels.size(); i++) {
    kernelsall += kernels[i] + "\n";
  }

  std::string baseCommand = (mProcessingSettings.RTCprependCommand != "" ? (mProcessingSettings.RTCprependCommand + " ") : "");
  baseCommand += (getenv("O2_GPU_RTC_OVERRIDE_CMD") ? std::string(getenv("O2_GPU_RTC_OVERRIDE_CMD")) : std::string(_binary_GPUReconstructionCUDArtc_command_start, _binary_GPUReconstructionCUDArtc_command_len));
  baseCommand += std::string(" ") + (mProcessingSettings.RTCoverrideArchitecture != "" ? mProcessingSettings.RTCoverrideArchitecture : std::string(_binary_GPUReconstructionCUDArtc_command_arch_start, _binary_GPUReconstructionCUDArtc_command_arch_len));

#ifdef GPUCA_HAVE_O2HEADERS
  char shasource[21], shaparam[21], shacmd[21], shakernels[21];
  if (mProcessingSettings.rtc.cacheOutput) {
    o2::framework::internal::SHA1(shasource, _binary_GPUReconstructionCUDArtc_src_start, _binary_GPUReconstructionCUDArtc_src_len);
    o2::framework::internal::SHA1(shaparam, rtcparam.c_str(), rtcparam.size());
    o2::framework::internal::SHA1(shacmd, baseCommand.c_str(), baseCommand.size());
    o2::framework::internal::SHA1(shakernels, kernelsall.c_str(), kernelsall.size());
  }
#endif

  nCompile = mProcessingSettings.rtc.compilePerKernel ? kernels.size() : 1;
  bool cacheLoaded = false;
  int fd = 0;
  if (mProcessingSettings.rtc.cacheOutput) {
    if (mProcessingSettings.RTCcacheFolder != ".") {
      std::filesystem::create_directories(mProcessingSettings.RTCcacheFolder);
    }
#ifndef GPUCA_HAVE_O2HEADERS
    throw std::runtime_error("Cannot use RTC cache without O2 headers");
#else
    if (mProcessingSettings.rtc.cacheMutex) {
      mode_t mask = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
      fd = open((mProcessingSettings.RTCcacheFolder + "/cache.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, mask);
      if (fd == -1) {
        throw std::runtime_error("Error opening rtc cache mutex lock file");
      }
      fchmod(fd, mask);
      if (lockf(fd, F_LOCK, 0)) {
        throw std::runtime_error("Error locking rtc cache mutex file");
      }
    }

    FILE* fp = fopen((mProcessingSettings.RTCcacheFolder + "/rtc.cuda.cache").c_str(), "rb");
    char sharead[20];
    if (fp) {
      size_t len;
      while (true) {
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!mProcessingSettings.rtc.ignoreCacheValid && memcmp(sharead, shasource, 20)) {
          GPUInfo("Cache file content outdated (source)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!mProcessingSettings.rtc.ignoreCacheValid && memcmp(sharead, shaparam, 20)) {
          GPUInfo("Cache file content outdated (param)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!mProcessingSettings.rtc.ignoreCacheValid && memcmp(sharead, shacmd, 20)) {
          GPUInfo("Cache file content outdated (commandline)");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!mProcessingSettings.rtc.ignoreCacheValid && memcmp(sharead, shakernels, 20)) {
          GPUInfo("Cache file content outdated (kernel definitions)");
          break;
        }
        GPUSettingsProcessingRTC cachedSettings;
        static_assert(std::is_trivially_copyable_v<GPUSettingsProcessingRTC> == true, "GPUSettingsProcessingRTC must be POD");
        if (fread(&cachedSettings, sizeof(cachedSettings), 1, fp) != 1) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!mProcessingSettings.rtc.ignoreCacheValid && memcmp(&cachedSettings, &mProcessingSettings.rtc, sizeof(cachedSettings))) {
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
          FILE* fp2 = fopen((filename + "_" + std::to_string(i) + mRtcBinExtension).c_str(), "w+b");
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
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (unsigned int i = 0; i < nCompile; i++) {
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Compiling %s\n", (filename + "_" + std::to_string(i) + mRtcSrcExtension).c_str());
      }
      FILE* fp = fopen((filename + "_" + std::to_string(i) + mRtcSrcExtension).c_str(), "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Error opening file");
      }

      std::string kernel = "extern \"C\" {";
      kernel += mProcessingSettings.rtc.compilePerKernel ? kernels[i] : kernelsall;
      kernel += "}";

      if (fwrite(rtcparam.c_str(), 1, rtcparam.size(), fp) != rtcparam.size() ||
          fwrite(_binary_GPUReconstructionCUDArtc_src_start, 1, _binary_GPUReconstructionCUDArtc_src_len, fp) != _binary_GPUReconstructionCUDArtc_src_len ||
          fwrite(kernel.c_str(), 1, kernel.size(), fp) != kernel.size()) {
        throw std::runtime_error("Error writing file");
      }
      fclose(fp);
      std::string command = baseCommand;
      command += " -c " + filename + "_" + std::to_string(i) + mRtcSrcExtension + " -o " + filename + "_" + std::to_string(i) + mRtcBinExtension;
      if (mProcessingSettings.debugLevel < 0) {
        command += " &> /dev/null";
      } else if (mProcessingSettings.debugLevel < 2) {
        command += " > /dev/null";
      }
      if (mProcessingSettings.debugLevel >= 3) {
        printf("Running command %s\n", command.c_str());
      }
      if (system(command.c_str())) {
        if (mProcessingSettings.debugLevel >= 3) {
          printf("Source code file: %s", filename.c_str());
        }
        throw std::runtime_error("Error during CUDA compilation");
      }
    }
    if (mProcessingSettings.debugLevel >= 0) {
      GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
    }
#ifdef GPUCA_HAVE_O2HEADERS
    if (mProcessingSettings.rtc.cacheOutput) {
      FILE* fp = fopen((mProcessingSettings.RTCcacheFolder + "/rtc.cuda.cache").c_str(), "w+b");
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
        FILE* fp2 = fopen((filename + "_" + std::to_string(i) + mRtcBinExtension).c_str(), "rb");
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
#endif
  }
  if (mProcessingSettings.rtc.cacheOutput && mProcessingSettings.rtc.cacheMutex) {
    if (lockf(fd, F_ULOCK, 0)) {
      throw std::runtime_error("Error unlocking RTC cache mutex file");
    }
    close(fd);
  }

#endif
  return 0;
}
