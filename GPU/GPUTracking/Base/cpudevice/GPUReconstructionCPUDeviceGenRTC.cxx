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
#include "GPUParamRTC.h"
#include "GPUDefParametersLoad.inc"
#include "GPUNoFastMathKernels.h"
#include "Framework/SHA1.h"
#include "utils/qGetLdBinarySymbols.h"
#include "utils/timer.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_src);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_command);
QGET_LD_BINARY_SYMBOLS(GPUReconstructionCPUDevicertc_command_no_fast_math);

using namespace o2::gpu;

static constexpr const char* RTC_SRC_EXTENSION = ".cxx";
static constexpr const char* RTC_BIN_EXTENSION = ".so";
static constexpr const char* RTC_MAP_EXTENSION = ".map";

static bool kernelWithoutRTC(const std::string& name) { return name == "GPUTPCCFCheckPadBaseline"; } // Device code requires more than 1 thread per block, use the host-compiled kernel

int32_t GPUReconstructionCPUDevice::genRTC(std::string& filename, uint32_t& nCompile)
{
  const std::string_view rtcSource(_binary_GPUReconstructionCPUDevicertc_src_start, _binary_GPUReconstructionCPUDevicertc_src_len);
  const std::string rtcparam = std::string("#define GPUCA_RTC_CODE\n") +
                               std::string(GetProcessingSettings().rtc.optSpecialCode ? "#define GPUCA_RTC_SPECIAL_CODE(...) __VA_ARGS__\n" : "#define GPUCA_RTC_SPECIAL_CODE(...)\n") +
                               std::string(GetProcessingSettings().rtc.optConstexpr ? "#define GPUCA_RTC_CONSTEXPR constexpr\n" : "#define GPUCA_RTC_CONSTEXPR\n") +
                               GPUParamRTC::generateRTCCode(param(), GetProcessingSettings().rtc.optConstexpr);
  if (filename == "") {
    filename = "/tmp/o2cagpu_rtc_cpudevice_";
  }
  filename += std::to_string(getpid());
  filename += "_";
  filename += std::to_string(rand());

  std::vector<std::string> kernels;
  getRTCKernelCalls(kernels);
  for (uint32_t i = 0; i < kernels.size(); i++) {
    if (kernelWithoutRTC(GetKernelName(i))) {
      kernels[i].clear();
    }
  }
  std::string kernelsall;
  for (uint32_t i = 0; i < kernels.size(); i++) {
    kernelsall += kernels[i] + "\n";
  }

  std::string baseCommand = (GetProcessingSettings().rtctech.prependCommand != "" ? (GetProcessingSettings().rtctech.prependCommand + " ") : "");
  baseCommand += (getenv("O2_GPU_RTC_OVERRIDE_CMD") ? std::string(getenv("O2_GPU_RTC_OVERRIDE_CMD")) : std::string(_binary_GPUReconstructionCPUDevicertc_command_start, _binary_GPUReconstructionCPUDevicertc_command_len));
  baseCommand += std::string(" ") + GetProcessingSettings().rtctech.overrideArchitecture;

  if (GetProcessingSettings().rtctech.loadLaunchBoundsFromFile.size()) {
    FILE* fp = fopen(GetProcessingSettings().rtctech.loadLaunchBoundsFromFile.c_str(), "rb");
    if (fp == nullptr) {
      throw std::runtime_error("Cannot open launch bounds parameter module file");
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    if (size != sizeof(*mParDevice)) {
      throw std::runtime_error("launch bounds parameter file has incorrect size");
    }
    fseek(fp, 0, SEEK_SET);
    if (fread(mParDevice, 1, size, fp) != size) {
      throw std::runtime_error("Error reading launch bounds parameter file");
    }
    fclose(fp);
  }
  const std::string launchBounds = o2::gpu::internal::GPUDefParametersExport(*mParDevice, true, 0) + "#define GPUCA_WARP_SIZE " + std::to_string(mWarpSize) + "\n";
  if (GetProcessingSettings().rtctech.printLaunchBounds || GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("RTC Launch Bounds:\n%s", launchBounds.c_str());
    if (GetProcessingSettings().rtctech.printLaunchBounds >= 2) {
      return 1;
    }
  }

  const std::string compilerVersions = getBackendVersions();

  char shasource[21], shaparam[21], shacmd[21], shakernels[21], shabounds[21], shaversion[21];
  if (GetProcessingSettings().rtc.cacheOutput) {
    o2::framework::internal::SHA1(shasource, rtcSource.data(), rtcSource.size());
    o2::framework::internal::SHA1(shaparam, rtcparam.c_str(), rtcparam.size());
    o2::framework::internal::SHA1(shacmd, baseCommand.c_str(), baseCommand.size());
    o2::framework::internal::SHA1(shakernels, kernelsall.c_str(), kernelsall.size());
    o2::framework::internal::SHA1(shabounds, launchBounds.c_str(), launchBounds.size());
    o2::framework::internal::SHA1(shaversion, compilerVersions.c_str(), compilerVersions.size());
  }

  nCompile = GetProcessingSettings().rtc.compilePerKernel ? kernels.size() : 1;
  bool cacheLoaded = false;
  int32_t fd = 0;
  const std::string cacheFile = GetProcessingSettings().rtctech.cacheFolder + "/rtc.cpudevice.cache";
  if (GetProcessingSettings().rtc.cacheOutput) {
    if (GetProcessingSettings().rtctech.cacheFolder != ".") {
      std::filesystem::create_directories(GetProcessingSettings().rtctech.cacheFolder);
    }
    if (GetProcessingSettings().rtctech.cacheMutex) {
      mode_t mask = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
      fd = open((GetProcessingSettings().rtctech.cacheFolder + "/cache.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, mask);
      if (fd == -1) {
        throw std::runtime_error("Error opening rtc cache mutex lock file");
      }
      fchmod(fd, mask);
      if (lockf(fd, F_LOCK, 0)) {
        throw std::runtime_error("Error locking rtc cache mutex file");
      }
    }

    FILE* fp = fopen(cacheFile.c_str(), "rb");
    char sharead[20];
    if (fp) {
      size_t len;
      while (true) {
        auto checkSHA = [&](const char* shacmp, const char* name) {
          if (fread(sharead, 1, 20, fp) != 20) {
            throw std::runtime_error("Cache file corrupt");
          }
          if (!GetProcessingSettings().rtctech.ignoreCacheValid && memcmp(sharead, shacmp, 20)) {
            GPUInfo("Cache file content outdated (%s)", name);
            return 1;
          }
          return 0;
        };
        if (checkSHA(shasource, "source") ||
            checkSHA(shaparam, "param") ||
            checkSHA(shacmd, "command line") ||
            checkSHA(shakernels, "kernel definitions") ||
            checkSHA(shabounds, "launch bounds") ||
            checkSHA(shaversion, "compiler versions")) {
          break;
        }
        GPUSettingsProcessingRTC cachedSettings;
        static_assert(std::is_trivially_copyable_v<GPUSettingsProcessingRTC> == true, "GPUSettingsProcessingRTC must be POD");
        if (fread(&cachedSettings, sizeof(cachedSettings), 1, fp) != 1) {
          throw std::runtime_error("Cache file corrupt");
        }
        if (!GetProcessingSettings().rtctech.ignoreCacheValid && !(cachedSettings == GetProcessingSettings().rtc)) {
          GPUInfo("Cache file content outdated (rtc parameters)");
          break;
        }
        std::vector<char> buffer;
        for (uint32_t i = 0; i < nCompile; i++) {
          if (fread(&len, sizeof(len), 1, fp) != 1) {
            throw std::runtime_error("Cache file corrupt");
          }
          buffer.resize(len);
          if (fread(buffer.data(), 1, len, fp) != len) {
            throw std::runtime_error("Cache file corrupt");
          }
          FILE* fp2 = fopen((filename + "_" + std::to_string(i) + RTC_BIN_EXTENSION).c_str(), "w+b");
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
  }
  if (!cacheLoaded) {
    if (GetProcessingSettings().debugLevel >= 0) {
      GPUInfo("Starting CPUDevice RTC Compilation");
    }
    HighResTimer rtcTimer;
    rtcTimer.ResetStart();
    tbb::parallel_for<uint32_t>(0, nCompile, [&](auto i) {
      const std::string fileBase = filename + "_" + std::to_string(i);
      if (GetProcessingSettings().debugLevel >= 3) {
        printf("Compiling %s\n", (fileBase + RTC_SRC_EXTENSION).c_str());
      }
      FILE* fp = fopen((fileBase + RTC_SRC_EXTENSION).c_str(), "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Error opening file");
      }

      const std::string& kernel = GetProcessingSettings().rtc.compilePerKernel ? kernels[i] : kernelsall;

      bool deterministic = GetProcessingSettings().rtc.deterministic || (GetProcessingSettings().rtc.compilePerKernel && o2::gpu::internal::noFastMathKernels.find(GetKernelName(i)) != o2::gpu::internal::noFastMathKernels.end());
      const std::string deterministicStr = std::string(deterministic ? "#define GPUCA_DETERMINISTIC_CODE(det, indet) det\n" : "#define GPUCA_DETERMINISTIC_CODE(det, indet) indet\n");

      if (fwrite(deterministicStr.c_str(), 1, deterministicStr.size(), fp) != deterministicStr.size() ||
          fwrite(rtcparam.c_str(), 1, rtcparam.size(), fp) != rtcparam.size() ||
          fwrite(launchBounds.c_str(), 1, launchBounds.size(), fp) != launchBounds.size() ||
          fwrite(rtcSource.data(), 1, rtcSource.size(), fp) != rtcSource.size() ||
          fwrite(kernel.c_str(), 1, kernel.size(), fp) != kernel.size()) {
        throw std::runtime_error("Error writing file");
      }
      fclose(fp);

      // Export only the kernel entry points, everything else stays local to the RTC library to avoid symbol collisions
      fp = fopen((fileBase + RTC_MAP_EXTENSION).c_str(), "w+b");
      if (fp == nullptr || fprintf(fp, "{ global: krnl_*; local: *; };\n") < 0) {
        throw std::runtime_error("Error writing version script");
      }
      fclose(fp);

      std::string command = baseCommand;
      if (deterministic) {
        command += std::string(" ") + std::string(_binary_GPUReconstructionCPUDevicertc_command_no_fast_math_start, _binary_GPUReconstructionCPUDevicertc_command_no_fast_math_len);
      }
      command += " -Wl,--version-script=" + fileBase + RTC_MAP_EXTENSION + " -x c++ " + fileBase + RTC_SRC_EXTENSION + " -o " + fileBase + RTC_BIN_EXTENSION;
      if (GetProcessingSettings().debugLevel < 0) {
        command += " &> /dev/null";
      } else if (GetProcessingSettings().debugLevel < 2) {
        command += " > /dev/null";
      }
      if (GetProcessingSettings().debugLevel >= 3) {
        printf("Running command %s\n", command.c_str());
      }
      if (system(command.c_str())) {
        if (GetProcessingSettings().debugLevel >= 3) {
          printf("Source code file: %s", (fileBase + RTC_SRC_EXTENSION).c_str());
        }
        throw std::runtime_error("Error during CPUDevice RTC compilation");
      } // clang-format off
    }, tbb::simple_partitioner()); // clang-format on
    if (GetProcessingSettings().debugLevel >= 0) {
      GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
    }
    if (GetProcessingSettings().rtc.cacheOutput) {
      FILE* fp = fopen(cacheFile.c_str(), "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Cannot open cache file for writing");
      }
      GPUInfo("Storing RTC compilation result in cache file");

      if (fwrite(shasource, 1, 20, fp) != 20 ||
          fwrite(shaparam, 1, 20, fp) != 20 ||
          fwrite(shacmd, 1, 20, fp) != 20 ||
          fwrite(shakernels, 1, 20, fp) != 20 ||
          fwrite(shabounds, 1, 20, fp) != 20 ||
          fwrite(shaversion, 1, 20, fp) != 20 ||
          fwrite(&GetProcessingSettings().rtc, sizeof(GetProcessingSettings().rtc), 1, fp) != 1) {
        throw std::runtime_error("Error writing cache file");
      }

      std::vector<char> buffer;
      for (uint32_t i = 0; i < nCompile; i++) {
        FILE* fp2 = fopen((filename + "_" + std::to_string(i) + RTC_BIN_EXTENSION).c_str(), "rb");
        if (fp2 == nullptr) {
          throw std::runtime_error("Cannot open RTC library file");
        }
        fseek(fp2, 0, SEEK_END);
        size_t size = ftell(fp2);
        buffer.resize(size);
        fseek(fp2, 0, SEEK_SET);
        if (fread(buffer.data(), 1, size, fp2) != size) {
          throw std::runtime_error("Error reading RTC library file");
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
  if (GetProcessingSettings().rtc.cacheOutput && GetProcessingSettings().rtctech.cacheMutex) {
    if (lockf(fd, F_ULOCK, 0)) {
      throw std::runtime_error("Error unlocking RTC cache mutex file");
    }
    close(fd);
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
