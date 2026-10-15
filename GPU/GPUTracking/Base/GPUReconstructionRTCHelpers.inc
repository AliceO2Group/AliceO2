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

/// \file GPUReconstructionRTCHelpers.inc
/// \author David Rohr

// Backend-independent helpers for the genRTC functions of the RTC backends, include only once per backend source file

#include "GPUSettings.h"
#include "GPUParamRTC.h"
#include "GPUDefParametersLoad.inc"
#include "GPUNoFastMathKernels.h"
#include "GPULogging.h"
#include "Framework/SHA1.h"
#include "utils/timer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <oneapi/tbb.h>

namespace o2::gpu::internal::rtc
{

static std::string paramCode(const GPUSettingsProcessing& proc, const GPUParam& param) // RTC defines and parameters, common to all compilations
{
  return std::string("#define GPUCA_RTC_CODE\n") +
         (proc.rtc.optSpecialCode ? "#define GPUCA_RTC_SPECIAL_CODE(...) __VA_ARGS__\n" : "#define GPUCA_RTC_SPECIAL_CODE(...)\n") +
         (proc.rtc.optConstexpr ? "#define GPUCA_RTC_CONSTEXPR constexpr\n" : "#define GPUCA_RTC_CONSTEXPR\n") +
         GPUParamRTC::generateRTCCode(param, proc.rtc.optConstexpr);
}

static void tmpFilename(std::string& filename, const char* defaultPrefix)
{
  filename = (filename == "" ? defaultPrefix : filename) + std::to_string(getpid()) + "_" + std::to_string(rand());
}

static std::string joinKernels(const std::vector<std::string>& kernels)
{
  std::string retVal;
  for (const auto& kernel : kernels) {
    retVal += kernel + "\n";
  }
  return retVal;
}

static std::string baseCommand(const GPUSettingsProcessing& proc, std::string_view embeddedCommand) // Prepended command + compiler command (embedded or from O2_GPU_RTC_OVERRIDE_CMD)
{
  const char* overrideCmd = getenv("O2_GPU_RTC_OVERRIDE_CMD");
  return (proc.rtctech.prependCommand != "" ? (proc.rtctech.prependCommand + " ") : "") + (overrideCmd ? std::string(overrideCmd) : std::string(embeddedCommand));
}

static void loadLaunchBounds(const GPUSettingsProcessing& proc, GPUDefParameters& par)
{
  if (proc.rtctech.loadLaunchBoundsFromFile.size() == 0) {
    return;
  }
  FILE* fp = fopen(proc.rtctech.loadLaunchBoundsFromFile.c_str(), "rb");
  if (fp == nullptr) {
    throw std::runtime_error("Cannot open launch bounds parameter module file");
  }
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  if (size != sizeof(par)) {
    throw std::runtime_error("launch bounds parameter file has incorrect size");
  }
  fseek(fp, 0, SEEK_SET);
  if (fread(&par, 1, size, fp) != size) {
    throw std::runtime_error("Error reading launch bounds parameter file");
  }
  fclose(fp);
}

static std::string launchBoundsCode(const GPUSettingsProcessing& proc, const GPUDefParameters& par, int32_t minBlockFactor, uint32_t warpSize) // Prints the launch bounds if requested
{
  std::string retVal = GPUDefParametersExport(par, true, minBlockFactor) + "#define GPUCA_WARP_SIZE " + std::to_string(warpSize) + "\n";
  if (proc.rtctech.printLaunchBounds || proc.debugLevel >= 3) {
    GPUInfo("RTC Launch Bounds:\n%s", retVal.c_str());
  }
  return retVal;
}

static bool deterministic(const GPUSettingsProcessing& proc, const std::string& kernelName) // Whether compilation of kernel (or of all kernels with !compilePerKernel) uses deterministic code and no fast math
{
  return proc.rtc.deterministic || (proc.rtc.compilePerKernel && noFastMathKernels.find(kernelName) != noFastMathKernels.end());
}

static void writeSource(const std::string& file, bool deterministic, const std::string& param, const std::string& launchBounds, std::string_view source, const std::string& kernel)
{
  const std::string deterministicStr = deterministic ? "#define GPUCA_DETERMINISTIC_CODE(det, indet) det\n" : "#define GPUCA_DETERMINISTIC_CODE(det, indet) indet\n";
  FILE* fp = fopen(file.c_str(), "w+b");
  if (fp == nullptr) {
    throw std::runtime_error("Error opening file");
  }
  for (std::string_view part : {std::string_view(deterministicStr), std::string_view(param), std::string_view(launchBounds), source, std::string_view(kernel)}) {
    if (fwrite(part.data(), 1, part.size(), fp) != part.size()) {
      throw std::runtime_error("Error writing file");
    }
  }
  fclose(fp);
}

static void runCommand(const GPUSettingsProcessing& proc, std::string command, const std::string& sourceFile, const char* errorMessage) // Runs the compiler command, output suppressed depending on the debug level
{
  if (proc.debugLevel < 0) {
    command += " &> /dev/null";
  } else if (proc.debugLevel < 2) {
    command += " > /dev/null";
  }
  if (proc.debugLevel >= 3) {
    printf("Running command %s\n", command.c_str());
  }
  if (system(command.c_str())) {
    if (proc.debugLevel >= 3) {
      printf("Source code file: %s", sourceFile.c_str());
    }
    throw std::runtime_error(errorMessage);
  }
}

template <class T>
static void compile(const GPUSettingsProcessing& proc, uint32_t nCompile, const char* backend, T&& compileOne) // Runs compileOne(i) for all compilations in parallel
{
  if (proc.debugLevel >= 0) {
    GPUInfo("Starting %s RTC Compilation", backend);
  }
  HighResTimer rtcTimer;
  rtcTimer.ResetStart();
  tbb::parallel_for<uint32_t>(0, nCompile, compileOne, tbb::simple_partitioner());
  if (proc.debugLevel >= 0) {
    GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
  }
}

class Cache // RTC cache file <cacheFolder>/rtc.<backend>.cache.<id> (no id with rtctech.ignoreCacheValid) (if rtc.cacheOutput), id from the SHAs of the inputs, locked per backend while the object exists. Usage: if (!cache.load(...)) { compile; cache.store(...); }
{
 public:
  static constexpr uint32_t N_SHA = 6;
  using Inputs = std::array<std::string_view, N_SHA>; // source, param, command line, kernel definitions, launch bounds, compiler versions

  Cache(const GPUSettingsProcessing& proc, const char* backend, const Inputs& inputs) : mProc(proc), mPrefix(std::string("rtc.") + backend + ".cache.")
  {
    if (!mProc.rtc.cacheOutput) {
      return;
    }
    char shaSettings[21]; // Only for the id, the settings are compared with operator== on load (the raw bytes include padding)
    for (uint32_t i = 0; i < N_SHA; i++) {
      o2::framework::internal::SHA1(mSHA[i], inputs[i].data(), inputs[i].size());
    }
    o2::framework::internal::SHA1(shaSettings, (const char*)&mProc.rtc, sizeof(mProc.rtc));
    uint64_t id = 0;
    for (uint32_t i = 0; i <= N_SHA; i++) {
      for (uint32_t j = 0; j < 20; j += 5) { // XOR of all SHAs in 5-byte blocks
        uint64_t block = 0;
        memcpy(&block, (i < N_SHA ? mSHA[i] : shaSettings) + j, 5);
        id ^= block;
      }
    }
    char idStr[11];
    snprintf(idStr, sizeof(idStr), "%010llX", (unsigned long long)id);
    mFileName = mProc.rtctech.ignoreCacheValid ? mPrefix.substr(0, mPrefix.size() - 1) : (mPrefix + idStr); // Without id for ignoreCacheValid, to find the cache file for any input
    mFile = mProc.rtctech.cacheFolder + "/" + mFileName;
    if (mProc.rtctech.cacheFolder != ".") {
      std::filesystem::create_directories(mProc.rtctech.cacheFolder);
    }
    if (mProc.rtctech.cacheMutex) {
      mode_t mask = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
      mFd = open((mProc.rtctech.cacheFolder + "/cache_" + backend + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, mask);
      if (mFd == -1) {
        throw std::runtime_error("Error opening rtc cache mutex lock file");
      }
      fchmod(mFd, mask);
      if (lockf(mFd, F_LOCK, 0)) {
        close(mFd);
        mFd = -1;
        throw std::runtime_error("Error locking rtc cache mutex file");
      }
    }
  }
  ~Cache()
  {
    if (mFd != -1) {
      if (lockf(mFd, F_ULOCK, 0)) {
        GPUError("Error unlocking RTC cache mutex file");
      }
      close(mFd);
    }
  }
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  bool load(const std::string& filename, const char* binExtension, uint32_t nCompile) // Writes the cached binaries to filename_<i><binExtension>, false if no valid cache
  {
    if (!mProc.rtc.cacheOutput) {
      return false;
    }
    FILE* fp = fopen(mFile.c_str(), "rb");
    if (fp == nullptr) {
      return false;
    }
    bool retVal = read(fp, filename, binExtension, nCompile);
    fclose(fp);
    if (retVal) {
      GPUInfo("Using RTC cache file");
      std::error_code ec;
      std::filesystem::last_write_time(mFile, std::filesystem::file_time_type::clock::now(), ec); // Mark as recently used for removeOldFiles()
    }
    return retVal;
  }

  void store(const std::string& filename, const char* binExtension, uint32_t nCompile)
  {
    if (!mProc.rtc.cacheOutput) {
      return;
    }
    FILE* fp = fopen(mFile.c_str(), "w+b");
    if (fp == nullptr) {
      throw std::runtime_error("Cannot open cache file for writing");
    }
    GPUInfo("Storing RTC compilation result in cache file");
    for (uint32_t i = 0; i < N_SHA; i++) {
      if (fwrite(mSHA[i], 1, 20, fp) != 20) {
        throw std::runtime_error("Error writing cache file");
      }
    }
    if (fwrite(&mProc.rtc, sizeof(mProc.rtc), 1, fp) != 1) {
      throw std::runtime_error("Error writing cache file");
    }
    std::vector<char> buffer;
    for (uint32_t i = 0; i < nCompile; i++) {
      FILE* fp2 = fopen((filename + "_" + std::to_string(i) + binExtension).c_str(), "rb");
      if (fp2 == nullptr) {
        throw std::runtime_error("Cannot open RTC binary file");
      }
      fseek(fp2, 0, SEEK_END);
      size_t size = ftell(fp2);
      buffer.resize(size);
      fseek(fp2, 0, SEEK_SET);
      if (fread(buffer.data(), 1, size, fp2) != size) {
        throw std::runtime_error("Error reading RTC binary file");
      }
      fclose(fp2);
      if (fwrite(&size, sizeof(size), 1, fp) != 1 || fwrite(buffer.data(), 1, size, fp) != size) {
        throw std::runtime_error("Error writing cache file");
      }
    }
    fclose(fp);
    removeOldFiles();
  }

 private:
  void removeOldFiles() const // Keeps the rtctech.cacheSize most recent cache files of this backend, including the one just written
  {
    if (mProc.rtctech.cacheSize == 0 || mProc.rtctech.ignoreCacheValid) {
      return;
    }
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(mProc.rtctech.cacheFolder)) {
      const std::string name = entry.path().filename().string();
      if (name.starts_with(mPrefix) && name != mFileName && entry.is_regular_file(ec)) {
        files.emplace_back(entry.last_write_time(ec), entry.path());
      }
    }
    if (files.size() < mProc.rtctech.cacheSize) {
      return;
    }
    std::sort(files.begin(), files.end(), std::greater<>()); // Most recent first
    for (uint32_t i = mProc.rtctech.cacheSize - 1; i < files.size(); i++) {
      std::filesystem::remove(files[i].second, ec);
    }
  }

  bool read(FILE* fp, const std::string& filename, const char* binExtension, uint32_t nCompile)
  {
    static constexpr const char* names[N_SHA] = {"source", "param", "command line", "kernel definitions", "launch bounds", "compiler versions"};
    for (uint32_t i = 0; i < N_SHA; i++) {
      char sharead[20];
      if (fread(sharead, 1, 20, fp) != 20) {
        throw std::runtime_error("Cache file corrupt");
      }
      if (mProc.debugLevel >= 3) {
        char shaprint1[41], shaprint2[41];
        for (uint32_t j = 0; j < 20; j++) {
          snprintf(shaprint1 + 2 * j, 3, "%02X", (uint8_t)mSHA[i][j]);
          snprintf(shaprint2 + 2 * j, 3, "%02X", (uint8_t)sharead[j]);
        }
        GPUInfo("SHA for %s: expected %s, read %s", names[i], shaprint1, shaprint2);
      }
      if (!mProc.rtctech.ignoreCacheValid && memcmp(sharead, mSHA[i], 20)) {
        GPUInfo("Cache file content outdated (%s)", names[i]);
        return false;
      }
    }
    GPUSettingsProcessingRTC cachedSettings;
    static_assert(std::is_trivially_copyable_v<GPUSettingsProcessingRTC> == true, "GPUSettingsProcessingRTC must be POD");
    if (fread(&cachedSettings, sizeof(cachedSettings), 1, fp) != 1) {
      throw std::runtime_error("Cache file corrupt");
    }
    if (!mProc.rtctech.ignoreCacheValid && !(cachedSettings == mProc.rtc)) {
      GPUInfo("Cache file content outdated (rtc parameters)");
      return false;
    }
    std::vector<char> buffer;
    for (uint32_t i = 0; i < nCompile; i++) {
      size_t len;
      if (fread(&len, sizeof(len), 1, fp) != 1) {
        throw std::runtime_error("Cache file corrupt");
      }
      buffer.resize(len);
      if (fread(buffer.data(), 1, len, fp) != len) {
        throw std::runtime_error("Cache file corrupt");
      }
      FILE* fp2 = fopen((filename + "_" + std::to_string(i) + binExtension).c_str(), "w+b");
      if (fp2 == nullptr) {
        throw std::runtime_error("Cannot open tmp file");
      }
      if (fwrite(buffer.data(), 1, len, fp2) != len) {
        throw std::runtime_error("Error writing file");
      }
      fclose(fp2);
    }
    return true;
  }

  const GPUSettingsProcessing& mProc;
  const std::string mPrefix;
  std::string mFileName, mFile;
  char mSHA[N_SHA][21];
  int32_t mFd = -1;
};

} // namespace o2::gpu::internal::rtc
