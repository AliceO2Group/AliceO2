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
#include <cuda.h>
#include <cuda_fp16.h>
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "GPUParamRTC.h"
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

int GPUReconstructionCUDABackend::genRTC()
{
#ifndef GPUCA_ALIROOT_LIB
  std::string rtcparam = GPUParamRTC::generateRTCCode(param(), mProcessingSettings.rtcConstexpr);
  std::string filename = "/tmp/o2cagpu_rtc_";
  filename += std::to_string(getpid());
  filename += "_";
  filename += std::to_string(rand());
#ifdef GPUCA_HAVE_O2HEADERS
  char shasource[21], shaparam[21], shacmd[21];
  if (mProcessingSettings.cacheRTC) {
    o2::framework::internal::SHA1(shasource, _curtc_GPUReconstructionCUDArtc_cu_src, _curtc_GPUReconstructionCUDArtc_cu_src_size);
    o2::framework::internal::SHA1(shaparam, rtcparam.c_str(), rtcparam.size());
    o2::framework::internal::SHA1(shacmd, _curtc_GPUReconstructionCUDArtc_cu_command, strlen(_curtc_GPUReconstructionCUDArtc_cu_command));
  }
#endif

  bool cacheLoaded = false;
  if (mProcessingSettings.cacheRTC) {
#ifndef GPUCA_HAVE_O2HEADERS
    throw std::runtime_error("Cannot use RTC cache without O2 headers");
#else
    FILE* fp = fopen("rtc.cuda.cache", "rb");
    char sharead[20];
    if (fp) {
      size_t len;
      while (true) {
        if (fread(sharead, 1, 20, fp) != 20) {
          GPUError("Cache file corrupt");
          break;
        }
        if (memcmp(sharead, shasource, 20)) {
          GPUInfo("Cache file content outdated");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          GPUError("Cache file corrupt");
          break;
        }
        if (memcmp(sharead, shaparam, 20)) {
          GPUInfo("Cache file content outdated");
          break;
        }
        if (fread(sharead, 1, 20, fp) != 20) {
          GPUError("Cache file corrupt");
          break;
        }
        if (memcmp(sharead, shacmd, 20)) {
          GPUInfo("Cache file content outdated");
          break;
        }
        if (fread(&len, sizeof(len), 1, fp) != 1) {
          GPUError("Cache file corrupt");
          break;
        }
        std::unique_ptr<char[]> buffer;
        buffer.reset(new char[len]);
        if (fread(buffer.get(), 1, len, fp) != len) {
          GPUError("Cache file corrupt");
          break;
        }
        FILE* fp2 = fopen((filename + ".o").c_str(), "w+b");
        if (fp2 == nullptr) {
          GPUError("Cannot open tmp file");
          break;
        }
        fclose(fp);
        fp = fp2;
        if (fwrite(buffer.get(), 1, len, fp) != len) {
          GPUError("Error writing file");
          break;
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
    if (mProcessingSettings.debugLevel >= 3) {
      printf("Writing to %s\n", filename.c_str());
    }
    FILE* fp = fopen((filename + ".cu").c_str(), "w+b");
    if (fp == nullptr) {
      throw std::runtime_error("Error opening file");
    }
    if (fwrite(rtcparam.c_str(), 1, rtcparam.size(), fp) != rtcparam.size()) {
      throw std::runtime_error("Error writing file");
    }
    if (fwrite(_curtc_GPUReconstructionCUDArtc_cu_src, 1, _curtc_GPUReconstructionCUDArtc_cu_src_size, fp) != _curtc_GPUReconstructionCUDArtc_cu_src_size) {
      throw std::runtime_error("Error writing file");
    }
    fclose(fp);
    std::string command = _curtc_GPUReconstructionCUDArtc_cu_command;
    command += " -cubin -c " + filename + ".cu -o " + filename + ".o";
    if (mProcessingSettings.debugLevel >= 3) {
      printf("Running command %s\n", command.c_str());
    }
    if (system(command.c_str())) {
      return 1;
    }
    if (mProcessingSettings.debugLevel >= 0) {
      GPUInfo("RTC Compilation finished (%f seconds)", rtcTimer.GetCurrentElapsedTime());
    }
    if (mProcessingSettings.cacheRTC) {
      fp = fopen((filename + ".o").c_str(), "rb");
      if (fp == nullptr) {
        throw std::runtime_error("Cannot open cuda module file");
      }
      fseek(fp, 0, SEEK_END);
      size_t size = ftell(fp);
      std::unique_ptr<char[]> buffer{new char[size]};
      fseek(fp, 0, SEEK_SET);
      if (fread(buffer.get(), 1, size, fp) != size) {
        throw std::runtime_error("Error reading cuda module file");
      }
      fclose(fp);
      fp = fopen("rtc.cuda.cache", "w+b");
      if (fp == nullptr) {
        throw std::runtime_error("Cannot open cache file for writing");
      }
      GPUInfo("Storing RTC compilation result in cache file");
      if (fwrite(shasource, 1, 20, fp) != 20 ||
          fwrite(shaparam, 1, 20, fp) != 20 ||
          fwrite(shacmd, 1, 20, fp) != 20 ||
          fwrite(&size, sizeof(size), 1, fp) != 1 ||
          fwrite(buffer.get(), 1, size, fp) != size) {
        throw std::runtime_error("Error writing cache file");
      }
      fclose(fp);
    }
  }

  GPUFailedMsg(cuModuleLoad(&mInternals->rtcModule, (filename + ".o").c_str()));
  remove((filename + ".cu").c_str());
  remove((filename + ".o").c_str());

#endif
  return 0;
}
