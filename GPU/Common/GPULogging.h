// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPULogging.h
/// \author David Rohr

#ifndef GPULOGGING_H
#define GPULOGGING_H

#include "GPUCommonDef.h"
// clang-format off
#if !defined(GPUCA_NOCOMPAT)
  // Cannot do anything for ROOT5CINT / OpenCL1, so just disable
  #define GPUInfo(...)
  #define GPUImportant(...)
  #define GPUWarning(...)
  #define GPUError(...)
  #define GPUFatal(...)
#elif defined(GPUCA_STANDALONE) || defined(GPUCA_GPUCODE_DEVICE) || (defined(GPUCA_ALIROOT_LIB) && defined(__CUDACC__) && defined(__cplusplus) && __cplusplus < 201703L)
  // For standalone / CUDA / HIP, we just use printf, which should be available
  // Temporarily, we also have to handle CUDA on AliRoot with O2 defaults due to ROOT / CUDA incompatibilities
  #define GPUInfo(string, ...)            \
    {                                     \
      printf(string "\n", ##__VA_ARGS__); \
    }
  #define GPUImportant(...) GPUInfo(__VA_ARGS__)
  #ifdef GPUCA_GPUCODE_DEVICE
    #define GPUWarning(...) GPUInfo(__VA_ARGS__)
    #define GPUError(...) GPUInfo(__VA_ARGS__)
    #define GPUFatal(...) GPUInfo(__VA_ARGS__)
  #else
    #define GPUWarning(string, ...)                  \
      {                                              \
        fprintf(stderr, string "\n", ##__VA_ARGS__); \
      }
    #define GPUError(...) GPUWarning(__VA_ARGS__)
    #ifdef GPUCA_NOCOMPAT
      #define GPUFatal(string, ...)                    \
        {                                              \
          fprintf(stderr, string "\n", ##__VA_ARGS__); \
          throw std::exception();                      \
        }
    #else
      #define GPUFatal(string, ...)                  \
        {                                            \
          fprintf(stderr, string "\n", __VA_ARGS__); \
          exit(1);                                   \
        }
    #endif
  #endif
#elif defined(GPUCA_ALIROOT_LIB)
  // Forward to HLT Logging functions for AliRoot
  #include "AliHLTLogging.h"
  #define GPUInfo(...) HLTInfo(__VA_ARGS__)
  #define GPUImportant(...) HLTImportant(__VA_ARGS__)
  #define GPUWarning(...) HLTWarning(__VA_ARGS__)
  #define GPUError(...) HLTError(__VA_ARGS__)
  #define GPUFatal(...) HLTFatal(__VA_ARGS__)
  // Workaround for static functions / classes not deriving from AliHLTLogging
  namespace AliGPU
  {
  namespace gpu
  {
  // We pollute the AliGPU::gpu namespace with some anonymous functions that catch the HLT...() magic
  namespace
  {
  AliHLTLogging gAliGPULog; // This creates a couple of bogus instances, but there are plenty anyway
  template <typename... Args>
  void LoggingVarargs(Args... args)
  {
    gAliGPULog.LoggingVarargs(args...);
  }
  template <typename... Args>
  bool CheckFilter(Args... args)
  {
    return gAliGPULog.CheckFilter(args...);
  }
  const char* Class_Name() { return "GPU"; };
  } // namespace
  } // namespace gpu
  } // namespace AliGPU
#elif defined(GPUCA_O2_LIB)
  // Forward to O2 LOGF logginf for O2
  #include "GPUCommonLogger.h"
  #define GPUInfo(...) LOGF(info, __VA_ARGS__)
  #define GPUImportant(...) LOGF(info, __VA_ARGS__)
  #define GPUWarning(...) LOGF(warning, __VA_ARGS__)
  #define GPUError(...) LOGF(error, __VA_ARGS__)
  #define GPUFatal(...) LOGF(fatal, __VA_ARGS__)
#endif

// clang-format on

#endif // GPULOGGING_H
