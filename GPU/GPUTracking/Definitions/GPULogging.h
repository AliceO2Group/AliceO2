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

/// \file GPULogging.h
/// \author David Rohr

#ifndef GPULOGGING_H
#define GPULOGGING_H

#include "GPUCommonDef.h"
// clang-format off
#if defined(GPUCA_GPUCODE_DEVICE) && !defined(GPUCA_GPU_DEBUG_PRINT)
  // Compile-time disable for performance-reasons
  #define GPUInfo(...)
  #define GPUImportant(...)
  #define GPUWarning(...)
  #define GPUAlarm(...)
  #define GPUError(...)
  #define GPUCritical(...)
  #define GPUFatal(...)
#elif defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE_DEVICE) && !defined(GPUCA_NO_FMT)
  #include <cstdio>
  #pragma GCC diagnostic push
  #if defined(__FAST_MATH__) && defined(__clang__)
  #pragma GCC diagnostic ignored "-Wnan-infinity-disabled"
  #endif
  #include <fmt/printf.h>
  #pragma GCC diagnostic pop
  #define GPUInfo(string, ...)                 \
    {                                          \
      fmt::printf(string "\n", ##__VA_ARGS__); \
    }
  #define GPUImportant(...) GPUInfo(__VA_ARGS__)
  #define GPUWarning(string, ...)                       \
    {                                                   \
      fmt::fprintf(stderr, string "\n", ##__VA_ARGS__); \
    }
  #define GPUError(...) GPUWarning(__VA_ARGS__)
  #define GPUCritical(...) GPUWarning(__VA_ARGS__)
  #define GPUAlarm(...) GPUWarning(__VA_ARGS__)
  #define GPUFatal(string, ...)                         \
    {                                                   \
      fmt::fprintf(stderr, string "\n", ##__VA_ARGS__); \
      throw std::exception();                           \
    }
#elif defined(GPUCA_STANDALONE) || defined(GPUCA_GPUCODE_DEVICE)
  // For standalone / CUDA / HIP, we just use printf, which should be available
  #include <cstdio>
  #define GPUInfo(string, ...)            \
    {                                     \
      printf(string "\n", ##__VA_ARGS__); \
    }
  #define GPUImportant(...) GPUInfo(__VA_ARGS__)
  #ifdef GPUCA_GPUCODE_DEVICE
    #define GPUWarning(...) GPUInfo(__VA_ARGS__)
    #define GPUAlarm(...) GPUInfo(__VA_ARGS__)
    #define GPUError(...) GPUInfo(__VA_ARGS__)
    #define GPUFatal(...) GPUInfo(__VA_ARGS__)
  #else
    #define GPUWarning(string, ...)                  \
      {                                              \
        fprintf(stderr, string "\n", ##__VA_ARGS__); \
      }
    #define GPUAlarm(...) GPUWarning(__VA_ARGS__)
    #define GPUError(...) GPUWarning(__VA_ARGS__)
    #define GPUCritical(...) GPUWarning(__VA_ARGS__)
    #define GPUFatal(string, ...)                  \
      {                                            \
        fprintf(stderr, string "\n", ##__VA_ARGS__); \
        exit(1);                                   \
      }
  #endif
#elif defined(GPUCA_O2_LIB) || defined(GPUCA_O2_INTERFACE)
  // Forward to O2 LOGF logginf for O2
  #include "GPUCommonLogger.h"
  #define GPUInfo(...) LOGF(info, __VA_ARGS__)
  #define GPUImportant(...) LOGF(info, __VA_ARGS__)
  #define GPUWarning(...) LOGF(warning, __VA_ARGS__)
  #define GPUAlarm(...) LOGF(alarm, __VA_ARGS__)
  #define GPUError(...) LOGF(error, __VA_ARGS__)
  #define GPUCritical(...) LOGF(critical, __VA_ARGS__)
  #define GPUFatal(...) LOGF(fatal, __VA_ARGS__)
#endif

// clang-format on

#endif // GPULOGGING_H
