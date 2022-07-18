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

/// \file GPUCommonLogger.h
/// \author David Rohr

#ifndef GPUCOMMONFAIRLOGGER_H
#define GPUCOMMONFAIRLOGGER_H

#include "GPUCommonDef.h"

#if defined(GPUCA_GPUCODE_DEVICE) || defined(__HIPCC__)
namespace o2::gpu::detail
{
struct DummyLogger {
  template <typename... Args>
  GPUhd() DummyLogger& operator<<(Args... args)
  {
    return *this;
  }
};
} // namespace o2::gpu::detail
#endif

#if defined(__OPENCL__)
#define LOG(...) o2::gpu::detail::DummyLogger()
#define LOGF(...)
#define LOGP(...)

#elif defined(GPUCA_GPUCODE_DEVICE) || defined(__HIPCC__)
#define LOG(...) o2::gpu::detail::DummyLogger()
//#define LOG(...) static_assert(false, "LOG(...) << ... unsupported in GPU code");
#define LOGF(type, string, ...)         \
  {                                     \
    printf(string "\n", ##__VA_ARGS__); \
  }
#define LOGP(type, string, ...) \
  {                             \
    printf(string "\n");        \
  }

#elif defined(GPUCA_STANDALONE) ||                    \
  defined(GPUCA_ALIROOT_LIB) ||                       \
  (!defined(__cplusplus) || __cplusplus < 201703L) || \
  (defined(__HIPCC__) && (!defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0))
#include <iostream>
#include <cstdio>
#define LOG(type) std::cout
#define LOGF(type, string, ...)         \
  {                                     \
    printf(string "\n", ##__VA_ARGS__); \
  }
#ifdef GPUCA_ALIROOT_LIB
#define LOGP(...)
#else
#define LOGP(type, string, ...) \
  {                             \
    printf("%s", string);       \
    printf("\n");               \
  }
#endif

#else
#include <Framework/Logger.h>

#endif

#endif
