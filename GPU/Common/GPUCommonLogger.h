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
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdio>
#endif

namespace o2::gpu::internal
{
#if defined(GPUCA_GPUCODE_DEVICE)
struct DummyLogger {
  template <typename... Args>
  GPUd() DummyLogger& operator<<(Args... args)
  {
    return *this;
  }
};
#else
template <typename... Args>
void LOGP_internal(const char* str, Args... args)
{
  printf("%s\n", str);
}
#endif
enum class loglevel : int32_t {
  debug = 0,
  info = 1,
  warning = 2,
  important = 3,
  alarm = 4,
  error = 5,
  fatal = 6
};
} // namespace o2::gpu::internal

#ifdef GPUCA_GPUCODE_DEVICE // clang-format off
// ---------- begin GPUCA_GPUCODE_DEVICE ----------

#if defined(__OPENCL__) || !defined(GPUCA_GPU_DEBUG_PRINT)
#define LOG(...) o2::gpu::internal::DummyLogger()
#define LOGF(...)
#define LOGP(...)

#else
#define LOG(...) o2::gpu::internal::DummyLogger()
// #define LOG(...) static_assert(false, "LOG(...) << ... unsupported in GPU code");
#define LOGF(type, string, ...) do { if (o2::gpu::internal::loglevel::type >= o2::gpu::internal::loglevel::info) { printf(string "\n", ##__VA_ARGS__); }} while (false);
#define LOGP(...)
// #define LOGP(...) static_assert(false, "LOGP(...) unsupported in GPU code");
#endif

// ---------- end GPUCA_GPUCODE_DEVICE ----------
#elif defined(GPUCA_STANDALONE) || defined(GPUCA_GPUCODE_COMPILEKERNELS) || defined(GPUCA_COMPILEKERNELS)
// ---------- begin GPUCA_STANDALONE / COMPILEKERNELS ----------

#include <iostream>
#include <cstdio>
#define LOG(type) std::cout
#define LOGF(type, string, ...) do { if (o2::gpu::internal::loglevel::type >= o2::gpu::internal::loglevel::info) { printf(string "\n", ##__VA_ARGS__); }} while (false);
#if !defined(GPUCA_NO_FMT) && !defined(GPUCA_GPUCODE)
#define LOGP(type, string, ...) do { if (o2::gpu::internal::loglevel::type >= o2::gpu::internal::loglevel::info) { fmt::print(string, ##__VA_ARGS__); printf("\n"); }} while (false);
#else
#define LOGP(type, string, ...) do { if (o2::gpu::internal::loglevel::type >= o2::gpu::internal::loglevel::info) { o2::gpu::internal::LOGP_internal(string, ##__VA_ARGS__); }} while (false);
#endif
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
#if !defined(GPUCA_NO_FMT)
#include <fmt/format.h>
#else
namespace fmt
{
template <typename... Args>
static const char* format(Args... args)
{
  return "";
}
} // namespace fmt
#endif
#endif

// ---------- end GPUCA_STANDALONE / COMPILEKERNELS ----------
#else
#include <Framework/Logger.h>
#endif // clang-format on

#endif
