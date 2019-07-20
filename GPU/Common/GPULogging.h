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
#define GPUInfo(...)     \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUImportant(...) \
  {                       \
    printf(__VA_ARGS__);  \
    printf("\n");         \
  }
#define GPUWarning(...)  \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUError(...)    \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUFatal(...)    \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
    exit(1);             \
  }
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
#define GPUInfo(...) LOGF(INFO, __VA_ARGS__)
#define GPUImportant(...) LOGF(INFO, __VA_ARGS__)
#define GPUWarning(...) LOGF(WARNING, __VA_ARGS__)
#define GPUError(...) LOGF(ERROR, __VA_ARGS__)
#define GPUFatal(...) LOGF(FATAL, __VA_ARGS__)
#endif

#endif // GPULOGGING_H
