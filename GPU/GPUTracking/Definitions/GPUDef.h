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

/// \file GPUDef.h
/// \author David Rohr, Sergey Gorbunov

// clang-format off
#ifndef GPUDEF_H
#define GPUDEF_H

#include "GPUCommonDef.h"
#include "GPUDefConstantsAndSettings.h"
#include "GPUDefParametersWrapper.h"
#include "GPUCommonRtypes.h"

// Macros for kernel arguments. OpenCL can only pass buffer objects, so pointers
// are masked as uint64_t and cast back inside the kernel. MSL needs an explicit
// buffer index on every parameter, but can bind a pointer directly. The index is
// emitted per argument by o2_gpu_add_kernel; 0, 1 and 2 are taken by gpu_mem,
// the constant memory and the sector index.
#ifdef __OPENCL__
  #define GPUPtr1(idx, a, b) uint64_t b
  #define GPUPtr2(a, b) ((__generic a) (a) b)
  #define GPUArg1(idx, a, b) a b
#elif defined(__METAL__)
  // As for OpenCL, pointers travel as a 64-bit address: a pointer to a derived
  // class is not a valid kernel argument type in MSL either.
  #define GPUPtr1(idx, a, b) constant uint64_t& b [[buffer(idx)]]
  // through device and then to generic: the kernel's own buffers are device
  // memory, but the Thread() entry points take the pointer unannotated
  #define GPUPtr2(a, b) ((a)((device a)(b)))
  #define GPUArg1(idx, a, b) constant a& b [[buffer(idx)]]
#else
  #define GPUPtr1(idx, a, b) a b
  #define GPUPtr2(a, b) b
  #define GPUArg1(idx, a, b) a b
#endif

#define GPUCA_EVDUMP_FILE "event"

#ifdef GPUCA_GPUCODE
  #define GPUCA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUsharedref() vartype& __restrict__ varname = varshared;
  #define GPUCA_SHARED_STORAGE(storage) storage
  #define GPUCA_SHARED_CACHE(nThreads, iThread, target, src, size) \
    static_assert((size) % sizeof(int32_t) == 0, "Invalid shared cache size"); \
    for (uint32_t i_shared_cache = (iThread); i_shared_cache < (size) / sizeof(int32_t); i_shared_cache += (nThreads)) { \
      reinterpret_cast<GPUsharedref() int32_t*>(target)[i_shared_cache] = reinterpret_cast<GPUglobalref() const int32_t*>(src)[i_shared_cache]; \
    }
  #define GPUCA_SHARED_CACHE_REF(nThreads, iThread, target, src, size, reftype, ref) \
    GPUCA_SHARED_CACHE(nThreads, iThread, target, src, size) \
    GPUsharedref() const reftype* __restrict__ ref = (target)
#else
  #define GPUCA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUglobalref() vartype & __restrict__ varname = varglobal;
  #define GPUCA_SHARED_STORAGE(storage)
  #define GPUCA_SHARED_CACHE(nThreads, iThread, target, src, size)
  #define GPUCA_SHARED_CACHE_REF(nThreads, iThread, target, src, size, reftype, ref) GPUglobalref() const reftype* __restrict__ ref = src
#endif

#endif //GPUTPCDEF_H

#ifdef GPUCA_CADEBUG
  #ifdef CADEBUG
    #undef CADEBUG
  #endif
  #if GPUCA_CADEBUG == 1 && !defined(GPUCA_GPUCODE)
    #define CADEBUG(...) __VA_ARGS__
    #define CADEBUG2(cmd, ...) {__VA_ARGS__; cmd;}
    #define GPUCA_CADEBUG_ENABLED
  #endif
  #undef GPUCA_CADEBUG
#endif

#ifndef CADEBUG
  #define CADEBUG(...)
  #define CADEBUG2(cmd, ...) {cmd;}
#endif
// clang-format on
