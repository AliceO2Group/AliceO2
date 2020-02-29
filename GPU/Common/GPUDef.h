// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "GPUDefGPUParameters.h"
#include "GPUDefOpenCL12Templates.h"
#include "GPUCommonRtypes.h"

// Macros for masking ptrs in OpenCL kernel calls as unsigned long (The API only allows us to pass buffer objects)
#ifdef __OPENCL__
  #define GPUPtr1(a, b) unsigned long b
  #define GPUPtr2(a, b) ((__global a) (a) b)
#else
  #define GPUPtr1(a, b) a b
  #define GPUPtr2(a, b) b
#endif

#ifdef GPUCA_FULL_CLUSTERDATA
  #define GPUCA_EVDUMP_FILE "event_full"
#else
  #define GPUCA_EVDUMP_FILE "event"
#endif

#ifdef GPUCA_GPUCODE
  #define CA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUsharedref() MEM_LOCAL(vartype) & __restrict__ varname = varshared;
  #define CA_SHARED_STORAGE(storage) storage
  #define CA_SHARED_CACHE(target, src, size) \
    static_assert((size) % sizeof(int) == 0, "Invalid shared cache size"); \
    for (unsigned int i_shared_cache = get_local_id(0); i_shared_cache < (size) / sizeof(int); i_shared_cache += get_local_size(0)) { \
      reinterpret_cast<GPUsharedref() int*>(target)[i_shared_cache] = reinterpret_cast<GPUglobalref() const int*>(src)[i_shared_cache]; \
    }
  #define CA_SHARED_CACHE_REF(target, src, size, reftype, ref) \
    CA_SHARED_CACHE(target, src, size) \
    GPUsharedref() const reftype* __restrict__ ref = (target)
#else
  #define CA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUglobalref() MEM_GLOBAL(vartype) & __restrict__ varname = varglobal;
  #define CA_SHARED_STORAGE(storage)
  #define CA_SHARED_CACHE(target, src, size)
  #define CA_SHARED_CACHE_REF(target, src, size, reftype, ref) GPUglobalref() const reftype* __restrict__ ref = src
#endif

#ifdef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  #define CA_TEXTURE_FETCH(type, texture, address, entry) tex1Dfetch(texture, ((char*) address - tracker.Data().GPUTextureBase()) / sizeof(type) + entry);
#else
  #define CA_TEXTURE_FETCH(type, texture, address, entry) address[entry];
#endif

#endif //GPUTPCDEF_H

#ifdef GPUCA_CADEBUG
  #ifdef CADEBUG
    #undef CADEBUG
  #endif
  #ifdef GPUCA_CADEBUG_ENABLED
    #undef GPUCA_CADEBUG_ENABLED
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
