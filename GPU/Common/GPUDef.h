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

//Macros for GRID dimension
#if defined(__CUDACC__)
  #define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
  #define get_global_size(dim) (blockDim.x * gridDim.x)
  #define get_num_groups(dim) (gridDim.x)
  #define get_local_id(dim) (threadIdx.x)
  #define get_local_size(dim) (blockDim.x)
  #define get_group_id(dim) (blockIdx.x)
#elif defined(__HIPCC__)
  #define get_global_id(dim) (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x)
  #define get_global_size(dim) (hipBlockDim_x * hipGridDim_x)
  #define get_num_groups(dim) (hipGridDim_x)
  #define get_local_id(dim) (hipThreadIdx_x)
  #define get_local_size(dim) (hipBlockDim_x)
  #define get_group_id(dim) (hipBlockIdx_x)
#elif defined(__OPENCL__)
  //Using OpenCL defaults
#else
  #define get_global_id(dim) iBlock
  #define get_global_size(dim) nBlocks
  #define get_num_groups(dim) nBlocks
  #define get_local_id(dim) 0
  #define get_local_size(dim) 1
  #define get_group_id(dim) iBlock
#endif

#ifdef GPUCA_FULL_CLUSTERDATA
  #define GPUCA_EVDUMP_FILE "event_full"
#else
  #define GPUCA_EVDUMP_FILE "event"
#endif

#ifdef GPUCA_GPUCODE
  #define CA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUsharedref() MEM_LOCAL(vartype) &varname = varshared;
#else
  #define CA_MAKE_SHARED_REF(vartype, varname, varglobal, varshared) const GPUglobalref() MEM_GLOBAL(vartype) &varname = varglobal;
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
