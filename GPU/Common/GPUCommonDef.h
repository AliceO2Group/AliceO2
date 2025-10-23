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

/// \file GPUCommonDef.h
/// \author David Rohr

// This is the base header to be included by all files that should feature GPU suppurt.
// Incompatible code that cannot compile on GPU must be protected by one of the checks below.
// The usual approach would be to protect with GPUCA_GPUCODE. This will be sufficient for all functions. If header includes still show errors, use GPUCA_ALIGPUCODE

// The following checks are increasingly more strict hiding the code in more and more cases:
// #ifndef __OPENCL__ : Hide from OpenCL kernel code. All system headers and usage thereof must be protected like this, or stronger.
// #ifndef GPUCA_GPUCODE_DEVICE : Hide from kernel code on all GPU architectures. This includes the __OPENCL__ case and bodies of all GPU device functions (GPUd(), etc.)
// #ifndef GPUCA_GPUCODE : Hide from compilation with GPU compiler. This includes the kernel case of GPUCA_GPUCODE_DEVICE but also all host code compiled by the GPU compiler, e.g. for management.
// #ifndef GPUCA_ALIGPUCODE : Code is completely invisible to the GPUCATracking library, irrespective of GPU or CPU compilation or which compiler.

#ifndef GPUCOMMONDEF_H
#define GPUCOMMONDEF_H

// clang-format off

//Some GPU configuration settings, must be included first
#include "GPUCommonDefSettings.h"

#if !defined(__CLING__) && !defined(G__ROOT) // No GPU code for ROOT
  #if defined(__CUDACC__) || defined(__OPENCL__) || defined(__HIPCC__) || defined(__OPENCL_HOST__)
    #define GPUCA_GPUCODE // Compiled by GPU compiler
  #endif

  #if defined(GPUCA_GPUCODE)
    #if defined(__CUDA_ARCH__) || defined(__OPENCL__) || defined(__HIP_DEVICE_COMPILE__)
      #define GPUCA_GPUCODE_DEVICE // Executed on device
    #endif
    #if defined(__CUDACC__)
      #define GPUCA_GPUTYPE CUDA
    #elif defined(__HIPCC__)
      #define GPUCA_GPUTYPE HIP
    #elif defined(__OPENCL__) || defined(__OPENCL_HOST__)
      #define GPUCA_GPUTYPE OCL
    #endif
  #endif
#endif
#ifndef GPUCA_GPUTYPE
  #define GPUCA_GPUTYPE CPU
#endif

#if defined(GPUCA_STANDALONE) || (defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)) || defined (GPUCA_GPUCODE)
  #define GPUCA_ALIGPUCODE // Part of GPUTracking library but not of interface
#endif

#if (defined(__CUDACC__) && defined(GPUCA_CUDA_NO_CONSTANT_MEMORY)) || (defined(__HIPCC__) && defined(GPUCA_HIP_NO_CONSTANT_MEMORY)) || (defined(__OPENCL__) && defined(GPUCA_OPENCL_NO_CONSTANT_MEMORY))
  #define GPUCA_NO_CONSTANT_MEMORY
#elif (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GPUCA_GPUCODE_HOSTONLY)
  #define GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(DEBUG_STREAMER)
  #define GPUCA_DEBUG_STREAMER_CHECK(...) __VA_ARGS__
#else
  #define GPUCA_DEBUG_STREAMER_CHECK(...)
#endif

#ifndef GPUCA_RTC_SPECIAL_CODE // By default, we ignore special RTC code
  #define GPUCA_RTC_SPECIAL_CODE(...)
#endif

#ifndef GPUCA_RTC_CONSTEXPR
  #define GPUCA_RTC_CONSTEXPR
#endif

#ifndef GPUCA_DETERMINISTIC_CODE
  #ifdef GPUCA_DETERMINISTIC_MODE
    #define GPUCA_DETERMINISTIC_CODE(det, indet) det // In deterministic mode, take deterministic code path
  #else
    #define GPUCA_DETERMINISTIC_CODE(det, indet) indet // otherwise the fast default code path
  #endif
#endif

// API Definitions for GPU Compilation
#include "GPUCommonDefAPI.h"

// clang-format on

#endif
