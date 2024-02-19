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
// #ifndef GPUCA_GPUCODE : Hide from compilation with GPU compiler. This includes the case kernel case of GPUCA_GPUCODE_DEVICE but also all host code compiled by the GPU compiler, e.g. for management.
// #ifndef GPUCA_ALIGPUCODE : Code is completely invisible to the GPUCATracking library, irrespective of GPU or CPU compilation or which compiler.

#ifndef GPUCOMMONDEF_H
#define GPUCOMMONDEF_H

// clang-format off

//Some GPU configuration settings, must be included first
#include "GPUCommonDefSettings.h"

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && (!(defined(__CINT__) || defined(__ROOTCINT__)) || defined(__CLING__)) && defined(__cplusplus) && __cplusplus >= 201103L
  #define GPUCA_NOCOMPAT // C++11 + No old ROOT5 + No old OpenCL
  #ifndef __OPENCL__
    #define GPUCA_NOCOMPAT_ALLOPENCL // + No OpenCL at all
  #endif
  #ifndef __CINT__
    #define GPUCA_NOCOMPAT_ALLCINT // + No ROOT CINT at all
  #endif
#endif

#if !(defined(__CINT__) || defined(__ROOTCINT__) || defined(__CLING__) || defined(__ROOTCLING__) || defined(G__ROOT)) //No GPU code for ROOT
  #if defined(__CUDACC__) || defined(__OPENCL__) || defined(__HIPCC__) || defined(__OPENCL_HOST__)
    #define GPUCA_GPUCODE //Compiled by GPU compiler
  #endif

  #if defined(__CUDA_ARCH__) || defined(__OPENCL__) || defined(__HIP_DEVICE_COMPILE__)
    #define GPUCA_GPUCODE_DEVICE //Executed on device
  #endif
#endif

//Definitions for C++11 features not supported by CINT / OpenCL
#ifdef GPUCA_NOCOMPAT
  #define CON_DELETE = delete
  #define CON_DEFAULT = default
  #if defined(__cplusplus) && __cplusplus >= 201703L
    #define CONSTEXPR constexpr
  #else
    #define CONSTEXPR
  #endif
#else
  #define CON_DELETE
  #define CON_DEFAULT
  #define CONSTEXPR
#endif
#if defined(__ROOT__) && !defined(GPUCA_NOCOMPAT)
  #define VOLATILE // ROOT5 has a problem with volatile in CINT
#else
  #define VOLATILE volatile
#endif

//Set AliRoot / O2 namespace
#if defined(GPUCA_STANDALONE) || (defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)) || defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPULIBRARY) || defined (GPUCA_GPUCODE)
  #define GPUCA_ALIGPUCODE
#endif
#ifdef GPUCA_ALIROOT_LIB
  #define GPUCA_NAMESPACE AliGPU
#else
  #define GPUCA_NAMESPACE o2
#endif

#if (defined(__CUDACC__) && defined(GPUCA_CUDA_NO_CONSTANT_MEMORY)) || (defined(__HIPCC__) && defined(GPUCA_HIP_NO_CONSTANT_MEMORY)) || (defined(__OPENCL__) && !defined(__OPENCLCPP__) && defined(GPUCA_OPENCL_NO_CONSTANT_MEMORY)) || (defined(__OPENCLCPP__) && defined(GPUCA_OPENCLCPP_NO_CONSTANT_MEMORY))
  #define GPUCA_NO_CONSTANT_MEMORY
#elif defined(__CUDACC__) || defined(__HIPCC__)
  #define GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM
#endif
#if !defined(GPUCA_HAVE_O2HEADERS) && (defined(GPUCA_O2_LIB) || (!defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_STANDALONE)))
  #define GPUCA_HAVE_O2HEADERS
#endif

#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(DEBUG_STREAMER)
#define GPUCA_DEBUG_STREAMER_CHECK(...) __VA_ARGS__
#else
#define GPUCA_DEBUG_STREAMER_CHECK(...)
#endif


//API Definitions for GPU Compilation
#include "GPUCommonDefAPI.h"

// clang-format on

#endif
