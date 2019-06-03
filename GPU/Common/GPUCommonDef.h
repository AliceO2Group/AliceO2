// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDef.h
/// \author David Rohr

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
  #if defined(__CUDACC__) || defined(__OPENCL__) || defined(__HIPCC__)
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
  #define CONSTEXPR constexpr
#else
  #define CON_DELETE
  #define CON_DEFAULT
  #define CONSTEXPR const
#endif

//Set AliRoot / O2 namespace
#if defined(GPUCA_STANDALONE) || defined(GPUCA_O2_LIB) || defined(GPUCA_GPULIBRARY)
  #define GPUCA_ALIGPUCODE
#endif
#ifdef GPUCA_ALIROOT_LIB
  #define GPUCA_NAMESPACE AliGPU
#else
  #define GPUCA_NAMESPACE o2
#endif

//API Definitions for GPU Compilation
#include "GPUCommonDefAPI.h"

// clang-format on

#endif
