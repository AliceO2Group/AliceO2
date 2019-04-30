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

#if !(defined(__CINT__) || defined(__ROOTCINT__) || defined(__CLING__) || defined(__ROOTCLING__) || defined(G__ROOT)) //No GPU code for ROOT
  #if defined(__CUDACC__) || defined(__OPENCL__) || defined(__HIPCC__)
    #define GPUCA_GPUCODE //Compiled by GPU compiler
  #endif

  #if defined(__CUDA_ARCH__) || defined(__OPENCL__) || defined(__HIP_DEVICE_COMPILE__)
    #define GPUCA_GPUCODE_DEVICE //Executed on device
  #endif
#endif

//Definitions for C++11 features not supported by CINT / OpenCL
#if ((defined(__CINT__) || defined(__ROOTCINT__)) && !defined(__CLING__)) || (defined(__OPENCL__) && !defined(__OPENCLCPP__))
  #define CON_DELETE
  #define CON_DEFAULT
  #define CONSTEXPR const
#else
  #define CON_DELETE = delete
  #define CON_DEFAULT = default
  #define CONSTEXPR constexpr
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

//Definitions steering enabling of GPU processing components
#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !defined(GPUCA_ALIROOT_LIB)
  #define GPUCA_BUILD_MERGER
  #define GPUCA_BUILD_DEDX
  #if defined(HAVE_O2HEADERS)
    #define GPUCA_BUILD_TRD
    #define GPUCA_BUILD_ITS
  #endif
#endif

// clang-format on

#endif
