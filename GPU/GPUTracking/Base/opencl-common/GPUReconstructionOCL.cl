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

/// \file GPUReconstructionOCL.cl
/// \author David Rohr

// clang-format off
#define __OPENCL__
#define GPUCA_GPUTYPE_OPENCL

#ifdef __OPENCLCPP__
  #ifdef GPUCA_OPENCLCPP_NO_CONSTANT_MEMORY
    #define GPUCA_NO_CONSTANT_MEMORY
  #endif
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable // Allow double precision variables
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable // Allow half precision
  #ifdef __clang__
    #pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable          //
    #pragma OPENCL EXTENSION __cl_clang_function_pointers : enable               // Allow function pointers
    #pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable              // Allow variadic functions
    #pragma OPENCL EXTENSION __cl_clang_non_portable_kernel_param_types : enable // Allow pointers to non-standard types as kernel arguments
    #pragma OPENCL EXTENSION __cl_clang_bitfields : enable                       // Allow usage of bitfields
    #define global __global
    #define local __local
    #define constant __constant
    #define private __private
    #undef global
    #undef local
    #undef constant
    #undef private
  #else
    #include <opencl_def>
    #include <opencl_common>
    #include <opencl_math>
    #include <opencl_atomic>
    #include <opencl_memory>
    #include <opencl_work_item>
    #include <opencl_synchronization>
    #include <opencl_printf>
    #include <opencl_integer>
    using namespace cl;
  #endif
  #ifndef M_PI
    #define M_PI 3.1415926535f
  #endif
#else
  #ifdef GPUCA_OPENCL_NO_CONSTANT_MEMORY
    #define GPUCA_NO_CONSTANT_MEMORY
  #endif
  #define nullptr NULL
  #define NULL (0x0)
#endif
typedef unsigned long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed long int64_t;
typedef signed int int32_t;
typedef signed short int16_t;
typedef signed char int8_t;

// Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)
#ifndef __OPENCLCPP__
#define static_assert(...)
#define GPUCA_OPENCL1
#endif

#include "GPUConstantMem.h"
#ifdef __OPENCLCPP__
#include "GPUReconstructionIncludesDeviceAll.h"
#else // Workaround, since OpenCL1 cannot digest all files
#include "GPUTPCTrackParam.cxx"
#include "GPUTPCTrack.cxx"
#include "GPUTPCGrid.cxx"
#include "GPUTPCRow.cxx"
#include "GPUTPCTracker.cxx"

#include "GPUGeneralKernels.cxx"
#include "GPUErrors.cxx"

#include "GPUTPCTrackletSelector.cxx"
#include "GPUTPCNeighboursFinder.cxx"
#include "GPUTPCNeighboursCleaner.cxx"
#include "GPUTPCStartHitsFinder.cxx"
#include "GPUTPCStartHitsSorter.cxx"
#include "GPUTPCTrackletConstructor.cxx"
#include "GPUTPCGlobalTracking.cxx"
#endif

// if (gpu_mem != pTracker.GPUParametersConst()->gpumem) return; //TODO!

#define GPUCA_KRNL(...) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, __VA_ARGS__)
#define GPUCA_KRNL_LOAD_single(...) GPUCA_KRNLGPU_SINGLE(__VA_ARGS__)
#define GPUCA_KRNL_LOAD_multi(...) GPUCA_KRNLGPU_MULTI(__VA_ARGS__)
#define GPUCA_CONSMEM_PTR GPUglobal() char *gpu_mem, GPUconstant() MEM_CONSTANT(GPUConstantMem) * pConstant,
#define GPUCA_CONSMEM (*pConstant)
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

// clang-format on
