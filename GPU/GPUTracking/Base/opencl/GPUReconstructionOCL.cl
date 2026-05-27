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

#ifdef __OPENCL__
  #ifdef GPUCA_OPENCL_NO_CONSTANT_MEMORY
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

#include "GPUCommonDef.h"
#include "GPUCommonTypeTraits.h" // TODO: Once possible in OpenCL, should use GPUStdSystemHeaders.h here
#include "GPUCommonArray.h"      // TODO: Same
#include "GPUConstantMem.h"
#include "GPUReconstructionIncludesDeviceAll.h"

#define GPUCA_KRNL(...) GPUCA_KRNLGPU(__VA_ARGS__)
#define GPUCA_CONSMEM_PTR GPUglobal() char *gpu_mem, GPUconstant() GPUConstantMem* pConstant,
#define GPUCA_CONSMEM (*pConstant)
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

// clang-format on
