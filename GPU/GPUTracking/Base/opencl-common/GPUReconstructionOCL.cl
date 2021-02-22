// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #ifdef __clang__
    #pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
    #pragma OPENCL EXTENSION __cl_clang_function_pointers : enable
    #define global __global
    #define local __local
    #define constant __constant
    #define private __private
    //#include <clc/clc.h> //Use -finclude-default-header instead! current clang libclc.h is incompatible to SPIR-V
    typedef __SIZE_TYPE__ size_t; //BUG: OpenCL C++ does not declare placement new
    void* operator new (size_t size, void *ptr);
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
#define uint32_t unsigned int
#define uint16_t unsigned short
#define uint8_t unsigned char

// Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)
#ifndef __OPENCLCPP__
#define static_assert(...)
#define GPUCA_OPENCL1
#endif

#include "GPUReconstructionIncludesDevice.h"
#include "GPUConstantMem.h"

// if (gpu_mem != pTracker.GPUParametersConst()->gpumem) return; //TODO!

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_CONSMEM_PTR GPUglobal() char *gpu_mem, GPUconstant() MEM_CONSTANT(GPUConstantMem) * pConstant,
#define GPUCA_CONSMEM (*pConstant)
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi

// clang-format on
