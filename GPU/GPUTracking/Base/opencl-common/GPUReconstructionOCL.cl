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
#define GPUCA_GPUTYPE_RADEON

#ifdef __OPENCLCPP__
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #ifdef __clang__
    #pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
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
  #define nullptr NULL
  #define NULL (0x0)
#endif
#define uint32_t unsigned int
#define uint16_t unsigned short
#define uint8_t unsigned char
// clang-format on

// Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#include "GPUReconstructionIncludesDevice.h"
#include "GPUConstantMem.h"

#define OCL_DEVICE_KERNELS_PRE GPUglobal() char *gpu_mem, GPUconstant() MEM_CONSTANT(GPUConstantMem) * pConstant
#define OCL_CALL_KERNEL(T, I, num)                            \
  GPUshared() typename T::MEM_LOCAL(GPUTPCSharedMemory) smem; \
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(*pConstant)[num]);

#define OCL_CALL_KERNEL_MULTI(T, I)                                                                                                                                                \
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0); \
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;                                                                                                          \
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;                                                                                                                    \
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;                                                            \
  GPUshared() typename T::MEM_LOCAL(GPUTPCSharedMemory) smem;                                                                                                                      \
  T::template Thread<I>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Processor(*pConstant)[firstSlice + iSlice]);

#define OCL_CALL_KERNEL_ARGS(T, I, ...)            \
  GPUshared() typename T::GPUTPCSharedMemory smem; \
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(*pConstant)[0], __VA_ARGS__);

// if (gpu_mem != pTracker.GPUParametersConst()->gpumem) return; //TODO!

GPUg() void GPUTPCProcess_N2o23gpu13GPUMemClean16E0(OCL_DEVICE_KERNELS_PRE, unsigned long ptr, unsigned long size) { OCL_CALL_KERNEL_ARGS(GPUMemClean16, 0, (GPUglobalref() void*)(void*)ptr, size); }

GPUg() void GPUTPCProcess_N2o23gpu22GPUTPCNeighboursFinderE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCNeighboursFinder, 0, iSlice); }

GPUg() void GPUTPCProcess_N2o23gpu23GPUTPCNeighboursCleanerE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCNeighboursCleaner, 0, iSlice); }

GPUg() void GPUTPCProcess_N2o23gpu21GPUTPCStartHitsFinderE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCStartHitsFinder, 0, iSlice); }

GPUg() void GPUTPCProcess_N2o23gpu21GPUTPCStartHitsSorterE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCStartHitsSorter, 0, iSlice); }

GPUg() void GPUTPCProcess_N2o23gpu25GPUTPCTrackletConstructorE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCTrackletConstructor, 0, iSlice); }

GPUg() void GPUTPCProcess_N2o23gpu25GPUTPCTrackletConstructorE1(OCL_DEVICE_KERNELS_PRE) { OCL_CALL_KERNEL(GPUTPCTrackletConstructor, 1, 0); }

GPUg() void GPUTPCProcess_N2o23gpu22GPUTPCTrackletSelectorE0(OCL_DEVICE_KERNELS_PRE, int iSlice) { OCL_CALL_KERNEL(GPUTPCTrackletSelector, 0, iSlice); }

GPUg() void GPUTPCProcess_Multi_N2o23gpu22GPUTPCTrackletSelectorE0(OCL_DEVICE_KERNELS_PRE, int firstSlice, int nSliceCount) { OCL_CALL_KERNEL_MULTI(GPUTPCTrackletSelector, 0); }
