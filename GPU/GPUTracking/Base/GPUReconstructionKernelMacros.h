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

/// \file GPUReconstructionKernelMacros.h
/// \author David Rohr

// clang-format off
#ifndef O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H
#define O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H

#include "GPUDefMacros.h"

#define GPUCA_M_KRNL_TEMPLATE_B(a, b, ...) a, a::b
#define GPUCA_M_KRNL_TEMPLATE_A(...) GPUCA_M_KRNL_TEMPLATE_B(__VA_ARGS__, defaultKernel)
#define GPUCA_M_KRNL_TEMPLATE(...) GPUCA_M_KRNL_TEMPLATE_A(GPUCA_M_STRIP(__VA_ARGS__))

#define GPUCA_M_KRNL_NUM_B(a, b, ...) a::b
#define GPUCA_M_KRNL_NUM_A(...) GPUCA_M_KRNL_NUM_B(__VA_ARGS__, defaultKernel)
#define GPUCA_M_KRNL_NUM(...) GPUCA_M_KRNL_NUM_A(GPUCA_M_STRIP(__VA_ARGS__))

#define GPUCA_M_KRNL_NAME_B0(a, b, ...) GPUCA_M_CAT3(a, _, b)
#define GPUCA_M_KRNL_NAME_B1(a) a
#define GPUCA_M_KRNL_NAME_A(...) GPUCA_M_CAT(GPUCA_M_KRNL_NAME_B, GPUCA_M_SINGLEOPT(__VA_ARGS__))(__VA_ARGS__)
#define GPUCA_M_KRNL_NAME(...) GPUCA_M_KRNL_NAME_A(GPUCA_M_STRIP(__VA_ARGS__))

#if defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_HOSTONLY)

#if defined(__HIPCC__) && !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)
  static_assert(GPUCA_PAR_AMD_EUS_PER_CU > 0);
  #define GPUCA_MIN_WARPS_PER_EU(maxThreadsPerBlock, minBlocksPerCU) GPUCA_CEIL_INT_DIV((minBlocksPerCU) * (maxThreadsPerBlock), (GPUCA_WARP_SIZE * GPUCA_PAR_AMD_EUS_PER_CU))

  #define GPUCA_LB_ARGS_1(maxThreadsPerBlock) maxThreadsPerBlock
  #define GPUCA_LB_ARGS_2(maxThreadsPerBlock, minBlocksPerCU) maxThreadsPerBlock, GPUCA_MIN_WARPS_PER_EU(maxThreadsPerBlock, minBlocksPerCU)

  #define GPUCA_LAUNCH_BOUNDS_SELECT(n, ...) GPUCA_M_CAT(GPUCA_LB_ARGS_, n)(__VA_ARGS__)
  #define GPUCA_LAUNCH_BOUNDS_DISP(...) GPUCA_LAUNCH_BOUNDS_SELECT(GPUCA_M_COUNT(__VA_ARGS__), __VA_ARGS__)
  #define GPUCA_KRNL_REG_DEFAULT(args) __launch_bounds__(GPUCA_LAUNCH_BOUNDS_DISP(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args))))
#elif !defined(GPUCA_GPUCODE_NO_LAUNCH_BOUNDS)
  #define GPUCA_KRNL_REG_DEFAULT(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#endif

#ifndef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(...)
#endif
#ifndef GPUCA_KRNL_CUSTOM
#define GPUCA_KRNL_CUSTOM(...)
#endif
#define GPUCA_ATTRRES_REG(reg, num, ...) GPUCA_M_EXPAND(GPUCA_KRNL_REG)(num) GPUCA_ATTRRES_XREG (__VA_ARGS__)
#define GPUCA_ATTRRES_CUSTOM(custom, args, ...) GPUCA_M_EXPAND(GPUCA_KRNL_CUSTOM)(args) GPUCA_ATTRRES_XCUSTOM(__VA_ARGS__)
#define GPUCA_ATTRRES_NONE(none, ...) GPUCA_ATTRRES_XNONE(__VA_ARGS__)
#define GPUCA_ATTRRES_(...)
#define GPUCA_ATTRRES_XNONE(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)
#define GPUCA_ATTRRES_XCUSTOM(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)
#define GPUCA_ATTRRES_XREG(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)
#define GPUCA_ATTRRES(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)

// GPU Kernel entry point
#define GPUCA_KRNLGPU_DEF(x_class, x_attributes, x_arguments, ...) \
  GPUg() void GPUCA_ATTRRES(GPUCA_M_STRIP(x_attributes)) GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))(GPUCA_CONSMEM_PTR int32_t _iSector_internal GPUCA_M_STRIP(x_arguments))

#ifdef GPUCA_KRNL_DEFONLY
#define GPUCA_KRNLGPU(...) GPUCA_KRNLGPU_DEF(__VA_ARGS__);
#else
#define GPUCA_KRNLGPU(x_class, x_attributes, x_arguments, x_forward, ...) \
  GPUCA_KRNLGPU_DEF(x_class, x_attributes, x_arguments, x_forward, __VA_ARGS__) \
  { \
    GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::GPUSharedMemory smem; \
    GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[_iSector_internal] GPUCA_M_STRIP(x_forward)); \
  }
#endif

#endif // GPUCA_GPUCODE

#define GPUCA_KRNL_LB(x_class, x_attributes, ...) GPUCA_KRNL(x_class, (REG, (GPUCA_M_CAT(GPUCA_LB_, GPUCA_M_KRNL_NAME(x_class))), GPUCA_M_STRIP(x_attributes)), __VA_ARGS__)

#endif // O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H
// clang-format on
