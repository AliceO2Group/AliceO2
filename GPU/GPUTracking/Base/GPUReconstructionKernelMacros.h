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
#ifndef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(...)
#endif
#define GPUCA_KRNL_REG_INTERNAL_PROP(...) GPUCA_M_STRIP(__VA_ARGS__)
#ifndef GPUCA_KRNL_CUSTOM
#define GPUCA_KRNL_CUSTOM(...)
#endif
#define GPUCA_KRNL_CUSTOM_INTERNAL_PROP(...)
#define GPUCA_ATTRRES_REG(XX, reg, num, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_REG, XX))(num) GPUCA_ATTRRES2(XX, __VA_ARGS__)
#define GPUCA_ATTRRES2_REG(XX, reg, num, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_REG, XX))(num) GPUCA_ATTRRES3(XX, __VA_ARGS__)
#define GPUCA_ATTRRES_CUSTOM(XX, custom, args, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_CUSTOM, XX))(args) GPUCA_ATTRRES2(XX, __VA_ARGS__)
#define GPUCA_ATTRRES2_CUSTOM(XX, custom, args, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_CUSTOM, XX))(args) GPUCA_ATTRRES3(XX, __VA_ARGS__)
#define GPUCA_ATTRRES_NONE(XX, ...)
#define GPUCA_ATTRRES2_NONE(XX, ...)
#define GPUCA_ATTRRES_(XX, ...)
#define GPUCA_ATTRRES2_(XX, ...)
#define GPUCA_ATTRRES3(XX) // 3 attributes not supported
#define GPUCA_ATTRRES2(XX, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES2_, GPUCA_M_FIRST(__VA_ARGS__)))(XX, __VA_ARGS__)
#define GPUCA_ATTRRES(XX, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(XX, __VA_ARGS__)
// GPU Kernel entry point for single sector
#define GPUCA_KRNLGPU_SINGLE_DEF(x_class, x_attributes, x_arguments, ...) \
  GPUg() void GPUCA_ATTRRES(,GPUCA_M_SHIFT(GPUCA_M_STRIP(x_attributes))) GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))(GPUCA_CONSMEM_PTR int32_t iSlice_internal GPUCA_M_STRIP(x_arguments))
#ifdef GPUCA_KRNL_DEFONLY
#define GPUCA_KRNLGPU_SINGLE(...) GPUCA_KRNLGPU_SINGLE_DEF(__VA_ARGS__);
#else
#define GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward, ...) GPUCA_KRNLGPU_SINGLE_DEF(x_class, x_attributes, x_arguments, x_forward, __VA_ARGS__) \
  { \
    GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::MEM_LOCAL(GPUSharedMemory) smem; \
    GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[iSlice_internal] GPUCA_M_STRIP(x_forward)); \
  }
#endif

// GPU Kernel entry point for multiple sector
#define GPUCA_KRNLGPU_MULTI_DEF(x_class, x_attributes, x_arguments, ...) \
  GPUg() void GPUCA_ATTRRES(,GPUCA_M_SHIFT(GPUCA_M_STRIP(x_attributes))) GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)(GPUCA_CONSMEM_PTR int32_t firstSlice, int32_t nSliceCount GPUCA_M_STRIP(x_arguments))
#ifdef GPUCA_KRNL_DEFONLY
#define GPUCA_KRNLGPU_MULTI(...) GPUCA_KRNLGPU_MULTI_DEF(__VA_ARGS__);
#else
#define GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward, ...) GPUCA_KRNLGPU_MULTI_DEF(x_class, x_attributes, x_arguments, x_forward, __VA_ARGS__) \
  { \
    const int32_t iSlice_internal = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0); \
    const int32_t nSliceBlockOffset = get_num_groups(0) * iSlice_internal / nSliceCount; \
    const int32_t sliceBlockId = get_group_id(0) - nSliceBlockOffset; \
    const int32_t sliceGridDim = get_num_groups(0) * (iSlice_internal + 1) / nSliceCount - get_num_groups(0) * (iSlice_internal) / nSliceCount; \
    GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::MEM_LOCAL(GPUSharedMemory) smem; \
    GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[firstSlice + iSlice_internal] GPUCA_M_STRIP(x_forward)); \
  }
#endif

// GPU Host wrapper pre- and post-parts
#define GPUCA_KRNL_PRE(x_class, ...) \
  template <> class GPUCA_KRNL_BACKEND_CLASS::backendInternal<GPUCA_M_KRNL_TEMPLATE(x_class)> { \
   public: \
    template <typename T, typename... Args> \
    static inline void runKernelBackendMacro(const krnlSetupTime& _xyz, T* me, const Args&... args) \
    { \
      auto& x = _xyz.x; \
      auto& y = _xyz.y;

#define GPUCA_KRNL_POST() \
    } \
  };

// GPU Host wrappers for single kernel, multi-sector, or auto-detection
#define GPUCA_KRNL_single(...) \
  GPUCA_KRNLGPU_SINGLE(__VA_ARGS__) \
  GPUCA_KRNL_PRE(__VA_ARGS__) \
  if (y.num > 1) { \
    throw std::runtime_error("Kernel called with invalid number of sectors"); \
  } else { \
    GPUCA_KRNL_CALL_single(__VA_ARGS__) \
  } \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_multi(...) \
  GPUCA_KRNLGPU_MULTI(__VA_ARGS__) \
  GPUCA_KRNL_PRE(__VA_ARGS__) \
  GPUCA_KRNL_CALL_multi(__VA_ARGS__) \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_(...) GPUCA_KRNL_single(__VA_ARGS__)
#define GPUCA_KRNL_simple(...) GPUCA_KRNL_single(__VA_ARGS__)
#define GPUCA_KRNL_both(...) \
  GPUCA_KRNLGPU_SINGLE(__VA_ARGS__) \
  GPUCA_KRNLGPU_MULTI(__VA_ARGS__) \
  GPUCA_KRNL_PRE(__VA_ARGS__) \
  if (y.num <= 1) { \
    GPUCA_KRNL_CALL_single(__VA_ARGS__) \
  } else { \
    GPUCA_KRNL_CALL_multi(__VA_ARGS__) \
  } \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_LOAD_(...) GPUCA_KRNL_LOAD_single(__VA_ARGS__)
#define GPUCA_KRNL_LOAD_simple(...) GPUCA_KRNL_LOAD_single(__VA_ARGS__)
#define GPUCA_KRNL_LOAD_both(...) \
  GPUCA_KRNL_LOAD_single(__VA_ARGS__) \
  GPUCA_KRNL_LOAD_multi(__VA_ARGS__)

#define GPUCA_KRNL_PROP(x_class, x_attributes) \
  template <> gpu_reconstruction_kernels::krnlProperties GPUCA_KRNL_BACKEND_CLASS::getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>() { \
    gpu_reconstruction_kernels::krnlProperties ret = gpu_reconstruction_kernels::krnlProperties{GPUCA_ATTRRES(_INTERNAL_PROP,GPUCA_M_SHIFT(GPUCA_M_STRIP(x_attributes)))}; \
    return ret.nThreads > 0 ? ret : gpu_reconstruction_kernels::krnlProperties{(int32_t)mThreadCount}; \
  }

// Generate GPU kernel and host wrapper
#define GPUCA_KRNL_WRAP(x_func, x_class, x_attributes, ...) GPUCA_M_CAT(x_func, GPUCA_M_STRIP_FIRST(x_attributes))(x_class, x_attributes, __VA_ARGS__)
#endif // GPUCA_GPUCODE

#define GPUCA_KRNL_LB(x_class, x_attributes, ...) GPUCA_KRNL(x_class, (GPUCA_M_STRIP(x_attributes), REG, (GPUCA_M_CAT(GPUCA_LB_, GPUCA_M_KRNL_NAME(x_class)))), __VA_ARGS__)

#endif // O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H
// clang-format on
