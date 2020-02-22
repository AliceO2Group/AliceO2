// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#ifdef GPUCA_GPUCODE
#ifndef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(...)
#endif
#ifndef GPUCA_KRNL_BACKEND_XARGS
#define GPUCA_KRNL_BACKEND_XARGS
#endif
#define GPUCA_ATTRRES_REG(reg, num, ...) GPUCA_KRNL_REG(num) GPUCA_ATTRRES2(__VA_ARGS__)
#define GPUCA_ATTRRES2_REG(reg, num, ...) GPUCA_KRNL_REG(num) GPUCA_ATTRRES3(__VA_ARGS__)
#define GPUCA_ATTRRES_NONE(...)
#define GPUCA_ATTRRES2_NONE(...)
#define GPUCA_ATTRRES_(...)
#define GPUCA_ATTRRES2_(...)
#define GPUCA_ATTRRES3() // 3 attributes not supported
#define GPUCA_ATTRRES2(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES2_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)
#define GPUCA_ATTRRES(...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(__VA_ARGS__)
// GPU Kernel entry point for single sector
#define GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward) \
GPUg() void GPUCA_ATTRRES(GPUCA_M_SHIFT(GPUCA_M_STRIP(x_attributes))) GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))(GPUCA_CONSMEM_PTR int iSlice GPUCA_M_STRIP(x_arguments)) \
{ \
  GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::MEM_LOCAL(GPUSharedMemory) smem; \
  GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[iSlice] GPUCA_M_STRIP(x_forward)); \
}

// GPU Kernel entry point for multiple sector
#define GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward) \
GPUg() void GPUCA_ATTRRES(GPUCA_M_SHIFT(GPUCA_M_STRIP(x_attributes))) GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)(GPUCA_CONSMEM_PTR int firstSlice, int nSliceCount GPUCA_M_STRIP(x_arguments)) \
{ \
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0); \
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount; \
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset; \
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount; \
  GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::MEM_LOCAL(GPUSharedMemory) smem; \
  GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[firstSlice + iSlice] GPUCA_M_STRIP(x_forward)); \
}

// GPU Host wrapper pre- and post-parts
#define GPUCA_KRNL_PRE(x_class, x_attributes, x_arguments, x_forward) \
  template <> class GPUCA_KRNL_BACKEND_CLASS::backendInternal<GPUCA_M_KRNL_TEMPLATE(x_class)> { \
   public: \
    template <typename T, typename... Args> \
    static inline void runKernelBackendInternal(krnlSetup& _xyz, T* me, GPUCA_KRNL_BACKEND_XARGS const Args&... args) \
    { \
      auto& x = _xyz.x; \
      auto& y = _xyz.y;

#define GPUCA_KRNL_POST() \
    } \
  };

// GPU Host wrappers for single kernel, multi-sector, or auto-detection
#define GPUCA_KRNL_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PRE(x_class, x_attributes, x_arguments, x_forward) \
  if (y.num > 1) { \
    throw std::runtime_error("Kernel called with invalid number of sectors"); \
  } else { \
    GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  } \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PRE(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_single(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_simple(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_single(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_both(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PRE(x_class, x_attributes, x_arguments, x_forward) \
  if (y.num <= 1) { \
    GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  } else { \
    GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  } \
  GPUCA_KRNL_POST()

#define GPUCA_KRNL_LOAD_(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_simple(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_both(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)

// Generate GPU kernel and host wrapper
#define GPUCA_KRNL_WRAP(x_func, x_class, x_attributes, x_arguments, x_forward) GPUCA_M_CAT(x_func, GPUCA_M_STRIP_FIRST(x_attributes))(x_class, x_attributes, x_arguments, x_forward)
#endif

#endif
// clang-format on
