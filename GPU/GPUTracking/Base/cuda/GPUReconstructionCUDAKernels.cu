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

/// \file GPUReconstructionCUDAKernels.cu
/// \author David Rohr

#define GPUCA_COMPILEKERNELS
#include "GPUReconstructionCUDAIncludesSystem.h"
#include "GPUReconstructionCUDADef.h"

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"

using namespace o2::gpu;

#include "GPUReconstructionIncludesDeviceAll.h"

#include "GPUReconstructionCUDAKernelsSpecialize.inc"
#include "GPUReconstructionProcessingKernels.inc"
template void GPUReconstructionProcessing::KernelInterface<GPUReconstructionCUDA, GPUReconstructionDeviceBase>::runKernelVirtual(const int num, const void* args);

#if defined(__HIPCC__) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
__global__ void gGPUConstantMemBuffer_dummy(int32_t* p) { *p = *(int32_t*)&gGPUConstantMemBuffer; }
#endif

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionCUDA::runKernelBackendTimed(const krnlSetupTime& _xyz, const Args&... args)
{
#if !defined(GPUCA_KERNEL_COMPILE_MODE) || GPUCA_KERNEL_COMPILE_MODE != 1
  if (!GetProcessingSettings().rtc.enable) {
    kernelBackendMacro<T, I>::run(_xyz, this, args...);
  } else
#endif
  {
    auto& x = _xyz.x;
    auto& y = _xyz.y;
    const void* pArgs[sizeof...(Args) + 3]; // 3 is max: cons mem + y.index + y.num
    int32_t arg_offset = 0;
#ifdef GPUCA_NO_CONSTANT_MEMORY
    arg_offset = 1;
    pArgs[0] = &mDeviceConstantMem;
#endif
    pArgs[arg_offset] = &y.index;
    GPUReconstructionCUDAInternals::getArgPtrs(&pArgs[arg_offset + 1], args...);
    GPUChkErr(cuLaunchKernel(*mInternals->kernelFunctions[GetKernelNum<T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
  }
}

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionCUDA::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& z = _xyz.z;
  if (z.evList) {
    for (int32_t k = 0; k < z.nEvents; k++) {
      GPUChkErr(cudaStreamWaitEvent(mInternals->Streams[x.stream], ((cudaEvent_t*)z.evList)[k], 0));
    }
  }
  {
    GPUDebugTiming timer(GetProcessingSettings().deviceTimers && GetProcessingSettings().debugLevel > 0, (deviceEvent*)mDebugEvents, mInternals->Streams, _xyz, this);
    runKernelBackendTimed<T, I, Args...>(_xyz, args...);
  }
  GPUChkErr(cudaGetLastError());
  if (z.ev) {
    GPUChkErr(cudaEventRecord(*(cudaEvent_t*)z.ev, mInternals->Streams[x.stream]));
  }
}

#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(...) GPUCA_KRNL_REG_DEFAULT(__VA_ARGS__)

// clang-format off
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE != 1 // ---------- COMPILE_MODE = perkernel ----------
  #if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 2
    #define GPUCA_KRNL_DEFONLY // COMPILE_MODE = rdc
  #endif

  #ifndef __HIPCC__ // CUDA version
    #define GPUCA_KRNL_CALL(x_class, ...) \
      GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.index, args...);
  #else // HIP version
    #undef GPUCA_KRNL_CUSTOM
    #define GPUCA_KRNL_CUSTOM(args) GPUCA_M_STRIP(args)
    #define GPUCA_KRNL_CALL(x_class, ...) \
      hipLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], GPUCA_CONSMEM_CALL y.index, args...);
  #endif // __HIPCC__

  #define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types, ...) \
    GPUCA_KRNLGPU(x_class, x_attributes, x_arguments, x_forward, x_types, __VA_ARGS__) \
    template <> struct GPUReconstructionCUDA::kernelBackendMacro<GPUCA_M_KRNL_TEMPLATE(x_class)> { \
      template <typename... Args> \
      static inline void run(const GPUReconstructionProcessing::krnlSetupTime& _xyz, auto* me, const Args&... args) \
      { \
        auto& x = _xyz.x; \
        auto& y = _xyz.y; \
        GPUCA_KRNL_CALL(x_class, x_attributes, x_arguments, x_forward, x_types, __VA_ARGS__) \
      } \
    };

  #include "GPUReconstructionKernelList.h"
  #undef GPUCA_KRNL
#endif // ---------- COMPILE_MODE = onefile | rdc ----------
// clang-format on

#ifndef GPUCA_NO_CONSTANT_MEMORY
static GPUReconstructionDeviceBase::deviceConstantMemRegistration registerConstSymbol([]() {
  void* retVal = nullptr;
  if (GPUChkErrS(cudaGetSymbolAddress(&retVal, gGPUConstantMemBuffer))) {
    throw std::runtime_error("Could not obtain GPU constant memory symbol");
  }
  return retVal;
});
#endif
