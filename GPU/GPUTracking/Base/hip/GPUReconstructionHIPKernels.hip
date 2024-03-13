#include "hip/hip_runtime.h"
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

/// \file GPUReconstructionHIPKernels.cu
/// \author David Rohr

#include "GPUReconstructionHIPDef.h"
#include "GPUReconstructionHIPIncludes.h"

#include "GPUReconstructionHIP.h"
#include "GPUReconstructionHIPInternals.h"
#include "HIPThrustHelpers.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_USE_TEXTURES
texture<cahit2, hipTextureType1D, hipReadModeElementType> gAliTexRefu2;
texture<calink, hipTextureType1D, hipReadModeElementType> gAliTexRefu;
#endif

#include "GPUReconstructionIncludesDeviceAll.h"

#if defined(__HIPCC__) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
__global__ void gGPUConstantMemBuffer_dummy(int* p) { *p = *(int*)&gGPUConstantMemBuffer; }
#endif

template <>
inline void GPUReconstructionHIPBackend::runKernelBackendInternal<GPUMemClean16, 0>(krnlSetup& _xyz, void* const& ptr, unsigned long const& size)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  GPUFailedMsg(hipMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

template <class T, int I, typename... Args>
inline void GPUReconstructionHIPBackend::runKernelBackendInternal(krnlSetup& _xyz, const Args&... args)
{
#ifndef __HIPCC__ // HIP version
  GPUDebugTiming timer(mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0, (void**)mDebugEvents, mInternals->Streams, _xyz, this);
#if !defined(GPUCA_KERNEL_COMPILE_MODE) || GPUCA_KERNEL_COMPILE_MODE != 1
  if (!mProcessingSettings.rtc.enable) {
    backendInternal<T, I>::runKernelBackendMacro(_xyz, this, args...);
  } else
#endif
  {
    auto& x = _xyz.x;
    auto& y = _xyz.y;
    const void* pArgs[sizeof...(Args) + 3]; // 3 is max: cons mem + y.start + y.num
    int arg_offset = 0;
#ifdef GPUCA_NO_CONSTANT_MEMORY
    arg_offset = 1;
    pArgs[0] = &mDeviceConstantMem;
#endif
    pArgs[arg_offset] = &y.start;
    GPUReconstructionHIPInternals::getArgPtrs(&pArgs[arg_offset + 1 + (y.num > 1)], args...);
    if (y.num <= 1) {
      GPUFailedMsg(hipModuleLaunchKernel(*mInternals->kernelFunctions[mInternals->getRTCkernelNum<false, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    } else {
      pArgs[arg_offset + 1] = &y.num;
      GPUFailedMsg(hipModuleLaunchKernel(*mInternals->kernelFunctions[mInternals->getRTCkernelNum<true, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    }
  }
#else // HIP version
  if (mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0) {
    backendInternal<T, I>::runKernelBackendMacro(_xyz, this, (hipEvent_t*)&mDebugEvents->DebugStart, (hipEvent_t*)&mDebugEvents->DebugStop, args...);
    GPUFailedMsg(hipEventSynchronize((hipEvent_t)mDebugEvents->DebugStop));
    float v;
    GPUFailedMsg(hipEventElapsedTime(&v, (hipEvent_t)mDebugEvents->DebugStart, (hipEvent_t)mDebugEvents->DebugStop));
    _xyz.t = v * 1.e-3f;
  } else {
    backendInternal<T, I>::runKernelBackendMacro(_xyz, this, nullptr, nullptr, args...);
  }
#endif
  if (mProcessingSettings.checkKernelFailures) {
    if (GPUDebug(GetKernelName<T, I>(), _xyz.x.stream, true)) {
      throw std::runtime_error("Kernel Failure");
    }
  }
}

template <class T, int I, typename... Args>
int GPUReconstructionHIPBackend::runKernelBackend(krnlSetup& _xyz, Args... args)
{
  auto& x = _xyz.x;
  auto& z = _xyz.z;
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(hipStreamWaitEvent(mInternals->Streams[x.stream], ((hipEvent_t*)z.evList)[k], 0));
    }
  }
  runKernelBackendInternal<T, I>(_xyz, args...);
  GPUFailedMsg(hipGetLastError());
  if (z.ev) {
    GPUFailedMsg(hipEventRecord(*(hipEvent_t*)z.ev, mInternals->Streams[x.stream]));
  }
  return 0;
}

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                          \
  template int GPUReconstructionHIPBackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(krnlSetup & _xyz GPUCA_M_STRIP(x_arguments));
#else
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 2
#define GPUCA_KRNL_DEFONLY
#endif

#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)             \
  GPUCA_KRNL_PROP(x_class, x_attributes)                                      \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_, x_class, x_attributes, x_arguments, x_forward) \
  template int GPUReconstructionHIPBackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(krnlSetup & _xyz GPUCA_M_STRIP(x_arguments));
#ifndef __HIPCC__ // HIP version
#define GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);
#else // HIP version
#undef GPUCA_KRNL_CUSTOM
#define GPUCA_KRNL_CUSTOM(args) GPUCA_M_STRIP(args)
#undef GPUCA_KRNL_BACKEND_XARGS
#define GPUCA_KRNL_BACKEND_XARGS hipEvent_t *debugStartEvent, hipEvent_t *debugStopEvent,
#define GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward)                                                                                                                                                                   \
  if (debugStartEvent == nullptr) {                                                                                                                                                                                                             \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], GPUCA_CONSMEM_CALL y.start, args...);                                          \
  } else {                                                                                                                                                                                                                                      \
    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], *debugStartEvent, *debugStopEvent, 0, GPUCA_CONSMEM_CALL y.start, args...); \
  }
#define GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward)                                                                                                                                                                                    \
  if (debugStartEvent == nullptr) {                                                                                                                                                                                                                             \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], GPUCA_CONSMEM_CALL y.start, y.num, args...);                                          \
  } else {                                                                                                                                                                                                                                                      \
    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], *debugStartEvent, *debugStopEvent, 0, GPUCA_CONSMEM_CALL y.start, y.num, args...); \
  }
#endif // __HIPCC__
#endif

#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

#ifndef GPUCA_NO_CONSTANT_MEMORY
static GPUReconstructionDeviceBase::deviceConstantMemRegistration registerConstSymbol([]() {
  void* retVal = nullptr;
  GPUReconstructionHIP::GPUFailedMsgI(hipGetSymbolAddress(&retVal, HIP_SYMBOL(gGPUConstantMemBuffer)));
  return retVal;
});
#endif
