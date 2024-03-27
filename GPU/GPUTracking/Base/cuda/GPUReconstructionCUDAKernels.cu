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

#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDAIncludes.h"

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "CUDAThrustHelpers.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

#include "GPUReconstructionIncludesDeviceAll.h"

#if defined(__HIPCC__) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
__global__ void gGPUConstantMemBuffer_dummy(int* p) { *p = *(int*)&gGPUConstantMemBuffer; }
#endif

template <>
inline void GPUReconstructionCUDABackend::runKernelBackendInternal<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, unsigned long const& size)
{
  GPUFailedMsg(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

template <class T, int I, typename... Args>
inline void GPUReconstructionCUDABackend::runKernelBackendInternal(const krnlSetupTime& _xyz, const Args&... args)
{
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
    GPUReconstructionCUDAInternals::getArgPtrs(&pArgs[arg_offset + 1 + (y.num > 1)], args...);
    if (y.num <= 1) {
      GPUFailedMsg(cuLaunchKernel(*mInternals->kernelFunctions[getRTCkernelNum<false, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    } else {
      pArgs[arg_offset + 1] = &y.num;
      GPUFailedMsg(cuLaunchKernel(*mInternals->kernelFunctions[getRTCkernelNum<true, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    }
  }
}

template <class T, int I, typename... Args>
int GPUReconstructionCUDABackend::runKernelBackend(const krnlSetupArgs<T, I, Args...>& args)
{
  auto& x = args.s.x;
  auto& z = args.s.z;
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[x.stream], ((cudaEvent_t*)z.evList)[k], 0));
    }
  }
  {
    GPUDebugTiming timer(mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0, (deviceEvent*)mDebugEvents, mInternals->Streams, args.s, this);
    std::apply([this, &args](auto&... vals) { this->runKernelBackendInternal<T, I, Args...>(args.s, vals...); }, args.v);
  }
  GPUFailedMsg(cudaGetLastError());
  if (mProcessingSettings.checkKernelFailures) {
    if (GPUDebug(GetKernelName<T, I>(), args.s.x.stream, true)) {
      throw std::runtime_error("Kernel Failure");
    }
  }
  if (z.ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)z.ev, mInternals->Streams[x.stream]));
  }
  return 0;
}

#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                                   \
  template int GPUReconstructionCUDABackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args);
#else
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 2
#define GPUCA_KRNL_DEFONLY
#endif

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types)             \
  GPUCA_KRNL_PROP(x_class, x_attributes)                                               \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_, x_class, x_attributes, x_arguments, x_forward, x_types) \
  template int GPUReconstructionCUDABackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args);
#ifndef __HIPCC__ // CUDA version
#define GPUCA_KRNL_CALL_single(x_class, ...) \
  GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, ...) \
  GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);
#else // HIP version
#undef GPUCA_KRNL_CUSTOM
#define GPUCA_KRNL_CUSTOM(args) GPUCA_M_STRIP(args)
#define GPUCA_KRNL_CALL_single(x_class, ...) \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, ...) \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), dim3(x.nBlocks), dim3(x.nThreads), 0, me->mInternals->Streams[x.stream], GPUCA_CONSMEM_CALL y.start, y.num, args...);
#endif // __HIPCC__
#endif

#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

template <bool multi, class T, int I>
int GPUReconstructionCUDABackend::getRTCkernelNum(int k)
{
  static int num = k;
  if (num < 0) {
    throw std::runtime_error("Invalid kernel");
  }
  return num;
}

#define GPUCA_KRNL(x_class, ...)                                                                            \
  template int GPUReconstructionCUDABackend::getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(int k); \
  template int GPUReconstructionCUDABackend::getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(int k);
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

void GPUReconstructionCUDABackend::getRTCKernelCalls(std::vector<std::string>& kernels)
{
#define GPUCA_KRNL(...) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, __VA_ARGS__)
#define GPUCA_KRNL_LOAD_single(...) kernels.emplace_back(GPUCA_M_STR(GPUCA_KRNLGPU_SINGLE(__VA_ARGS__)));
#define GPUCA_KRNL_LOAD_multi(...) kernels.emplace_back(GPUCA_M_STR(GPUCA_KRNLGPU_MULTI(__VA_ARGS__)));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi
}

#ifndef GPUCA_NO_CONSTANT_MEMORY
static GPUReconstructionDeviceBase::deviceConstantMemRegistration registerConstSymbol([]() {
  void* retVal = nullptr;
  GPUReconstructionCUDA::GPUFailedMsgI(cudaGetSymbolAddress(&retVal, gGPUConstantMemBuffer));
  return retVal;
});
#endif
