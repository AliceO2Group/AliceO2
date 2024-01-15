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

#include "GPUReconstructionIncludesDevice.h"

template <>
void GPUReconstructionCUDABackend::runKernelBackendInternal<GPUMemClean16, 0>(krnlSetup& _xyz, void* const& ptr, unsigned long const& size)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  GPUFailedMsg(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

template <class T, int I, typename... Args>
void GPUReconstructionCUDABackend::runKernelBackendInternal(krnlSetup& _xyz, const Args&... args)
{
  GPUDebugTiming timer(mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0, (void**)mDebugEvents, mInternals->Streams, _xyz, this);
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE != 1
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
      GPUFailedMsg(cuLaunchKernel(*mInternals->kernelFunctions[mInternals->getRTCkernelNum<false, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    } else {
      pArgs[arg_offset + 1] = &y.num;
      GPUFailedMsg(cuLaunchKernel(*mInternals->kernelFunctions[mInternals->getRTCkernelNum<true, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    }
  }
  if (mProcessingSettings.checkKernelFailures) {
    if (GPUDebug(GetKernelName<T, I>(), _xyz.x.stream, true)) {
      throw std::runtime_error("Kernel Failure");
    }
  }
}

template <class T, int I, typename... Args>
int GPUReconstructionCUDABackend::runKernelBackend(krnlSetup& _xyz, Args... args)
{
  auto& x = _xyz.x;
  auto& z = _xyz.z;
  if (z.evList) {
    for (int k = 0; k < z.nEvents; k++) {
      GPUFailedMsg(cudaStreamWaitEvent(mInternals->Streams[x.stream], ((cudaEvent_t*)z.evList)[k], 0));
    }
  }
  runKernelBackendInternal<T, I>(_xyz, args...);
  GPUFailedMsg(cudaGetLastError());
  if (z.ev) {
    GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*)z.ev, mInternals->Streams[x.stream]));
  }
  return 0;
}

#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 1
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                          \
  template int GPUReconstructionCUDABackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(krnlSetup & _xyz GPUCA_M_STRIP(x_arguments));
#else
#if defined(GPUCA_KERNEL_COMPILE_MODE) && GPUCA_KERNEL_COMPILE_MODE == 2
#define GPUCA_KRNL_DEFONLY
#endif
#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)             \
  GPUCA_KRNL_PROP(x_class, x_attributes)                                      \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_, x_class, x_attributes, x_arguments, x_forward) \
  template int GPUReconstructionCUDABackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(krnlSetup & _xyz GPUCA_M_STRIP(x_arguments));
#define GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);
#endif
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

#ifndef GPUCA_NO_CONSTANT_MEMORY
static GPUReconstructionDeviceBase::deviceConstantMemRegistration registerConstSymbol([]() {
  void* retVal = nullptr;
  GPUReconstructionCUDA::GPUFailedMsgI(cudaGetSymbolAddress(&retVal, gGPUConstantMemBuffer));
  return retVal;
});
#endif
