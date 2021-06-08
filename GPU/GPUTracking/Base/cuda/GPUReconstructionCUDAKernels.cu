// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

class GPUDebugTiming
{
 public:
  GPUDebugTiming(bool d, void** t, cudaStream_t* s, GPUReconstruction::krnlSetup& x, GPUReconstructionCUDABackend* r = nullptr) : mDeviceTimers(t), mStreams(s), mXYZ(x), mRec(r), mDo(d)
  {
    if (mDo) {
      if (mDeviceTimers) {
        GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[0], mStreams[mXYZ.x.stream]));
      } else {
        mTimer.ResetStart();
      }
    }
  }
  ~GPUDebugTiming()
  {
    if (mDo) {
      if (mDeviceTimers) {
        GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[1], mStreams[mXYZ.x.stream]));
        GPUFailedMsg(cudaEventSynchronize((cudaEvent_t)mDeviceTimers[1]));
        float v;
        GPUFailedMsg(cudaEventElapsedTime(&v, (cudaEvent_t)mDeviceTimers[0], (cudaEvent_t)mDeviceTimers[1]));
        mXYZ.t = v * 1.e-3;
      } else {
        GPUFailedMsg(cudaStreamSynchronize(mStreams[mXYZ.x.stream]));
        mXYZ.t = mTimer.GetCurrentElapsedTime();
      }
    }
  }

 private:
  void** mDeviceTimers;
  cudaStream_t* mStreams;
  GPUReconstruction::krnlSetup& mXYZ;
  GPUReconstructionCUDABackend* mRec;
  HighResTimer mTimer;
  bool mDo;
};

#include "GPUReconstructionIncludesDevice.h"

/*
// Not using templated kernel any more, since nvidia profiler does not resolve template names
template <class T, int I, typename... Args>
GPUg() void runKernelCUDA(GPUCA_CONSMEM_PTR int iSlice_internal, Args... args)
{
  GPUshared() typename T::GPUSharedMemory smem;
  T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Processor(GPUCA_CONSMEM)[iSlice_internal], args...);
}
*/

#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                          \
  GPUCA_KRNL_WRAP(GPUCA_KRNL_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_CALL_single(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, args...);
#define GPUCA_KRNL_CALL_multi(x_class, x_attributes, x_arguments, x_forward) \
  GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)<<<x.nBlocks, x.nThreads, 0, me->mInternals->Streams[x.stream]>>>(GPUCA_CONSMEM_CALL y.start, y.num, args...);

#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

template <bool multi, class T, int I>
int GPUReconstructionCUDAInternals::getRTCkernelNum(int k)
{
  static int num = k;
  if (num < 0) {
    throw std::runtime_error("Invalid kernel");
  }
  return num;
}

template <>
void GPUReconstructionCUDABackend::runKernelBackendInternal<GPUMemClean16, 0>(krnlSetup& _xyz, void* const& ptr, unsigned long const& size)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  GPUFailedMsg(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

static void getArgPtrs(const void** pArgs) {}
template <typename T, typename... Args>
static void getArgPtrs(const void** pArgs, const T& arg, const Args&... args)
{
  *pArgs = &arg;
  getArgPtrs(pArgs + 1, args...);
}

template <class T, int I, typename... Args>
void GPUReconstructionCUDABackend::runKernelBackendInternal(krnlSetup& _xyz, const Args&... args)
{
  GPUDebugTiming timer(mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0, (void**)mDebugEvents, mInternals->Streams, _xyz);
  if (mProcessingSettings.rtc.enable) {
    auto& x = _xyz.x;
    auto& y = _xyz.y;
    const void* pArgs[sizeof...(Args) + 3]; // 3 is max: cons mem + y.start + y.num
    int arg_offset = 0;
#ifdef GPUCA_NO_CONSTANT_MEMORY
    arg_offset = 1;
    pArgs[0] = &mDeviceConstantMemRTC[0];
#endif
    pArgs[arg_offset] = &y.start;
    getArgPtrs(&pArgs[arg_offset + 1 + (y.num > 1)], args...);
    if (y.num <= 1) {
      GPUFailedMsg(cuLaunchKernel(*mInternals->rtcFunctions[mInternals->getRTCkernelNum<false, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    } else {
      pArgs[arg_offset + 1] = &y.num;
      GPUFailedMsg(cuLaunchKernel(*mInternals->rtcFunctions[mInternals->getRTCkernelNum<true, T, I>()], x.nBlocks, 1, 1, x.nThreads, 1, 1, 0, mInternals->Streams[x.stream], (void**)pArgs, nullptr));
    }
  } else {
    backendInternal<T, I>::runKernelBackendMacro(_xyz, this, args...);
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

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)                                                                           \
  template int GPUReconstructionCUDABackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(krnlSetup & _xyz GPUCA_M_STRIP(x_arguments)); \
  template int GPUReconstructionCUDAInternals::getRTCkernelNum<false, GPUCA_M_KRNL_TEMPLATE(x_class)>(int k);                               \
  template int GPUReconstructionCUDAInternals::getRTCkernelNum<true, GPUCA_M_KRNL_TEMPLATE(x_class)>(int k);
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

void* GPUReconstructionCUDABackend::GetBackendConstSymbolAddress()
{
  void* retVal = nullptr;
#ifndef GPUCA_NO_CONSTANT_MEMORY
  GPUFailedMsg(cudaGetSymbolAddress(&retVal, gGPUConstantMemBuffer));
#endif
  return retVal;
}

void GPUReconstructionCUDABackend::PrintKernelOccupancies()
{
  int maxBlocks, threads, suggestedBlocks;
  cudaFuncAttributes attr;
  GPUFailedMsg(cuCtxPushCurrent(mInternals->CudaContext));
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward)                                                          \
  GPUFailedMsg(cudaOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));        \
  GPUFailedMsg(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class)), threads, 0)); \
  GPUFailedMsg(cudaFuncGetAttributes(&attr, GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))));                                          \
  GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", GPUCA_M_STR(GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))), threads, maxBlocks, suggestedBlocks, attr.numRegs, (int)attr.sharedSizeBytes);
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward)                                                                    \
  GPUFailedMsg(cudaOccupancyMaxPotentialBlockSize(&suggestedBlocks, &threads, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)));        \
  GPUFailedMsg(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi), threads, 0)); \
  GPUFailedMsg(cudaFuncGetAttributes(&attr, GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)));                                          \
  GPUInfo("Kernel: %50s Block size: %4d, Maximum active blocks: %3d, Suggested blocks: %3d, Regs: %3d, smem: %3d", GPUCA_M_STR(GPUCA_M_CAT3(krnl_, GPUCA_M_KRNL_NAME(x_class), _multi)), threads, maxBlocks, suggestedBlocks, attr.numRegs, (int)attr.sharedSizeBytes);
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
#undef GPUCA_KRNL_LOAD_single
#undef GPUCA_KRNL_LOAD_multi
  GPUFailedMsg(cuCtxPopCurrent(&mInternals->CudaContext));
}

template class GPUReconstructionKernels<GPUReconstructionCUDABackend>;
