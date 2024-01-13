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

/// \file GPUReconstructionCUDAInternals.h
/// \author David Rohr

// All CUDA-header related stuff goes here, so we can run CING over GPUReconstructionCUDA

#ifndef GPURECONSTRUCTIONCUDAINTERNALS_H
#define GPURECONSTRUCTIONCUDAINTERNALS_H

#include <cuda.h>
#include "GPULogging.h"
#include <vector>
#include <memory>
#include <string>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

struct GPUReconstructionCUDAInternals {
  std::vector<std::unique_ptr<CUmodule>> kernelModules;     // module for RTC compilation
  std::vector<std::unique_ptr<CUfunction>> kernelFunctions; // vector of ptrs to RTC kernels
  std::vector<std::string> kernelNames;                     // names of kernels
  cudaStream_t Streams[GPUCA_MAX_STREAMS];                  // Pointer to array of CUDA Streams

  template <bool multi, class T, int I = 0>
  static int getRTCkernelNum(int k = -1);

  static void getArgPtrs(const void** pArgs) {}
  template <typename T, typename... Args>
  static void getArgPtrs(const void** pArgs, const T& arg, const Args&... args)
  {
    *pArgs = &arg;
    getArgPtrs(pArgs + 1, args...);
  }
};

class GPUDebugTiming
{
 public:
  GPUDebugTiming(bool d, void** t, cudaStream_t* s, GPUReconstruction::krnlSetup& x, GPUReconstructionCUDABackend* r) : mDeviceTimers(t), mStreams(s), mXYZ(x), mRec(r), mDo(d)
  {
    if (mDo) {
      if (mDeviceTimers) {
        mRec->GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[0], mStreams[mXYZ.x.stream]));
      } else {
        mTimer.ResetStart();
      }
    }
  }
  ~GPUDebugTiming()
  {
    if (mDo) {
      if (mDeviceTimers) {
        mRec->GPUFailedMsg(cudaEventRecord((cudaEvent_t)mDeviceTimers[1], mStreams[mXYZ.x.stream]));
        mRec->GPUFailedMsg(cudaEventSynchronize((cudaEvent_t)mDeviceTimers[1]));
        float v;
        mRec->GPUFailedMsg(cudaEventElapsedTime(&v, (cudaEvent_t)mDeviceTimers[0], (cudaEvent_t)mDeviceTimers[1]));
        mXYZ.t = v * 1.e-3;
      } else {
        mRec->GPUFailedMsg(cudaStreamSynchronize(mStreams[mXYZ.x.stream]));
        mXYZ.t = mTimer.GetCurrentElapsedTime();
      }
    }
  }

 private:
  GPUReconstruction::deviceEvent* mDeviceTimers;
  cudaStream_t* mStreams;
  GPUReconstruction::krnlSetup& mXYZ;
  GPUReconstructionCUDABackend* mRec;
  HighResTimer mTimer;
  bool mDo;
};

static_assert(std::is_convertible<cudaEvent_t, void*>::value, "CUDA event type incompatible to deviceEvent");

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
