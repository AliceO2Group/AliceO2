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
#include "GPUCommonHelpers.h"

namespace o2::gpu
{

struct GPUReconstructionCUDAInternals {
  std::vector<std::unique_ptr<CUmodule>> kernelModules;     // module for RTC compilation
  std::vector<std::unique_ptr<CUfunction>> kernelFunctions; // vector of ptrs to RTC kernels
  cudaStream_t Streams[constants::GPU_MAX_STREAMS];         // Pointer to array of CUDA Streams

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
  GPUDebugTiming(bool d, GPUReconstructionProcessing::deviceEvent* t, cudaStream_t* s, const GPUReconstructionProcessing::krnlSetupTime& x, GPUReconstructionCUDA* r) : mDeviceTimers(t), mStreams(s), mXYZ(x), mRec(r), mDo(d)
  {
    if (mDo) {
      if (mDeviceTimers) {
        mRec->GPUChkErr(cudaEventRecord(mDeviceTimers[0].get<cudaEvent_t>(), mStreams[mXYZ.x.stream]));
      } else {
        mTimer.ResetStart();
      }
    }
  }
  ~GPUDebugTiming()
  {
    if (mDo && mXYZ.t == 0.) {
      if (mDeviceTimers) {
        mRec->GPUChkErr(cudaEventRecord(mDeviceTimers[1].get<cudaEvent_t>(), mStreams[mXYZ.x.stream]));
        mRec->GPUChkErr(cudaEventSynchronize(mDeviceTimers[1].get<cudaEvent_t>()));
        float v;
        mRec->GPUChkErr(cudaEventElapsedTime(&v, mDeviceTimers[0].get<cudaEvent_t>(), mDeviceTimers[1].get<cudaEvent_t>()));
        mXYZ.t = v * 1.e-3f;
      } else {
        mRec->GPUChkErr(cudaStreamSynchronize(mStreams[mXYZ.x.stream]));
        mXYZ.t = mTimer.GetCurrentElapsedTime();
      }
    }
  }

 private:
  GPUReconstructionProcessing::deviceEvent* mDeviceTimers;
  cudaStream_t* mStreams;
  const GPUReconstructionProcessing::krnlSetupTime& mXYZ;
  GPUReconstructionCUDA* mRec;
  HighResTimer mTimer;
  bool mDo;
};

static_assert(std::is_convertible_v<cudaEvent_t, void*>, "CUDA event type incompatible to deviceEvent");

} // namespace o2::gpu

#endif
