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

/// \file GPUReconstructionProcessing.cxx
/// \author David Rohr

#include "GPUReconstructionProcessing.h"
#include "GPUReconstructionThreading.h"
#include "GPUDefParametersLoad.inc"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUSettings.h"
#include "GPULogging.h"

using namespace o2::gpu;

GPUReconstructionProcessing::GPUReconstructionProcessing(const GPUSettingsDeviceBackend& cfg) : GPUReconstruction(cfg)
{
  if (mMaster == nullptr) {
    mParCPU = new GPUDefParameters(o2::gpu::internal::GPUDefParametersLoad());
    mParDevice = new GPUDefParameters();
  } else {
    GPUReconstructionProcessing* master = dynamic_cast<GPUReconstructionProcessing*>(mMaster);
    mParCPU = master->mParCPU;
    mParDevice = master->mParDevice;
  }
}

GPUReconstructionProcessing::~GPUReconstructionProcessing()
{
  if (mMaster == nullptr) {
    delete mParCPU;
    delete mParDevice;
  }
}

int32_t GPUReconstructionProcessing::getNKernelHostThreads(bool splitCores)
{
  int32_t nThreads = 0;
  if (GetProcessingSettings().inKernelParallel == 2 && mNActiveThreadsOuterLoop) {
    if (splitCores) {
      nThreads = mMaxHostThreads / mNActiveThreadsOuterLoop;
      nThreads += (uint32_t)getHostThreadIndex() < mMaxHostThreads % mNActiveThreadsOuterLoop;
    } else {
      nThreads = mMaxHostThreads;
    }
    nThreads = std::max(1, nThreads);
  } else {
    nThreads = GetProcessingSettings().inKernelParallel ? mMaxHostThreads : 1;
  }
  return nThreads;
}

void GPUReconstructionProcessing::SetNActiveThreads(int32_t n)
{
  mActiveHostKernelThreads = std::max(1, n < 0 ? mMaxHostThreads : std::min(n, mMaxHostThreads));
  mThreading->activeThreads = std::make_unique<tbb::task_arena>(mActiveHostKernelThreads);
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Set number of active parallel kernels threads on host to %d (%d requested)", mActiveHostKernelThreads, n);
  }
}

void GPUReconstructionProcessing::runParallelOuterLoop(bool doGPU, uint32_t nThreads, std::function<void(uint32_t)> lambda)
{
  uint32_t nThreadsAdjusted = SetAndGetNActiveThreadsOuterLoop(!doGPU, nThreads);
  if (nThreadsAdjusted > 1) {
    tbb::task_arena(nThreadsAdjusted).execute([&] {
      tbb::parallel_for<uint32_t>(0, nThreads, lambda, tbb::simple_partitioner());
    });
  } else {
    for (uint32_t i = 0; i < nThreads; i++) {
      lambda(i);
    }
  }
}

uint32_t GPUReconstructionProcessing::SetAndGetNActiveThreadsOuterLoop(bool condition, uint32_t max)
{
  if (condition && GetProcessingSettings().inKernelParallel != 1) {
    mNActiveThreadsOuterLoop = GetProcessingSettings().inKernelParallel == 2 ? std::min<uint32_t>(max, mMaxHostThreads) : mMaxHostThreads;
  } else {
    mNActiveThreadsOuterLoop = 1;
  }
  if (GetProcessingSettings().debugLevel >= 5) {
    printf("Running %d threads in outer loop\n", mNActiveThreadsOuterLoop);
  }
  return mNActiveThreadsOuterLoop;
}

std::atomic_flag GPUReconstructionProcessing::mTimerFlag = ATOMIC_FLAG_INIT;

GPUReconstructionProcessing::timerMeta* GPUReconstructionProcessing::insertTimer(uint32_t id, std::string&& name, int32_t J, int32_t num, int32_t type, RecoStep step)
{
  while (mTimerFlag.test_and_set()) {
  }
  if (mTimers.size() <= id) {
    mTimers.resize(id + 1);
  }
  if (mTimers[id] == nullptr) {
    if (J >= 0) {
      name += std::to_string(J);
    }
    mTimers[id].reset(new timerMeta{std::unique_ptr<HighResTimer[]>{new HighResTimer[num]}, name, num, type, 1u, step, (size_t)0});
  } else {
    mTimers[id]->count++;
  }
  timerMeta* retVal = mTimers[id].get();
  mTimerFlag.clear();
  return retVal;
}

GPUReconstructionProcessing::timerMeta* GPUReconstructionProcessing::getTimerById(uint32_t id, bool increment)
{
  timerMeta* retVal = nullptr;
  while (mTimerFlag.test_and_set()) {
  }
  if (mTimers.size() > id && mTimers[id]) {
    retVal = mTimers[id].get();
    retVal->count += increment;
  }
  mTimerFlag.clear();
  return retVal;
}

uint32_t GPUReconstructionProcessing::getNextTimerId()
{
  static std::atomic<uint32_t> id{0};
  return id.fetch_add(1);
}

std::unique_ptr<GPUReconstructionProcessing::threadContext> GPUReconstructionProcessing::GetThreadContext()
{
  return std::make_unique<threadContext>();
}

GPUReconstruction::threadContext::threadContext() = default;
GPUReconstruction::threadContext::~threadContext() = default;

const std::vector<std::string> GPUReconstructionProcessing::mKernelNames = {
#define GPUCA_KRNL(x_class, ...) GPUCA_M_STR(GPUCA_M_KRNL_NAME(x_class)),
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
};

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types, x_num)          \
  template <>                                                                              \
  uint32_t GPUReconstructionProcessing::GetKernelNum<GPUCA_M_KRNL_TEMPLATE(x_class)>()     \
  {                                                                                        \
    return x_num;                                                                          \
  }                                                                                        \
  template <>                                                                              \
  const char* GPUReconstructionProcessing::GetKernelName<GPUCA_M_KRNL_TEMPLATE(x_class)>() \
  {                                                                                        \
    return GPUCA_M_STR(GPUCA_M_KRNL_NAME(x_class));                                        \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
