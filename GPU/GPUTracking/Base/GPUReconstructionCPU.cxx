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

/// \file GPUReconstructionCPU.cxx
/// \author David Rohr

#include "GPUReconstructionCPU.h"
#include "GPUReconstructionIncludes.h"
#include "GPUChain.h"

#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCSliceOutCluster.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUMemoryResource.h"
#include "GPUConstantMem.h"
#include "GPUMemorySizeScalers.h"
#include <atomic>

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#if defined(WITH_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
static inline int32_t omp_get_thread_num() { return 0; }
static inline int32_t omp_get_max_threads() { return 1; }
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::gpu_reconstruction_kernels;

constexpr GPUReconstructionCPU::krnlRunRange GPUReconstructionCPU::krnlRunRangeNone;
constexpr GPUReconstructionCPU::krnlEvent GPUReconstructionCPU::krnlEventNone;

GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCPU(cfg); }

GPUReconstructionCPU::~GPUReconstructionCPU()
{
  Exit(); // Needs to be identical to GPU backend bahavior in order to avoid calling abstract methods later in the destructor
}

template <class T, int32_t I, typename... Args>
inline int32_t GPUReconstructionCPUBackend::runKernelBackendInternal(const krnlSetupTime& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  if (x.device == krnlDeviceType::Device) {
    throw std::runtime_error("Cannot run device kernel on host");
  }
  if (x.nThreads != 1) {
    throw std::runtime_error("Cannot run device kernel on host with nThreads != 1");
  }
  uint32_t num = y.num == 0 || y.num == -1 ? 1 : y.num;
  for (uint32_t k = 0; k < num; k++) {
    int32_t ompThreads = 0;
    if (mProcessingSettings.ompKernels == 2) {
      ompThreads = mProcessingSettings.ompThreads / mNestedLoopOmpFactor;
      if ((uint32_t)getOMPThreadNum() < mProcessingSettings.ompThreads % mNestedLoopOmpFactor) {
        ompThreads++;
      }
      ompThreads = std::max(1, ompThreads);
    } else {
      ompThreads = mProcessingSettings.ompKernels ? mProcessingSettings.ompThreads : 1;
    }
    if (ompThreads > 1) {
      if (mProcessingSettings.debugLevel >= 5) {
        printf("Running %d ompThreads\n", ompThreads);
      }
      GPUCA_OPENMP(parallel for num_threads(ompThreads))
      for (uint32_t iB = 0; iB < x.nBlocks; iB++) {
        typename T::GPUSharedMemory smem;
        T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.start + k], args...);
      }
    } else {
      for (uint32_t iB = 0; iB < x.nBlocks; iB++) {
        typename T::GPUSharedMemory smem;
        T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.start + k], args...);
      }
    }
  }
  return 0;
}

template <>
inline int32_t GPUReconstructionCPUBackend::runKernelBackendInternal<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  memset(ptr, 0, size);
  return 0;
}

template <class T, int32_t I, typename... Args>
int32_t GPUReconstructionCPUBackend::runKernelBackend(const krnlSetupArgs<T, I, Args...>& args)
{
  return std::apply([this, &args](auto&... vals) { return runKernelBackendInternal<T, I, Args...>(args.s, vals...); }, args.v);
}

template <class T, int32_t I>
krnlProperties GPUReconstructionCPUBackend::getKernelPropertiesBackend()
{
  return krnlProperties{1, 1};
}

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types)                                                                                                          \
  template int32_t GPUReconstructionCPUBackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args); \
  template krnlProperties GPUReconstructionCPUBackend::getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>();
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

size_t GPUReconstructionCPU::TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  memcpy(dst, src, size);
  return 0;
}
size_t GPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev) { return 0; }
int32_t GPUReconstructionCPU::GPUDebug(const char* state, int32_t stream, bool force) { return 0; }
size_t GPUReconstructionCPU::TransferMemoryResourcesHelper(GPUProcessor* proc, int32_t stream, bool all, bool toGPU)
{
  int32_t inc = toGPU ? GPUMemoryResource::MEMORY_INPUT_FLAG : GPUMemoryResource::MEMORY_OUTPUT_FLAG;
  int32_t exc = toGPU ? GPUMemoryResource::MEMORY_OUTPUT_FLAG : GPUMemoryResource::MEMORY_INPUT_FLAG;
  size_t n = 0;
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    GPUMemoryResource& res = mMemoryResources[i];
    if (res.mPtr == nullptr) {
      continue;
    }
    if (proc && res.mProcessor != proc) {
      continue;
    }
    if (!(res.mType & GPUMemoryResource::MEMORY_GPU) || (res.mType & GPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) {
      continue;
    }
    if (!mProcessingSettings.keepAllMemory && !all && (res.mType & exc) && !(res.mType & inc)) {
      continue;
    }
    if (toGPU) {
      n += TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
    } else {
      n += TransferMemoryResourceToHost(&mMemoryResources[i], stream);
    }
  }
  return n;
}

int32_t GPUReconstructionCPU::GetThread()
{
// Get Thread ID
#if defined(__APPLE__)
  return (0); // syscall is deprecated on MacOS..., only needed for GPU support which we don't do on Mac anyway
#elif defined(_WIN32)
  return ((int32_t)(size_t)GetCurrentThread());
#else
  return ((int32_t)syscall(SYS_gettid));
#endif
}

int32_t GPUReconstructionCPU::InitDevice()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      if (mDeviceMemorySize > mHostMemorySize) {
        mHostMemorySize = mDeviceMemorySize;
      }
      mHostMemoryBase = operator new(mHostMemorySize GPUCA_OPERATOR_NEW_ALIGNMENT);
    }
    mHostMemoryPermanent = mHostMemoryBase;
    ClearAllocatedMemory();
  }
  if (mProcessingSettings.ompKernels) {
    mBlockCount = getOMPMaxThreads();
  }
  mThreadId = GetThread();
  mProcShadow.mProcessorsProc = processors();
  return 0;
}

int32_t GPUReconstructionCPU::ExitDevice()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      operator delete(mHostMemoryBase GPUCA_OPERATOR_NEW_ALIGNMENT);
    }
    mHostMemoryPool = mHostMemoryBase = mHostMemoryPoolEnd = mHostMemoryPermanent = nullptr;
    mHostMemorySize = 0;
  }
  return 0;
}

int32_t GPUReconstructionCPU::RunChains()
{
  mMemoryScalers->temporaryFactor = 1.;
  mStatNEvents++;
  mNEventsProcessed++;

  timerTotal.Start();
  if (mProcessingSettings.doublePipeline) {
    if (EnqueuePipeline()) {
      return 1;
    }
  } else {
    if (mThreadId != GetThread()) {
      if (mProcessingSettings.debugLevel >= 2) {
        GPUInfo("Thread changed, migrating context, Previous Thread: %d, New Thread: %d", mThreadId, GetThread());
      }
      mThreadId = GetThread();
    }
    if (mSlaves.size() || mMaster) {
      WriteConstantParams(); // Reinitialize
    }
    for (uint32_t i = 0; i < mChains.size(); i++) {
      int32_t retVal = mChains[i]->RunChain();
      if (retVal) {
        return retVal;
      }
    }
  }
  timerTotal.Stop();

  mStatWallTime = (timerTotal.GetElapsedTime() * 1000000. / mStatNEvents);
  std::string nEventReport;
  if (GetProcessingSettings().debugLevel >= 0 && mStatNEvents > 1) {
    nEventReport += "   (avergage of " + std::to_string(mStatNEvents) + " runs)";
  }
  if (GetProcessingSettings().debugLevel >= 1) {
    double kernelTotal = 0;
    std::vector<double> kernelStepTimes(GPUDataTypes::N_RECO_STEPS);

    for (uint32_t i = 0; i < mTimers.size(); i++) {
      double time = 0;
      if (mTimers[i] == nullptr) {
        continue;
      }
      for (int32_t j = 0; j < mTimers[i]->num; j++) {
        HighResTimer& timer = mTimers[i]->timer[j];
        time += timer.GetElapsedTime();
        if (mProcessingSettings.resetTimers) {
          timer.Reset();
        }
      }

      uint32_t type = mTimers[i]->type;
      if (type == 0) {
        kernelTotal += time;
        int32_t stepNum = getRecoStepNum(mTimers[i]->step);
        kernelStepTimes[stepNum] += time;
      }
      char bandwidth[256] = "";
      if (mTimers[i]->memSize && mStatNEvents && time != 0.) {
        snprintf(bandwidth, 256, " (%8.3f GB/s - %'14zu bytes - %'14zu per call)", mTimers[i]->memSize / time * 1e-9, mTimers[i]->memSize / mStatNEvents, mTimers[i]->memSize / mStatNEvents / mTimers[i]->count);
      }
      printf("Execution Time: Task (%c %8ux): %50s Time: %'10.0f us%s\n", type == 0 ? 'K' : 'C', mTimers[i]->count, mTimers[i]->name.c_str(), time * 1000000 / mStatNEvents, bandwidth);
      if (mProcessingSettings.resetTimers) {
        mTimers[i]->count = 0;
        mTimers[i]->memSize = 0;
      }
    }
    for (int32_t i = 0; i < GPUDataTypes::N_RECO_STEPS; i++) {
      if (kernelStepTimes[i] != 0. || mTimersRecoSteps[i].timerTotal.GetElapsedTime() != 0.) {
        printf("Execution Time: Step              : %11s %38s Time: %'10.0f us %64s ( Total Time : %'14.0f us)\n", "Tasks", GPUDataTypes::RECO_STEP_NAMES[i], kernelStepTimes[i] * 1000000 / mStatNEvents, "", mTimersRecoSteps[i].timerTotal.GetElapsedTime() * 1000000 / mStatNEvents);
      }
      if (mTimersRecoSteps[i].bytesToGPU) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10.0f us (%8.3f GB/s - %'14zu bytes - %'14zu per call)\n", mTimersRecoSteps[i].countToGPU, "DMA to GPU", GPUDataTypes::RECO_STEP_NAMES[i], mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1000000 / mStatNEvents,
               mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1e-9, mTimersRecoSteps[i].bytesToGPU / mStatNEvents, mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].countToGPU);
      }
      if (mTimersRecoSteps[i].bytesToHost) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10.0f us (%8.3f GB/s - %'14zu bytes - %'14zu per call)\n", mTimersRecoSteps[i].countToHost, "DMA to Host", GPUDataTypes::RECO_STEP_NAMES[i], mTimersRecoSteps[i].timerToHost.GetElapsedTime() * 1000000 / mStatNEvents,
               mTimersRecoSteps[i].bytesToHost / mTimersRecoSteps[i].timerToHost.GetElapsedTime() * 1e-9, mTimersRecoSteps[i].bytesToHost / mStatNEvents, mTimersRecoSteps[i].bytesToHost / mTimersRecoSteps[i].countToHost);
      }
      if (mProcessingSettings.resetTimers) {
        mTimersRecoSteps[i].bytesToGPU = mTimersRecoSteps[i].bytesToHost = 0;
        mTimersRecoSteps[i].timerToGPU.Reset();
        mTimersRecoSteps[i].timerToHost.Reset();
        mTimersRecoSteps[i].timerTotal.Reset();
        mTimersRecoSteps[i].countToGPU = 0;
        mTimersRecoSteps[i].countToHost = 0;
      }
    }
    for (int32_t i = 0; i < GPUDataTypes::N_GENERAL_STEPS; i++) {
      if (mTimersGeneralSteps[i].GetElapsedTime() != 0.) {
        printf("Execution Time: General Step      : %50s Time: %'10.0f us\n", GPUDataTypes::GENERAL_STEP_NAMES[i], mTimersGeneralSteps[i].GetElapsedTime() * 1000000 / mStatNEvents);
      }
    }
    mStatKernelTime = kernelTotal * 1000000 / mStatNEvents;
    printf("Execution Time: Total   : %50s Time: %'10.0f us%s\n", "Total Kernel", mStatKernelTime, nEventReport.c_str());
    printf("Execution Time: Total   : %50s Time: %'10.0f us%s\n", "Total Wall", mStatWallTime, nEventReport.c_str());
  } else if (GetProcessingSettings().debugLevel >= 0) {
    GPUInfo("Total Wall Time: %lu us%s", (uint64_t)mStatWallTime, nEventReport.c_str());
  }
  if (mProcessingSettings.resetTimers) {
    mStatNEvents = 0;
    timerTotal.Reset();
  }

  return 0;
}

void GPUReconstructionCPU::ResetDeviceProcessorTypes()
{
  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && mProcessors[i].proc->mLinkedProcessor) {
      mProcessors[i].proc->mLinkedProcessor->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
    }
  }
}

int32_t GPUReconstructionCPUBackend::getOMPThreadNum()
{
  return omp_get_thread_num();
}

int32_t GPUReconstructionCPUBackend::getOMPMaxThreads()
{
  return omp_get_max_threads();
}

static std::atomic_flag timerFlag = ATOMIC_FLAG_INIT; // TODO: Should be a class member not global, but cannot be moved to header due to ROOT limitation

GPUReconstructionCPU::timerMeta* GPUReconstructionCPU::insertTimer(uint32_t id, std::string&& name, int32_t J, int32_t num, int32_t type, RecoStep step)
{
  while (timerFlag.test_and_set()) {
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
  timerFlag.clear();
  return retVal;
}

GPUReconstructionCPU::timerMeta* GPUReconstructionCPU::getTimerById(uint32_t id, bool increment)
{
  timerMeta* retVal = nullptr;
  while (timerFlag.test_and_set()) {
  }
  if (mTimers.size() > id && mTimers[id]) {
    retVal = mTimers[id].get();
    retVal->count += increment;
  }
  timerFlag.clear();
  return retVal;
}

uint32_t GPUReconstructionCPU::getNextTimerId()
{
  static std::atomic<uint32_t> id{0};
  return id.fetch_add(1);
}

uint32_t GPUReconstructionCPU::SetAndGetNestedLoopOmpFactor(bool condition, uint32_t max)
{
  if (condition && mProcessingSettings.ompKernels != 1) {
    mNestedLoopOmpFactor = mProcessingSettings.ompKernels == 2 ? std::min<uint32_t>(max, mProcessingSettings.ompThreads) : mProcessingSettings.ompThreads;
  } else {
    mNestedLoopOmpFactor = 1;
  }
  if (mProcessingSettings.debugLevel >= 5) {
    printf("Running %d OMP threads in outer loop\n", mNestedLoopOmpFactor);
  }
  return mNestedLoopOmpFactor;
}

void GPUReconstructionCPU::UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, int32_t stream)
{
  param().occupancyMap = mapHost;
  param().occupancyTotal = occupancyTotal;
  if (IsGPU()) {
    if (!((size_t)&param().occupancyTotal - (size_t)&param().occupancyMap == sizeof(param().occupancyMap) && sizeof(param().occupancyMap) == sizeof(size_t) && sizeof(param().occupancyTotal) < sizeof(size_t))) {
      throw std::runtime_error("occupancy data not consecutive in GPUParam");
    }
    const auto threadContext = GetThreadContext();
    size_t tmp[2] = {(size_t)mapGPU, 0};
    memcpy(&tmp[1], &occupancyTotal, sizeof(occupancyTotal));
    WriteToConstantMemory((char*)&processors()->param.occupancyMap - (char*)processors(), &tmp, sizeof(param().occupancyMap) + sizeof(param().occupancyTotal), stream);
  }
}
