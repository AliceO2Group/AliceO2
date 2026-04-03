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
#include "GPUReconstructionThreading.h"
#include "GPUChain.h"
#include "GPUDefParametersRuntime.h"
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
#include "GPULogging.h"
#include "GPUMemorySizeScalers.h"
#include "GPUReconstructionProcessingKernels.inc"
#include "GPUTPCClusterOccupancyMap.h"

#include <atomic>
#include <ctime>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace o2::gpu;

constexpr GPUReconstructionCPU::krnlRunRange GPUReconstructionCPU::krnlRunRangeNone;
constexpr GPUReconstructionCPU::krnlEvent GPUReconstructionCPU::krnlEventNone;

GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCPU(cfg); }

GPUReconstructionCPU::~GPUReconstructionCPU()
{
  Exit(); // Needs to be identical to GPU backend bahavior in order to avoid calling abstract methods later in the destructor
}

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionCPU::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  if (x.device == krnlDeviceType::Device) {
    throw std::runtime_error("Cannot run device kernel on host");
  }
  if (x.nThreads != 1) {
    throw std::runtime_error("Cannot run device kernel on host with nThreads != 1");
  }
  int32_t nThreads = getNKernelHostThreads(false);
  if (nThreads > 1) {
    if (GetProcessingSettings().debugLevel >= 5) {
      GPUInfo("Running %d Threads", mThreading->activeThreads->max_concurrency());
    }
    tbb::this_task_arena::isolate([&] {
      mThreading->activeThreads->execute([&] {
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, x.nBlocks, 1), [&](const tbb::blocked_range<uint32_t>& r) {
          typename T::GPUSharedMemory smem;
          for (uint32_t iB = r.begin(); iB < r.end(); iB++) {
            T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.index], args...);
          }
        });
      });
    });
  } else {
    for (uint32_t iB = 0; iB < x.nBlocks; iB++) {
      typename T::GPUSharedMemory smem;
      T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.index], args...);
    }
  }
}

template <>
inline void GPUReconstructionCPU::runKernelBackend<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  int32_t nThreads = std::max<int32_t>(1, std::min<int32_t>(size / (16 * 1024 * 1024), getNKernelHostThreads(true)));
  if (nThreads > 1) {
    tbb::parallel_for(0, nThreads, [&](int iThread) {
      size_t threadSize = size / nThreads;
      if (threadSize % 4096) {
        threadSize += 4096 - threadSize % 4096;
      }
      size_t offset = threadSize * iThread;
      size_t mySize = std::min<size_t>(threadSize, size - offset);
      if (mySize) {
        memset((char*)ptr + offset, 0, mySize);
      } // clang-format off
    }, tbb::static_partitioner()); // clang-format on
  } else {
    memset(ptr, 0, size);
  }
}

template <class S, int32_t I>
GPUReconstructionProcessing::krnlProperties GPUReconstructionCPU::getKernelProperties(int gpu)
{
  if (gpu == -1) {
    gpu = IsGPU();
  }
  const auto num = GetKernelNum<S, I>();
  const auto* p = gpu ? mParDevice : mParCPU;
  GPUReconstructionProcessing::krnlProperties ret = {p->par_LB_maxThreads[num], p->par_LB_minBlocks[num], p->par_LB_forceBlocks[num]};
  if (ret.nThreads == 0) {
    ret.nThreads = gpu ? mThreadCount : 1u;
  }
  if (ret.minBlocks == 0) {
    ret.minBlocks = 1;
  }
  return ret;
}

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types, ...) \
  template GPUReconstructionProcessing::krnlProperties GPUReconstructionCPU::getKernelProperties<GPUCA_M_KRNL_TEMPLATE(x_class)>(int gpu);
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
    if (!GetProcessingSettings().keepAllMemory && !all && (res.mType & exc) && !(res.mType & inc)) {
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
  mActiveHostKernelThreads = mMaxHostThreads;
  mThreading->activeThreads = std::make_unique<tbb::task_arena>(mActiveHostKernelThreads);
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      if (mDeviceMemorySize > mHostMemorySize) {
        mHostMemorySize = mDeviceMemorySize;
      }
      mHostMemoryBase = ::operator new(mHostMemorySize, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
    }
    mHostMemoryPermanent = mHostMemoryBase;
    ClearAllocatedMemory();
  }
  if (GetProcessingSettings().inKernelParallel) {
    mMultiprocessorCount = mMaxHostThreads;
  }
  mProcShadow.mProcessorsProc = processors();
  return 0;
}

int32_t GPUReconstructionCPU::ExitDevice()
{
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      ::operator delete(mHostMemoryBase, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
    }
    mHostMemoryPool = mHostMemoryBase = mHostMemoryPoolEnd = mHostMemoryPermanent = nullptr;
    mHostMemorySize = 0;
  }
  return 0;
}

int32_t GPUReconstructionCPU::RunChains()
{
  mMemoryScalers->temporaryFactor = 1.;
  if (GetProcessingSettings().memoryScalingFuzz) {
    static std::mt19937 rng;
    static std::uniform_int_distribution<uint64_t> dist(0, 1000000);
    uint64_t fuzzFactor = GetProcessingSettings().memoryScalingFuzz == 1 ? dist(rng) : GetProcessingSettings().memoryScalingFuzz;
    GPUInfo("Fuzzing memory scaling factor with %lu", fuzzFactor);
    mMemoryScalers->fuzzScalingFactor(fuzzFactor);
  }

  mStatNEvents++;
  mNEventsProcessed++;

  if (GetProcessingSettings().debugLevel >= 3 || GetProcessingSettings().allocDebugLevel) {
    GPUInfo("Allocated memory when starting processing %34s", "");
    PrintMemoryOverview();
  }

  mTimerTotal.Start();
  const std::clock_t cpuTimerStart = std::clock();
  int32_t retVal = 0;
  if (GetProcessingSettings().doublePipeline) {
    retVal = EnqueuePipeline();
  } else {
    if (mSlaves.size() || mMaster) {
      WriteConstantParams(); // Reinitialize // TODO: Get this in sync with GPUChainTracking::DoQueuedUpdates, and consider the doublePipeline
    }
    for (uint32_t i = 0; i < mChains.size(); i++) {
      retVal = mChains[i]->RunChain();
    }
  }
  if (retVal != 0 && retVal != 2) {
    return retVal;
  }
  mTimerTotal.Stop();
  if (GetProcessingSettings().tpcFreeAllocatedMemoryAfterProcessing) {
    ClearAllocatedMemory();
  }
  mStatCPUTime += (double)(std::clock() - cpuTimerStart) / CLOCKS_PER_SEC;
  if (GetProcessingSettings().debugLevel >= 3 || GetProcessingSettings().allocDebugLevel) {
    GPUInfo("Allocated memory when ending processing %36s", "");
    PrintMemoryOverview();
  }

  mStatWallTime = (mTimerTotal.GetElapsedTime() * 1000000. / mStatNEvents);
  std::string nEventReport;
  if (GetProcessingSettings().debugLevel >= 0 && mStatNEvents > 1) {
    nEventReport += "   (avergage of " + std::to_string(mStatNEvents) + " runs)";
  }
  double kernelTotal = 0;
  std::vector<double> kernelStepTimes(gpudatatypes::N_RECO_STEPS, 0.);

  debugWriter writer(GetProcessingSettings().debugCSV, GetProcessingSettings().debugMarkdown, mStatNEvents);

  if (GetProcessingSettings().debugLevel >= 1) {
    writer.header();
    for (uint32_t i = 0; i < mTimers.size(); i++) {
      double time = 0;
      if (mTimers[i] == nullptr) {
        continue;
      }
      for (int32_t j = 0; j < mTimers[i]->num; j++) {
        HighResTimer& timer = mTimers[i]->timer[j];
        time += timer.GetElapsedTime();
        if (GetProcessingSettings().resetTimers) {
          timer.Reset();
        }
      }

      uint32_t type = mTimers[i]->type;
      if (type == 0) {
        kernelTotal += time;
        int32_t stepNum = getRecoStepNum(mTimers[i]->step);
        kernelStepTimes[stepNum] += time;
      }
      writer.row('K', mTimers[i]->count, mTimers[i]->name.c_str(), time, -1.0, -1.0, mTimers[i]->memSize);
      if (GetProcessingSettings().resetTimers) {
        mTimers[i]->count = 0;
        mTimers[i]->memSize = 0;
      }
    }
  }
  if (GetProcessingSettings().recoTaskTiming) {
    for (int32_t i = 0; i < gpudatatypes::N_RECO_STEPS; i++) {
      if (kernelStepTimes[i] != 0. || mTimersRecoSteps[i].timerTotal.GetElapsedTime() != 0.) {
        writer.row(' ', 0, std::string(gpudatatypes::RECO_STEP_NAMES[i]) + " (Tasks)", kernelStepTimes[i], mTimersRecoSteps[i].timerCPU, mTimersRecoSteps[i].timerTotal.GetElapsedTime(), 0);
      }
      if (mTimersRecoSteps[i].bytesToGPU) {
        writer.row('D', mTimersRecoSteps[i].countToGPU, std::string(gpudatatypes::RECO_STEP_NAMES[i]) + " (DMA to GPU)", mTimersRecoSteps[i].timerToGPU.GetElapsedTime(), -1.0, -1.0, mTimersRecoSteps[i].bytesToGPU);
      }
      if (mTimersRecoSteps[i].bytesToHost) {
        writer.row('D', mTimersRecoSteps[i].countToHost, std::string(gpudatatypes::RECO_STEP_NAMES[i]) + " (DMA to Host)", mTimersRecoSteps[i].timerToHost.GetElapsedTime(), -1.0, -1.0, mTimersRecoSteps[i].bytesToHost);
      }
      if (GetProcessingSettings().resetTimers) {
        mTimersRecoSteps[i].bytesToGPU = mTimersRecoSteps[i].bytesToHost = 0;
        mTimersRecoSteps[i].timerToGPU.Reset();
        mTimersRecoSteps[i].timerToHost.Reset();
        mTimersRecoSteps[i].timerTotal.Reset();
        mTimersRecoSteps[i].timerCPU = 0;
        mTimersRecoSteps[i].countToGPU = 0;
        mTimersRecoSteps[i].countToHost = 0;
      }
    }
    for (int32_t i = 0; i < gpudatatypes::N_GENERAL_STEPS; i++) {
      if (mTimersGeneralSteps[i].GetElapsedTime() != 0.) {
        writer.row(' ', 0, gpudatatypes::GENERAL_STEP_NAMES[i], mTimersGeneralSteps[i].GetElapsedTime(), -1.0, -1.0, 0);
      }
    }
    double gpu_time = GetProcessingSettings().debugLevel >= 1 ? kernelTotal : -1.0;
    writer.row(' ', 0, "Wall", gpu_time, mStatCPUTime, mTimerTotal.GetElapsedTime(), 0, nEventReport);
  } else if (GetProcessingSettings().debugLevel >= 0) {
    GPUInfo("Total Wall Time: %10.0f us%s", mStatWallTime, nEventReport.c_str());
  }
  if (GetProcessingSettings().resetTimers) {
    mStatNEvents = 0;
    mStatCPUTime = 0;
    mTimerTotal.Reset();
  }

  if (GetProcessingSettings().memoryStat) {
    PrintMemoryStatistics();
  } else if (GetProcessingSettings().debugLevel >= 2) {
    PrintMemoryOverview();
  }

  return retVal;
}

void GPUReconstructionCPU::ResetDeviceProcessorTypes()
{
  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && mProcessors[i].proc->mLinkedProcessor) {
      mProcessors[i].proc->mLinkedProcessor->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
    }
  }
}

void GPUReconstructionCPU::UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, uint32_t mapSize, int32_t stream, deviceEvent* ev)
{
  if (mapHost && mapSize != GPUTPCClusterOccupancyMapBin::getNBins(param())) {
    throw std::runtime_error("Updating occupancy map with object of invalid size");
  }
  param().occupancyMap = mapHost;
  param().occupancyMapSize = mapSize;
  param().occupancyTotal = occupancyTotal;
  if (IsGPU()) {
    if (!((size_t)&param().occupancyMapSize - (size_t)&param().occupancyMap == sizeof(param().occupancyMap) + sizeof(param().occupancyTotal) && sizeof(param().occupancyMap) == sizeof(void*) && sizeof(param().occupancyTotal) == sizeof(uint32_t))) { // TODO: Make static assert, and check alignment
      throw std::runtime_error("occupancy data not consecutive in GPUParam");
    }
    struct tmpOccuapncyParam {
      const void* ptr;
      uint32_t total;
      uint32_t size;
    };
    tmpOccuapncyParam tmp = {mapGPU, occupancyTotal, mapSize};
    const auto holdContext = GetThreadContext();
    WriteToConstantMemory((char*)&processors()->param.occupancyMap - (char*)processors(), &tmp, sizeof(tmp), stream, ev);
  }
}
