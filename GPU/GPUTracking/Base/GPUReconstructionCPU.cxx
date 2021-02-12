// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCPU.cxx
/// \author David Rohr

#define GPUCA_GPURECONSTRUCTIONCPU_IMPLEMENTATION
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
#include <atomic>

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#if defined(WITH_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

using namespace GPUCA_NAMESPACE::gpu;

constexpr GPUReconstructionCPU::krnlRunRange GPUReconstructionCPU::krnlRunRangeNone;
constexpr GPUReconstructionCPU::krnlEvent GPUReconstructionCPU::krnlEventNone;

GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCPU(cfg); }

GPUReconstructionCPU::~GPUReconstructionCPU()
{
  Exit(); // Needs to be identical to GPU backend bahavior in order to avoid calling abstract methods later in the destructor
}

template <class T, int I, typename... Args>
int GPUReconstructionCPUBackend::runKernelBackend(krnlSetup& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  if (x.device == krnlDeviceType::Device) {
    throw std::runtime_error("Cannot run device kernel on host");
  }
  if (x.nThreads != 1) {
    throw std::runtime_error("Cannot run device kernel on host with nThreads != 1");
  }
  unsigned int num = y.num == 0 || y.num == -1 ? 1 : y.num;
  for (unsigned int k = 0; k < num; k++) {
    int ompThreads = mProcessingSettings.ompKernels ? (mProcessingSettings.ompKernels == 2 ? ((mProcessingSettings.ompThreads + mNestedLoopOmpFactor - 1) / mNestedLoopOmpFactor) : mProcessingSettings.ompThreads) : 1;
    if (ompThreads > 1) {
      if (mProcessingSettings.debugLevel >= 5) {
        printf("Running %d ompThreads\n", ompThreads);
      }
      GPUCA_OPENMP(parallel for num_threads(ompThreads))
      for (unsigned int iB = 0; iB < x.nBlocks; iB++) {
        typename T::GPUSharedMemory smem;
        T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.start + k], args...);
      }
    } else {
      for (unsigned int iB = 0; iB < x.nBlocks; iB++) {
        typename T::GPUSharedMemory smem;
        T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.start + k], args...);
      }
    }
  }
  return 0;
}

template <>
int GPUReconstructionCPUBackend::runKernelBackend<GPUMemClean16, 0>(krnlSetup& _xyz, void* const& ptr, unsigned long const& size)
{
  memset(ptr, 0, size);
  return 0;
}

template <class T, int I>
GPUReconstruction::krnlProperties GPUReconstructionCPUBackend::getKernelPropertiesBackend()
{
  return krnlProperties{1, 1};
}

size_t GPUReconstructionCPU::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  memcpy(dst, src, size);
  return 0;
}
size_t GPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) { return 0; }
int GPUReconstructionCPU::GPUDebug(const char* state, int stream) { return 0; }
size_t GPUReconstructionCPU::TransferMemoryResourcesHelper(GPUProcessor* proc, int stream, bool all, bool toGPU)
{
  int inc = toGPU ? GPUMemoryResource::MEMORY_INPUT_FLAG : GPUMemoryResource::MEMORY_OUTPUT_FLAG;
  int exc = toGPU ? GPUMemoryResource::MEMORY_OUTPUT_FLAG : GPUMemoryResource::MEMORY_INPUT_FLAG;
  size_t n = 0;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
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

int GPUReconstructionCPU::GetThread()
{
// Get Thread ID
#if defined(__APPLE__)
  return (0); // syscall is deprecated on MacOS..., only needed for GPU support which we don't do on Mac anyway
#elif defined(_WIN32)
  return ((int)(size_t)GetCurrentThread());
#else
  return ((int)syscall(SYS_gettid));
#endif
}

int GPUReconstructionCPU::InitDevice()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mDeviceMemorySize > mHostMemorySize) {
      mHostMemorySize = mDeviceMemorySize;
    }
    if (mMaster == nullptr) {
      mHostMemoryBase = operator new(mHostMemorySize);
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

int GPUReconstructionCPU::ExitDevice()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      operator delete(mHostMemoryBase);
    }
    mHostMemoryPool = mHostMemoryBase = mHostMemoryPoolEnd = mHostMemoryPermanent = nullptr;
    mHostMemorySize = 0;
  }
  return 0;
}

int GPUReconstructionCPU::RunChains()
{
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
    for (unsigned int i = 0; i < mChains.size(); i++) {
      int retVal = mChains[i]->RunChain();
      if (retVal) {
        return retVal;
      }
    }
  }
  timerTotal.Stop();

  mStatWallTime = (timerTotal.GetElapsedTime() * 1000000. / mStatNEvents);
  if (GetProcessingSettings().debugLevel >= 1) {
    double kernelTotal = 0;
    std::vector<double> kernelStepTimes(GPUDataTypes::N_RECO_STEPS);

    for (unsigned int i = 0; i < mTimers.size(); i++) {
      double time = 0;
      if (mTimers[i] == nullptr) {
        continue;
      }
      for (int j = 0; j < mTimers[i]->num; j++) {
        HighResTimer& timer = mTimers[i]->timer[j];
        time += timer.GetElapsedTime();
        if (mProcessingSettings.resetTimers) {
          timer.Reset();
        }
      }

      char type = mTimers[i]->type;
      if (type == 0) {
        kernelTotal += time;
        int stepNum = getRecoStepNum(mTimers[i]->step);
        kernelStepTimes[stepNum] += time;
      }
      type = type == 0 ? 'K' : 'C';
      char bandwidth[256] = "";
      if (mTimers[i]->memSize && mStatNEvents && time != 0.) {
        snprintf(bandwidth, 256, " (%6.3f GB/s - %'14lu bytes)", mTimers[i]->memSize / time * 1e-9, (unsigned long)(mTimers[i]->memSize / mStatNEvents));
      }
      printf("Execution Time: Task (%c %8ux): %50s Time: %'10d us%s\n", type, mTimers[i]->count, mTimers[i]->name.c_str(), (int)(time * 1000000 / mStatNEvents), bandwidth);
      if (mProcessingSettings.resetTimers) {
        mTimers[i]->count = 0;
        mTimers[i]->memSize = 0;
      }
    }
    for (int i = 0; i < GPUDataTypes::N_RECO_STEPS; i++) {
      if (kernelStepTimes[i] != 0. || mTimersRecoSteps[i].timerTotal.GetElapsedTime() != 0.) {
        printf("Execution Time: Step              : %11s %38s Time: %'10d us ( Total Time : %'14d us)\n", "Tasks", GPUDataTypes::RECO_STEP_NAMES[i], (int)(kernelStepTimes[i] * 1000000 / mStatNEvents), (int)(mTimersRecoSteps[i].timerTotal.GetElapsedTime() * 1000000 / mStatNEvents));
      }
      if (mTimersRecoSteps[i].bytesToGPU) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10d us (%6.3f GB/s - %'14lu bytes - %'14lu per call)\n", mTimersRecoSteps[i].countToGPU, "DMA to GPU", GPUDataTypes::RECO_STEP_NAMES[i], (int)(mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1000000 / mStatNEvents),
               mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1e-9, mTimersRecoSteps[i].bytesToGPU / mStatNEvents, mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].countToGPU);
      }
      if (mTimersRecoSteps[i].bytesToHost) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10d us (%6.3f GB/s - %'14lu bytes - %'14lu per call)\n", mTimersRecoSteps[i].countToHost, "DMA to Host", GPUDataTypes::RECO_STEP_NAMES[i], (int)(mTimersRecoSteps[i].timerToHost.GetElapsedTime() * 1000000 / mStatNEvents),
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
    for (int i = 0; i < GPUDataTypes::N_GENERAL_STEPS; i++) {
      if (mTimersGeneralSteps[i].GetElapsedTime() != 0.) {
        printf("Execution Time: General Step      : %50s Time: %'10d us\n", GPUDataTypes::GENERAL_STEP_NAMES[i], (int)(mTimersGeneralSteps[i].GetElapsedTime() * 1000000 / mStatNEvents));
      }
    }
    mStatKernelTime = kernelTotal * 1000000 / mStatNEvents;
    printf("Execution Time: Total   : %50s Time: %'10d us\n", "Total Kernel", (int)mStatKernelTime);
    printf("Execution Time: Total   : %50s Time: %'10d us\n", "Total Wall", (int)mStatWallTime);
  } else if (GetProcessingSettings().debugLevel >= 0) {
    GPUInfo("Total Wall Time: %d us", (int)mStatWallTime);
  }
  if (mProcessingSettings.resetTimers) {
    mStatNEvents = 0;
    timerTotal.Reset();
  }

  return 0;
}

void GPUReconstructionCPU::ResetDeviceProcessorTypes()
{
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && mProcessors[i].proc->mLinkedProcessor) {
      mProcessors[i].proc->mLinkedProcessor->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
    }
  }
}

int GPUReconstructionCPU::getOMPThreadNum()
{
  return omp_get_thread_num();
}

int GPUReconstructionCPU::getOMPMaxThreads()
{
  return omp_get_max_threads();
}

static std::atomic_flag timerFlag = ATOMIC_FLAG_INIT; // TODO: Should be a class member not global, but cannot be moved to header due to ROOT limitation

GPUReconstructionCPU::timerMeta* GPUReconstructionCPU::insertTimer(unsigned int id, std::string&& name, int J, int num, int type, RecoStep step)
{
  while (timerFlag.test_and_set()) {
    ;
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

GPUReconstructionCPU::timerMeta* GPUReconstructionCPU::getTimerById(unsigned int id)
{
  timerMeta* retVal = nullptr;
  while (timerFlag.test_and_set()) {
    ;
  }
  if (mTimers.size() > id && mTimers[id]) {
    retVal = mTimers[id].get();
    retVal->count++;
  }
  timerFlag.clear();
  return retVal;
}

unsigned int GPUReconstructionCPU::getNextTimerId()
{
  static std::atomic<unsigned int> id{0};
  return id.fetch_add(1);
}

unsigned int GPUReconstructionCPU::SetAndGetNestedLoopOmpFactor(bool condition, unsigned int max)
{
  if (condition && mProcessingSettings.ompKernels != 1) {
    mNestedLoopOmpFactor = mProcessingSettings.ompKernels == 2 ? std::min<unsigned int>(max, mProcessingSettings.ompThreads) : mProcessingSettings.ompThreads;
  } else {
    mNestedLoopOmpFactor = 1;
  }
  if (mProcessingSettings.debugLevel >= 5) {
    printf("Running %d OMP threads in outer loop\n", mNestedLoopOmpFactor);
  }
  return mNestedLoopOmpFactor;
}
