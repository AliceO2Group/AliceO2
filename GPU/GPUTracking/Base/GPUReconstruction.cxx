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

/// \file GPUReconstruction.cxx
/// \author David Rohr

#include <cstring>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <array>

#include "GPUReconstruction.h"
#include "GPUReconstructionIncludes.h"
#include "GPUReconstructionThreading.h"
#include "GPUReconstructionIO.h"
#include "GPUROOTDumpCore.h"
#include "GPUConfigDump.h"
#include "GPUChainTracking.h"
#include "GPUConstantMem.h"
#include "GPUCommonHelpers.h"
#include "GPUSettings.h"

#include "GPUMemoryResource.h"
#include "GPUChain.h"
#include "GPUMemorySizeScalers.h"

#include "GPULogging.h"
#include "utils/strtag.h"
#include "utils/stdspinlock.h"

#ifdef GPUCA_O2_LIB
#include "GPUO2InterfaceConfiguration.h"
#endif

#include "GPUReconstructionIncludesITS.h"

namespace o2::gpu
{
namespace // anonymous
{
struct GPUReconstructionPipelineQueue {
  uint32_t op = 0; // For now, 0 = process, 1 = terminate
  GPUChain* chain = nullptr;
  std::mutex m;
  std::condition_variable c;
  bool done = false;
  int32_t retVal = 0;
};
} // namespace

struct GPUReconstructionPipelineContext {
  std::queue<GPUReconstructionPipelineQueue*> pipelineQueue;
  std::mutex mutex;
  std::condition_variable cond;
  bool terminate = false;
};
} // namespace o2::gpu

using namespace o2::gpu;

constexpr const char* const GPUReconstruction::GEOMETRY_TYPE_NAMES[];
constexpr const char* const GPUReconstruction::IOTYPENAMES[];
constexpr GPUReconstruction::GeometryType GPUReconstruction::geometryType;

static ptrdiff_t ptrDiff(void* a, void* b) { return (char*)a - (char*)b; }

GPUReconstruction::GPUReconstruction(const GPUSettingsDeviceBackend& cfg) : mHostConstantMem(new GPUConstantMem), mGRPSettings(new GPUSettingsGRP), mDeviceBackendSettings(new GPUSettingsDeviceBackend(cfg)), mProcessingSettings(new GPUSettingsProcessing)
{
  if (cfg.master) {
    if (cfg.master->GetDeviceBackendSettings().deviceType != cfg.deviceType) {
      throw std::invalid_argument("device type of master and slave GPUReconstruction does not match");
    }
    if (cfg.master->mMaster) {
      throw std::invalid_argument("Cannot be slave to a slave");
    }
    mMaster = cfg.master;
    mSlaveId = cfg.master->mSlaves.size();
    cfg.master->mSlaves.emplace_back(this);
  }
  param().SetDefaults(mGRPSettings.get());
  mMemoryScalers.reset(new GPUMemorySizeScalers);
  for (uint32_t i = 0; i < NSECTORS; i++) {
    processors()->tpcTrackers[i].SetSector(i); // TODO: Move to a better place
    processors()->tpcClusterer[i].mISector = i;
#ifdef GPUCA_HAS_ONNX
    processors()->tpcNNClusterer[i].mISector = i;
#endif
  }
#ifndef GPUCA_NO_ROOT
  mROOTDump = GPUROOTDumpCore::getAndCreate();
#endif
}

GPUReconstruction::~GPUReconstruction()
{
  if (mInitialized) {
    GPUError("GPU Reconstruction not properly deinitialized!");
  }
}

void GPUReconstruction::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits<7>>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits<7>>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame<7>>* timeFrame)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraits<7>);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraits<7>);
  }
  if (timeFrame) {
    timeFrame->reset(new o2::its::TimeFrame<7>);
  }
}

int32_t GPUReconstruction::getHostThreadIndex()
{
  return std::max<int32_t>(0, tbb::this_task_arena::current_thread_index());
}

int32_t GPUReconstruction::Init()
{
  if (mMaster) {
    throw std::runtime_error("Must not call init on slave!");
  }
  int32_t retVal = InitPhaseBeforeDevice();
  if (retVal) {
    return retVal;
  }
  for (uint32_t i = 0; i < mSlaves.size(); i++) {
    retVal = mSlaves[i]->InitPhaseBeforeDevice();
    if (retVal) {
      GPUError("Error initialization slave (before deviceinit)");
      return retVal;
    }
    mNStreams = std::max(mNStreams, mSlaves[i]->mNStreams);
    mHostMemorySize = std::max(mHostMemorySize, mSlaves[i]->mHostMemorySize);
    mDeviceMemorySize = std::max(mDeviceMemorySize, mSlaves[i]->mDeviceMemorySize);
  }
  if (InitDevice()) {
    return 1;
  }
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    mHostMemoryPoolEnd = (char*)mHostMemoryBase + mHostMemorySize;
    mDeviceMemoryPoolEnd = (char*)mDeviceMemoryBase + mDeviceMemorySize;
  } else {
    mHostMemoryPoolEnd = mDeviceMemoryPoolEnd = nullptr;
  }
  if (InitPhasePermanentMemory()) {
    return 1;
  }
  for (uint32_t i = 0; i < mSlaves.size(); i++) {
    mSlaves[i]->mDeviceMemoryBase = mDeviceMemoryPermanent;
    mSlaves[i]->mHostMemoryBase = mHostMemoryPermanent;
    mSlaves[i]->mDeviceMemorySize = mDeviceMemorySize - ptrDiff(mSlaves[i]->mDeviceMemoryBase, mDeviceMemoryBase);
    mSlaves[i]->mHostMemorySize = mHostMemorySize - ptrDiff(mSlaves[i]->mHostMemoryBase, mHostMemoryBase);
    mSlaves[i]->mHostMemoryPoolEnd = mHostMemoryPoolEnd;
    mSlaves[i]->mDeviceMemoryPoolEnd = mDeviceMemoryPoolEnd;
    if (mSlaves[i]->InitDevice()) {
      GPUError("Error initialization slave (deviceinit)");
      return 1;
    }
    if (mSlaves[i]->InitPhasePermanentMemory()) {
      GPUError("Error initialization slave (permanent memory)");
      return 1;
    }
    mDeviceMemoryPermanent = mSlaves[i]->mDeviceMemoryPermanent;
    mHostMemoryPermanent = mSlaves[i]->mHostMemoryPermanent;
  }
  retVal = InitPhaseAfterDevice();
  if (retVal) {
    return retVal;
  }
  ClearAllocatedMemory();
  for (uint32_t i = 0; i < mSlaves.size(); i++) {
    mSlaves[i]->mDeviceMemoryPermanent = mDeviceMemoryPermanent;
    mSlaves[i]->mHostMemoryPermanent = mHostMemoryPermanent;
    retVal = mSlaves[i]->InitPhaseAfterDevice();
    if (retVal) {
      GPUError("Error initialization slave (after device init)");
      return retVal;
    }
    mSlaves[i]->ClearAllocatedMemory();
  }
  debugInit();
  return 0;
}

namespace o2::gpu::internal
{
static uint32_t getDefaultNThreads()
{
  const char* tbbEnv = getenv("TBB_NUM_THREADS");
  uint32_t tbbNum = tbbEnv ? atoi(tbbEnv) : 0;
  if (tbbNum) {
    return tbbNum;
  }
  const char* ompEnv = getenv("OMP_NUM_THREADS");
  uint32_t ompNum = ompEnv ? atoi(ompEnv) : 0;
  if (ompNum) {
    return ompNum;
  }
  return tbb::info::default_concurrency();
}
} // namespace o2::gpu::internal

int32_t GPUReconstruction::InitPhaseBeforeDevice()
{
  if (GetProcessingSettings().printSettings) {
    if (mSlaves.size() || mMaster) {
      printf("\nConfig Dump %s\n", mMaster ? "Slave" : "Master");
    }
    const GPUChainTracking* chTrk;
    for (uint32_t i = 0; i < mChains.size(); i++) {
      if ((chTrk = dynamic_cast<GPUChainTracking*>(mChains[i].get()))) {
        break;
      }
    }
    GPUConfigDump::dumpConfig(&param().rec, mProcessingSettings.get(), chTrk ? chTrk->GetQAConfig() : nullptr, chTrk ? chTrk->GetEventDisplayConfig() : nullptr, mDeviceBackendSettings.get(), &mRecoSteps);
  }
  mRecoSteps.stepsGPUMask &= mRecoSteps.steps;
  mRecoSteps.stepsGPUMask &= AvailableGPURecoSteps();
  if (!IsGPU()) {
    mRecoSteps.stepsGPUMask.set((uint8_t)0);
  }

  if (GetProcessingSettings().forceMemoryPoolSize >= 1024 || GetProcessingSettings().forceHostMemoryPoolSize >= 1024) {
    mProcessingSettings->memoryAllocationStrategy = GPUMemoryResource::ALLOCATION_GLOBAL;
  }
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_AUTO) {
    mProcessingSettings->memoryAllocationStrategy = IsGPU() ? GPUMemoryResource::ALLOCATION_GLOBAL : GPUMemoryResource::ALLOCATION_INDIVIDUAL;
  }
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mProcessingSettings->forceMemoryPoolSize = mProcessingSettings->forceHostMemoryPoolSize = 0;
  }
  if (GetProcessingSettings().debugLevel >= 4) {
    mProcessingSettings->keepAllMemory = true;
  }
  if (GetProcessingSettings().debugLevel >= 5 && GetProcessingSettings().allocDebugLevel < 2) {
    mProcessingSettings->allocDebugLevel = 2;
  }
  if (GetProcessingSettings().eventDisplay || GetProcessingSettings().keepAllMemory) {
    mProcessingSettings->keepDisplayMemory = true;
  }
  if (GetProcessingSettings().debugLevel < 6) {
    mProcessingSettings->debugMask = 0;
  }
  if (GetProcessingSettings().debugLevel < 1) {
    mProcessingSettings->deviceTimers = false;
  }
  if (GetProcessingSettings().debugLevel > 0) {
    mProcessingSettings->recoTaskTiming = true;
  }
  if (GetProcessingSettings().deterministicGPUReconstruction == -1) {
#ifdef GPUCA_DETERMINISTIC_MODE
    mProcessingSettings->deterministicGPUReconstruction = 1;
#else
    mProcessingSettings->deterministicGPUReconstruction = GetProcessingSettings().debugLevel >= 6;
#endif
  }
  if (GetProcessingSettings().deterministicGPUReconstruction) {
#ifndef GPUCA_DETERMINISTIC_MODE
    GPUError("WARNING, deterministicGPUReconstruction needs GPUCA_DETERMINISTIC_MODE for being fully deterministic, without only most indeterminism by concurrency is removed, but floating point effects remain!");
#endif
    if (mProcessingSettings->debugLevel >= 6 && ((mProcessingSettings->debugMask + 1) & mProcessingSettings->debugMask)) {
      GPUError("WARNING: debugMask %d - debug output might not be deterministic with intermediate steps missing", mProcessingSettings->debugMask);
    }
    mProcessingSettings->overrideClusterizerFragmentLen = TPC_MAX_FRAGMENT_LEN_GPU;
    if (GetProcessingSettings().createO2Output > 1) {
      mProcessingSettings->createO2Output = 1;
    }
    mProcessingSettings->rtc.deterministic = 1;
  } else {
#ifdef GPUCA_DETERMINISTIC_MODE
    GPUError("WARNING, compiled with GPUCA_DETERMINISTIC_MODE but deterministicGPUReconstruction not set, only compile-time determinism and deterministic math enforced, not fully deterministic!");
#endif
  }
  if (GetProcessingSettings().deterministicGPUReconstruction && GetProcessingSettings().debugLevel >= 6) {
    mProcessingSettings->nTPCClustererLanes = 1;
  }
  if (GetProcessingSettings().createO2Output > 1 && GetProcessingSettings().runQA && GetProcessingSettings().qcRunFraction == 100.f) {
    mProcessingSettings->createO2Output = 1;
  }
  if (!GetProcessingSettings().createO2Output || !IsGPU()) {
    mProcessingSettings->clearO2OutputFromGPU = false;
  }
  if (!(mRecoSteps.stepsGPUMask & gpudatatypes::RecoStep::TPCMerging)) {
    mProcessingSettings->mergerSortTracks = false;
  }
  if (GetProcessingSettings().debugLevel > 3 || !IsGPU() || GetProcessingSettings().deterministicGPUReconstruction) {
    mProcessingSettings->delayedOutput = false;
  }
  if (!GetProcessingSettings().rtc.enable) {
    mProcessingSettings->rtc.optConstexpr = false;
  }
  if (GetProcessingSettings().allSanityChecks) {
    mProcessingSettings->clusterizerZSSanityCheck = mProcessingSettings->mergerSanityCheck = mProcessingSettings->outputSanityCheck = true;
  }

  mMemoryScalers->scalingFactor = GetProcessingSettings().memoryScalingFactor;
  mMemoryScalers->conservative = GetProcessingSettings().conservativeMemoryEstimate;
  mMemoryScalers->returnMaxVal = GetProcessingSettings().forceMaxMemScalers != 0;
  if (GetProcessingSettings().forceMaxMemScalers > 1) {
    mMemoryScalers->rescaleMaxMem(GetProcessingSettings().forceMaxMemScalers);
  }

  if (GetProcessingSettings().nHostThreads != -1 && GetProcessingSettings().ompThreads != -1) {
    GPUFatal("Must not use both nHostThreads and ompThreads at the same time!");
  } else if (GetProcessingSettings().ompThreads != -1) {
    mProcessingSettings->nHostThreads = GetProcessingSettings().ompThreads;
    GPUWarning("You are using the deprecated ompThreads option, please switch to nHostThreads!");
  }

  if (GetProcessingSettings().nHostThreads <= 0) {
    mProcessingSettings->nHostThreads = internal::getDefaultNThreads();
  } else {
    mProcessingSettings->autoAdjustHostThreads = false;
  }
  mMaxHostThreads = GetProcessingSettings().nHostThreads;
  if (mMaster == nullptr) {
    mThreading = std::make_shared<GPUReconstructionThreading>();
    mThreading->control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, mMaxHostThreads);
    mThreading->allThreads = std::make_unique<tbb::task_arena>(mMaxHostThreads);
    mThreading->activeThreads = std::make_unique<tbb::task_arena>(mMaxHostThreads);
  } else {
    mThreading = mMaster->mThreading;
  }
  mMaxBackendThreads = std::max(mMaxBackendThreads, mMaxHostThreads);
  if (IsGPU()) {
    mNStreams = std::max<int32_t>(GetProcessingSettings().nStreams, 3);
  }

  if (GetProcessingSettings().nTPCClustererLanes == -1) {
    mProcessingSettings->nTPCClustererLanes = (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) ? 3 : std::max<int32_t>(1, std::min<int32_t>(GPUCA_NSECTORS, GetProcessingSettings().inKernelParallel ? (mMaxHostThreads >= 4 ? std::min<int32_t>(mMaxHostThreads / 2, mMaxHostThreads >= 32 ? GPUCA_NSECTORS : 4) : 1) : mMaxHostThreads));
  }
  if (GetProcessingSettings().overrideClusterizerFragmentLen == -1) {
    mProcessingSettings->overrideClusterizerFragmentLen = ((GetRecoStepsGPU() & RecoStep::TPCClusterFinding) || (mMaxHostThreads / GetProcessingSettings().nTPCClustererLanes >= 3)) ? TPC_MAX_FRAGMENT_LEN_GPU : TPC_MAX_FRAGMENT_LEN_HOST;
  }
  if (GetProcessingSettings().nTPCClustererLanes > GPUCA_NSECTORS) {
    GPUError("Invalid value for nTPCClustererLanes: %d", GetProcessingSettings().nTPCClustererLanes);
    mProcessingSettings->nTPCClustererLanes = GPUCA_NSECTORS;
  }

  if (GetProcessingSettings().doublePipeline) {
    mProcessingSettings->rtctech.allowOptimizedSlaveReconstruction = true;
  }
  if (GetProcessingSettings().doublePipeline && (mChains.size() != 1 || mChains[0]->SupportsDoublePipeline() == false || !IsGPU() || GetProcessingSettings().memoryAllocationStrategy != GPUMemoryResource::ALLOCATION_GLOBAL)) {
    GPUError("Must use double pipeline mode only with exactly one chain that must support it");
    return 1;
  }
  if (mMaster == nullptr && GetProcessingSettings().doublePipeline) {
    mPipelineContext.reset(new GPUReconstructionPipelineContext);
  }

  if (mMaster && GetProcessingSettings().rtc.enable && (GetProcessingSettings().rtc.optConstexpr || GetProcessingSettings().rtc.optSpecialCode) && !GetProcessingSettings().rtctech.allowOptimizedSlaveReconstruction) {
    GPUError("Not allowed to create optimized RTC code with more than one GPUReconstruction instances");
    return 1;
  }

  mDeviceMemorySize = mHostMemorySize = 0;
  for (uint32_t i = 0; i < mChains.size(); i++) {
    if (mChains[i]->EarlyConfigure()) {
      return 1;
    }
    mChains[i]->RegisterPermanentMemoryAndProcessors();
    size_t memPrimary, memPageLocked;
    mChains[i]->MemorySize(memPrimary, memPageLocked);
    if (!IsGPU() || mOutputControl.useInternal()) {
      memPageLocked = memPrimary;
    }
    mDeviceMemorySize += memPrimary;
    mHostMemorySize += memPageLocked;
  }
  if (GetProcessingSettings().forceMemoryPoolSize && GetProcessingSettings().forceMemoryPoolSize <= 2 && CanQueryMaxMemory()) {
    mDeviceMemorySize = GetProcessingSettings().forceMemoryPoolSize;
  } else if (GetProcessingSettings().forceMemoryPoolSize > 2) {
    mDeviceMemorySize = GetProcessingSettings().forceMemoryPoolSize;
    if (!IsGPU() || mOutputControl.useInternal()) {
      mHostMemorySize = mDeviceMemorySize;
    }
  }
  if (GetProcessingSettings().forceHostMemoryPoolSize) {
    mHostMemorySize = GetProcessingSettings().forceHostMemoryPoolSize;
  }

  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].RegisterMemoryAllocation))();
  }

  return 0;
}

int32_t GPUReconstruction::InitPhasePermanentMemory()
{
  if (IsGPU()) {
    for (uint32_t i = 0; i < mChains.size(); i++) {
      mChains[i]->RegisterGPUProcessors();
    }
  }
  AllocateRegisteredPermanentMemory();
  return 0;
}

int32_t GPUReconstruction::InitPhaseAfterDevice()
{
  if (GetProcessingSettings().forceMaxMemScalers <= 1 && GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    mMemoryScalers->rescaleMaxMem(IsGPU() ? mDeviceMemorySize : mHostMemorySize);
  }
  for (uint32_t i = 0; i < mChains.size(); i++) {
    if (mChains[i]->Init()) {
      return 1;
    }
  }
  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].InitializeProcessor))();
  }

  WriteConstantParams(); // Initialize with initial values, can optionally be updated later

  mInitialized = true;
  return 0;
}

void GPUReconstruction::WriteConstantParams()
{
  if (IsGPU()) {
    const auto threadContext = GetThreadContext();
    WriteToConstantMemory(ptrDiff(&processors()->param, processors()), &param(), sizeof(param()), -1);
  }
}

int32_t GPUReconstruction::Finalize()
{
  for (uint32_t i = 0; i < mChains.size(); i++) {
    mChains[i]->Finalize();
  }
  return 0;
}

int32_t GPUReconstruction::Exit()
{
  if (!mInitialized) {
    return 1;
  }
  for (uint32_t i = 0; i < mSlaves.size(); i++) {
    if (mSlaves[i]->Exit()) {
      GPUError("Error exiting slave");
    }
  }

  mChains.clear();          // Make sure we destroy a possible ITS GPU tracker before we call the destructors
  mHostConstantMem.reset(); // Reset these explicitly before the destruction of other members unloads the library
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
      if (mMemoryResources[i].mReuse >= 0) {
        continue;
      }
      operator delete(mMemoryResources[i].mPtrDevice, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
      mMemoryResources[i].mPtr = mMemoryResources[i].mPtrDevice = nullptr;
    }
  }
  mMemoryResources.clear();
  if (mInitialized) {
    ExitDevice();
  }
  debugExit();
  mInitialized = false;
  return 0;
}

void GPUReconstruction::RegisterGPUDeviceProcessor(GPUProcessor* proc, GPUProcessor* slaveProcessor) { proc->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE, slaveProcessor); }
void GPUReconstruction::ConstructGPUProcessor(GPUProcessor* proc) { proc->mConstantMem = proc->mGPUProcessorType == GPUProcessor::PROCESSOR_TYPE_DEVICE ? mDeviceConstantMem : mHostConstantMem.get(); }

void GPUReconstruction::ComputeReuseMax(GPUProcessor* proc)
{
  for (auto it = mMemoryReuse1to1.begin(); it != mMemoryReuse1to1.end(); it++) {
    auto& re = it->second;
    if (proc == nullptr || re.proc == proc) {
      GPUMemoryResource& resMain = mMemoryResources[re.res[0]];
      resMain.mOverrideSize = 0;
      for (uint32_t i = 0; i < re.res.size(); i++) {
        GPUMemoryResource& res = mMemoryResources[re.res[i]];
        resMain.mOverrideSize = std::max<size_t>(resMain.mOverrideSize, ptrDiff(res.SetPointers((void*)1), (char*)1));
      }
    }
  }
}

int16_t GPUReconstruction::RegisterMemoryAllocationHelper(GPUProcessor* proc, void* (GPUProcessor::*setPtr)(void*), int32_t type, const char* name, const GPUMemoryReuse& re)
{
  if (!(type & (GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU))) {
    if ((type & GPUMemoryResource::MEMORY_SCRATCH) && !GetProcessingSettings().keepDisplayMemory) { // keepAllMemory --> keepDisplayMemory
      type |= (proc->mGPUProcessorType == GPUProcessor::PROCESSOR_TYPE_CPU ? GPUMemoryResource::MEMORY_HOST : GPUMemoryResource::MEMORY_GPU);
    } else {
      type |= GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU;
    }
  }
  if (proc->mGPUProcessorType == GPUProcessor::PROCESSOR_TYPE_CPU) {
    type &= ~GPUMemoryResource::MEMORY_GPU;
  }
  mMemoryResources.emplace_back(proc, setPtr, (GPUMemoryResource::MemoryType)type, name);
  if (mMemoryResources.size() >= 32768) {
    throw std::bad_alloc();
  }
  uint16_t retVal = mMemoryResources.size() - 1;
  if (re.type != GPUMemoryReuse::NONE && !GetProcessingSettings().disableMemoryReuse) {
    const auto& it = mMemoryReuse1to1.find(re.id);
    if (it == mMemoryReuse1to1.end()) {
      mMemoryReuse1to1[re.id] = {proc, retVal};
    } else {
      mMemoryResources[retVal].mReuse = it->second.res[0];
      it->second.res.emplace_back(retVal);
    }
  }
  return retVal;
}

size_t GPUReconstruction::AllocateRegisteredMemory(GPUProcessor* proc, bool resetCustom)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Allocating memory %p", (void*)proc);
  }
  size_t total = 0;
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    if (proc == nullptr ? !mMemoryResources[i].mProcessor->mAllocateAndInitializeLate : mMemoryResources[i].mProcessor == proc) {
      if (!(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) {
        total += AllocateRegisteredMemory(i);
      } else if (resetCustom && (mMemoryResources[i].mPtr || mMemoryResources[i].mPtrDevice)) {
        ResetRegisteredMemoryPointers(i);
      }
    }
  }
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Allocating memory done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredPermanentMemory()
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Allocating Permanent Memory");
  }
  if (mVolatileMemoryStart) {
    GPUError("Must not allocate permanent memory while volatile chunks are allocated");
    throw std::bad_alloc();
  }
  int32_t total = 0;
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    if ((mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && mMemoryResources[i].mPtr == nullptr) {
      total += AllocateRegisteredMemory(i);
    }
  }
  mHostMemoryPermanent = mHostMemoryPool;
  mDeviceMemoryPermanent = mDeviceMemoryPool;
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Permanent Memory Done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredMemoryHelper(GPUMemoryResource* res, void*& ptr, void*& memorypool, void* memorybase, size_t memorysize, void* (GPUMemoryResource::*setPtr)(void*) const, void*& memorypoolend, const char* device)
{
  if (res->mReuse >= 0) {
    ptr = (&ptr == &res->mPtrDevice) ? mMemoryResources[res->mReuse].mPtrDevice : mMemoryResources[res->mReuse].mPtr;
    if (ptr == nullptr) {
      GPUError("Invalid reuse ptr (%s)", res->mName);
      throw std::bad_alloc();
    }
    size_t retVal = ptrDiff((res->*setPtr)(ptr), ptr);
    if (retVal > mMemoryResources[res->mReuse].mSize) {
      GPUError("Insufficient reuse memory %lu < %lu (%s) (%s)", mMemoryResources[res->mReuse].mSize, retVal, res->mName, device);
      throw std::bad_alloc();
    }
    if (GetProcessingSettings().allocDebugLevel >= 2) {
      std::cout << "Reused (" << device << ") " << res->mName << ": " << retVal << "\n";
    }
    return retVal;
  }
  if (memorypool == nullptr) {
    GPUError("Cannot allocate memory from uninitialized pool");
    throw std::bad_alloc();
  }
  size_t retVal;
  stdspinlock spinlock(mMemoryMutex);
  if ((res->mType & GPUMemoryResource::MEMORY_STACK) && memorypoolend) {
    retVal = ptrDiff((res->*setPtr)((char*)1), (char*)(1));
    memorypoolend = (void*)((char*)memorypoolend - GPUProcessor::getAlignmentMod<GPUCA_MEMALIGN>(memorypoolend));
    if (retVal < res->mOverrideSize) {
      retVal = res->mOverrideSize;
    }
    retVal += GPUProcessor::getAlignment<GPUCA_MEMALIGN>(retVal);
    memorypoolend = (char*)memorypoolend - retVal;
    ptr = memorypoolend;
    retVal = std::max<size_t>(ptrDiff((res->*setPtr)(ptr), ptr), res->mOverrideSize);
  } else {
    ptr = memorypool;
    memorypool = (char*)((res->*setPtr)(ptr));
    retVal = ptrDiff(memorypool, ptr);
    if (retVal < res->mOverrideSize) {
      retVal = res->mOverrideSize;
      memorypool = (char*)ptr + res->mOverrideSize;
    }
    memorypool = (void*)((char*)memorypool + GPUProcessor::getAlignment<GPUCA_MEMALIGN>(memorypool));
  }
  if (memorypoolend ? (memorypool > memorypoolend) : ((size_t)ptrDiff(memorypool, memorybase) > memorysize)) {
    std::cerr << "Memory pool size exceeded (" << device << ") (" << res->mName << ": " << (memorypoolend ? (memorysize + ptrDiff(memorypool, memorypoolend)) : ptrDiff(memorypool, memorybase)) << " > " << memorysize << "\n";
    throw std::bad_alloc();
  }
  if (GetProcessingSettings().allocDebugLevel >= 2) {
    std::cout << "Allocated (" << device << ") " << res->mName << ": " << retVal << " - available: " << (memorypoolend ? ptrDiff(memorypoolend, memorypool) : (memorysize - ptrDiff(memorypool, memorybase))) << "\n";
  }
  return retVal;
}

void GPUReconstruction::AllocateRegisteredMemoryInternal(GPUMemoryResource* res, GPUOutputControl* control, GPUReconstruction* recPool)
{
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL && (control == nullptr || control->useInternal())) {
    if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
      if (res->mPtrDevice && res->mReuse < 0) {
        operator delete(res->mPtrDevice, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
      }
      res->mSize = std::max((size_t)res->SetPointers((void*)1) - 1, res->mOverrideSize);
      if (res->mReuse >= 0) {
        if (res->mSize > mMemoryResources[res->mReuse].mSize) {
          GPUError("Invalid reuse, insufficient size: %ld < %ld", (int64_t)mMemoryResources[res->mReuse].mSize, (int64_t)res->mSize);
          throw std::bad_alloc();
        }
        res->mPtrDevice = mMemoryResources[res->mReuse].mPtrDevice;
      } else {
        res->mPtrDevice = operator new(res->mSize + GPUCA_BUFFER_ALIGNMENT, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
      }
      res->mPtr = GPUProcessor::alignPointer<GPUCA_BUFFER_ALIGNMENT>(res->mPtrDevice);
      res->SetPointers(res->mPtr);
      if (GetProcessingSettings().allocDebugLevel >= 2) {
        std::cout << (res->mReuse >= 0 ? "Reused " : "Allocated ") << res->mName << ": " << res->mSize << " (individual" << ((res->mType & GPUMemoryResource::MEMORY_STACK) ? " stack" : "") << ")\n";
      }
      if (res->mType & GPUMemoryResource::MEMORY_STACK) {
        stdspinlock spinlock(mMemoryMutex);
        mNonPersistentIndividualAllocations.emplace_back(res);
      }
      if ((size_t)res->mPtr % GPUCA_BUFFER_ALIGNMENT) {
        GPUError("Got buffer with insufficient alignment");
        throw std::bad_alloc();
      }
    }
  } else {
    if (res->mPtr != nullptr) {
      GPUError("Double allocation! (%s)", res->mName);
      throw std::bad_alloc();
    }
    if (IsGPU() && res->mOverrideSize < GPUCA_BUFFER_ALIGNMENT) {
      res->mOverrideSize = GPUCA_BUFFER_ALIGNMENT;
    }
    if ((!IsGPU() || (res->mType & GPUMemoryResource::MEMORY_HOST) || GetProcessingSettings().keepDisplayMemory) && !(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) { // keepAllMemory --> keepDisplayMemory
      if (control && control->useExternal()) {
        if (control->allocator) {
          res->mSize = std::max((size_t)res->SetPointers((void*)1) - 1, res->mOverrideSize);
          res->mPtr = control->allocator(CAMath::nextMultipleOf<GPUCA_BUFFER_ALIGNMENT>(res->mSize));
          res->mSize = std::max<size_t>(ptrDiff(res->SetPointers(res->mPtr), res->mPtr), res->mOverrideSize);
          if (GetProcessingSettings().allocDebugLevel >= 2) {
            std::cout << "Allocated (from callback) " << res->mName << ": " << res->mSize << "\n";
          }
        } else {
          void* dummy = nullptr;
          res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, control->ptrCurrent, control->ptrBase, control->size, &GPUMemoryResource::SetPointers, dummy, "host");
        }
      } else {
        res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, recPool->mHostMemoryPool, recPool->mHostMemoryBase, recPool->mHostMemorySize, &GPUMemoryResource::SetPointers, recPool->mHostMemoryPoolEnd, "host");
      }
      if ((size_t)res->mPtr % GPUCA_BUFFER_ALIGNMENT) {
        GPUError("Got buffer with insufficient alignment");
        throw std::bad_alloc();
      }
    }
    if (IsGPU() && (res->mType & GPUMemoryResource::MEMORY_GPU)) {
      if (res->mProcessor->mLinkedProcessor == nullptr) {
        GPUError("Device Processor not set (%s)", res->mName);
        throw std::bad_alloc();
      }
      if (mVolatileMemoryStart && !mDeviceMemoryAsVolatile && !(res->mType & GPUMemoryResource::MEMORY_STACK)) {
        GPUError("Must not allocate non-stacked device memory while volatile chunks are allocated");
        throw std::bad_alloc();
      }
      size_t size = AllocateRegisteredMemoryHelper(res, res->mPtrDevice, recPool->mDeviceMemoryPool, recPool->mDeviceMemoryBase, recPool->mDeviceMemorySize, &GPUMemoryResource::SetDevicePointers, recPool->mDeviceMemoryPoolEnd, " gpu");

      if (!(res->mType & GPUMemoryResource::MEMORY_HOST) || (res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
        res->mSize = size;
      } else if (size != res->mSize) {
        GPUError("Inconsistent device memory allocation (%s: device %lu vs %lu)", res->mName, size, res->mSize);
        throw std::bad_alloc();
      }
      if ((size_t)res->mPtrDevice % GPUCA_BUFFER_ALIGNMENT) {
        GPUError("Got buffer with insufficient alignment");
        throw std::bad_alloc();
      }
    }
    UpdateMaxMemoryUsed();
  }
}

void GPUReconstruction::AllocateRegisteredForeignMemory(int16_t ires, GPUReconstruction* rec, GPUOutputControl* control)
{
  rec->AllocateRegisteredMemoryInternal(&rec->mMemoryResources[ires], control, this);
}

size_t GPUReconstruction::AllocateRegisteredMemory(int16_t ires, GPUOutputControl* control)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if ((res->mType & GPUMemoryResource::MEMORY_PERMANENT) && res->mPtr != nullptr) {
    ResetRegisteredMemoryPointers(ires);
  } else {
    AllocateRegisteredMemoryInternal(res, control, this);
  }
  return res->mReuse >= 0 ? 0 : res->mSize;
}

void* GPUReconstruction::AllocateDirectMemory(size_t size, int32_t type)
{
  stdspinlock spinlock(mMemoryMutex);
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    char* retVal = new (std::align_val_t(GPUCA_BUFFER_ALIGNMENT)) char[size];
    if ((type & GPUMemoryResource::MEMORY_STACK)) {
      mNonPersistentIndividualDirectAllocations.emplace_back(retVal, alignedDeleter());
    } else {
      mDirectMemoryChunks.emplace_back(retVal, alignedDeleter());
    }
    return retVal;
  }

  if ((type & ~(GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK)) || ((type & GPUMemoryResource::MEMORY_HOST) && (type & GPUMemoryResource::MEMORY_GPU))) {
    throw std::runtime_error("Requested invalid memory typo for direct allocation");
  }
  if (mVolatileMemoryStart && !mDeviceMemoryAsVolatile && (type & GPUMemoryResource::MEMORY_GPU) && !(type & GPUMemoryResource::MEMORY_STACK)) {
    GPUError("Must not allocate direct memory while volatile chunks are allocated");
    throw std::bad_alloc();
  }

  void*& pool = (type & GPUMemoryResource::MEMORY_GPU) ? mDeviceMemoryPool : mHostMemoryPool;
  void*& poolend = (type & GPUMemoryResource::MEMORY_GPU) ? mDeviceMemoryPoolEnd : mHostMemoryPoolEnd;
  char* retVal;
  if ((type & GPUMemoryResource::MEMORY_STACK)) {
    poolend = (char*)poolend - size; // TODO: Implement overflow check
    poolend = (char*)poolend - GPUProcessor::getAlignmentMod<GPUCA_MEMALIGN>(poolend);
    retVal = (char*)poolend;
  } else {
    GPUProcessor::computePointerWithAlignment(pool, retVal, size);
  }
  if (pool > poolend) {
    GPUError("Insufficient unmanaged memory: missing %ld bytes", ptrDiff(pool, poolend));
    throw std::bad_alloc();
  }
  UpdateMaxMemoryUsed();
  if (GetProcessingSettings().allocDebugLevel >= 2) {
    std::cout << "Allocated (unmanaged " << ((type & GPUMemoryResource::MEMORY_GPU) ? "gpu" : "host") << "): " << size << " - available: " << ptrDiff(poolend, pool) << "\n";
  }
  return retVal;
}

void* GPUReconstruction::AllocateVolatileDeviceMemory(size_t size)
{
  stdspinlock spinlock(mMemoryMutex);
  if (mVolatileMemoryStart == nullptr) {
    mVolatileMemoryStart = mDeviceMemoryPool;
  }
  if (size == 0) {
    return nullptr; // Future GPU memory allocation is volatile
  }
  char* retVal;
  GPUProcessor::computePointerWithAlignment(mDeviceMemoryPool, retVal, size);
  if (mDeviceMemoryPool > mDeviceMemoryPoolEnd) {
    GPUError("Insufficient volatile device memory: missing %ld", ptrDiff(mDeviceMemoryPool, mDeviceMemoryPoolEnd));
    throw std::bad_alloc();
  }
  UpdateMaxMemoryUsed();
  if (GetProcessingSettings().allocDebugLevel >= 2) {
    std::cout << "Allocated (volatile GPU): " << size << " - available: " << ptrDiff(mDeviceMemoryPoolEnd, mDeviceMemoryPool) << "\n";
  }
  return retVal;
}

void* GPUReconstruction::AllocateVolatileMemory(size_t size, bool device)
{
  if (device) {
    return AllocateVolatileDeviceMemory(size);
  }
  char* retVal = new (std::align_val_t(GPUCA_BUFFER_ALIGNMENT)) char[size];
  stdspinlock spinlock(mMemoryMutex);
  mVolatileChunks.emplace_back(retVal, alignedDeleter());
  return retVal;
}

void GPUReconstruction::MakeFutureDeviceMemoryAllocationsVolatile()
{
  mDeviceMemoryAsVolatile = true;
  AllocateVolatileDeviceMemory(0);
}

void GPUReconstruction::ReturnVolatileDeviceMemory()
{
  mDeviceMemoryAsVolatile = false;
  if (mVolatileMemoryStart) {
    mDeviceMemoryPool = mVolatileMemoryStart;
    mVolatileMemoryStart = nullptr;
  }
  if (GetProcessingSettings().allocDebugLevel >= 2) {
    std::cout << "Freed (volatile GPU) - available: " << ptrDiff(mDeviceMemoryPoolEnd, mDeviceMemoryPool) << "\n";
  }
}

void GPUReconstruction::ReturnVolatileMemory()
{
  ReturnVolatileDeviceMemory();
  mVolatileChunks.clear();
}

void GPUReconstruction::ResetRegisteredMemoryPointers(GPUProcessor* proc)
{
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    if (proc == nullptr || mMemoryResources[i].mProcessor == proc) {
      ResetRegisteredMemoryPointers(i);
    }
  }
}

void GPUReconstruction::ResetRegisteredMemoryPointers(int16_t ires)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL) && (res->mType & GPUMemoryResource::MEMORY_HOST)) {
    void* basePtr = res->mReuse >= 0 ? mMemoryResources[res->mReuse].mPtr : res->mPtr;
    size_t size = ptrDiff(res->SetPointers(basePtr), basePtr);
    if (basePtr && size > std::max(res->mSize, res->mOverrideSize)) {
      std::cerr << "Updated pointers exceed available memory size: " << size << " > " << std::max(res->mSize, res->mOverrideSize) << " - host - " << res->mName << "\n";
      throw std::bad_alloc();
    }
  }
  if (IsGPU() && (res->mType & GPUMemoryResource::MEMORY_GPU)) {
    void* basePtr = res->mReuse >= 0 ? mMemoryResources[res->mReuse].mPtrDevice : res->mPtrDevice;
    size_t size = ptrDiff(res->SetDevicePointers(basePtr), basePtr);
    if (basePtr && size > std::max(res->mSize, res->mOverrideSize)) {
      std::cerr << "Updated pointers exceed available memory size: " << size << " > " << std::max(res->mSize, res->mOverrideSize) << " - GPU - " << res->mName << "\n";
      throw std::bad_alloc();
    }
  }
}

void GPUReconstruction::FreeRegisteredMemory(GPUProcessor* proc, bool freeCustom, bool freePermanent)
{
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    if ((proc == nullptr || mMemoryResources[i].mProcessor == proc) && (freeCustom || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) && (freePermanent || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT))) {
      FreeRegisteredMemory(i);
    }
  }
}

void GPUReconstruction::FreeRegisteredMemory(int16_t ires)
{
  FreeRegisteredMemory(&mMemoryResources[ires]);
}

void GPUReconstruction::FreeRegisteredMemory(GPUMemoryResource* res)
{
  if (GetProcessingSettings().allocDebugLevel >= 2 && (res->mPtr || res->mPtrDevice)) {
    std::cout << "Freeing " << res->mName << ": size " << res->mSize << " (reused " << res->mReuse << ")\n";
  }
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL && res->mReuse < 0) {
    operator delete(res->mPtrDevice, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
  }
  res->mPtr = nullptr;
  res->mPtrDevice = nullptr;
}

void GPUReconstruction::PushNonPersistentMemory(uint64_t tag)
{
  mNonPersistentMemoryStack.emplace_back(mHostMemoryPoolEnd, mDeviceMemoryPoolEnd, mNonPersistentIndividualAllocations.size(), mNonPersistentIndividualDirectAllocations.size(), tag);
}

void GPUReconstruction::PopNonPersistentMemory(RecoStep step, uint64_t tag, const GPUProcessor* proc)
{
  if (proc && GetProcessingSettings().memoryAllocationStrategy != GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    GPUFatal("Processor-depending memory-free works only with allocation strategy ALLOCATION_INDIVIDUAL");
  }
  if (GetProcessingSettings().keepDisplayMemory || GetProcessingSettings().disableMemoryReuse) {
    return;
  }
  if (mNonPersistentMemoryStack.size() == 0) {
    GPUFatal("Trying to pop memory state from empty stack");
  }
  if (tag != 0 && std::get<4>(mNonPersistentMemoryStack.back()) != tag) {
    GPUFatal("Tag mismatch when popping non persistent memory from stack : pop %s vs on stack %s", qTag2Str(tag).c_str(), qTag2Str(std::get<4>(mNonPersistentMemoryStack.back())).c_str());
  }
  if (!proc && (GetProcessingSettings().debugLevel >= 3 || GetProcessingSettings().allocDebugLevel) && (IsGPU() || GetProcessingSettings().forceHostMemoryPoolSize)) {
    printf("Allocated memory after %30s (%8s) (Stack %zu): ", gpudatatypes::RECO_STEP_NAMES[getRecoStepNum(step, true)], qTag2Str(std::get<4>(mNonPersistentMemoryStack.back())).c_str(), mNonPersistentMemoryStack.size());
    PrintMemoryOverview();
    printf("%76s", "");
    PrintMemoryMax();
  }
  for (uint32_t i = std::get<2>(mNonPersistentMemoryStack.back()); i < mNonPersistentIndividualAllocations.size(); i++) {
    GPUMemoryResource* res = mNonPersistentIndividualAllocations[i];
    if (proc && res->mProcessor != proc) {
      continue;
    }
    if (GetProcessingSettings().allocDebugLevel >= 2 && (res->mPtr || res->mPtrDevice)) {
      std::cout << "Freeing NonPersistent " << res->mName << ": size " << res->mSize << " (reused " << res->mReuse << ")\n";
    }
    if (res->mReuse < 0) {
      operator delete(res->mPtrDevice, std::align_val_t(GPUCA_BUFFER_ALIGNMENT));
    }
    res->mPtr = nullptr;
    res->mPtrDevice = nullptr;
  }
  if (!proc) {
    stdspinlock spinlock(mMemoryMutex);
    mHostMemoryPoolEnd = std::get<0>(mNonPersistentMemoryStack.back());
    mDeviceMemoryPoolEnd = std::get<1>(mNonPersistentMemoryStack.back());
    mNonPersistentIndividualAllocations.resize(std::get<2>(mNonPersistentMemoryStack.back()));
    mNonPersistentIndividualDirectAllocations.resize(std::get<3>(mNonPersistentMemoryStack.back()));
    mNonPersistentMemoryStack.pop_back();
  }
}

void GPUReconstruction::BlockStackedMemory(GPUReconstruction* rec)
{
  if (mHostMemoryPoolBlocked || mDeviceMemoryPoolBlocked) {
    throw std::runtime_error("temporary memory stack already blocked");
  }
  mHostMemoryPoolBlocked = rec->mHostMemoryPoolEnd;
  mDeviceMemoryPoolBlocked = rec->mDeviceMemoryPoolEnd;
}

void GPUReconstruction::UnblockStackedMemory()
{
  if (mNonPersistentMemoryStack.size()) {
    throw std::runtime_error("cannot unblock while there is stacked memory");
  }
  mHostMemoryPoolEnd = (char*)mHostMemoryBase + mHostMemorySize;
  mDeviceMemoryPoolEnd = (char*)mDeviceMemoryBase + mDeviceMemorySize;
  mHostMemoryPoolBlocked = nullptr;
  mDeviceMemoryPoolBlocked = nullptr;
}

void GPUReconstruction::SetMemoryExternalInput(int16_t res, void* ptr)
{
  mMemoryResources[res].mPtr = ptr;
}

void GPUReconstruction::ClearAllocatedMemory(bool clearOutputs)
{
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    if (!(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && (clearOutputs || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_OUTPUT))) {
      FreeRegisteredMemory(i);
    }
  }
  mNonPersistentMemoryStack.clear();
  mNonPersistentIndividualAllocations.clear();
  mDirectMemoryChunks.clear();
  mNonPersistentIndividualDirectAllocations.clear();
  mVolatileChunks.clear();
  mVolatileMemoryStart = nullptr;
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    mHostMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mHostMemoryPermanent);
    mDeviceMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mDeviceMemoryPermanent);
    mHostMemoryPoolEnd = mHostMemoryPoolBlocked ? mHostMemoryPoolBlocked : ((char*)mHostMemoryBase + mHostMemorySize);
    mDeviceMemoryPoolEnd = mDeviceMemoryPoolBlocked ? mDeviceMemoryPoolBlocked : ((char*)mDeviceMemoryBase + mDeviceMemorySize);
  } else {
    mHostMemoryPool = mDeviceMemoryPool = mHostMemoryPoolEnd = mDeviceMemoryPoolEnd = nullptr;
  }
}

void GPUReconstruction::UpdateMaxMemoryUsed()
{
  mHostMemoryUsedMax = std::max<size_t>(mHostMemoryUsedMax, ptrDiff(mHostMemoryPool, mHostMemoryBase) + ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd));
  mDeviceMemoryUsedMax = std::max<size_t>(mDeviceMemoryUsedMax, ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase) + ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd));
}

void GPUReconstruction::PrintMemoryMax()
{
  printf("Maximum Memory Allocation: Host %'zu / Device %'zu\n", mHostMemoryUsedMax, mDeviceMemoryUsedMax);
}

void GPUReconstruction::PrintMemoryOverview()
{
  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    printf("Memory Allocation: Host %'13zd / %'13zu (Permanent %'13zd, Data %'13zd, Scratch %'13zd), Device %'13zd / %'13zu, (Permanent %'13zd, Data %'13zd, Scratch %'13zd) %zu chunks\n",
           ptrDiff(mHostMemoryPool, mHostMemoryBase) + ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd), mHostMemorySize, ptrDiff(mHostMemoryPermanent, mHostMemoryBase), ptrDiff(mHostMemoryPool, mHostMemoryPermanent), ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd),
           ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase) + ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd), mDeviceMemorySize, ptrDiff(mDeviceMemoryPermanent, mDeviceMemoryBase), ptrDiff(mDeviceMemoryPool, mDeviceMemoryPermanent), ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd),
           mMemoryResources.size());
  }
}

void GPUReconstruction::PrintMemoryStatistics()
{
  std::map<std::string, std::array<size_t, 3>> sizes;
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    auto& res = mMemoryResources[i];
    if (res.mReuse >= 0) {
      continue;
    }
    auto& x = sizes[res.mName];
    if (res.mPtr) {
      x[0] += res.mSize;
    }
    if (res.mPtrDevice) {
      x[1] += res.mSize;
    }
    if (res.mType & GPUMemoryResource::MemoryType::MEMORY_PERMANENT) {
      x[2] = 1;
    }
  }
  printf("%59s CPU / %9s GPU\n", "", "");
  for (auto it = sizes.begin(); it != sizes.end(); it++) {
    printf("Allocation %50s %s: Size %'14zu / %'14zu\n", it->first.c_str(), it->second[2] ? "P" : " ", it->second[0], it->second[1]);
  }
  PrintMemoryOverview();
  for (uint32_t i = 0; i < mChains.size(); i++) {
    mChains[i]->PrintMemoryStatistics();
  }
}

int32_t GPUReconstruction::registerMemoryForGPU(const void* ptr, size_t size)
{
  if (GetProcessingSettings().noGPUMemoryRegistration) {
    return 0;
  }
  int32_t retVal = registerMemoryForGPU_internal(ptr, size);
  if (retVal == 0) {
    mRegisteredMemoryPtrs.emplace(ptr);
  }
  return retVal;
}

int32_t GPUReconstruction::unregisterMemoryForGPU(const void* ptr)
{
  if (GetProcessingSettings().noGPUMemoryRegistration) {
    return 0;
  }
  const auto& pos = mRegisteredMemoryPtrs.find(ptr);
  if (pos != mRegisteredMemoryPtrs.end()) {
    mRegisteredMemoryPtrs.erase(pos);
    return unregisterMemoryForGPU_internal(ptr);
  }
  return 1;
}

namespace o2::gpu::internal
{
namespace // anonymous
{
template <class T>
constexpr static inline int32_t getStepNum(T step, bool validCheck, int32_t N, const char* err = "Invalid step num")
{
  static_assert(sizeof(step) == sizeof(uint32_t), "Invalid step enum size");
  int32_t retVal = 8 * sizeof(uint32_t) - 1 - CAMath::Clz((uint32_t)step);
  if ((uint32_t)step == 0 || retVal >= N) {
    if (!validCheck) {
      return -1;
    }
    throw std::runtime_error("Invalid General Step");
  }
  return retVal;
}
} // anonymous namespace
} // namespace o2::gpu::internal

int32_t GPUReconstruction::getRecoStepNum(RecoStep step, bool validCheck) { return internal::getStepNum(step, validCheck, gpudatatypes::N_RECO_STEPS, "Invalid Reco Step"); }
int32_t GPUReconstruction::getGeneralStepNum(GeneralStep step, bool validCheck) { return internal::getStepNum(step, validCheck, gpudatatypes::N_GENERAL_STEPS, "Invalid General Step"); }

void GPUReconstruction::RunPipelineWorker()
{
  if (!mInitialized || !GetProcessingSettings().doublePipeline || mMaster != nullptr || !mSlaves.size()) {
    throw std::invalid_argument("Cannot start double pipeline mode");
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Pipeline worker started");
  }
  bool terminate = false;
  while (!terminate) {
    {
      std::unique_lock<std::mutex> lk(mPipelineContext->mutex);
      mPipelineContext->cond.wait(lk, [this] { return this->mPipelineContext->pipelineQueue.size() > 0; });
    }
    GPUReconstructionPipelineQueue* q;
    {
      std::lock_guard<std::mutex> lk(mPipelineContext->mutex);
      q = mPipelineContext->pipelineQueue.front();
      mPipelineContext->pipelineQueue.pop();
    }
    if (q->op == 1) {
      terminate = 1;
    } else {
      q->retVal = q->chain->RunChain();
    }
    {
      std::lock_guard<std::mutex> lk(q->m);
      q->done = true;
    }
    q->c.notify_one();
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Pipeline worker ended");
  }
}

void GPUReconstruction::TerminatePipelineWorker()
{
  EnqueuePipeline(true);
}

int32_t GPUReconstruction::EnqueuePipeline(bool terminate)
{
  ClearAllocatedMemory(true);
  GPUReconstruction* rec = mMaster ? mMaster : this;
  std::unique_ptr<GPUReconstructionPipelineQueue> qu(new GPUReconstructionPipelineQueue);
  GPUReconstructionPipelineQueue* q = qu.get();
  q->chain = terminate ? nullptr : mChains[0].get();
  q->op = terminate ? 1 : 0;
  std::unique_lock<std::mutex> lkdone(q->m);
  {
    std::lock_guard<std::mutex> lkpipe(rec->mPipelineContext->mutex);
    if (rec->mPipelineContext->terminate) {
      throw std::runtime_error("Must not enqueue work after termination request");
    }
    rec->mPipelineContext->pipelineQueue.push(q);
    rec->mPipelineContext->terminate = terminate;
    rec->mPipelineContext->cond.notify_one();
  }
  q->c.wait(lkdone, [&q]() { return q->done; });
  if (terminate || (q->retVal && (q->retVal != 3 || !GetProcessingSettings().ignoreNonFatalGPUErrors))) {
    return q->retVal;
  }
  int32_t retVal2 = mChains[0]->FinalizePipelinedProcessing();
  return retVal2 ? retVal2 : q->retVal;
}

GPUChain* GPUReconstruction::GetNextChainInQueue()
{
  GPUReconstruction* rec = mMaster ? mMaster : this;
  std::lock_guard<std::mutex> lk(rec->mPipelineContext->mutex);
  return rec->mPipelineContext->pipelineQueue.size() && rec->mPipelineContext->pipelineQueue.front()->op == 0 ? rec->mPipelineContext->pipelineQueue.front()->chain : nullptr;
}

void GPUReconstruction::PrepareEvent() // TODO: Clean this up, this should not be called from chainTracking but before
{
  ClearAllocatedMemory(true);
  for (uint32_t i = 0; i < mChains.size(); i++) {
    mChains[i]->PrepareEvent();
  }
  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mAllocateAndInitializeLate) {
      continue;
    }
    (mProcessors[i].proc->*(mProcessors[i].SetMaxData))(mHostConstantMem->ioPtrs);
    if (mProcessors[i].proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && mProcessors[i].proc->mLinkedProcessor) {
      (mProcessors[i].proc->mLinkedProcessor->*(mProcessors[i].SetMaxData))(mHostConstantMem->ioPtrs);
    }
  }
  ComputeReuseMax(nullptr);
  AllocateRegisteredMemory(nullptr);
}

int32_t GPUReconstruction::CheckErrorCodes(bool cpuOnly, bool forceShowErrors, std::vector<std::array<uint32_t, 4>>* fillErrors)
{
  int32_t retVal = 0;
  for (uint32_t i = 0; i < mChains.size(); i++) {
    if (mChains[i]->CheckErrorCodes(cpuOnly, forceShowErrors, fillErrors)) {
      retVal++;
    }
  }
  return retVal;
}

int32_t GPUReconstruction::GPUChkErrA(const int64_t error, const char* file, int32_t line, bool failOnError)
{
  if (error == 0 || !GPUChkErrInternal(error, file, line)) {
    return 0;
  }
  if (failOnError) {
    if (mInitialized && mInErrorHandling == false) {
      mInErrorHandling = true;
      CheckErrorCodes(false, true);
    }
    throw std::runtime_error("GPU Backend Failure");
  }
  return 1;
}

void GPUReconstruction::DumpSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  DumpStructToFile(mGRPSettings.get(), f.c_str());
  for (uint32_t i = 0; i < mChains.size(); i++) {
    mChains[i]->DumpSettings(dir);
  }
}

void GPUReconstruction::UpdateDynamicSettings(const GPUSettingsRecDynamic* d)
{
  UpdateSettings(nullptr, nullptr, d);
}

void GPUReconstruction::UpdateSettings(const GPUSettingsGRP* g, const GPUSettingsProcessing* p, const GPUSettingsRecDynamic* d)
{
  if (g) {
    *mGRPSettings = *g;
  }
  if (p) {
    mProcessingSettings->debugLevel = p->debugLevel;
    mProcessingSettings->resetTimers = p->resetTimers;
  }
  GPURecoStepConfiguration* w = nullptr;
  if (mRecoSteps.steps.isSet(gpudatatypes::RecoStep::TPCdEdx)) {
    w = &mRecoSteps;
  }
  param().UpdateSettings(g, p, w, d);
  if (mInitialized) {
    WriteConstantParams();
  }
}

int32_t GPUReconstruction::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  new (mGRPSettings.get()) GPUSettingsGRP;
  if (ReadStructFromFile(f.c_str(), mGRPSettings.get())) {
    return 1;
  }
  param().UpdateSettings(mGRPSettings.get());
  for (uint32_t i = 0; i < mChains.size(); i++) {
    mChains[i]->ReadSettings(dir);
  }
  return 0;
}

void GPUReconstruction::SetSettings(float solenoidBzNominalGPU, const GPURecoStepConfiguration* workflow)
{
#ifdef GPUCA_O2_LIB
  GPUO2InterfaceConfiguration config;
  config.ReadConfigurableParam(config);
  config.configGRP.solenoidBzNominalGPU = solenoidBzNominalGPU;
  SetSettings(&config.configGRP, &config.configReconstruction, &config.configProcessing, workflow);
#else
  GPUSettingsGRP grp;
  grp.solenoidBzNominalGPU = solenoidBzNominalGPU;
  SetSettings(&grp, nullptr, nullptr, workflow);
#endif
}

void GPUReconstruction::SetSettings(const GPUSettingsGRP* grp, const GPUSettingsRec* rec, const GPUSettingsProcessing* proc, const GPURecoStepConfiguration* workflow)
{
  if (mInitialized) {
    GPUError("Cannot update settings while initialized");
    throw std::runtime_error("Settings updated while initialized");
  }
  *mGRPSettings = *grp;
  if (proc) {
    *mProcessingSettings = *proc;
  }
  if (workflow) {
    mRecoSteps.steps = workflow->steps;
    mRecoSteps.stepsGPUMask &= workflow->stepsGPUMask;
    mRecoSteps.inputs = workflow->inputs;
    mRecoSteps.outputs = workflow->outputs;
  }
  param().SetDefaults(mGRPSettings.get(), rec, proc, workflow);
}

void GPUReconstruction::SetOutputControl(void* ptr, size_t size)
{
  GPUOutputControl outputControl;
  outputControl.set(ptr, size);
  SetOutputControl(outputControl);
}

void GPUReconstruction::SetInputControl(void* ptr, size_t size) { mInputControl.set(ptr, size); }
GPUReconstruction::DeviceType GPUReconstruction::GetDeviceType() const { return (DeviceType)GetDeviceBackendSettings().deviceType; }
const GPUParam& GPUReconstruction::GetParam() const { return mHostConstantMem->param; }
void GPUReconstruction::SetResetTimers(bool reset) { mProcessingSettings->resetTimers = reset; }
void GPUReconstruction::SetDebugLevelTmp(int32_t level) { mProcessingSettings->debugLevel = level; }
GPUParam& GPUReconstruction::param() { return mHostConstantMem->param; }
const GPUTrackingInOutPointers GPUReconstruction::GetIOPtrs() const { return mHostConstantMem->ioPtrs; }
const GPUCalibObjectsConst& GPUReconstruction::GetCalib() const { return processors()->calibObjects; }

ThrustVolatileAllocator::ThrustVolatileAllocator(GPUReconstruction* r)
{
  mAlloc = [&r](size_t n) { return (char*)r->AllocateVolatileDeviceMemory(n); };
}
ThrustVolatileAllocator GPUReconstruction::getThrustVolatileDeviceAllocator()
{
  return ThrustVolatileAllocator(this);
}
