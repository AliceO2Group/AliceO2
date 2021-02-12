// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#include <conio.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "GPUReconstruction.h"
#include "GPUReconstructionIncludes.h"

#include "GPUMemoryResource.h"
#include "GPUChain.h"
#include "GPUMemorySizeScalers.h"

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

#ifdef GPUCA_O2_LIB
#include "GPUO2InterfaceConfiguration.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUReconstructionPipelineQueue {
  unsigned int op = 0; // For now, 0 = process, 1 = terminate
  GPUChain* chain = nullptr;
  std::mutex m;
  std::condition_variable c;
  bool done = false;
  int retVal = 0;
};

struct GPUReconstructionPipelineContext {
  std::queue<GPUReconstructionPipelineQueue*> queue;
  std::mutex mutex;
  std::condition_variable cond;
  bool terminate = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

using namespace GPUCA_NAMESPACE::gpu;

constexpr const char* const GPUReconstruction::DEVICE_TYPE_NAMES[];
constexpr const char* const GPUReconstruction::GEOMETRY_TYPE_NAMES[];
constexpr const char* const GPUReconstruction::IOTYPENAMES[];
constexpr GPUReconstruction::GeometryType GPUReconstruction::geometryType;

static long long int ptrDiff(void* a, void* b) { return (long long int)((char*)a - (char*)b); }

GPUReconstruction::GPUReconstruction(const GPUSettingsDeviceBackend& cfg) : mHostConstantMem(new GPUConstantMem), mDeviceBackendSettings(cfg)
{
  if (cfg.master) {
    if (cfg.master->mDeviceBackendSettings.deviceType != cfg.deviceType) {
      throw std::invalid_argument("device type of master and slave GPUReconstruction does not match");
    }
    if (cfg.master->mMaster) {
      throw std::invalid_argument("Cannot be slave to a slave");
    }
    mMaster = cfg.master;
    cfg.master->mSlaves.emplace_back(this);
  }
  new (&mProcessingSettings) GPUSettingsProcessing;
  new (&mEventSettings) GPUSettingsEvent;
  param().SetDefaults(&mEventSettings);
  mMemoryScalers.reset(new GPUMemorySizeScalers);
  for (unsigned int i = 0; i < NSLICES; i++) {
    processors()->tpcTrackers[i].SetSlice(i); // TODO: Move to a better place
#ifdef HAVE_O2HEADERS
    processors()->tpcClusterer[i].mISlice = i;
#endif
  }
}

GPUReconstruction::~GPUReconstruction()
{
  if (mInitialized) {
    GPUError("GPU Reconstruction not properly deinitialized!");
  }
}

void GPUReconstruction::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsCPU);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraits);
  }
}

int GPUReconstruction::SetNOMPThreads(int n)
{
#ifdef WITH_OPENMP
  omp_set_num_threads(mProcessingSettings.ompThreads = std::max(1, n < 0 ? mMaxOMPThreads : std::min(n, mMaxOMPThreads)));
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Set number of OpenMP threads to %d (%d requested)", mProcessingSettings.ompThreads, n);
  }
  return n > mMaxOMPThreads;
#else
  return 1;
#endif
}

int GPUReconstruction::Init()
{
  if (mMaster) {
    throw std::runtime_error("Must not call init on slave!");
  }
  int retVal = InitPhaseBeforeDevice();
  if (retVal) {
    return retVal;
  }
  for (unsigned int i = 0; i < mSlaves.size(); i++) {
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
  mHostMemoryPoolEnd = (char*)mHostMemoryBase + mHostMemorySize;
  mDeviceMemoryPoolEnd = (char*)mDeviceMemoryBase + mDeviceMemorySize;
  if (InitPhasePermanentMemory()) {
    return 1;
  }
  for (unsigned int i = 0; i < mSlaves.size(); i++) {
    mSlaves[i]->mDeviceMemoryBase = mDeviceMemoryPermanent;
    mSlaves[i]->mHostMemoryBase = mHostMemoryPermanent;
    mSlaves[i]->mDeviceMemorySize = mDeviceMemorySize - ((char*)mSlaves[i]->mDeviceMemoryBase - (char*)mDeviceMemoryBase);
    mSlaves[i]->mHostMemorySize = mHostMemorySize - ((char*)mSlaves[i]->mHostMemoryBase - (char*)mHostMemoryBase);
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
  for (unsigned int i = 0; i < mSlaves.size(); i++) {
    mSlaves[i]->mDeviceMemoryPermanent = mDeviceMemoryPermanent;
    mSlaves[i]->mHostMemoryPermanent = mHostMemoryPermanent;
    retVal = mSlaves[i]->InitPhaseAfterDevice();
    if (retVal) {
      GPUError("Error initialization slave (after device init)");
      return retVal;
    }
  }
  return 0;
}

int GPUReconstruction::InitPhaseBeforeDevice()
{
#ifndef HAVE_O2HEADERS
  mRecoSteps.setBits(RecoStep::ITSTracking, false);
  mRecoSteps.setBits(RecoStep::TRDTracking, false);
  mRecoSteps.setBits(RecoStep::TPCConversion, false);
  mRecoSteps.setBits(RecoStep::TPCCompression, false);
  mRecoSteps.setBits(RecoStep::TPCdEdx, false);
  mProcessingSettings.createO2Output = false;
#endif
  mRecoStepsGPU &= mRecoSteps;
  mRecoStepsGPU &= AvailableRecoSteps();
  if (!IsGPU()) {
    mRecoStepsGPU.set((unsigned char)0);
  }

  if (mProcessingSettings.forceMemoryPoolSize >= 1024 || mProcessingSettings.forceHostMemoryPoolSize >= 1024) {
    mProcessingSettings.memoryAllocationStrategy = GPUMemoryResource::ALLOCATION_GLOBAL;
  }
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_AUTO) {
    mProcessingSettings.memoryAllocationStrategy = IsGPU() ? GPUMemoryResource::ALLOCATION_GLOBAL : GPUMemoryResource::ALLOCATION_INDIVIDUAL;
  }
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mProcessingSettings.forceMemoryPoolSize = mProcessingSettings.forceHostMemoryPoolSize = 0;
  }
  if (mProcessingSettings.debugLevel >= 4) {
    mProcessingSettings.keepAllMemory = true;
  }
  if (mProcessingSettings.debugLevel >= 5 && mProcessingSettings.allocDebugLevel < 2) {
    mProcessingSettings.allocDebugLevel = 2;
  }
  if (mProcessingSettings.eventDisplay || mProcessingSettings.keepAllMemory) {
    mProcessingSettings.keepDisplayMemory = true;
  }
  if (mProcessingSettings.debugLevel < 6) {
    mProcessingSettings.debugMask = 0;
  }
  if (mProcessingSettings.debugLevel < 1) {
    mProcessingSettings.deviceTimers = false;
  }
  if (mProcessingSettings.debugLevel >= 6 && mProcessingSettings.comparableDebutOutput) {
    mProcessingSettings.nTPCClustererLanes = 1;
    if (mProcessingSettings.trackletConstructorInPipeline < 0) {
      mProcessingSettings.trackletConstructorInPipeline = 1;
    }
    if (mProcessingSettings.trackletSelectorInPipeline < 0) {
      mProcessingSettings.trackletSelectorInPipeline = 1;
    }
    if (mProcessingSettings.trackletSelectorSlices < 0) {
      mProcessingSettings.trackletSelectorSlices = 1;
    }
  }
  if (mProcessingSettings.createO2Output > 1 && mProcessingSettings.runQA) {
    mProcessingSettings.createO2Output = 1;
  }
  if (!(mRecoStepsGPU & GPUDataTypes::RecoStep::TPCMerging)) {
    mProcessingSettings.mergerSortTracks = false;
  }
  if (!IsGPU()) {
    mProcessingSettings.nDeviceHelperThreads = 0;
    mProcessingSettings.nTPCClustererLanes = 1;
  }

  if (param().rec.NonConsecutiveIDs) {
    param().rec.DisableRefitAttachment = 0xFF;
  }
  if (!(mRecoStepsGPU & RecoStep::TPCMerging) || !param().rec.mergerReadFromTrackerDirectly) {
    mProcessingSettings.fullMergerOnGPU = false;
  }
  if (mProcessingSettings.debugLevel || !mProcessingSettings.fullMergerOnGPU) {
    mProcessingSettings.delayedOutput = false;
  }

  UpdateSettings();
  GPUCA_GPUReconstructionUpdateDefailts();
  if (!mProcessingSettings.trackletConstructorInPipeline) {
    mProcessingSettings.trackletSelectorInPipeline = false;
  }
  if (!mProcessingSettings.enableRTC) {
    mProcessingSettings.rtcConstexpr = false;
  }

  mMemoryScalers->factor = mProcessingSettings.memoryScalingFactor;

#ifdef WITH_OPENMP
  if (mProcessingSettings.ompThreads <= 0) {
    mProcessingSettings.ompThreads = omp_get_max_threads();
  } else {
    mProcessingSettings.ompAutoNThreads = false;
    omp_set_num_threads(mProcessingSettings.ompThreads);
  }
#else
  mProcessingSettings.ompThreads = 1;
#endif
  mMaxOMPThreads = mProcessingSettings.ompThreads;
  mMaxThreads = std::max(mMaxThreads, mProcessingSettings.ompThreads);
  if (IsGPU()) {
    mNStreams = std::max<int>(mProcessingSettings.nStreams, 3);
  }

  if (mProcessingSettings.doublePipeline && (mChains.size() != 1 || mChains[0]->SupportsDoublePipeline() == false || !IsGPU() || mProcessingSettings.memoryAllocationStrategy != GPUMemoryResource::ALLOCATION_GLOBAL)) {
    GPUError("Must use double pipeline mode only with exactly one chain that must support it");
    return 1;
  }

  if (mMaster == nullptr && mProcessingSettings.doublePipeline) {
    mPipelineContext.reset(new GPUReconstructionPipelineContext);
  }

  mDeviceMemorySize = mHostMemorySize = 0;
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->RegisterPermanentMemoryAndProcessors();
    size_t memPrimary, memPageLocked;
    mChains[i]->MemorySize(memPrimary, memPageLocked);
    if (!IsGPU() || mOutputControl.useInternal()) {
      memPageLocked = memPrimary;
    }
    mDeviceMemorySize += memPrimary;
    mHostMemorySize += memPageLocked;
  }
  if (mProcessingSettings.forceMemoryPoolSize && mProcessingSettings.forceMemoryPoolSize <= 2 && CanQueryMaxMemory()) {
    mDeviceMemorySize = mProcessingSettings.forceMemoryPoolSize;
  } else if (mProcessingSettings.forceMemoryPoolSize > 2) {
    mDeviceMemorySize = mProcessingSettings.forceMemoryPoolSize;
    if (!IsGPU() || mOutputControl.useInternal()) {
      mHostMemorySize = mDeviceMemorySize;
    }
  }
  if (mProcessingSettings.forceHostMemoryPoolSize) {
    mHostMemorySize = mProcessingSettings.forceHostMemoryPoolSize;
  }

  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].RegisterMemoryAllocation))();
  }

  return 0;
}

int GPUReconstruction::InitPhasePermanentMemory()
{
  if (IsGPU()) {
    for (unsigned int i = 0; i < mChains.size(); i++) {
      mChains[i]->RegisterGPUProcessors();
    }
  }
  AllocateRegisteredPermanentMemory();
  return 0;
}

int GPUReconstruction::InitPhaseAfterDevice()
{
  for (unsigned int i = 0; i < mChains.size(); i++) {
    if (mChains[i]->Init()) {
      return 1;
    }
  }
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].InitializeProcessor))();
  }

  WriteConstantParams(); // First initialization, if the user doesn't use RunChains

  mInitialized = true;
  return 0;
}

void GPUReconstruction::WriteConstantParams()
{
  if (IsGPU()) {
    const auto threadContext = GetThreadContext();
    WriteToConstantMemory((char*)&processors()->param - (char*)processors(), &param(), sizeof(param()), -1);
  }
}

int GPUReconstruction::Finalize()
{
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->Finalize();
  }
  return 0;
}

int GPUReconstruction::Exit()
{
  if (!mInitialized) {
    return 1;
  }
  for (unsigned int i = 0; i < mSlaves.size(); i++) {
    if (mSlaves[i]->Exit()) {
      GPUError("Error exiting slave");
    }
  }

  mChains.clear();          // Make sure we destroy a possible ITS GPU tracker before we call the destructors
  mHostConstantMem.reset(); // Reset these explicitly before the destruction of other members unloads the library
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
      if (mMemoryResources[i].mReuse >= 0) {
        continue;
      }
      operator delete(mMemoryResources[i].mPtrDevice GPUCA_OPERATOR_NEW_ALIGNMENT);
      mMemoryResources[i].mPtr = mMemoryResources[i].mPtrDevice = nullptr;
    }
  }
  mMemoryResources.clear();
  if (mInitialized) {
    ExitDevice();
  }
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
      for (unsigned int i = 0; i < re.res.size(); i++) {
        GPUMemoryResource& res = mMemoryResources[re.res[i]];
        resMain.mOverrideSize = std::max<size_t>(resMain.mOverrideSize, (char*)res.SetPointers((void*)1) - (char*)1);
      }
    }
  }
}

size_t GPUReconstruction::AllocateRegisteredMemory(GPUProcessor* proc, bool resetCustom)
{
  if (mProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating memory %p", (void*)proc);
  }
  size_t total = 0;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if (proc == nullptr ? !mMemoryResources[i].mProcessor->mAllocateAndInitializeLate : mMemoryResources[i].mProcessor == proc) {
      if (!(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) {
        total += AllocateRegisteredMemory(i);
      } else if (resetCustom && (mMemoryResources[i].mPtr || mMemoryResources[i].mPtrDevice)) {
        ResetRegisteredMemoryPointers(i);
      }
    }
  }
  if (mProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating memory done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredPermanentMemory()
{
  if (mProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating Permanent Memory");
  }
  int total = 0;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if ((mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && mMemoryResources[i].mPtr == nullptr) {
      total += AllocateRegisteredMemory(i);
    }
  }
  mHostMemoryPermanent = mHostMemoryPool;
  mDeviceMemoryPermanent = mDeviceMemoryPool;
  if (mProcessingSettings.debugLevel >= 5) {
    GPUInfo("Permanent Memory Done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredMemoryHelper(GPUMemoryResource* res, void*& ptr, void*& memorypool, void* memorybase, size_t memorysize, void* (GPUMemoryResource::*setPtr)(void*), void*& memorypoolend)
{
  if (res->mReuse >= 0) {
    ptr = (&ptr == &res->mPtrDevice) ? mMemoryResources[res->mReuse].mPtrDevice : mMemoryResources[res->mReuse].mPtr;
    if (ptr == nullptr) {
      GPUError("Invalid reuse ptr (%s)", res->mName);
      throw std::bad_alloc();
    }
    size_t retVal = (char*)((res->*setPtr)(ptr)) - (char*)(ptr);
    if (retVal > mMemoryResources[res->mReuse].mSize) {
      GPUError("Insufficient reuse memory %lu < %lu (%s)", mMemoryResources[res->mReuse].mSize, retVal, res->mName);
      throw std::bad_alloc();
    }
    if (mProcessingSettings.allocDebugLevel >= 2) {
      std::cout << "Reused " << res->mName << ": " << retVal << "\n";
    }
    return retVal;
  }
  if (memorypool == nullptr) {
    GPUInfo("Memory pool uninitialized");
    throw std::bad_alloc();
  }
  size_t retVal;
  if ((res->mType & GPUMemoryResource::MEMORY_STACK) && memorypoolend) {
    retVal = (char*)((res->*setPtr)((char*)1)) - (char*)(1);
    memorypoolend = (void*)((char*)memorypoolend - GPUProcessor::getAlignmentMod<GPUCA_MEMALIGN>(memorypoolend));
    if (retVal < res->mOverrideSize) {
      retVal = res->mOverrideSize;
    }
    retVal += GPUProcessor::getAlignment<GPUCA_MEMALIGN>(retVal);
    memorypoolend = (char*)memorypoolend - retVal;
    ptr = memorypoolend;
    retVal = std::max<size_t>((char*)((res->*setPtr)(ptr)) - (char*)ptr, res->mOverrideSize);
  } else {
    ptr = memorypool;
    memorypool = (char*)((res->*setPtr)(ptr));
    retVal = (char*)memorypool - (char*)ptr;
    if (retVal < res->mOverrideSize) {
      retVal = res->mOverrideSize;
      memorypool = (char*)ptr + res->mOverrideSize;
    }
    memorypool = (void*)((char*)memorypool + GPUProcessor::getAlignment<GPUCA_MEMALIGN>(memorypool));
  }
  if (memorypoolend ? (memorypool > memorypoolend) : ((size_t)((char*)memorypool - (char*)memorybase) > memorysize)) {
    std::cout << "Memory pool size exceeded (" << res->mName << ": " << (memorypoolend ? (memorysize + ((char*)memorypool - (char*)memorypoolend)) : (char*)memorypool - (char*)memorybase) << " < " << memorysize << "\n";
    throw std::bad_alloc();
  }
  if (mProcessingSettings.allocDebugLevel >= 2) {
    std::cout << "Allocated " << res->mName << ": " << retVal << " - available: " << (memorypoolend ? ((char*)memorypoolend - (char*)memorypool) : (memorysize - ((char*)memorypool - (char*)memorybase))) << "\n";
  }
  return retVal;
}

void GPUReconstruction::AllocateRegisteredMemoryInternal(GPUMemoryResource* res, GPUOutputControl* control, GPUReconstruction* recPool)
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL && (control == nullptr || control->useInternal())) {
    if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
      if (res->mPtrDevice && res->mReuse < 0) {
        operator delete(res->mPtrDevice GPUCA_OPERATOR_NEW_ALIGNMENT);
      }
      res->mSize = std::max((size_t)res->SetPointers((void*)1) - 1, res->mOverrideSize);
      if (res->mReuse >= 0) {
        if (res->mSize > mMemoryResources[res->mReuse].mSize) {
          GPUError("Invalid reuse, insufficient size: %lld < %lld", (long long int)mMemoryResources[res->mReuse].mSize, (long long int)res->mSize);
          throw std::bad_alloc();
        }
        res->mPtrDevice = mMemoryResources[res->mReuse].mPtrDevice;
      } else {
        res->mPtrDevice = operator new(res->mSize + GPUCA_BUFFER_ALIGNMENT GPUCA_OPERATOR_NEW_ALIGNMENT);
      }
      res->mPtr = GPUProcessor::alignPointer<GPUCA_BUFFER_ALIGNMENT>(res->mPtrDevice);
      res->SetPointers(res->mPtr);
      if (mProcessingSettings.allocDebugLevel >= 2) {
        std::cout << (res->mReuse >= 0 ? "Reused " : "Allocated ") << res->mName << ": " << res->mSize << "\n";
      }
      if (res->mType & GPUMemoryResource::MEMORY_STACK) {
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
    if ((!IsGPU() || (res->mType & GPUMemoryResource::MEMORY_HOST) || mProcessingSettings.keepDisplayMemory) && !(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) { // keepAllMemory --> keepDisplayMemory
      if (control && control->useExternal()) {
        if (control->allocator) {
          res->mSize = std::max((size_t)res->SetPointers((void*)1) - 1, res->mOverrideSize);
          res->mPtr = control->allocator(CAMath::nextMultipleOf<GPUCA_BUFFER_ALIGNMENT>(res->mSize));
          res->mSize = std::max<size_t>((char*)res->SetPointers(res->mPtr) - (char*)res->mPtr, res->mOverrideSize);
        } else {
          void* dummy = nullptr;
          res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, control->ptrCurrent, control->ptrBase, control->size, &GPUMemoryResource::SetPointers, dummy);
        }
      } else {
        res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, recPool->mHostMemoryPool, recPool->mHostMemoryBase, recPool->mHostMemorySize, &GPUMemoryResource::SetPointers, recPool->mHostMemoryPoolEnd);
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
      size_t size = AllocateRegisteredMemoryHelper(res, res->mPtrDevice, recPool->mDeviceMemoryPool, recPool->mDeviceMemoryBase, recPool->mDeviceMemorySize, &GPUMemoryResource::SetDevicePointers, recPool->mDeviceMemoryPoolEnd);

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

void GPUReconstruction::AllocateRegisteredForeignMemory(short ires, GPUReconstruction* rec, GPUOutputControl* control)
{
  rec->AllocateRegisteredMemoryInternal(&rec->mMemoryResources[ires], control, this);
}

size_t GPUReconstruction::AllocateRegisteredMemory(short ires, GPUOutputControl* control)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if ((res->mType & GPUMemoryResource::MEMORY_PERMANENT) && res->mPtr != nullptr) {
    ResetRegisteredMemoryPointers(ires);
  } else {
    AllocateRegisteredMemoryInternal(res, control, this);
  }
  return res->mReuse >= 0 ? 0 : res->mSize;
}

void* GPUReconstruction::AllocateUnmanagedMemory(size_t size, int type)
{
  if (type != GPUMemoryResource::MEMORY_HOST && (!IsGPU() || type != GPUMemoryResource::MEMORY_GPU)) {
    throw std::bad_alloc();
  }
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mUnmanagedChunks.emplace_back(new char[size + GPUCA_BUFFER_ALIGNMENT]);
    return GPUProcessor::alignPointer<GPUCA_BUFFER_ALIGNMENT>(mUnmanagedChunks.back().get());
  } else {
    void* pool = type == GPUMemoryResource::MEMORY_GPU ? mDeviceMemoryPool : mHostMemoryPool;
    void* poolend = type == GPUMemoryResource::MEMORY_GPU ? mDeviceMemoryPoolEnd : mHostMemoryPoolEnd;
    char* retVal;
    GPUProcessor::computePointerWithAlignment(pool, retVal, size);
    if (pool > poolend) {
      throw std::bad_alloc();
    }
    UpdateMaxMemoryUsed();
    return retVal;
  }
}

void* GPUReconstruction::AllocateVolatileDeviceMemory(size_t size)
{
  if (mVolatileMemoryStart == nullptr) {
    mVolatileMemoryStart = mDeviceMemoryPool;
  }
  if (size == 0) {
    return nullptr; // Future GPU memory allocation is volatile
  }
  char* retVal;
  GPUProcessor::computePointerWithAlignment(mDeviceMemoryPool, retVal, size);
  if (mDeviceMemoryPool > mDeviceMemoryPoolEnd) {
    throw std::bad_alloc();
  }
  UpdateMaxMemoryUsed();
  return retVal;
}

void GPUReconstruction::ResetRegisteredMemoryPointers(GPUProcessor* proc)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if (proc == nullptr || mMemoryResources[i].mProcessor == proc) {
      ResetRegisteredMemoryPointers(i);
    }
  }
}

void GPUReconstruction::ResetRegisteredMemoryPointers(short ires)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL) && (res->mType & GPUMemoryResource::MEMORY_HOST)) {
    if (res->mReuse >= 0) {
      res->SetPointers(mMemoryResources[res->mReuse].mPtr);
    } else {
      res->SetPointers(res->mPtr);
    }
  }
  if (IsGPU() && (res->mType & GPUMemoryResource::MEMORY_GPU)) {
    if (res->mReuse >= 0) {
      res->SetDevicePointers(mMemoryResources[res->mReuse].mPtrDevice);
    } else {
      res->SetDevicePointers(res->mPtrDevice);
    }
  }
}

void GPUReconstruction::FreeRegisteredMemory(GPUProcessor* proc, bool freeCustom, bool freePermanent)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if ((proc == nullptr || mMemoryResources[i].mProcessor == proc) && (freeCustom || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) && (freePermanent || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT))) {
      FreeRegisteredMemory(i);
    }
  }
}

void GPUReconstruction::FreeRegisteredMemory(short ires)
{
  FreeRegisteredMemory(&mMemoryResources[ires]);
}

void GPUReconstruction::FreeRegisteredMemory(GPUMemoryResource* res)
{
  if (mProcessingSettings.allocDebugLevel >= 2 && (res->mPtr || res->mPtrDevice)) {
    std::cout << "Freeing " << res->mName << ": size " << res->mSize << " (reused " << res->mReuse << ")\n";
  }
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL && res->mReuse < 0) {
    operator delete(res->mPtrDevice GPUCA_OPERATOR_NEW_ALIGNMENT);
  }
  res->mPtr = nullptr;
  res->mPtrDevice = nullptr;
}

void GPUReconstruction::ReturnVolatileDeviceMemory()
{
  if (mVolatileMemoryStart) {
    mDeviceMemoryPool = mVolatileMemoryStart;
    mVolatileMemoryStart = nullptr;
  }
}

void GPUReconstruction::PushNonPersistentMemory()
{
  mNonPersistentMemoryStack.emplace_back(mHostMemoryPoolEnd, mDeviceMemoryPoolEnd, mNonPersistentIndividualAllocations.size());
}

void GPUReconstruction::PopNonPersistentMemory(RecoStep step)
{
  if ((mProcessingSettings.debugLevel >= 3 || mProcessingSettings.allocDebugLevel) && (IsGPU() || mProcessingSettings.forceHostMemoryPoolSize)) {
    if (IsGPU()) {
      printf("Allocated Device memory after %30s: %'13lld (non temporary %'13lld, blocked %'13lld)\n", GPUDataTypes::RECO_STEP_NAMES[getRecoStepNum(step, true)], ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase) + ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd), ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase), mDeviceMemoryPoolBlocked == nullptr ? 0ll : ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolBlocked));
    }
    printf("Allocated Host memory after  %30s: %'13lld (non temporary %'13lld, blocked %'13lld)\n", GPUDataTypes::RECO_STEP_NAMES[getRecoStepNum(step, true)], ptrDiff(mHostMemoryPool, mHostMemoryBase) + ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd), ptrDiff(mHostMemoryPool, mHostMemoryBase), mHostMemoryPoolBlocked == nullptr ? 0ll : ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolBlocked));
  }
  if (mProcessingSettings.keepDisplayMemory || mProcessingSettings.disableMemoryReuse) {
    return;
  }
  mHostMemoryPoolEnd = std::get<0>(mNonPersistentMemoryStack.back());
  mDeviceMemoryPoolEnd = std::get<1>(mNonPersistentMemoryStack.back());
  for (unsigned int i = std::get<2>(mNonPersistentMemoryStack.back()); i < mNonPersistentIndividualAllocations.size(); i++) {
    GPUMemoryResource* res = mNonPersistentIndividualAllocations[i];
    if (res->mReuse < 0) {
      operator delete(res->mPtrDevice GPUCA_OPERATOR_NEW_ALIGNMENT);
    }
    res->mPtr = nullptr;
    res->mPtrDevice = nullptr;
  }
  mNonPersistentIndividualAllocations.resize(std::get<2>(mNonPersistentMemoryStack.back()));
  mNonPersistentMemoryStack.pop_back();
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

void GPUReconstruction::SetMemoryExternalInput(short res, void* ptr)
{
  mMemoryResources[res].mPtr = ptr;
}

void GPUReconstruction::ClearAllocatedMemory(bool clearOutputs)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if (!(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && (clearOutputs || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_OUTPUT))) {
      FreeRegisteredMemory(i);
    }
  }
  mHostMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mHostMemoryPermanent);
  mDeviceMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mDeviceMemoryPermanent);
  mUnmanagedChunks.clear();
  mVolatileMemoryStart = nullptr;
  mNonPersistentMemoryStack.clear();
  mNonPersistentIndividualAllocations.clear();
  mHostMemoryPoolEnd = mHostMemoryPoolBlocked ? mHostMemoryPoolBlocked : ((char*)mHostMemoryBase + mHostMemorySize);
  mDeviceMemoryPoolEnd = mDeviceMemoryPoolBlocked ? mDeviceMemoryPoolBlocked : ((char*)mDeviceMemoryBase + mDeviceMemorySize);
}

void GPUReconstruction::UpdateMaxMemoryUsed()
{
  mHostMemoryUsedMax = std::max<size_t>(mHostMemoryUsedMax, ptrDiff(mHostMemoryPool, mHostMemoryBase) + ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd));
  mDeviceMemoryUsedMax = std::max<size_t>(mDeviceMemoryUsedMax, ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase) + ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd));
}

void GPUReconstruction::PrintMemoryMax()
{
  printf("Maximum Memory Allocation: Host %'lld / Device %'lld\n", (long long int)mHostMemoryUsedMax, (long long int)mDeviceMemoryUsedMax);
}

void GPUReconstruction::PrintMemoryOverview()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    printf("Memory Allocation: Host %'lld / %'lld (Permanent %'lld), Device %'lld / %'lld, (Permanent %'lld) %d chunks\n",
           ptrDiff(mHostMemoryPool, mHostMemoryBase) + ptrDiff((char*)mHostMemoryBase + mHostMemorySize, mHostMemoryPoolEnd), (long long int)mHostMemorySize, ptrDiff(mHostMemoryPermanent, mHostMemoryBase),
           ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase) + ptrDiff((char*)mDeviceMemoryBase + mDeviceMemorySize, mDeviceMemoryPoolEnd), (long long int)mDeviceMemorySize, ptrDiff(mDeviceMemoryPermanent, mDeviceMemoryBase), (int)mMemoryResources.size());
  }
}

void GPUReconstruction::PrintMemoryStatistics()
{
  std::map<std::string, std::array<size_t, 3>> sizes;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
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
    printf("Allocation %30s %s: Size %'13lld / %'13lld\n", it->first.c_str(), it->second[2] ? "P" : " ", (long long int)it->second[0], (long long int)it->second[1]);
  }
  PrintMemoryOverview();
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->PrintMemoryStatistics();
  }
}

template <class T>
static inline int getStepNum(T step, bool validCheck, int N, const char* err = "Invalid step num")
{
  static_assert(sizeof(step) == sizeof(unsigned int), "Invalid step enum size");
  int retVal = 8 * sizeof(unsigned int) - 1 - CAMath::Clz((unsigned int)step);
  if ((unsigned int)step == 0 || retVal >= N) {
    if (!validCheck) {
      return -1;
    }
    throw std::runtime_error("Invalid General Step");
  }
  return retVal;
}

int GPUReconstruction::getRecoStepNum(RecoStep step, bool validCheck) { return getStepNum(step, validCheck, GPUDataTypes::N_RECO_STEPS, "Invalid Reco Step"); }
int GPUReconstruction::getGeneralStepNum(GeneralStep step, bool validCheck) { return getStepNum(step, validCheck, GPUDataTypes::N_GENERAL_STEPS, "Invalid General Step"); }

void GPUReconstruction::RunPipelineWorker()
{
  if (!mInitialized || !mProcessingSettings.doublePipeline || mMaster != nullptr || !mSlaves.size()) {
    throw std::invalid_argument("Cannot start double pipeline mode");
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Pipeline worker started");
  }
  bool terminate = false;
  while (!terminate) {
    {
      std::unique_lock<std::mutex> lk(mPipelineContext->mutex);
      mPipelineContext->cond.wait(lk, [this] { return this->mPipelineContext->queue.size() > 0; });
    }
    GPUReconstructionPipelineQueue* q;
    {
      std::lock_guard<std::mutex> lk(mPipelineContext->mutex);
      q = mPipelineContext->queue.front();
      mPipelineContext->queue.pop();
    }
    if (q->op == 1) {
      terminate = 1;
    } else {
      q->retVal = q->chain->RunChain();
    }
    std::lock_guard<std::mutex> lk(q->m);
    q->done = true;
    q->c.notify_one();
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Pipeline worker ended");
  }
}

void GPUReconstruction::TerminatePipelineWorker()
{
  EnqueuePipeline(true);
}

int GPUReconstruction::EnqueuePipeline(bool terminate)
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
    rec->mPipelineContext->queue.push(q);
    rec->mPipelineContext->terminate = terminate;
    rec->mPipelineContext->cond.notify_one();
  }
  q->c.wait(lkdone, [&q]() { return q->done; });
  if (q->retVal) {
    return q->retVal;
  }
  return mChains[0]->FinalizePipelinedProcessing();
}

GPUChain* GPUReconstruction::GetNextChainInQueue()
{
  GPUReconstruction* rec = mMaster ? mMaster : this;
  std::lock_guard<std::mutex> lk(rec->mPipelineContext->mutex);
  return rec->mPipelineContext->queue.size() && rec->mPipelineContext->queue.front()->op == 0 ? rec->mPipelineContext->queue.front()->chain : nullptr;
}

void GPUReconstruction::PrepareEvent() // TODO: Clean this up, this should not be called from chainTracking but before
{
  ClearAllocatedMemory(true);
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->PrepareEvent();
  }
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
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

int GPUReconstruction::CheckErrorCodes(bool cpuOnly)
{
  int retVal = 0;
  for (unsigned int i = 0; i < mChains.size(); i++) {
    if (mChains[i]->CheckErrorCodes(cpuOnly)) {
      retVal++;
    }
  }
  return retVal;
}

void GPUReconstruction::DumpSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  DumpStructToFile(&mEventSettings, f.c_str());
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->DumpSettings(dir);
  }
}

void GPUReconstruction::UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsProcessing* p)
{
  param().UpdateEventSettings(e, p);
  if (mInitialized) {
    WriteConstantParams();
  }
}

int GPUReconstruction::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  new (&mEventSettings) GPUSettingsEvent;
  if (ReadStructFromFile(f.c_str(), &mEventSettings)) {
    return 1;
  }
  param().UpdateEventSettings(&mEventSettings);
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->ReadSettings(dir);
  }
  return 0;
}

void GPUReconstruction::SetSettings(float solenoidBz, const GPURecoStepConfiguration* workflow)
{
#ifdef GPUCA_O2_LIB
  GPUO2InterfaceConfiguration config;
  config.ReadConfigurableParam_internal();
  if (config.configEvent.solenoidBz <= -1e6f) {
    config.configEvent.solenoidBz = solenoidBz;
  }
  SetSettings(&config.configEvent, &config.configReconstruction, &config.configProcessing, workflow);
#else
  GPUSettingsEvent ev;
  ev.solenoidBz = solenoidBz;
  SetSettings(&ev, nullptr, nullptr, workflow);
#endif
}

void GPUReconstruction::SetSettings(const GPUSettingsEvent* settings, const GPUSettingsRec* rec, const GPUSettingsProcessing* proc, const GPURecoStepConfiguration* workflow)
{
  if (mInitialized) {
    GPUError("Cannot update settings while initialized");
    throw std::runtime_error("Settings updated while initialized");
  }
  mEventSettings = *settings;
  if (proc) {
    mProcessingSettings = *proc;
  }
  if (workflow) {
    mRecoSteps = workflow->steps;
    mRecoStepsGPU &= workflow->stepsGPUMask;
    mRecoStepsInputs = workflow->inputs;
    mRecoStepsOutputs = workflow->outputs;
  }
  param().SetDefaults(&mEventSettings, rec, proc, workflow);
}

void GPUReconstruction::SetOutputControl(void* ptr, size_t size)
{
  GPUOutputControl outputControl;
  outputControl.set(ptr, size);
  SetOutputControl(outputControl);
}

void GPUReconstruction::SetInputControl(void* ptr, size_t size)
{
  mInputControl.set(ptr, size);
}

GPUReconstruction::GPUThreadContext::GPUThreadContext() = default;
GPUReconstruction::GPUThreadContext::~GPUThreadContext() = default;

std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstruction::GetThreadContext() { return std::unique_ptr<GPUReconstruction::GPUThreadContext>(new GPUThreadContext); }

GPUReconstruction* GPUReconstruction::CreateInstance(DeviceType type, bool forceType, GPUReconstruction* master)
{
  GPUSettingsDeviceBackend cfg;
  new (&cfg) GPUSettingsDeviceBackend;
  cfg.deviceType = type;
  cfg.forceDeviceType = forceType;
  cfg.master = master;
  return CreateInstance(cfg);
}

GPUReconstruction* GPUReconstruction::CreateInstance(const GPUSettingsDeviceBackend& cfg)
{
  GPUReconstruction* retVal = nullptr;
  unsigned int type = cfg.deviceType;
  if (type == DeviceType::CPU) {
    retVal = GPUReconstruction_Create_CPU(cfg);
  } else if (type == DeviceType::CUDA) {
    if ((retVal = sLibCUDA->GetPtr(cfg))) {
      retVal->mMyLib = sLibCUDA;
    }
  } else if (type == DeviceType::HIP) {
    if ((retVal = sLibHIP->GetPtr(cfg))) {
      retVal->mMyLib = sLibHIP;
    }
  } else if (type == DeviceType::OCL) {
    if ((retVal = sLibOCL->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL;
    }
  } else if (type == DeviceType::OCL2) {
    if ((retVal = sLibOCL2->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL2;
    }
  } else {
    GPUError("Error: Invalid device type %u", type);
    return nullptr;
  }

  if (retVal == nullptr) {
    if (cfg.forceDeviceType) {
      GPUError("Error: Could not load GPUReconstruction for specified device: %s (%u)", DEVICE_TYPE_NAMES[type], type);
    } else {
      GPUError("Could not load GPUReconstruction for device type %s (%u), falling back to CPU version", DEVICE_TYPE_NAMES[type], type);
      GPUSettingsDeviceBackend cfg2 = cfg;
      cfg2.deviceType = DeviceType::CPU;
      retVal = CreateInstance(cfg2);
    }
  } else {
    GPUInfo("Created GPUReconstruction instance for device type %s (%u) %s", DEVICE_TYPE_NAMES[type], type, cfg.master ? " (slave)" : "");
  }

  return retVal;
}

GPUReconstruction* GPUReconstruction::CreateInstance(const char* type, bool forceType, GPUReconstruction* master)
{
  DeviceType t = GetDeviceType(type);
  if (t == DeviceType::INVALID_DEVICE) {
    GPUError("Invalid device type: %s", type);
    return nullptr;
  }
  return CreateInstance(t, forceType, master);
}

GPUReconstruction::DeviceType GPUReconstruction::GetDeviceType(const char* type)
{
  for (unsigned int i = 1; i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]); i++) {
    if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0) {
      return (DeviceType)i;
    }
  }
  return DeviceType::INVALID_DEVICE;
}

#ifdef _WIN32
#define LIBRARY_EXTENSION ".dll"
#define LIBRARY_TYPE HMODULE
#define LIBRARY_LOAD(name) LoadLibraryEx(name, nullptr, nullptr)
#define LIBRARY_CLOSE FreeLibrary
#define LIBRARY_FUNCTION GetProcAddress
#else
#define LIBRARY_EXTENSION ".so"
#define LIBRARY_TYPE void*
#define LIBRARY_LOAD(name) dlopen(name, RTLD_NOW)
#define LIBRARY_CLOSE dlclose
#define LIBRARY_FUNCTION dlsym
#endif

#if defined(GPUCA_ALIROOT_LIB)
#define LIBRARY_PREFIX "Ali"
#elif defined(GPUCA_O2_LIB)
#define LIBRARY_PREFIX "O2"
#else
#define LIBRARY_PREFIX ""
#endif

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibCUDA(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "CUDA" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "CUDA"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibHIP(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "HIP" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "HIP"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "OCL" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "OCL"));

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL2(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "OCL2" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "OCL2"));

GPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr) {}

GPUReconstruction::LibraryLoader::~LibraryLoader() { CloseLibrary(); }

int GPUReconstruction::LibraryLoader::LoadLibrary()
{
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);

  if (mGPUEntry) {
    return 0;
  }

  LIBRARY_TYPE hGPULib;
  hGPULib = LIBRARY_LOAD(mLibName);
  if (hGPULib == nullptr) {
#ifndef _WIN32
    GPUImportant("The following error occured during dlopen: %s", dlerror());
#endif
    GPUError("Error Opening cagpu library for GPU Tracker (%s)", mLibName);
    return 1;
  } else {
    void* createFunc = LIBRARY_FUNCTION(hGPULib, mFuncName);
    if (createFunc == nullptr) {
      GPUError("Error fetching entry function in GPU library\n");
      LIBRARY_CLOSE(hGPULib);
      return 1;
    } else {
      mGPULib = (void*)(size_t)hGPULib;
      mGPUEntry = createFunc;
      GPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
    }
  }
  return 0;
}

GPUReconstruction* GPUReconstruction::LibraryLoader::GetPtr(const GPUSettingsDeviceBackend& cfg)
{
  if (LoadLibrary()) {
    return nullptr;
  }
  if (mGPUEntry == nullptr) {
    return nullptr;
  }
  GPUReconstruction* (*tmp)(const GPUSettingsDeviceBackend& cfg) = (GPUReconstruction * (*)(const GPUSettingsDeviceBackend& cfg)) mGPUEntry;
  return tmp(cfg);
}

int GPUReconstruction::LibraryLoader::CloseLibrary()
{
  if (mGPUEntry == nullptr) {
    return 1;
  }
  LIBRARY_CLOSE((LIBRARY_TYPE)(size_t)mGPULib);
  mGPULib = nullptr;
  mGPUEntry = nullptr;
  return 0;
}
