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
#include "GPUTPCSliceOutTrack.h"
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

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace GPUCA_NAMESPACE::gpu;

constexpr GPUReconstructionCPU::krnlRunRange GPUReconstructionCPU::krnlRunRangeNone;
constexpr GPUReconstructionCPU::krnlEvent GPUReconstructionCPU::krnlEventNone;

GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsProcessing& cfg) { return new GPUReconstructionCPU(cfg); }

GPUReconstructionCPU::~GPUReconstructionCPU()
{
  Exit(); // Needs to be identical to GPU backend bahavior in order to avoid calling abstract methods later in the destructor
}

template <class T, int I, typename... Args>
int GPUReconstructionCPUBackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
  if (x.device == krnlDeviceType::Device) {
    throw std::runtime_error("Cannot run device kernel on host");
  }
  unsigned int num = y.num == 0 || y.num == -1 ? 1 : y.num;
  for (unsigned int k = 0; k < num; k++) {
    for (unsigned int iB = 0; iB < x.nBlocks; iB++) {
      typename T::GPUTPCSharedMemory smem;
      T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.start + k], args...);
    }
  }
  return 0;
}

void GPUReconstructionCPU::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) {}
void GPUReconstructionCPU::GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents) {}
void GPUReconstructionCPU::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents) { memcpy(dst, src, size); }
void GPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) {}
int GPUReconstructionCPU::GPUDebug(const char* state, int stream) { return 0; }
void GPUReconstructionCPU::TransferMemoryResourcesHelper(GPUProcessor* proc, int stream, bool all, bool toGPU)
{
  int inc = toGPU ? GPUMemoryResource::MEMORY_INPUT_FLAG : GPUMemoryResource::MEMORY_OUTPUT_FLAG;
  int exc = toGPU ? GPUMemoryResource::MEMORY_OUTPUT_FLAG : GPUMemoryResource::MEMORY_INPUT_FLAG;
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
    if (!mDeviceProcessingSettings.keepAllMemory && !all && (res.mType & exc) && !(res.mType & inc)) {
      continue;
    }
    if (toGPU) {
      TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
    } else {
      TransferMemoryResourceToHost(&mMemoryResources[i], stream);
    }
  }
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
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    mHostMemoryPermanent = mHostMemoryBase = operator new(GPUCA_HOST_MEMORY_SIZE);
    mHostMemorySize = GPUCA_HOST_MEMORY_SIZE;
    ClearAllocatedMemory();
  }
  SetThreadCounts();
  mThreadId = GetThread();
  return 0;
}

int GPUReconstructionCPU::ExitDevice()
{
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    operator delete(mHostMemoryBase);
    mHostMemoryPool = mHostMemoryBase = mHostMemoryPermanent = nullptr;
    mHostMemorySize = 0;
  }
  return 0;
}

void GPUReconstructionCPU::SetThreadCounts() { mThreadCount = mBlockCount = mConstructorBlockCount = mSelectorBlockCount = mConstructorThreadCount = mSelectorThreadCount = mFinderThreadCount = mTRDThreadCount = 1; }

void GPUReconstructionCPU::SetThreadCounts(RecoStep step)
{
  if (IsGPU() && mRecoSteps != mRecoStepsGPU) {
    if (!(mRecoStepsGPU & step)) {
      GPUReconstructionCPU::SetThreadCounts();
    } else {
      SetThreadCounts();
    }
  }
}

int GPUReconstructionCPU::RunChains()
{
  mStatNEvents++;

  if (mThreadId != GetThread()) {
    if (mDeviceProcessingSettings.debugLevel >= 2) {
      GPUInfo("Thread changed, migrating context, Previous Thread: %d, New Thread: %d", mThreadId, GetThread());
    }
    mThreadId = GetThread();
  }

  for (unsigned int i = 0; i < mChains.size(); i++) {
    int retVal = mChains[i]->RunChain();
    if (retVal) {
      return retVal;
    }
  }

  return 0;
}

void GPUReconstructionCPU::ResetDeviceProcessorTypes()
{
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mDeviceProcessor) {
      mProcessors[i].proc->mDeviceProcessor->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
    }
  }
}
