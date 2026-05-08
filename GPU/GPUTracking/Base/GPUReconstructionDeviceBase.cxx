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

/// \file GPUReconstructionDeviceBase.cxx
/// \author David Rohr

#include "GPUReconstructionDeviceBase.h"
#include "GPUReconstructionIncludes.h"
#include "GPUConstantMem.h"

#include "GPUTPCTracker.h"

using namespace o2::gpu;

#if defined(_WIN32)
#include "../utils/pthread_mutex_win32_wrapper.h"
#else
#include <cerrno>
#include <unistd.h>
#endif
#include <cstring>

class GPUTPCRow;

#define SemLockName "AliceHLTTPCGPUTrackerInitLockSem"

GPUReconstructionDeviceBase::GPUReconstructionDeviceBase(const GPUSettingsDeviceBackend& cfg, size_t sizeCheck) : GPUReconstructionCPU(cfg)
{
  if (sizeCheck != sizeof(GPUReconstructionDeviceBase)) {
    GPUFatal("Mismatch of C++ object size between GPU compilers!");
  }
}

GPUReconstructionDeviceBase::~GPUReconstructionDeviceBase() = default;

int32_t GPUReconstructionDeviceBase::GetGlobalLock(void*& pLock)
{
#ifdef _WIN32
  HANDLE* semLock = new HANDLE;
  *semLock = CreateSemaphore(nullptr, 1, 1, SemLockName);
  if (*semLock == nullptr) {
    GPUError("Error creating GPUInit Semaphore");
    return (1);
  }
  WaitForSingleObject(*semLock, INFINITE);
#elif !defined(__APPLE__) // GPU not supported on MacOS anyway
  sem_t* semLock = sem_open(SemLockName, O_CREAT, 0x01B6, 1);
  if (semLock == SEM_FAILED) {
    GPUError("Error creating GPUInit Semaphore");
    return (1);
  }
  timespec semtime;
  clock_gettime(CLOCK_REALTIME, &semtime);
  semtime.tv_sec += 10;
  while (sem_timedwait(semLock, &semtime) != 0) {
    GPUError("Global Lock for GPU initialisation was not released for 10 seconds, assuming another thread died");
    GPUWarning("Resetting the global lock");
    sem_post(semLock);
  }
#else
  void* semLock = nullptr;
#endif
  pLock = semLock;
  return 0;
}

void GPUReconstructionDeviceBase::ReleaseGlobalLock(void* sem)
{
// Release the global named semaphore that locks GPU Initialization
#ifdef _WIN32
  HANDLE* h = (HANDLE*)sem;
  ReleaseSemaphore(*h, 1, nullptr);
  CloseHandle(*h);
  delete h;
#elif !defined(__APPLE__) // GPU not supported on MacOS anyway
  sem_t* pSem = (sem_t*)sem;
  sem_post(pSem);
  sem_unlink(SemLockName);
#endif
}

int32_t GPUReconstructionDeviceBase::InitDevice()
{
  // cpu_set_t mask;
  // CPU_ZERO(&mask);
  // CPU_SET(0, &mask);
  // sched_setaffinity(0, sizeof(mask), &mask);

  if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    GPUError("Individual memory allocation strategy unsupported for device\n");
    return (1);
  }
  if ((size_t)GetProcessingSettings().nStreams > constants::GPU_MAX_STREAMS) {
    GPUError("Too many straems requested %d > %d\n", GetProcessingSettings().nStreams, constants::GPU_MAX_STREAMS);
    return (1);
  }

  void* semLock = nullptr;
  if (GetProcessingSettings().globalInitMutex && GetGlobalLock(semLock)) {
    return (1);
  }

  if (GetProcessingSettings().deviceTimers) {
    AddGPUEvents(mDebugEvents);
  }

  int32_t retVal = InitDevice_Runtime();
  if (retVal) {
    GPUImportant("GPU Tracker initialization failed");
    return (1);
  }

  if (GetProcessingSettings().globalInitMutex) {
    ReleaseGlobalLock(semLock);
  }

  mDeviceMemoryPermanent = mDeviceMemoryBase;
  mHostMemoryPermanent = mHostMemoryBase;
  ClearAllocatedMemory();

  mProcShadow.InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_SLAVE);
  mProcShadow.mMemoryResProcessors = RegisterMemoryAllocation(&mProcShadow, &GPUProcessorProcessors::SetPointersDeviceProcessor, GPUMemoryResource::MEMORY_PERMANENT | GPUMemoryResource::MEMORY_HOST, "Processors");
  AllocateRegisteredMemory(mProcShadow.mMemoryResProcessors);

  if (mMaster == nullptr || GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU Tracker initialization successfull"); // Verbosity reduced because GPU backend will print GPUImportant message!
  }

  return (retVal);
}

void* GPUReconstructionDeviceBase::GPUProcessorProcessors::SetPointersDeviceProcessor(void* mem)
{
  // Don't run constructor / destructor here, this will be just local memcopy of Processors in GPU Memory
  computePointerWithAlignment(mem, mProcessorsProc, 1);
  return mem;
}

int32_t GPUReconstructionDeviceBase::ExitDevice()
{
  int32_t retVal = ExitDevice_Runtime();
  mProcessorsShadow = nullptr;
  mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPoolEnd = mDeviceMemoryPoolEnd = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
  mHostMemorySize = mDeviceMemorySize = 0;

  return retVal;
}

int32_t GPUReconstructionDeviceBase::registerMemoryForGPU_internal(const void* ptr, size_t size)
{
  return IsGPU();
}

int32_t GPUReconstructionDeviceBase::unregisterMemoryForGPU_internal(const void* ptr)
{
  return IsGPU();
}

void GPUReconstructionDeviceBase::unregisterRemainingRegisteredMemory()
{
  for (auto& ptr : mRegisteredMemoryPtrs) {
    unregisterMemoryForGPU_internal(ptr);
  }
  mRegisteredMemoryPtrs.clear();
}

void GPUReconstructionDeviceBase::runConstantRegistrators()
{
  auto& list = getDeviceConstantMemRegistratorsVector();
  for (uint32_t i = 0; i < list.size(); i++) {
    auto* ptr = list[i]();
    if (ptr == nullptr) {
      GPUFatal("Error registering constant memory");
    }
    mDeviceConstantMemList.emplace_back(ptr);
  }
}

size_t GPUReconstructionDeviceBase::TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (GetProcessingSettings().debugLevel >= 4) {
      GPUInfo("Skipped transfer of non-GPU memory resource: %s", res->Name());
    }
    return 0;
  }
  if (GetProcessingSettings().debugLevel >= 3 && (strcmp(res->Name(), "ErrorCodes") || GetProcessingSettings().debugLevel >= 4)) {
    GPUInfo("Copying to %s: %s - %ld bytes", toGPU ? "GPU" : "Host", res->Name(), (int64_t)res->Size());
  }
  return GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

const GPUParam* GPUReconstructionDeviceBase::DeviceParam() const { return &mDeviceConstantMem->param; }
