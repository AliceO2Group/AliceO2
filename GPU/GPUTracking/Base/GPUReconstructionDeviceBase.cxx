// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionDeviceBase.cxx
/// \author David Rohr

#include "GPUReconstructionDeviceBase.h"
#include "GPUReconstructionIncludes.h"

#include "GPUTPCTracker.h"
#include "GPUTPCSliceOutput.h"

using namespace GPUCA_NAMESPACE::gpu;

#if defined(_WIN32)
#include "../utils/pthread_mutex_win32_wrapper.h"
#else
#include <cerrno>
#include <unistd.h>
#endif
#include <cstring>

MEM_CLASS_PRE()
class GPUTPCRow;

#define SemLockName "AliceHLTTPCGPUTrackerInitLockSem"

GPUReconstructionDeviceBase::GPUReconstructionDeviceBase(const GPUSettingsProcessing& cfg) : GPUReconstructionCPU(cfg)
{
}

GPUReconstructionDeviceBase::~GPUReconstructionDeviceBase()
{
  // make d'tor such that vtable is created for this class
  // needed for build with AliRoot, otherwise dynamic loading of GPU libraries will fail
  (void)0; // Avoid compiler warnings
}

void* GPUReconstructionDeviceBase::helperWrapper_static(void* arg)
{
  GPUReconstructionHelpers::helperParam* par = (GPUReconstructionHelpers::helperParam*)arg;
  GPUReconstructionDeviceBase* cls = par->cls;
  return cls->helperWrapper(par);
}

void* GPUReconstructionDeviceBase::helperWrapper(GPUReconstructionHelpers::helperParam* par)
{
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("\tHelper thread %d starting", par->num);
  }

  // cpu_set_t mask; //TODO add option
  // CPU_ZERO(&mask);
  // CPU_SET(par->num * 2 + 2, &mask);
  // sched_setaffinity(0, sizeof(mask), &mask);

  par->mutex[0].lock();
  while (par->terminate == false) {
    for (int i = par->num + 1; i < par->count; i += mDeviceProcessingSettings.nDeviceHelperThreads + 1) {
      // if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Running, Slice %d+%d, Phase %d", par->num, i, par->phase);
      if ((par->functionCls->*par->function)(i, par->num + 1, par)) {
        par->error = 1;
      }
      if (par->reset) {
        break;
      }
      par->done = i + 1;
      // if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Finished, Slice %d+%d, Phase %d", par->num, i, par->phase);
    }
    ResetThisHelperThread(par);
    par->mutex[0].lock();
  }
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("\tHelper thread %d terminating", par->num);
  }
  par->mutex[1].unlock();
  pthread_exit(nullptr);
  return (nullptr);
}

void GPUReconstructionDeviceBase::ResetThisHelperThread(GPUReconstructionHelpers::helperParam* par)
{
  if (par->reset) {
    GPUImportant("GPU Helper Thread %d reseting", par->num);
  }
  par->reset = false;
  par->mutex[1].unlock();
}

int GPUReconstructionDeviceBase::GetGlobalLock(void*& pLock)
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

void GPUReconstructionDeviceBase::ResetHelperThreads(int helpers)
{
  GPUImportant("Error occurred, GPU tracker helper threads will be reset (Number of threads %d (%d))", mDeviceProcessingSettings.nDeviceHelperThreads, mNSlaveThreads);
  SynchronizeGPU();
  for (int i = 0; i < mDeviceProcessingSettings.nDeviceHelperThreads; i++) {
    mHelperParams[i].reset = true;
    if (helpers || i >= mDeviceProcessingSettings.nDeviceHelperThreads) {
      pthread_mutex_lock(&((pthread_mutex_t*)mHelperParams[i].mutex)[1]);
    }
  }
  GPUImportant("GPU Tracker helper threads have ben reset");
}

int GPUReconstructionDeviceBase::StartHelperThreads()
{
  int nThreads = mDeviceProcessingSettings.nDeviceHelperThreads;
  if (nThreads) {
    mHelperParams = new GPUReconstructionHelpers::helperParam[nThreads];
    if (mHelperParams == nullptr) {
      GPUError("Memory allocation error");
      ExitDevice();
      return (1);
    }
    for (int i = 0; i < nThreads; i++) {
      mHelperParams[i].cls = this;
      mHelperParams[i].terminate = false;
      mHelperParams[i].reset = false;
      mHelperParams[i].num = i;
      for (int j = 0; j < 2; j++) {
        mHelperParams[i].mutex[j].lock();
      }

      if (pthread_create(&mHelperParams[i].threadId, nullptr, helperWrapper_static, &mHelperParams[i])) {
        GPUError("Error starting slave thread");
        ExitDevice();
        return (1);
      }
    }
  }
  mNSlaveThreads = nThreads;
  return (0);
}

int GPUReconstructionDeviceBase::StopHelperThreads()
{
  if (mNSlaveThreads) {
    for (int i = 0; i < mNSlaveThreads; i++) {
      mHelperParams[i].terminate = true;
      mHelperParams[i].mutex[0].unlock();
      mHelperParams[i].mutex[1].lock();
      if (pthread_join(mHelperParams[i].threadId, nullptr)) {
        GPUError("Error waiting for thread to terminate");
        return (1);
      }
    }
    delete[] mHelperParams;
  }
  mNSlaveThreads = 0;
  return (0);
}

void GPUReconstructionDeviceBase::WaitForHelperThreads()
{
  for (int i = 0; i < mDeviceProcessingSettings.nDeviceHelperThreads; i++) {
    pthread_mutex_lock(&((pthread_mutex_t*)mHelperParams[i].mutex)[1]);
  }
}

void GPUReconstructionDeviceBase::RunHelperThreads(int (GPUReconstructionHelpers::helperDelegateBase::*function)(int i, int t, GPUReconstructionHelpers::helperParam* p), GPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
{
  for (int i = 0; i < mDeviceProcessingSettings.nDeviceHelperThreads; i++) {
    mHelperParams[i].done = 0;
    mHelperParams[i].error = 0;
    mHelperParams[i].function = function;
    mHelperParams[i].functionCls = functionCls;
    mHelperParams[i].count = count;
    pthread_mutex_unlock(&((pthread_mutex_t*)mHelperParams[i].mutex)[0]);
  }
}

int GPUReconstructionDeviceBase::InitDevice()
{
  // cpu_set_t mask;
  // CPU_ZERO(&mask);
  // CPU_SET(0, &mask);
  // sched_setaffinity(0, sizeof(mask), &mask);

  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    GPUError("Individual memory allocation strategy unsupported for device\n");
    return (1);
  }
  if (mDeviceProcessingSettings.nStreams > GPUCA_MAX_STREAMS) {
    GPUError("Too many straems requested %d > %d\n", mDeviceProcessingSettings.nStreams, GPUCA_MAX_STREAMS);
    return (1);
  }
  mThreadId = GetThread();

  void* semLock = nullptr;
  if (mDeviceProcessingSettings.globalInitMutex && GetGlobalLock(semLock)) {
    return (1);
  }

  int retVal = InitDevice_Runtime();
  if (retVal) {
    GPUImportant("GPU Tracker initialization failed");
    return (1);
  }

  if (mDeviceProcessingSettings.globalInitMutex) {
    ReleaseGlobalLock(semLock);
  }

  mDeviceMemoryPermanent = mDeviceMemoryBase;
  mHostMemoryPermanent = mHostMemoryBase;
  ClearAllocatedMemory();

  mProcShadow.InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_SLAVE);
  mProcShadow.mMemoryResProcessors = RegisterMemoryAllocation(&mProcShadow, &GPUProcessorProcessors::SetPointersDeviceProcessor, GPUMemoryResource::MEMORY_PERMANENT | GPUMemoryResource::MEMORY_HOST, "Processors");
  AllocateRegisteredMemory(mProcShadow.mMemoryResProcessors);

  if (StartHelperThreads()) {
    return (1);
  }

  SetThreadCounts();

  GPUInfo("GPU Tracker initialization successfull"); // Verbosity reduced because GPU backend will print GPUImportant message!

  return (retVal);
}

void* GPUReconstructionDeviceBase::GPUProcessorProcessors::SetPointersDeviceProcessor(void* mem)
{
  // Don't run constructor / destructor here, this will be just local memcopy of Processors in GPU Memory
  computePointerWithAlignment(mem, mProcessorsProc, 1);
  return mem;
}

int GPUReconstructionDeviceBase::ExitDevice()
{
  if (StopHelperThreads()) {
    return (1);
  }

  int retVal = ExitDevice_Runtime();
  mProcessorsShadow = nullptr;
  mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
  mHostMemorySize = mDeviceMemorySize = 0;

  return retVal;
}

int GPUReconstructionDeviceBase::GetMaxThreads()
{
  int retVal = mTRDThreadCount * mBlockCount;
  return std::max(retVal, GPUReconstruction::GetMaxThreads());
}
