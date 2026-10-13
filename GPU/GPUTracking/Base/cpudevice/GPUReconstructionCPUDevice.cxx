// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCPUDevice.cxx
/// \author David Rohr

#include "GPUReconstructionCPUDevice.h"
#include "GPUReconstructionCPUDeviceInternals.h"
#include "GPUReconstructionIncludes.h"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUDef.h"
#include "GPUConstantMem.h"
#include "GPUDefParametersRuntime.h"
#include "GPULogging.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace o2::gpu;

GPUReconstruction* GPUReconstruction_Create_CPUDEVICE(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCPUDevice(cfg); }

GPUReconstructionCPUDevice::GPUReconstructionCPUDevice(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing::KernelInterface<GPUReconstructionCPUDevice, GPUReconstructionDeviceBase>(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionCPUDeviceInternals;
    *mParDevice = *mParCPU;                     // We run the CPU kernels on the "device", so we use the CPU kernel parameters
    mParDevice->par_ALTERNATE_BORDER_SORT = 1;  // Required by the GPU code path without RTC
    mParDevice->par_CF_SCAN_WORKGROUP_SIZE = 1; // No block-wide scan with 1 thread per block, the clusterizer uses the CPU stream compaction
    for (auto& v : mParDevice->par_LB_maxThreads) {
      v = 1;
    }
  }
  mDeviceBackendSettings->deviceType = DeviceType::CPUDEVICE;
}

GPUReconstructionCPUDevice::~GPUReconstructionCPUDevice()
{
  Exit();
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

int32_t GPUReconstructionCPUDevice::GPUChkErrInternal(const int64_t retval, const char* file, int32_t line) const
{
  if (retval) {
    GPUError("CPUDevice Error: %ld (%s:%d)", (int64_t)retval, file, line);
  }
  return retval != 0;
}

int32_t GPUReconstructionCPUDevice::InitDevice_Runtime()
{
  if (GetProcessingSettings().rtc.enable && GetProcessingSettings().rtctech.runTest == 2) {
    genAndLoadRTC();
    exit(0);
  }

  if (mMaster == nullptr) {
    mMultiprocessorCount = std::max<int32_t>(1, mMaxHostThreads); // One device block runs as one TBB task
    mThreadCount = 1;
    mWarpSize = 1;
    mMaxBackendThreads = std::max<int32_t>(mMaxBackendThreads, mMultiprocessorCount);
    mDeviceName = "CPUDevice";
    if (GetProcessingSettings().deviceTimers) {
      if (GetProcessingSettings().debugLevel >= 2) {
        GPUInfo("Disabling device timers for CPUDevice");
      }
      mProcessingSettings->deviceTimers = 0;
    }

    // The arena does not own threads, it limits the concurrency of the command queues and kernels to the configured number of host threads
    mInternals->arena = std::make_unique<tbb::task_arena>(std::max<int32_t>(1, mMaxHostThreads));

    mInternals->deviceMemory = alignedDefaultBufferAllocator<char>(mDeviceMemorySize);
    mInternals->hostMemory = alignedDefaultBufferAllocator<char>(mHostMemorySize);
    mInternals->constantMemory = alignedDefaultBufferAllocator<char>(sizeof(GPUConstantMem));
    if ((mDeviceMemorySize && mInternals->deviceMemory == nullptr) || (mHostMemorySize && mInternals->hostMemory == nullptr) || mInternals->constantMemory == nullptr) {
      GPUError("CPUDevice memory allocation error (trying %ld bytes device / %ld bytes host memory)", (int64_t)mDeviceMemorySize, (int64_t)mHostMemorySize);
      return 1;
    }
    mDeviceMemoryBase = mInternals->deviceMemory;
    mHostMemoryBase = mInternals->hostMemory;
    mDeviceConstantMem = (GPUConstantMem*)mInternals->constantMemory;
    if (GetProcessingSettings().debugLevel >= 1) {
      GPUInfo("Memory ptrs: Device (%ld bytes): %p - Host (%ld bytes): %p", (int64_t)mDeviceMemorySize, mDeviceMemoryBase, (int64_t)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
      memset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize);
    }

    if (GetProcessingSettings().rtc.enable) {
      genAndLoadRTC();
    }

    GPUInfo("CPUDevice Initialisation successfull (%d blocks, %d streams, %ld / %ld bytes host / device memory, Constant memory %ld)", mMultiprocessorCount, mNStreams, (int64_t)mHostMemorySize, (int64_t)mDeviceMemorySize, (int64_t)sizeof(GPUConstantMem));
  } else {
    GPUReconstructionCPUDevice* master = dynamic_cast<GPUReconstructionCPUDevice*>(mMaster);
    mMultiprocessorCount = master->mMultiprocessorCount;
    mThreadCount = master->mThreadCount;
    mWarpSize = master->mWarpSize;
    mMaxBackendThreads = master->mMaxBackendThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mInternals = master->mInternals;
    GPUInfo("CPUDevice Initialisation successfull (from master)");
  }

  mEventStorage.reserve(mEvents.size());
  for (uint32_t i = 0; i < mEvents.size(); i++) {
    mEventStorage.emplace_back(mEvents[i].size());
    for (uint32_t j = 0; j < mEvents[i].size(); j++) {
      mEvents[i][j].set(&mEventStorage[i][j]);
    }
  }

  return 0;
}

int32_t GPUReconstructionCPUDevice::ExitDevice_Runtime()
{
  SynchronizeGPU();
  mEventStorage.clear();

  if (mMaster == nullptr) {
#ifndef _WIN32
    for (auto* lib : mInternals->rtcLibraries) {
      dlclose(lib);
    }
#endif
    mInternals->rtcLibraries.clear();
    mInternals->rtcKernels.clear();
    mInternals->arena.reset();
    alignedDefaultBufferDeleter()(mInternals->deviceMemory);
    alignedDefaultBufferDeleter()(mInternals->hostMemory);
    alignedDefaultBufferDeleter()(mInternals->constantMemory);
    mInternals->deviceMemory = mInternals->hostMemory = mInternals->constantMemory = nullptr;
    GPUInfo("CPUDevice Uninitialized");
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;
  mDeviceConstantMem = nullptr;

  return 0;
}

GPUReconstructionCPUDeviceEvent* GPUReconstructionCPUDevice::getEvent(deviceEvent& ev)
{
  return ev.get<GPUReconstructionCPUDeviceEvent*>();
}

void GPUReconstructionCPUDeviceInternals::enqueue(int32_t stream, GPUReconstructionCPUDeviceTask&& task)
{
  auto& q = queues[stream];
  bool start = false;
  {
    std::lock_guard lock(q.mutex);
    q.tasks.emplace_back(std::move(task));
    if (!q.active) {
      q.active = start = true;
    }
  }
  if (start) {
    startQueue(stream);
  }
}

void GPUReconstructionCPUDeviceInternals::startQueue(int32_t stream)
{
  arena->enqueue([this, stream]() { processQueue(stream); });
}

void GPUReconstructionCPUDeviceInternals::processQueue(int32_t stream)
{
  auto& q = queues[stream];
  while (true) {
    GPUReconstructionCPUDeviceTask* task;
    {
      std::lock_guard lock(q.mutex);
      if (q.tasks.empty()) {
        q.active = false;
        q.cv.notify_all();
        return;
      }
      task = &q.tasks.front(); // Only this TBB task removes elements, and push_back does not invalidate references into a deque
    }
    while (task->nextWait < task->waits.size()) {
      auto [e, value] = task->waits[task->nextWait];
      std::lock_guard lock(e->mutex);
      if (e->nCompleted < value) { // Do not block the TBB thread, suspend the queue, completeEvent will start it again
        e->waiting.push_back({value, this, stream});
        return;
      }
      task->nextWait++;
    }
    if (!error) {
      try {
        task->work();
      } catch (const std::exception& ex) {
        GPUError("CPUDevice: Error in command queue %d: %s", stream, ex.what());
        error = 1;
      }
    }
    if (task->record) {
      completeEvent(task->record, task->recordValue); // Also after errors, so that nothing waits forever
    }
    std::lock_guard lock(q.mutex);
    q.tasks.pop_front();
  }
}

void GPUReconstructionCPUDeviceInternals::completeEvent(GPUReconstructionCPUDeviceEvent* e, uint64_t value)
{
  std::vector<GPUReconstructionCPUDeviceEvent::waitingQueue> resume;
  {
    std::lock_guard lock(e->mutex);
    e->nCompleted = std::max(e->nCompleted, value);
    for (auto it = e->waiting.begin(); it != e->waiting.end();) {
      if (it->value <= e->nCompleted) {
        resume.emplace_back(*it);
        it = e->waiting.erase(it);
      } else {
        it++;
      }
    }
    e->cv.notify_all();
  }
  for (auto& w : resume) {
    w.internals->startQueue(w.stream);
  }
}

void GPUReconstructionCPUDevice::EnqueueTask(int32_t stream, deviceEvent* evList, int32_t nEvents, deviceEvent* ev, std::function<void()>&& work)
{
  if (stream < 0 || stream >= mNStreams) {
    throw std::runtime_error("CPUDevice: invalid stream " + std::to_string(stream));
  }
  GPUReconstructionCPUDeviceTask task;
  if (evList) { // Capture the event states at enqueue time, such that the queue waits for the markers recorded so far, like cudaStreamWaitEvent
    for (int32_t k = 0; k < nEvents; k++) {
      auto* e = getEvent(evList[k]);
      std::lock_guard lock(e->mutex);
      task.waits.emplace_back(e, e->nRecorded);
    }
  }
  if (ev) {
    task.record = getEvent(*ev);
    std::lock_guard lock(task.record->mutex);
    task.recordValue = ++task.record->nRecorded;
  }
  task.work = std::move(work);
  mInternals->enqueue(stream, std::move(task));
}

void GPUReconstructionCPUDevice::SynchronizeStream(int32_t stream)
{
  auto& q = mInternals->queues[stream];
  std::unique_lock lock(q.mutex);
  q.cv.wait(lock, [&q]() { return !q.active; });
}

void GPUReconstructionCPUDevice::SynchronizeGPU()
{
  for (uint32_t i = 0; i < mInternals->queues.size(); i++) {
    SynchronizeStream(i);
  }
}

void GPUReconstructionCPUDevice::SynchronizeEvents(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    auto* e = getEvent(evList[i]);
    std::unique_lock lock(e->mutex);
    const uint64_t value = e->nRecorded;
    e->cv.wait(lock, [e, value]() { return e->nCompleted >= value; });
  }
}

void GPUReconstructionCPUDevice::StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents)
{
  if (nEvents) {
    EnqueueTask(stream, evList, nEvents, nullptr, []() {});
  }
}

bool GPUReconstructionCPUDevice::IsEventDone(deviceEvent* evList, int32_t nEvents)
{
  for (int32_t i = 0; i < nEvents; i++) {
    auto* e = getEvent(evList[i]);
    std::lock_guard lock(e->mutex);
    if (e->nCompleted < e->nRecorded) {
      return false;
    }
  }
  return true;
}

void GPUReconstructionCPUDevice::RecordMarker(deviceEvent* ev, int32_t stream)
{
  EnqueueTask(stream, nullptr, 0, ev, []() {});
}

int32_t GPUReconstructionCPUDevice::GPUDebug(const char* state, int32_t stream, bool force)
{
  if (mInternals->error) {
    GPUError("CPUDevice Error while running (%s) (Stream %d)", state, stream);
    return 1;
  }
  if (!force && GetProcessingSettings().debugLevel <= 0) {
    return 0;
  }
  if (stream == -1) {
    SynchronizeGPU();
  } else {
    SynchronizeStream(stream);
  }
  if (mInternals->error) {
    GPUError("CPUDevice Error while synchronizing (%s) (Stream %d)", state, stream);
    return 1;
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return 0;
}

size_t GPUReconstructionCPUDevice::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (GetProcessingSettings().debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
    if (size) {
      memcpy(dst, src, size);
    }
    if (ev) {
      auto* e = getEvent(*ev);
      uint64_t value;
      {
        std::lock_guard lock(e->mutex);
        value = ++e->nRecorded;
      }
      mInternals->completeEvent(e, value);
    }
  } else {
    EnqueueTask(stream, evList, evList ? nEvents : 0, ev, [dst, src, size]() {
      if (size) {
        memcpy(dst, src, size);
      }
    });
  }
  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug(("GPUMemCpy " + std::to_string(toGPU)).c_str(), stream, true);
  }
  return size;
}

size_t GPUReconstructionCPUDevice::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev)
{
  char* dst = (char*)mDeviceConstantMem + offset;
  if (stream == -1) {
    SynchronizeGPU();
    memcpy(dst, src, size);
  } else {
    // The source may be a temporary, the CUDA backend stages the copy too, so we take a copy for the asynchronous transfer
    auto buffer = std::make_shared<std::vector<char>>((const char*)src, (const char*)src + size);
    EnqueueTask(stream, nullptr, 0, ev, [dst, buffer]() { memcpy(dst, buffer->data(), buffer->size()); });
  }
  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug("WriteToConstantMemory", stream, true);
  }
  return size;
}

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionCPUDevice::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  auto& z = _xyz.z;
  if (x.nThreads != 1) {
    throw std::runtime_error("CPUDevice backend supports only kernels with nThreads = 1");
  }
  const uint32_t nBlocks = x.nBlocks;
  const int32_t iSector = y.index;
  GPUConstantMem* constantMem = mDeviceConstantMem;
  using rtcKernel = void (*)(GPUConstantMem*, int32_t, uint32_t, uint32_t, uint32_t, Args...);
  rtcKernel rtc = mInternals->rtcKernels.size() ? reinterpret_cast<rtcKernel>(mInternals->rtcKernels[GetKernelNum<T, I>()]) : nullptr;

  // The kernel arguments are captured by value, since the kernel runs asynchronously in the command queue
  EnqueueTask(x.stream, z.evList, z.evList ? z.nEvents : 0, z.ev, [nBlocks, iSector, constantMem, rtc, ... args = args]() {
    tbb::this_task_arena::isolate([&] { // Do not pick up tasks of other command queues while waiting inside parallel_for
      tbb::parallel_for(tbb::blocked_range<uint32_t>(0, nBlocks, 1), [&](const tbb::blocked_range<uint32_t>& r) {
        if (rtc) {
          rtc(constantMem, iSector, nBlocks, r.begin(), r.end(), args...);
        } else {
          for (uint32_t iBlock = r.begin(); iBlock < r.end(); iBlock++) {
            typename T::GPUSharedMemory smem;
            T::template Thread<I>(nBlocks, 1, iBlock, 0, smem, T::Processor(*constantMem)[iSector], args...);
          }
        }
      });
    });
  });
}

template <>
inline void GPUReconstructionCPUDevice::runKernelBackend<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  EnqueueTask(_xyz.x.stream, _xyz.z.evList, _xyz.z.evList ? _xyz.z.nEvents : 0, _xyz.z.ev, [ptr, size]() { memset(ptr, 0, size); });
}

#include "GPUReconstructionProcessingKernels.inc"
template void GPUReconstructionProcessing::KernelInterface<GPUReconstructionCPUDevice, GPUReconstructionDeviceBase>::runKernelVirtual(const int num, const void* args);

// Entry point of an RTC-compiled kernel, running the blocks [iBlockStart, iBlockEnd) with one thread per block. Must match the rtcKernel function pointer type in GPUReconstructionCPUDevice.cxx.
#define GPUCA_CPUDEVICE_KRNL_RTC(x_class, x_attributes, x_arguments, x_forward, x_types, ...)                                                                                                                                                         \
  extern "C" __attribute__((visibility("default"))) void GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))(GPUConstantMem * pConstant, int32_t _iSector_internal, uint32_t nBlocks, uint32_t iBlockStart, uint32_t iBlockEnd GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                                                                                                                                                   \
    for (uint32_t iBlock = iBlockStart; iBlock < iBlockEnd; iBlock++) {                                                                                                                                                                               \
      typename GPUCA_M_STRIP_FIRST(x_class)::GPUSharedMemory smem;                                                                                                                                                                                    \
      GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(nBlocks, 1, iBlock, 0, smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(*pConstant)[_iSector_internal] GPUCA_M_STRIP(x_forward));                                         \
    }                                                                                                                                                                                                                                                 \
  }

void GPUReconstructionCPUDevice::getRTCKernelCalls(std::vector<std::string>& kernels)
{
#define GPUCA_KRNL(...) kernels.emplace_back(GPUCA_M_STR(GPUCA_CPUDEVICE_KRNL_RTC(__VA_ARGS__)));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
}
