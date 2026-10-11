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

/// \file GPUReconstructionCPUDeviceInternals.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCPUDEVICEINTERNALS_H
#define GPURECONSTRUCTIONCPUDEVICEINTERNALS_H

#include "GPUReconstructionCPUDevice.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <oneapi/tbb.h>

namespace o2::gpu
{
struct GPUReconstructionCPUDeviceInternals;

// Event of the CPUDevice backend. Recording a marker increments nRecorded when the marker is enqueued, the command queue sets nCompleted when the marker is processed. A wait captures the value of nRecorded at the time the wait is enqueued, so it behaves like cudaStreamWaitEvent.
struct GPUReconstructionCPUDeviceEvent {
  struct waitingQueue {
    uint64_t value;
    GPUReconstructionCPUDeviceInternals* internals;
    int32_t stream;
  };
  std::mutex mutex;
  std::condition_variable cv;        // For host threads waiting for the event
  std::vector<waitingQueue> waiting; // Suspended command queues, which are resumed when the event reaches the value
  uint64_t nRecorded = 0;
  uint64_t nCompleted = 0;
};

// Command in a command queue: wait for events, run the work, mark an event as completed
struct GPUReconstructionCPUDeviceTask {
  std::vector<std::pair<GPUReconstructionCPUDeviceEvent*, uint64_t>> waits;
  uint32_t nextWait = 0;
  std::function<void()> work;
  GPUReconstructionCPUDeviceEvent* record = nullptr;
  uint64_t recordValue = 0;
};

// In-order command queue. It does not own a thread: When commands are enqueued, a TBB task is started which processes them, and which returns once the queue is empty, or when the next command must wait for an event. In the latter case, the queue is registered at the event,
// and a new TBB task is started when the event completes. At most one TBB task processes a queue at a time.
struct GPUReconstructionCPUDeviceQueue {
  std::mutex mutex;
  std::condition_variable cv; // For host threads synchronizing the queue
  std::deque<GPUReconstructionCPUDeviceTask> tasks;
  bool active = false; // Queue has pending commands, and is either being processed by a TBB task or suspended waiting for an event
};

struct GPUReconstructionCPUDeviceInternals {
  std::unique_ptr<tbb::task_arena> arena;
  std::array<GPUReconstructionCPUDeviceQueue, constants::GPU_MAX_STREAMS> queues;
  std::atomic<int32_t> error{0}; // Set if a command in a command queue has failed

  char* deviceMemory = nullptr;   // "Device" memory, a separate host memory region
  char* hostMemory = nullptr;     // Host memory, corresponds to page-locked host memory of real GPUs
  char* constantMemory = nullptr; // "Device" constant memory, holding the device copy of GPUConstantMem

  std::vector<void*> rtcLibraries; // dlopen handles of RTC-compiled kernel libraries
  std::vector<void*> rtcKernels;   // RTC kernel entry points, indexed by kernel number

  void enqueue(int32_t stream, GPUReconstructionCPUDeviceTask&& task);
  void startQueue(int32_t stream);
  void processQueue(int32_t stream);
  void completeEvent(GPUReconstructionCPUDeviceEvent* e, uint64_t value);
};

} // namespace o2::gpu

#endif
