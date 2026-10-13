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

/// \file GPUReconstructionCPUDevice.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCPUDEVICE_H
#define GPURECONSTRUCTIONCPUDEVICE_H

#include "GPUReconstructionDeviceBase.h"
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
extern "C" __declspec(dllexport) o2::gpu::GPUReconstruction* GPUReconstruction_Create_CPUDEVICE(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" o2::gpu::GPUReconstruction* GPUReconstruction_Create_CPUDEVICE(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace o2::gpu
{
struct GPUReconstructionCPUDeviceInternals;
struct GPUReconstructionCPUDeviceEvent;

// Backend which runs on the host, but behaves like a GPU device: It has a separate device memory region with explicit memcopies,
// and multiple in-order command queues (streams) which are processed asynchronously by TBB. One device block is one TBB task,
// the number of device threads per block is fixed to 1, such that the normal CPU kernels can be used.
class GPUReconstructionCPUDevice : public GPUReconstructionProcessing::KernelInterface<GPUReconstructionCPUDevice, GPUReconstructionDeviceBase>
{
 public:
  GPUReconstructionCPUDevice(const GPUSettingsDeviceBackend& cfg);
  ~GPUReconstructionCPUDevice() override;

  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackend(const krnlSetupTime& _xyz, const Args&... args);

 protected:
  int32_t InitDevice_Runtime() override;
  int32_t ExitDevice_Runtime() override;

  int32_t GPUChkErrInternal(const int64_t retval, const char* file, int32_t line) const override;

  void SynchronizeGPU() override;
  int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false) override;
  void SynchronizeStream(int32_t stream) override;
  void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) override;
  void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) override;
  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override { return 0; }
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override { return 0; }

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override;
  void ReleaseEvent(deviceEvent ev) override {}
  void RecordMarker(deviceEvent* ev, int32_t stream) override;

  GPUReconstructionCPUDeviceInternals* mInternals = nullptr;

 private:
  // Enqueue a task into the command queue of the stream, which waits for evList, runs work, and then marks ev as completed
  void EnqueueTask(int32_t stream, deviceEvent* evList, int32_t nEvents, deviceEvent* ev, std::function<void()>&& work);
  static GPUReconstructionCPUDeviceEvent* getEvent(deviceEvent& ev);
  std::vector<std::vector<GPUReconstructionCPUDeviceEvent>> mEventStorage; // Backs the deviceEvents in mEvents, inner vectors are never resized

  int32_t genRTC(std::string& filename, uint32_t& nCompile);
  void getRTCKernelCalls(std::vector<std::string>& kernels);
  void genAndLoadRTC();
};

} // namespace o2::gpu

#endif
