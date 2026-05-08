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

/// \file GPUReconstructionCPU.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONICPU_H
#define GPURECONSTRUCTIONICPU_H

#include "GPUReconstructionProcessing.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Ort
{
struct SessionOptions;
}

namespace o2::gpu
{

class GPUReconstructionCPU : public GPUReconstructionProcessing::KernelInterface<GPUReconstructionCPU, GPUReconstructionProcessing>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  template <class S, int32_t I = 0>
  krnlProperties getKernelProperties(int gpu = -1);
  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackend(const krnlSetupTime& _xyz, const Args&... args);

  virtual int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false);
  int32_t GPUStuck() { return mGPUStuck; }
  void ResetDeviceProcessorTypes();

  int32_t RunChains() override;

  void UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, uint32_t mapSize, int32_t stream = -1, deviceEvent* ev = nullptr);

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    int16_t mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing::KernelInterface<GPUReconstructionCPU, GPUReconstructionProcessing>(cfg) {}

  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override { return 0; }
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override { return 0; }

  virtual void SynchronizeStream(int32_t stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) {}
  virtual void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) {}
  virtual bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) { return true; }
  virtual void RecordMarker(deviceEvent* ev, int32_t stream) {}
  virtual void SynchronizeGPU() {}
  virtual void ReleaseEvent(deviceEvent ev) {}

  size_t TransferMemoryResourceToGPU(GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice()); }
  size_t TransferMemoryResourceToHost(GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr()); }
  size_t TransferMemoryResourcesToGPU(GPUProcessor* proc, int32_t stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, true); }
  size_t TransferMemoryResourcesToHost(GPUProcessor* proc, int32_t stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, false); }
  size_t TransferMemoryResourceLinkToGPU(int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents); }
  size_t TransferMemoryResourceLinkToHost(int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents); }
  virtual size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1);
  virtual size_t GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1);
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  virtual size_t TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst);

  // ONNX runtime
  virtual void SetONNXGPUStream(Ort::SessionOptions&, int32_t, int32_t*) {}

  int32_t InitDevice() override;
  int32_t ExitDevice() override;
  int32_t GetThread();

  // Pointers to tracker classes
  GPUProcessorProcessors mProcShadow; // Host copy of tracker objects that will be used on the GPU
  GPUConstantMem*& mProcessorsShadow = mProcShadow.mProcessorsProc;

  uint32_t mMultiprocessorCount = 1;
  uint32_t mThreadCount = 1;
  uint32_t mWarpSize = 1;

 private:
  size_t TransferMemoryResourcesHelper(GPUProcessor* proc, int32_t stream, bool all, bool toGPU);
  template <class S, int32_t I = 0, typename... Args>
  void runKernelInterface(krnlSetup&& setup, Args const&... args);

  struct debugWriter {
    debugWriter(std::string filenameCSV, bool markdown, uint32_t statNEvents);
    void header();
    void row(char type, uint32_t count, std::string name, double gpu_time, double cpu_time, double total_time, std::size_t memSize, std::string nEventReport = "");

   private:
    std::ofstream streamCSV;
    bool mMarkdown;
    uint32_t mStatNEvents;
  };
};

} // namespace o2::gpu

#endif
