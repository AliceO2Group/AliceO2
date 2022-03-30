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

/// \file GPUReconstructionCUDA.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCUDA_H
#define GPURECONSTRUCTIONCUDA_H

#include "GPUReconstructionDeviceBase.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUReconstructionCUDAInternals;

class GPUReconstructionCUDABackend : public GPUReconstructionDeviceBase
{
 public:
  ~GPUReconstructionCUDABackend() override;

 protected:
  GPUReconstructionCUDABackend(const GPUSettingsDeviceBackend& cfg);

  void* GetBackendConstSymbolAddress();
  void PrintKernelOccupancies() override;

  template <class T, int I = 0, typename... Args>
  int runKernelBackend(krnlSetup& _xyz, Args... args);
  template <class T, int I = 0, typename... Args>
  void runKernelBackendInternal(krnlSetup& _xyz, const Args&... args);
  template <class T, int I = 0>
  const krnlProperties getKernelPropertiesBackend();
  template <class T, int I>
  class backendInternal;

  GPUReconstructionCUDAInternals* mInternals;
};

class GPUReconstructionCUDA : public GPUReconstructionKernels<GPUReconstructionCUDABackend>
{
 public:
  ~GPUReconstructionCUDA() override;
  GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg);

 protected:
  int InitDevice_Runtime() override;
  int ExitDevice_Runtime() override;
  void UpdateSettings() override;

  class GPUThreadContextCUDA : public GPUThreadContext
  {
   public:
    GPUThreadContextCUDA(GPUReconstructionCUDAInternals* context);
    ~GPUThreadContextCUDA() override;

   private:
    GPUReconstructionCUDAInternals* mContext = nullptr;
  };

  std::unique_ptr<GPUThreadContext> GetThreadContext() override;
  bool CanQueryMaxMemory() override { return true; }
  void SynchronizeGPU() override;
  int GPUDebug(const char* state = "UNKNOWN", int stream = -1, bool force = false) override;
  void SynchronizeStream(int stream) override;
  void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
  void StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;

  int PrepareTextures() override;
  int registerMemoryForGPU(const void* ptr, size_t size) override;
  int unregisterMemoryForGPU(const void* ptr) override;
  void startGPUProfiling() override;
  void endGPUProfiling() override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
  size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
  void ReleaseEvent(deviceEvent* ev) override;
  void RecordMarker(deviceEvent* ev, int stream) override;

  void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits) override;
  void GetITSTimeframe(std::unique_ptr<o2::its::TimeFrame>* timeFrame) override;

 private:
  std::vector<void*> mDeviceConstantMemRTC;
  int genRTC();
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
