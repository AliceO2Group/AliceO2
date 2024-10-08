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
#include <vector>
#include <string>

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
  static int32_t GPUFailedMsgAI(const int64_t error, const char* file, int32_t line);
  void GPUFailedMsgA(const int64_t error, const char* file, int32_t line);

 protected:
  GPUReconstructionCUDABackend(const GPUSettingsDeviceBackend& cfg);

  void PrintKernelOccupancies() override;

  template <class T, int32_t I = 0, typename... Args>
  int32_t runKernelBackend(const krnlSetupArgs<T, I, Args...>& args);
  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackendInternal(const krnlSetupTime& _xyz, const Args&... args);
  template <class T, int32_t I = 0>
  gpu_reconstruction_kernels::krnlProperties getKernelPropertiesBackend();
  template <class T, int32_t I>
  class backendInternal;

  template <bool multi, class T, int32_t I = 0>
  static int32_t getRTCkernelNum(int32_t k = -1);

  void getRTCKernelCalls(std::vector<std::string>& kernels);

  GPUReconstructionCUDAInternals* mInternals;
};

class GPUReconstructionCUDA : public GPUReconstructionKernels<GPUReconstructionCUDABackend>
{
 public:
  ~GPUReconstructionCUDA() override;
  GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg);

 protected:
  int32_t InitDevice_Runtime() override;
  int32_t ExitDevice_Runtime() override;
  void UpdateAutomaticProcessingSettings() override;

  std::unique_ptr<GPUThreadContext> GetThreadContext() override;
  void SynchronizeGPU() override;
  int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false) override;
  void SynchronizeStream(int32_t stream) override;
  void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) override;
  void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) override;
  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override;
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override;
  void ReleaseEvent(deviceEvent ev) override;
  void RecordMarker(deviceEvent ev, int32_t stream) override;

  void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame>* timeFrame) override;

#ifndef __HIPCC__ // CUDA
  bool CanQueryMaxMemory() override { return true; }
  int32_t PrepareTextures() override;
  void startGPUProfiling() override;
  void endGPUProfiling() override;
#else // HIP
  void* getGPUPointer(void* ptr) override;
#endif

 private:
  int32_t genRTC(std::string& filename, uint32_t& nCompile);
  void genAndLoadRTC();
  void loadKernelModules(bool perKernel, bool perSingleMulti = true);
  const char *mRtcSrcExtension = ".src", *mRtcBinExtension = ".o";
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
