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
#include "GPUCommonAlgorithm.h"
#include <vector>
#include <string>

#ifdef _WIN32
extern "C" __declspec(dllexport) o2::gpu::GPUReconstruction* GPUReconstruction_Create_CUDA(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" o2::gpu::GPUReconstruction* GPUReconstruction_Create_CUDA(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace Ort
{
struct SessionOptions;
}

namespace o2::gpu
{
struct GPUReconstructionCUDAInternals;

class GPUReconstructionCUDA : public GPUReconstructionProcessing::KernelInterface<GPUReconstructionCUDA, GPUReconstructionDeviceBase>
{
 public:
  GPUReconstructionCUDA(const GPUSettingsDeviceBackend& cfg);
  ~GPUReconstructionCUDA() override;

  void PrintKernelOccupancies() override;
  virtual int32_t GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const override;

  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackend(const krnlSetupTime& _xyz, const Args&... args);
  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackendTimed(const krnlSetupTime& _xyz, const Args&... args);
  template <class T, int32_t I>
  struct kernelBackendMacro;

  template <class T, class S>
  friend GPUh() void GPUCommonAlgorithm::sortOnDevice(auto* rec, int32_t stream, T* begin, size_t N, const S& comp);

 protected:
  GPUReconstructionCUDAInternals* mInternals;

  int32_t InitDevice_Runtime() override;
  int32_t ExitDevice_Runtime() override;

  std::unique_ptr<threadContext> GetThreadContext() override;
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
  void RecordMarker(deviceEvent* ev, int32_t stream) override;
  void SetONNXGPUStream(Ort::SessionOptions& session_options, int32_t stream, int32_t* deviceId) override;

  void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits<7>>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits<7>>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame<7>>* timeFrame) override;

#ifndef __HIPCC__ // CUDA
  bool CanQueryMaxMemory() override { return true; }
  void startGPUProfiling() override;
  void endGPUProfiling() override;
#else // HIP
  void* getGPUPointer(void* ptr) override;
#endif

 private:
  int32_t genRTC(std::string& filename, uint32_t& nCompile);
  void getRTCKernelCalls(std::vector<std::string>& kernels);
  void genAndLoadRTC();
  void loadKernelModules(bool perKernel);
  const char *mRtcSrcExtension = ".src", *mRtcBinExtension = ".o";
};

} // namespace o2::gpu

#endif
