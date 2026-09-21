// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef GPURECONSTRUCTIONMETAL_H
#define GPURECONSTRUCTIONMETAL_H

#include "GPUReconstructionDeviceBase.h"

extern "C" o2::gpu::GPUReconstruction* GPUReconstruction_Create_METAL(const o2::gpu::GPUSettingsDeviceBackend& cfg);

namespace o2::gpu
{
struct GPUReconstructionMetalInternals;

class GPUReconstructionMetal : public GPUReconstructionProcessing::KernelInterface<GPUReconstructionMetal, GPUReconstructionDeviceBase>
{
 public:
  GPUReconstructionMetal(const GPUSettingsDeviceBackend& cfg);
  ~GPUReconstructionMetal() override;

  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackend(const krnlSetupTime& _xyz, const Args&... args);

 protected:
  int32_t InitDevice_Runtime() override;
  int32_t ExitDevice_Runtime() override;

  virtual int32_t GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const override;

  void SynchronizeGPU() override;
  int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false) override;
  void SynchronizeStream(int32_t stream) override;
  void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) override;
  void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override;
  void ReleaseEvent(deviceEvent ev) override;
  void RecordMarker(deviceEvent* ev, int32_t stream) override;

  template <class T, int32_t I = 0>
  int32_t AddKernel();

  GPUReconstructionMetalInternals* mInternals;
  float mOclVersion;

  template <class S, class T, int32_t I>
  S& getKernelObject();

  int32_t GetMetalPrograms();

 private:
  int32_t AddKernels();
};

} // namespace o2::gpu

#endif // GPURECONSTRUCTIONMETAL_H
