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

/// \file GPUReconstructionOCL.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL_H
#define GPURECONSTRUCTIONOCL_H

#include "GPUReconstructionDeviceBase.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) o2::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" o2::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace o2::gpu
{
struct GPUReconstructionOCLInternals;

class GPUReconstructionOCL : public GPUReconstructionProcessing::KernelInterface<GPUReconstructionOCL, GPUReconstructionDeviceBase>
{
 public:
  GPUReconstructionOCL(const GPUSettingsDeviceBackend& cfg);
  ~GPUReconstructionOCL() override;

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

  GPUReconstructionOCLInternals* mInternals;
  float mOclVersion;

  template <class S, class T, int32_t I>
  S& getKernelObject();

  int32_t GetOCLPrograms();

 private:
  static const char* convertErrorToString(int32_t errorCode);
  template <typename T, typename... Args>
  static inline int64_t OCLsetKernelParameters_helper(cl_kernel& kernel, int32_t i, const T& firstParameter, const Args&... restOfParameters);
  template <typename... Args>
  static int64_t OCLsetKernelParameters(cl_kernel& kernel, const Args&... args);
  static int64_t clExecuteKernelA(cl_command_queue queue, cl_kernel krnl, size_t local_size, size_t global_size, cl_event* pEvent = nullptr, cl_event* wait = nullptr, cl_int nWaitEvents = 1);
  int32_t AddKernels();
};

} // namespace o2::gpu

#endif
