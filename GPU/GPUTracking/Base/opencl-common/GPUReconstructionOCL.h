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
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCLconst GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace GPUCA_NAMESPACE::gpu
{
struct GPUReconstructionOCLInternals;

class GPUReconstructionOCL : public GPUReconstructionDeviceBase
{
 public:
  ~GPUReconstructionOCL() override;
  GPUReconstructionOCL(const GPUSettingsDeviceBackend& cfg);

 protected:
  int InitDevice_Runtime() override;
  int ExitDevice_Runtime() override;
  void UpdateSettings() override;

  int GPUFailedMsgAI(const long int error, const char* file, int line);
  void GPUFailedMsgA(const long int error, const char* file, int line);

  void SynchronizeGPU() override;
  int DoStuckProtection(int stream, void* event) override;
  int GPUDebug(const char* state = "UNKNOWN", int stream = -1, bool force = false) override;
  void SynchronizeStream(int stream) override;
  void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
  void StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
  size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
  void ReleaseEvent(deviceEvent* ev) override;
  void RecordMarker(deviceEvent* ev, int stream) override;

  virtual int GetOCLPrograms() = 0;
  virtual bool CheckPlatform(unsigned int i) = 0;
  virtual bool ContextForAllPlatforms() { return false; }

  template <class T, int I = 0>
  int AddKernel(bool multi = false);
  template <class T, int I = 0>
  unsigned int FindKernel(int num);
  template <typename K, typename... Args>
  int runKernelBackendCommon(krnlSetup& _xyz, K& k, const Args&... args);
  template <class T, int I = 0>
  const krnlProperties getKernelPropertiesBackend();

  GPUReconstructionOCLInternals* mInternals;
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
