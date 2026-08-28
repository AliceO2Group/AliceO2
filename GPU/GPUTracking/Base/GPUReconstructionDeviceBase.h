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

/// \file GPUReconstructionDeviceBase.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONDEVICEBASE_H
#define GPURECONSTRUCTIONDEVICEBASE_H

#include "GPUReconstructionCPU.h"
#include <pthread.h>
#include "GPUChain.h"
#include <vector>

namespace o2::gpu
{
class GPUReconstructionDeviceBase : public GPUReconstructionCPU
{
 public:
  ~GPUReconstructionDeviceBase() override;

  const GPUParam* DeviceParam() const;
  struct deviceConstantMemRegistration {
    deviceConstantMemRegistration(void* (*reg)())
    {
      GPUReconstructionDeviceBase::getDeviceConstantMemRegistratorsVector().emplace_back(reg);
    }
  };

 protected:
  GPUReconstructionDeviceBase(const GPUSettingsDeviceBackend& cfg, size_t sizeCheck);

  int32_t InitDevice() override;
  virtual int32_t InitDevice_Runtime() = 0;
  int32_t ExitDevice() override;
  virtual int32_t ExitDevice_Runtime() = 0;
  virtual int32_t GPUChkErrInternal(const int64_t retval, const char* file, int32_t line) const override = 0;
  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override;
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override;
  void unregisterRemainingRegisteredMemory();

  int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false) override = 0;
  size_t TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override = 0;
  size_t GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override;
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override = 0;

  int32_t GetGlobalLock(void*& pLock);
  void ReleaseGlobalLock(void* sem);

  int32_t mDeviceId = -1; // Device ID used by backend

  struct DebugEvents {
    deviceEvent DebugStart, DebugStop; // Debug timer events
  };
  DebugEvents* mDebugEvents = nullptr;

  std::vector<void*> mDeviceConstantMemList;
  static std::vector<void* (*)()>& getDeviceConstantMemRegistratorsVector()
  {
    static std::vector<void* (*)()> deviceConstantMemRegistrators{};
    return deviceConstantMemRegistrators;
  }
  void runConstantRegistrators();
};

inline size_t GPUReconstructionDeviceBase::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (onGpu) {
    return GPUMemCpy(dst, src, size, stream, toGPU, ev, evList, nEvents);
  } else {
    return GPUReconstructionCPU::GPUMemCpyAlways(false, dst, src, size, stream, toGPU, ev, evList, nEvents);
  }
}
} // namespace o2::gpu

#endif
