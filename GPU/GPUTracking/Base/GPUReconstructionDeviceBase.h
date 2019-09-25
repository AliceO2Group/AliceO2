// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "GPUReconstructionHelpers.h"
#include "GPUChain.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#if !(defined(__CINT__) || defined(__ROOTCINT__) || defined(__CLING__) || defined(__ROOTCLING__) || defined(G__ROOT))
extern template class GPUReconstructionKernels<GPUReconstructionCPUBackend>;
#endif

class GPUReconstructionDeviceBase : public GPUReconstructionCPU
{
 public:
  ~GPUReconstructionDeviceBase() override;

  const GPUParam* DeviceParam() const { return &mDeviceConstantMem->param; }
  int GetMaxThreads() override;

 protected:
  GPUReconstructionDeviceBase(const GPUSettingsProcessing& cfg);

  int InitDevice() override;
  virtual int InitDevice_Runtime() = 0;
  int ExitDevice() override;
  virtual int ExitDevice_Runtime() = 0;

  virtual const GPUTPCTracker* CPUTracker(int iSlice) { return &processors()->tpcTrackers[iSlice]; }

  int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override = 0;
  void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) override = 0;
  void GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override = 0;
  void GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
  void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override = 0;

  int StartHelperThreads() override;
  int StopHelperThreads() override;
  void RunHelperThreads(int (GPUReconstructionHelpers::helperDelegateBase::*function)(int, int, GPUReconstructionHelpers::helperParam*), GPUReconstructionHelpers::helperDelegateBase* functionCls, int count) override;
  int HelperError(int iThread) const override { return mHelperParams[iThread].error; }
  int HelperDone(int iThread) const override { return mHelperParams[iThread].done; }
  void WaitForHelperThreads() override;
  void ResetHelperThreads(int helpers) override;
  void ResetThisHelperThread(GPUReconstructionHelpers::helperParam* par);

  int GetGlobalLock(void*& pLock);
  void ReleaseGlobalLock(void* sem);

  static void* helperWrapper_static(void* arg);
  void* helperWrapper(GPUReconstructionHelpers::helperParam* par);

  int mDeviceId = -1;                                             // Device ID used by backend
  GPUReconstructionHelpers::helperParam* mHelperParams = nullptr; // Control Struct for helper threads
  int mNSlaveThreads = 0;                                         // Number of slave threads currently active
};

inline void GPUReconstructionDeviceBase::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  if (onGpu) {
    GPUMemCpy(dst, src, size, stream, toGPU, ev, evList, nEvents);
  } else {
    GPUReconstructionCPU::GPUMemCpyAlways(false, dst, src, size, stream, toGPU, ev, evList, nEvents);
  }
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
