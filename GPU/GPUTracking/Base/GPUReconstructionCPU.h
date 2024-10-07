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

#include "GPUReconstruction.h"
#include "GPUReconstructionHelpers.h"
#include "GPUConstantMem.h"
#include <stdexcept>
#include "utils/timer.h"
#include <vector>

#include "GPUGeneralKernels.h"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUReconstructionKernels.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUReconstructionCPUBackend : public GPUReconstruction
{
 public:
  ~GPUReconstructionCPUBackend() override = default;

 protected:
  GPUReconstructionCPUBackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstruction(cfg) {}
  template <class T, int32_t I = 0, typename... Args>
  int32_t runKernelBackend(const gpu_reconstruction_kernels::krnlSetupArgs<T, I, Args...>& args);
  template <class T, int32_t I = 0, typename... Args>
  int32_t runKernelBackendInternal(const gpu_reconstruction_kernels::krnlSetupTime& _xyz, const Args&... args);
  template <class T, int32_t I>
  gpu_reconstruction_kernels::krnlProperties getKernelPropertiesBackend();
  uint32_t mNestedLoopOmpFactor = 1;
  static int32_t getOMPThreadNum();
  static int32_t getOMPMaxThreads();
};

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  template <class S, int32_t I = 0, typename... Args>
  int32_t runKernel(krnlSetup&& setup, Args&&... args);
  template <class S, int32_t I = 0>
  const gpu_reconstruction_kernels::krnlProperties getKernelProperties()
  {
    return getKernelPropertiesImpl(gpu_reconstruction_kernels::classArgument<S, I>());
  }

  template <class T, int32_t I>
  constexpr static const char* GetKernelName();

  virtual int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false);
  int32_t GPUStuck() { return mGPUStuck; }
  void ResetDeviceProcessorTypes();
  template <class T>
  void AddGPUEvents(T*& events);

  int32_t RunChains() override;

  HighResTimer& getRecoStepTimer(RecoStep step) { return mTimersRecoSteps[getRecoStepNum(step)].timerTotal; }
  HighResTimer& getGeneralStepTimer(GeneralStep step) { return mTimersGeneralSteps[getGeneralStepNum(step)]; }

  void SetNestedLoopOmpFactor(uint32_t f) { mNestedLoopOmpFactor = f; }
  uint32_t SetAndGetNestedLoopOmpFactor(bool condition, uint32_t max);

  void UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, int32_t stream = -1);

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    int16_t mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg) {}

#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward, x_types)                                                                                                                        \
  inline int32_t runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, bool cpuFallback, double& timer, krnlSetup&& setup GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                                                                                             \
    if (cpuFallback) {                                                                                                                                                                          \
      return GPUReconstructionCPU::runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));              \
    } else {                                                                                                                                                                                    \
      return runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));                                    \
    }                                                                                                                                                                                           \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override { return 0; }
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override { return 0; }

  virtual void SynchronizeStream(int32_t stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) {}
  virtual void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) {}
  virtual bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) { return true; }
  virtual void RecordMarker(deviceEvent ev, int32_t stream) {}
  virtual void SynchronizeGPU() {}
  virtual void ReleaseEvent(deviceEvent ev) {}
  virtual int32_t StartHelperThreads() { return 0; }
  virtual int32_t StopHelperThreads() { return 0; }
  virtual void RunHelperThreads(int32_t (GPUReconstructionHelpers::helperDelegateBase::*function)(int32_t, int32_t, GPUReconstructionHelpers::helperParam*), GPUReconstructionHelpers::helperDelegateBase* functionCls, int32_t count) {}
  virtual void WaitForHelperThreads() {}
  virtual int32_t HelperError(int32_t iThread) const { return 0; }
  virtual int32_t HelperDone(int32_t iThread) const { return 0; }
  virtual void ResetHelperThreads(int32_t helpers) {}

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

  int32_t InitDevice() override;
  int32_t ExitDevice() override;
  int32_t GetThread();

  virtual int32_t PrepareTextures() { return 0; }
  virtual int32_t DoStuckProtection(int32_t stream, deviceEvent event) { return 0; }

  // Pointers to tracker classes
  GPUProcessorProcessors mProcShadow; // Host copy of tracker objects that will be used on the GPU
  GPUConstantMem*& mProcessorsShadow = mProcShadow.mProcessorsProc;

  uint32_t mBlockCount = 1;
  uint32_t mThreadCount = 1;
  uint32_t mWarpSize = 1;

  struct timerMeta {
    std::unique_ptr<HighResTimer[]> timer;
    std::string name;
    int32_t num;        // How many parallel instances to sum up (CPU threads / GPU streams)
    int32_t type;       // 0 = kernel, 1 = CPU step, 2 = DMA transfer
    uint32_t count;     // How often was the timer queried
    RecoStep step;      // Which RecoStep is this
    size_t memSize;     // Memory size for memory bandwidth computation
  };

  struct RecoStepTimerMeta {
    HighResTimer timerToGPU;
    HighResTimer timerToHost;
    HighResTimer timerTotal;
    size_t bytesToGPU = 0;
    size_t bytesToHost = 0;
    uint32_t countToGPU = 0;
    uint32_t countToHost = 0;
  };

  HighResTimer mTimersGeneralSteps[GPUDataTypes::N_GENERAL_STEPS];

  std::vector<std::unique_ptr<timerMeta>> mTimers;
  RecoStepTimerMeta mTimersRecoSteps[GPUDataTypes::N_RECO_STEPS];
  HighResTimer timerTotal;
  template <class T, int32_t I = 0>
  HighResTimer& getKernelTimer(RecoStep step, int32_t num = 0, size_t addMemorySize = 0, bool increment = true);
  template <class T, int32_t J = -1>
  HighResTimer& getTimer(const char* name, int32_t num = -1);

  std::vector<std::vector<deviceEvent>> mEvents;

 private:
  size_t TransferMemoryResourcesHelper(GPUProcessor* proc, int32_t stream, bool all, bool toGPU);
  uint32_t getNextTimerId();
  timerMeta* getTimerById(uint32_t id, bool increment = true);
  timerMeta* insertTimer(uint32_t id, std::string&& name, int32_t J, int32_t num, int32_t type, RecoStep step);
};

template <class S, int32_t I, typename... Args>
inline int32_t GPUReconstructionCPU::runKernel(krnlSetup&& setup, Args&&... args)
{
  HighResTimer* t = nullptr;
  GPUCA_RECO_STEP myStep = S::GetRecoStep() == GPUCA_RECO_STEP::NoRecoStep ? setup.x.step : S::GetRecoStep();
  if (myStep == GPUCA_RECO_STEP::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int32_t cpuFallback = IsGPU() ? (setup.x.device == krnlDeviceType::CPU ? 2 : (mRecoSteps.stepsGPUMask & myStep) != myStep) : 0;
  uint32_t& nThreads = setup.x.nThreads;
  uint32_t& nBlocks = setup.x.nBlocks;
  const uint32_t stream = setup.x.stream;
  auto prop = getKernelProperties<S, I>();
  const int32_t autoThreads = cpuFallback ? 1 : prop.nThreads;
  const int32_t autoBlocks = cpuFallback ? 1 : (prop.forceBlocks ? prop.forceBlocks : (prop.minBlocks * mBlockCount));
  if (nBlocks == (uint32_t)-1) {
    nBlocks = (nThreads + autoThreads - 1) / autoThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-2) {
    nBlocks = nThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-3) {
    nBlocks = autoBlocks;
    nThreads = autoThreads;
  } else if ((int32_t)nThreads < 0) {
    nThreads = cpuFallback ? 1 : -nThreads;
  }
  if (nThreads > GPUCA_MAX_THREADS) {
    throw std::runtime_error("GPUCA_MAX_THREADS exceeded");
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Running kernel %s (Stream %d, Range %d/%d, Grid %d/%d) on %s", GetKernelName<S, I>(), stream, setup.y.start, setup.y.num, nBlocks, nThreads, cpuFallback == 2 ? "CPU (forced)" : cpuFallback ? "CPU (fallback)" : mDeviceName.c_str());
  }
  if (nThreads == 0 || nBlocks == 0) {
    return 0;
  }
  if (mProcessingSettings.debugLevel >= 1) {
    t = &getKernelTimer<S, I>(myStep, !IsGPU() || cpuFallback ? getOMPThreadNum() : stream);
    if ((!mProcessingSettings.deviceTimers || !IsGPU() || cpuFallback) && (mNestedLoopOmpFactor < 2 || getOMPThreadNum() == 0)) {
      t->Start();
    }
  }
  double deviceTimerTime = 0.;
  int32_t retVal = runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<S, I>(), cpuFallback, deviceTimerTime, std::forward<krnlSetup&&>(setup), std::forward<Args>(args)...);
  if (GPUDebug(GetKernelName<S, I>(), stream)) {
    throw std::runtime_error("kernel failure");
  }
  if (mProcessingSettings.debugLevel >= 1) {
    if (t) {
      if (deviceTimerTime != 0.) {
        t->AddTime(deviceTimerTime);
        if (t->IsRunning()) {
          t->Abort();
        }
      } else if (t->IsRunning()) {
        t->Stop();
      }
    }
    if (CheckErrorCodes(cpuFallback) && !mProcessingSettings.ignoreNonFatalGPUErrors) {
      throw std::runtime_error("kernel error code");
    }
  }
  return retVal;
}

#define GPUCA_KRNL(x_class, ...)                                                              \
  template <>                                                                                 \
  constexpr const char* GPUReconstructionCPU::GetKernelName<GPUCA_M_KRNL_TEMPLATE(x_class)>() \
  {                                                                                           \
    return GPUCA_M_STR(GPUCA_M_KRNL_NAME(x_class));                                           \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

template <class T>
inline void GPUReconstructionCPU::AddGPUEvents(T*& events)
{
  mEvents.emplace_back(std::vector<deviceEvent>(sizeof(T) / sizeof(deviceEvent)));
  events = (T*)mEvents.back().data();
}

template <class T, int32_t I>
HighResTimer& GPUReconstructionCPU::getKernelTimer(RecoStep step, int32_t num, size_t addMemorySize, bool increment)
{
  static int32_t id = getNextTimerId();
  timerMeta* timer = getTimerById(id, increment);
  if (timer == nullptr) {
    timer = insertTimer(id, GetKernelName<T, I>(), -1, NSLICES, 0, step);
  }
  if (addMemorySize) {
    timer->memSize += addMemorySize;
  }
  if (num < 0 || num >= timer->num) {
    throw std::runtime_error("Invalid timer requested");
  }
  return timer->timer[num];
}

template <class T, int32_t J>
HighResTimer& GPUReconstructionCPU::getTimer(const char* name, int32_t num)
{
  static int32_t id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
  if (timer == nullptr) {
    int32_t max = std::max<int32_t>({getOMPMaxThreads(), mProcessingSettings.nDeviceHelperThreads + 1, mProcessingSettings.nStreams});
    timer = insertTimer(id, name, J, max, 1, RecoStep::NoRecoStep);
  }
  if (num == -1) {
    num = getOMPThreadNum();
  }
  if (num < 0 || num >= timer->num) {
    throw std::runtime_error("Invalid timer requested");
  }
  return timer->timer[num];
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
