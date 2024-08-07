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
  template <class T, int I = 0, typename... Args>
  int runKernelBackend(const gpu_reconstruction_kernels::krnlSetupArgs<T, I, Args...>& args);
  template <class T, int I = 0, typename... Args>
  int runKernelBackendInternal(const gpu_reconstruction_kernels::krnlSetupTime& _xyz, const Args&... args);
  template <class T, int I>
  gpu_reconstruction_kernels::krnlProperties getKernelPropertiesBackend();
  unsigned int mNestedLoopOmpFactor = 1;
  static int getOMPThreadNum();
  static int getOMPMaxThreads();
};

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  template <class S, int I = 0, typename... Args>
  int runKernel(krnlSetup&& setup, Args&&... args);
  template <class S, int I = 0>
  const gpu_reconstruction_kernels::krnlProperties getKernelProperties()
  {
    return getKernelPropertiesImpl(gpu_reconstruction_kernels::classArgument<S, I>());
  }

  template <class T, int I>
  constexpr static const char* GetKernelName();

  virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1, bool force = false);
  int GPUStuck() { return mGPUStuck; }
  void ResetDeviceProcessorTypes();
  template <class T>
  void AddGPUEvents(T*& events);

  int RunChains() override;

  HighResTimer& getRecoStepTimer(RecoStep step) { return mTimersRecoSteps[getRecoStepNum(step)].timerTotal; }
  HighResTimer& getGeneralStepTimer(GeneralStep step) { return mTimersGeneralSteps[getGeneralStepNum(step)]; }

  void SetNestedLoopOmpFactor(unsigned int f) { mNestedLoopOmpFactor = f; }
  unsigned int SetAndGetNestedLoopOmpFactor(bool condition, unsigned int max);

  void UpdateParamOccupancyMap(const unsigned int* mapHost, const unsigned int* mapGPU, unsigned int occupancyTotal, int stream = -1);

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    short mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg) {}

#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward, x_types)                                                                                                                    \
  inline int runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, bool cpuFallback, double& timer, krnlSetup&& setup GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                                                                                         \
    if (cpuFallback) {                                                                                                                                                                      \
      return GPUReconstructionCPU::runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));          \
    } else {                                                                                                                                                                                \
      return runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));                                \
    }                                                                                                                                                                                       \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

  int registerMemoryForGPU_internal(const void* ptr, size_t size) override { return 0; }
  int unregisterMemoryForGPU_internal(const void* ptr) override { return 0; }

  virtual void SynchronizeStream(int stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {}
  virtual void StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents = 1) {}
  virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) { return true; }
  virtual void RecordMarker(deviceEvent ev, int stream) {}
  virtual void SynchronizeGPU() {}
  virtual void ReleaseEvent(deviceEvent ev) {}
  virtual int StartHelperThreads() { return 0; }
  virtual int StopHelperThreads() { return 0; }
  virtual void RunHelperThreads(int (GPUReconstructionHelpers::helperDelegateBase::*function)(int, int, GPUReconstructionHelpers::helperParam*), GPUReconstructionHelpers::helperDelegateBase* functionCls, int count) {}
  virtual void WaitForHelperThreads() {}
  virtual int HelperError(int iThread) const { return 0; }
  virtual int HelperDone(int iThread) const { return 0; }
  virtual void ResetHelperThreads(int helpers) {}

  size_t TransferMemoryResourceToGPU(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice()); }
  size_t TransferMemoryResourceToHost(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr()); }
  size_t TransferMemoryResourcesToGPU(GPUProcessor* proc, int stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, true); }
  size_t TransferMemoryResourcesToHost(GPUProcessor* proc, int stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, false); }
  size_t TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { return TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents); }
  size_t TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { return TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents); }
  virtual size_t GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  virtual size_t GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
  virtual size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst);

  int InitDevice() override;
  int ExitDevice() override;
  int GetThread();

  virtual int PrepareTextures() { return 0; }
  virtual int DoStuckProtection(int stream, deviceEvent event) { return 0; }

  // Pointers to tracker classes
  GPUProcessorProcessors mProcShadow; // Host copy of tracker objects that will be used on the GPU
  GPUConstantMem*& mProcessorsShadow = mProcShadow.mProcessorsProc;

  unsigned int mBlockCount = 1;
  unsigned int mThreadCount = 1;
  unsigned int mWarpSize = 1;

  struct timerMeta {
    std::unique_ptr<HighResTimer[]> timer;
    std::string name;
    int num;            // How many parallel instances to sum up (CPU threads / GPU streams)
    int type;           // 0 = kernel, 1 = CPU step, 2 = DMA transfer
    unsigned int count; // How often was the timer queried
    RecoStep step;      // Which RecoStep is this
    size_t memSize;     // Memory size for memory bandwidth computation
  };

  struct RecoStepTimerMeta {
    HighResTimer timerToGPU;
    HighResTimer timerToHost;
    HighResTimer timerTotal;
    size_t bytesToGPU = 0;
    size_t bytesToHost = 0;
    unsigned int countToGPU = 0;
    unsigned int countToHost = 0;
  };

  HighResTimer mTimersGeneralSteps[GPUDataTypes::N_GENERAL_STEPS];

  std::vector<std::unique_ptr<timerMeta>> mTimers;
  RecoStepTimerMeta mTimersRecoSteps[GPUDataTypes::N_RECO_STEPS];
  HighResTimer timerTotal;
  template <class T, int I = 0>
  HighResTimer& getKernelTimer(RecoStep step, int num = 0, size_t addMemorySize = 0);
  template <class T, int J = -1>
  HighResTimer& getTimer(const char* name, int num = -1);

  std::vector<std::vector<deviceEvent>> mEvents;

 private:
  size_t TransferMemoryResourcesHelper(GPUProcessor* proc, int stream, bool all, bool toGPU);
  unsigned int getNextTimerId();
  timerMeta* getTimerById(unsigned int id);
  timerMeta* insertTimer(unsigned int id, std::string&& name, int J, int num, int type, RecoStep step);
};

template <class S, int I, typename... Args>
inline int GPUReconstructionCPU::runKernel(krnlSetup&& setup, Args&&... args)
{
  HighResTimer* t = nullptr;
  GPUCA_RECO_STEP myStep = S::GetRecoStep() == GPUCA_RECO_STEP::NoRecoStep ? setup.x.step : S::GetRecoStep();
  if (myStep == GPUCA_RECO_STEP::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int cpuFallback = IsGPU() ? (setup.x.device == krnlDeviceType::CPU ? 2 : (mRecoSteps.stepsGPUMask & myStep) != myStep) : 0;
  unsigned int& nThreads = setup.x.nThreads;
  unsigned int& nBlocks = setup.x.nBlocks;
  const unsigned int stream = setup.x.stream;
  auto prop = getKernelProperties<S, I>();
  const int autoThreads = cpuFallback ? 1 : prop.nThreads;
  const int autoBlocks = cpuFallback ? 1 : (prop.forceBlocks ? prop.forceBlocks : (prop.minBlocks * mBlockCount));
  if (nBlocks == (unsigned int)-1) {
    nBlocks = (nThreads + autoThreads - 1) / autoThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (unsigned int)-2) {
    nBlocks = nThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (unsigned int)-3) {
    nBlocks = autoBlocks;
    nThreads = autoThreads;
  } else if ((int)nThreads < 0) {
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
  int retVal = runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<S, I>(), cpuFallback, deviceTimerTime, std::forward<krnlSetup&&>(setup), std::forward<Args>(args)...);
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

template <class T, int I>
HighResTimer& GPUReconstructionCPU::getKernelTimer(RecoStep step, int num, size_t addMemorySize)
{
  static int id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
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

template <class T, int J>
HighResTimer& GPUReconstructionCPU::getTimer(const char* name, int num)
{
  static int id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
  if (timer == nullptr) {
    int max = std::max<int>({getOMPMaxThreads(), mProcessingSettings.nDeviceHelperThreads + 1, mProcessingSettings.nStreams});
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
