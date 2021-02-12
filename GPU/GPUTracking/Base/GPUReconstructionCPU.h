// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCPU.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONIMPL_H
#define GPURECONSTRUCTIONIMPL_H

#include "GPUReconstruction.h"
#include "GPUReconstructionHelpers.h"
#include "GPUConstantMem.h"
#include <stdexcept>
#include "utils/timer.h"
#include <vector>

#include "GPUGeneralKernels.h"
#include "GPUTPCCreateSliceData.h"
#include "GPUTPCNeighboursFinder.h"
#include "GPUTPCNeighboursCleaner.h"
#include "GPUTPCStartHitsFinder.h"
#include "GPUTPCStartHitsSorter.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUTPCTrackletSelector.h"
#include "GPUTPCGlobalTracking.h"
#include "GPUTRDTrackerKernels.h"
#ifdef GPUCA_NOCOMPAT
#include "GPUTPCGMMergerGPU.h"
#endif
#ifdef HAVE_O2HEADERS
#include "GPUITSFitterKernels.h"
#include "GPUTPCConvertKernel.h"
#include "GPUTPCCompressionKernels.h"
#include "GPUTPCClusterFinderKernels.h"
#include "GPUTrackingRefitKernel.h"
#include "GPUTPCGMO2Output.h"
#endif

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
  int runKernelBackend(krnlSetup& _xyz, const Args&... args);
  template <class T, int I>
  krnlProperties getKernelPropertiesBackend();
  unsigned int mNestedLoopOmpFactor = 1;
};

template <class T>
class GPUReconstructionKernels : public T
{
 public:
  using krnlSetup = GPUReconstruction::krnlSetup;
  template <class X, int Y = 0>
  using classArgument = GPUReconstruction::classArgument<X, Y>;
  virtual ~GPUReconstructionKernels() = default; // NOLINT: BUG: Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsDeviceBackend& cfg) : T(cfg) {}

 protected:
#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward)                                                        \
  virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, krnlSetup& _xyz GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                    \
    return T::template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(_xyz GPUCA_M_STRIP(x_forward));                \
  }                                                                                                                    \
  virtual GPUReconstruction::krnlProperties getKernelPropertiesImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>)     \
  {                                                                                                                    \
    return T::template getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>();                                   \
  }
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
};

#ifndef GPUCA_GPURECONSTRUCTIONCPU_IMPLEMENTATION
// Hide the body for all files but GPUReconstructionCPU.cxx, otherwise we get weird symbol clashes when the compiler inlines
template <>
class GPUReconstructionKernels<GPUReconstructionCPUBackend> : public GPUReconstructionCPUBackend
{
 public:
  using krnlSetup = GPUReconstruction::krnlSetup;
  template <class X, int Y = 0>
  using classArgument = GPUReconstruction::classArgument<X, Y>;
  virtual ~GPUReconstructionKernels() = default; // NOLINT: Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionCPUBackend(cfg) {}

 protected:
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward)                                                       \
  virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, krnlSetup& _xyz GPUCA_M_STRIP(x_arguments)); \
  virtual krnlProperties getKernelPropertiesImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>);
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
};
#endif

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

#ifdef __clang__ // BUG: clang does not accept default parameters before parameter pack
  template <class S, int I = 0, int J = -1>
  inline int runKernel(const krnlExec& x, const krnlRunRange& y = krnlRunRangeNone)
  {
    return runKernel<S, I, J>(x, y, krnlEventNone);
  }
  template <class S, int I = 0, int J = -1, typename... Args>
  int runKernel(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);
#else
  template <class S, int I = 0, int J = -1, typename... Args>
  int runKernel(const krnlExec& x, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEventNone, const Args&... args);
#endif
  template <class S, int I = 0>
  const krnlProperties getKernelProperties()
  {
    return getKernelPropertiesImpl(GPUReconstruction::template classArgument<S, I>());
  }

  template <class T, int I>
  constexpr static const char* GetKernelName();

  virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1);
  int registerMemoryForGPU(const void* ptr, size_t size) override { return 0; }
  int unregisterMemoryForGPU(const void* ptr) override { return 0; }
  int GPUStuck() { return mGPUStuck; }
  void ResetDeviceProcessorTypes();
  template <class T>
  void AddGPUEvents(T*& events);

  int RunChains() override;

  HighResTimer& getRecoStepTimer(RecoStep step) { return mTimersRecoSteps[getRecoStepNum(step)].timerTotal; }
  HighResTimer& getGeneralStepTimer(GeneralStep step) { return mTimersGeneralSteps[getGeneralStepNum(step)]; }

  void SetNestedLoopOmpFactor(unsigned int f) { mNestedLoopOmpFactor = f; }
  unsigned int SetAndGetNestedLoopOmpFactor(bool condition, unsigned int max);

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    short mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg) {}

  virtual void SynchronizeStream(int stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {}
  virtual void StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents = 1) {}
  virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) { return true; }
  virtual void RecordMarker(deviceEvent* ev, int stream) {}
  virtual void SynchronizeGPU() {}
  virtual void ReleaseEvent(deviceEvent* ev) {}
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
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) override;
  virtual size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst);

  int InitDevice() override;
  int ExitDevice() override;
  int GetThread();

  virtual int PrepareTextures() { return 0; }
  virtual int DoStuckProtection(int stream, void* event) { return 0; }

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
  template <class T, int I = 0, int J = -1>
  HighResTimer& getKernelTimer(RecoStep step, int num = 0, size_t addMemorySize = 0);
  template <class T, int J = -1>
  HighResTimer& getTimer(const char* name, int num = -1);

  std::vector<std::vector<deviceEvent*>> mEvents;

 private:
  size_t TransferMemoryResourcesHelper(GPUProcessor* proc, int stream, bool all, bool toGPU);
  unsigned int getNextTimerId();
  timerMeta* getTimerById(unsigned int id);
  timerMeta* insertTimer(unsigned int id, std::string&& name, int J, int num, int type, RecoStep step);
  int getOMPThreadNum();
  int getOMPMaxThreads();
};

template <class S, int I, int J, typename... Args>
inline int GPUReconstructionCPU::runKernel(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
  HighResTimer* t = nullptr;
  GPUCA_RECO_STEP myStep = S::GetRecoStep() == GPUCA_RECO_STEP::NoRecoStep ? x.step : S::GetRecoStep();
  if (myStep == GPUCA_RECO_STEP::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int cpuFallback = IsGPU() ? (x.device == krnlDeviceType::CPU ? 2 : (mRecoStepsGPU & myStep) != myStep) : 0;
  unsigned int nThreads = x.nThreads;
  unsigned int nBlocks = x.nBlocks;
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
    GPUInfo("Running kernel %s (Stream %d, Range %d/%d, Grid %d/%d) on %s", GetKernelName<S, I>(), x.stream, y.start, y.num, nBlocks, nThreads, cpuFallback == 2 ? "CPU (forced)" : cpuFallback ? "CPU (fallback)" : mDeviceName.c_str());
  }
  if (nThreads == 0 || nBlocks == 0) {
    return 0;
  }
  if (mProcessingSettings.debugLevel >= 1) {
    t = &getKernelTimer<S, I, J>(myStep, !IsGPU() || cpuFallback ? getOMPThreadNum() : x.stream);
    if ((!mProcessingSettings.deviceTimers || !IsGPU() || cpuFallback) && (mNestedLoopOmpFactor < 2 || getOMPThreadNum() == 0)) {
      t->Start();
    }
  }
  krnlSetup setup{{nBlocks, nThreads, x.stream, x.device, x.step}, y, z, 0.};
  if (cpuFallback) {
    if (GPUReconstructionCPU::runKernelImpl(classArgument<S, I>(), setup, args...)) {
      return 1;
    }
  } else {
    if (runKernelImpl(classArgument<S, I>(), setup, args...)) {
      return 1;
    }
  }
  if (GPUDebug(GetKernelName<S, I>(), x.stream)) {
    throw std::runtime_error("kernel failure");
  }
  if (mProcessingSettings.debugLevel >= 1) {
    if (t) {
      if (!(!mProcessingSettings.deviceTimers || !IsGPU() || cpuFallback)) {
        t->AddTime(setup.t);
      } else if (mNestedLoopOmpFactor < 2 || getOMPThreadNum() == 0) {
        t->Stop();
      }
    }
    if (CheckErrorCodes(cpuFallback)) {
      throw std::runtime_error("kernel error code");
    }
  }
  return 0;
}

#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward)                               \
  template <>                                                                                 \
  constexpr const char* GPUReconstructionCPU::GetKernelName<GPUCA_M_KRNL_TEMPLATE(x_class)>() \
  {                                                                                           \
    return GPUCA_M_STR(GPUCA_M_KRNL_NAME(x_class));                                           \
  }
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

template <class T>
inline void GPUReconstructionCPU::AddGPUEvents(T*& events)
{
  mEvents.emplace_back(std::vector<deviceEvent*>(sizeof(T) / sizeof(deviceEvent*)));
  events = (T*)mEvents.back().data();
}

template <class T, int I, int J>
HighResTimer& GPUReconstructionCPU::getKernelTimer(RecoStep step, int num, size_t addMemorySize)
{
  static int id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
  if (timer == nullptr) {
    timer = insertTimer(id, GetKernelName<T, I>(), J, NSLICES, 0, step);
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
