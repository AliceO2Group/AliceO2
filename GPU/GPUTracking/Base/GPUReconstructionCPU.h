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
#include "GPUTPCNeighboursFinder.h"
#include "GPUTPCNeighboursCleaner.h"
#include "GPUTPCStartHitsFinder.h"
#include "GPUTPCStartHitsSorter.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUTPCTrackletSelector.h"
#include "GPUTPCGMMergerGPU.h"
#include "GPUTRDTrackerGPU.h"
#ifdef HAVE_O2HEADERS
#include "GPUITSFitterKernels.h"
#include "GPUTPCConvertKernel.h"
#include "GPUTPCCompressionKernels.h"
#include "GPUTPCClusterFinderKernels.h"
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
  GPUReconstructionCPUBackend(const GPUSettingsProcessing& cfg) : GPUReconstruction(cfg) {}
  template <class T, int I = 0, typename... Args>
  int runKernelBackend(krnlSetup& _xyz, const Args&... args);
};

template <class T>
class GPUReconstructionKernels : public T
{
 public:
  using krnlSetup = GPUReconstruction::krnlSetup;
  template <class X, int Y = 0>
  using classArgument = GPUReconstruction::classArgument<X, Y>;
  virtual ~GPUReconstructionKernels() = default; // NOLINT: BUG: Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : T(cfg) {}

 protected:
#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward)                                                        \
  virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, krnlSetup& _xyz GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                    \
    return T::template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(_xyz GPUCA_M_STRIP(x_forward));                \
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
  GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : GPUReconstructionCPUBackend(cfg) {}

 protected:
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, krnlSetup& _xyz GPUCA_M_STRIP(x_arguments));
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
};
#endif

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsProcessing& cfg);
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

  template <class T, int I>
  constexpr static const char* GetKernelName();

  virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1);
  int registerMemoryForGPU(void* ptr, size_t size) override { return 0; }
  int unregisterMemoryForGPU(void* ptr) override { return 0; }
  int GPUStuck() { return mGPUStuck; }
  int NStreams() { return mNStreams; }
  void SetThreadCounts(RecoStep step);
  void ResetDeviceProcessorTypes();
  template <class T>
  void AddGPUEvents(T*& events);

  int RunChains() override;

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    short mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsProcessing& cfg) : GPUReconstructionKernels(cfg) {}

  virtual void SynchronizeStream(int stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {}
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
  virtual size_t GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  virtual size_t GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) override;
  virtual size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst);

  virtual void SetThreadCounts();

  int InitDevice() override;
  int ExitDevice() override;
  int GetThread();

  virtual int PrepareTextures() { return 0; }
  virtual int DoStuckProtection(int stream, void* event) { return 0; }

  // Pointers to tracker classes
  GPUProcessorProcessors mProcShadow; // Host copy of tracker objects that will be used on the GPU
  GPUConstantMem*& mProcessorsShadow = mProcShadow.mProcessorsProc;

  unsigned int mBlockCount = 0;            // Default GPU block count
  unsigned int mThreadCount = 0;           // Default GPU thread count
  unsigned int mConstructorBlockCount = 0; // GPU blocks used in Tracklet Constructor
  unsigned int mSelectorBlockCount = 0;    // GPU blocks used in Tracklet Selector
  unsigned int mConstructorThreadCount = 0;
  unsigned int mSelectorThreadCount = 0;
  unsigned int mFinderThreadCount = 0;
  unsigned int mTRDThreadCount = 0;
  unsigned int mClustererThreadCount = 0;
  unsigned int mScanThreadCount = 0;
  unsigned int mConverterThreadCount = 0;
  unsigned int mCompression1ThreadCount = 0;
  unsigned int mCompression2ThreadCount = 0;
  unsigned int mCFDecodeThreadCount = 0;
  unsigned int mFitThreadCount = 0;
  unsigned int mITSThreadCount = 0;

  int mThreadId = -1; // Thread ID that is valid for the local CUDA context
  int mGPUStuck = 0;  // Marks that the GPU is stuck, skip future events
  int mNStreams = 1;

  struct timerMeta {
    std::unique_ptr<HighResTimer[]> timer;
    std::string name;
    int num;            // How many parallel instances to sum up (CPU threads / GPU streams)
    int type;           // 0 = kernel, 1 = CPU step, 2 = DMA transfer
    unsigned int count; // How often was the timer queried
    RecoStep step;      // Which RecoStep is this
  };

  struct RecoStepTimerMeta {
    HighResTimer timer;
    HighResTimer timerToGPU;
    HighResTimer timerToHost;
    size_t bytesToGPU = 0;
    size_t bytesToHost = 0;
    unsigned int countToGPU = 0;
    unsigned int countToHost = 0;
  };

  constexpr static int N_RECO_STEPS = sizeof(GPUDataTypes::RECO_STEP_NAMES) / sizeof(GPUDataTypes::RECO_STEP_NAMES[0]);
  std::vector<std::unique_ptr<timerMeta>> mTimers;
  RecoStepTimerMeta mTimersRecoSteps[N_RECO_STEPS];
  template <class T, int I = 0, int J = -1>
  HighResTimer& getKernelTimer(RecoStep step, int num = 0);
  template <class T, int J = -1>
  HighResTimer& getTimer(const char* name, int num = -1);
  int getRecoStepNum(RecoStep step, bool validCheck = true);

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
  if (x.nThreads > GPUCA_MAX_THREADS) {
    throw std::runtime_error("GPUCA_MAX_THREADS exceeded");
  }
  HighResTimer* t = nullptr;
  GPUCA_RECO_STEP myStep = S::GetRecoStep() == GPUCA_RECO_STEP::NoRecoStep ? x.step : S::GetRecoStep();
  if (myStep == GPUCA_RECO_STEP::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int cpuFallback = IsGPU() ? (x.device == krnlDeviceType::CPU ? 2 : (mRecoStepsGPU & myStep) != myStep) : 0;
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo("Running %s (Stream %d, Range %d/%d, Grid %d/%d) on %s", GetKernelName<S, I>(), x.stream, y.start, y.num, x.nBlocks, x.nThreads, cpuFallback == 2 ? "CPU (forced)" : cpuFallback ? "CPU (fallback)" : mDeviceName.c_str());
  }
  if (mDeviceProcessingSettings.debugLevel >= 0) {
    t = &getKernelTimer<S, I, J>(myStep, !IsGPU() || cpuFallback ? getOMPThreadNum() : x.stream);
    if (!mDeviceProcessingSettings.deviceTimers || cpuFallback) {
      t->Start();
    }
  }
  krnlSetup setup{x, y, z, 0.};
  if (cpuFallback) {
    if (GPUReconstructionCPU::runKernelImpl(classArgument<S, I>(), setup, args...)) {
      return 1;
    }
  } else {
    if (runKernelImpl(classArgument<S, I>(), setup, args...)) {
      return 1;
    }
  }
  if (mDeviceProcessingSettings.debugLevel >= 0) {
    if (GPUDebug(typeid(S).name(), x.stream)) {
      throw std::runtime_error("kernel failure");
    }
    if (t) {
      if (!mDeviceProcessingSettings.deviceTimers || cpuFallback) {
        t->Stop();
      } else {
        t->AddTime(setup.t);
      }
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
HighResTimer& GPUReconstructionCPU::getKernelTimer(RecoStep step, int num)
{
  static int id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
  if (timer == nullptr) {
    int max = step == GPUCA_RECO_STEP::NoRecoStep || step == GPUCA_RECO_STEP::TPCSliceTracking || step == GPUCA_RECO_STEP::TPCClusterFinding ? NSLICES : 1;
    timer = insertTimer(id, GetKernelName<T, I>(), J, max, 0, step);
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
    int max = std::max({getOMPMaxThreads(), mDeviceProcessingSettings.nDeviceHelperThreads + 1, mDeviceProcessingSettings.nStreams});
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
