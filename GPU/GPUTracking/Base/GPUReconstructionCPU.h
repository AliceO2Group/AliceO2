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

#include "GPUDefMacros.h"
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

#define GPUCA_M_KRNL_TEMPLATE_B(a, b, ...) a, a::b
#define GPUCA_M_KRNL_TEMPLATE_A(...) GPUCA_M_KRNL_TEMPLATE_B(__VA_ARGS__, defaultKernel)
#define GPUCA_M_KRNL_TEMPLATE(...) GPUCA_M_KRNL_TEMPLATE_A(GPUCA_M_STRIP(__VA_ARGS__))

template <class T>
class GPUReconstructionKernels : public T
{
 public:
  using krnlSetup = GPUReconstruction::krnlSetup;
  template <class X, int Y = 0>
  using classArgument = GPUReconstruction::classArgument<X, Y>;
  virtual ~GPUReconstructionKernels() = default; // NOLINT: Do not declare override in template class! AMD hcc will not create the destructor otherwise.
  GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : T(cfg) {}

 protected:
#define GPUCA_KRNL(class, attributes, arguments, forward)                                                          \
  virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(class)>, krnlSetup& _xyz GPUCA_M_STRIP(arguments)) \
  {                                                                                                                \
    return T::template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(class)>(_xyz GPUCA_M_STRIP(forward));                \
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
#define GPUCA_KRNL(class, attributes, arguments, forward) virtual int runKernelImpl(classArgument<GPUCA_M_KRNL_TEMPLATE(class)>, krnlSetup& _xyz GPUCA_M_STRIP(arguments));
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
};
#endif
#undef GPUCA_M_KRNL_TEMPLATE
#undef GPUCA_M_KRNL_TEMPLATE_A
#undef GPUCA_M_KRNL_TEMPLATE_B

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend class GPUReconstruction;
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

#ifdef __clang__ // BUG: clang does not accept default parameters before parameter pack
  template <class S, int I = 0>
  inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone)
  {
    return runKernel<S, I>(x, t, y, krnlEventNone);
  }
  template <class S, int I = 0, typename... Args>
  inline int runKernel(const krnlExec& x, HighResTimer* t, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
#else
  template <class S, int I = 0, typename... Args>
  inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEventNone, const Args&... args)
#endif
  {
    if (x.nThreads > GPUCA_MAX_THREADS) {
      throw std::runtime_error("GPUCA_MAX_THREADS exceeded");
    }
    int cpuFallback = IsGPU() ? (x.device == krnlDeviceType::CPU ? 2 : (mRecoStepsGPU & S::GetRecoStep()) != S::GetRecoStep()) : 0;
    if (mDeviceProcessingSettings.debugLevel >= 3) {
      GPUInfo("Running %s-%d (Stream %d, Range %d/%d, Grid %d/%d) on %s", typeid(S).name(), I, x.stream, y.start, y.num, x.nBlocks, x.nThreads, cpuFallback == 2 ? "CPU (forced)" : cpuFallback ? "CPU (fallback)" : mDeviceName.c_str());
    }
    if (t && mDeviceProcessingSettings.debugLevel) {
      t->Start();
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
    if (mDeviceProcessingSettings.debugLevel) {
      if (GPUDebug(typeid(S).name(), x.stream)) {
        throw std::runtime_error("kernel failure");
      }
      if (t) {
        t->Stop();
      }
    }
    return 0;
  }

  virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1);
  void TransferMemoryResourceToGPU(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice()); }
  void TransferMemoryResourceToHost(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr()); }
  void TransferMemoryResourcesToGPU(GPUProcessor* proc, int stream = -1, bool all = false) { TransferMemoryResourcesHelper(proc, stream, all, true); }
  void TransferMemoryResourcesToHost(GPUProcessor* proc, int stream = -1, bool all = false) { TransferMemoryResourcesHelper(proc, stream, all, false); }
  void TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents); }
  void TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents); }
  virtual void GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  virtual void GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1);
  void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) override;
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
  virtual void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst);

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

  int mThreadId = -1; // Thread ID that is valid for the local CUDA context
  int mGPUStuck = 0;  // Marks that the GPU is stuck, skip future events
  int mNStreams = 1;

  std::vector<std::vector<deviceEvent*>> mEvents;

 private:
  void TransferMemoryResourcesHelper(GPUProcessor* proc, int stream, bool all, bool toGPU);
};

template <class T>
inline void GPUReconstructionCPU::AddGPUEvents(T*& events)
{
  mEvents.emplace_back(std::vector<deviceEvent*>(sizeof(T) / sizeof(deviceEvent*)));
  events = (T*)mEvents.back().data();
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
