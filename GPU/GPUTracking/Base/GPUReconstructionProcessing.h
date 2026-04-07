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

/// \file GPUReconstructionProcessing.h
/// \author David Rohr

#if !defined(GPURECONSTRUCTIONPROCESSING_H)
#define GPURECONSTRUCTIONPROCESSING_H

#include "GPUReconstruction.h"

#include "utils/timer.h"
#include <functional>
#include <atomic>

namespace Ort
{
struct SessionOptions;
}

namespace o2::gpu
{

struct GPUDefParameters;

namespace gpu_reconstruction_kernels // TODO: Get rid of this namespace
{
struct deviceEvent {
  constexpr deviceEvent() = default;
  constexpr deviceEvent(std::nullptr_t p) : v(nullptr) {};
  template <class T>
  void set(T val)
  {
    v = reinterpret_cast<void*&>(val);
  }
  template <class T>
  T& get()
  {
    return reinterpret_cast<T&>(v);
  }
  template <class T>
  T* getEventList()
  {
    return reinterpret_cast<T*>(this);
  }
  bool isSet() const { return v; }

 private:
  void* v = nullptr; // We use only pointers anyway, and since cl_event and cudaEvent_t and hipEvent_t are actually pointers, we can cast them to deviceEvent (void*) this way.
};

class threadContext
{
 public:
  threadContext();
  virtual ~threadContext();
};

} // namespace gpu_reconstruction_kernels

class GPUReconstructionProcessing : public GPUReconstruction
{
 public:
  ~GPUReconstructionProcessing() override;

  using deviceEvent = gpu_reconstruction_kernels::deviceEvent;
  using threadContext = gpu_reconstruction_kernels::threadContext;

  struct RecoStepTimerMeta {
    HighResTimer timerToGPU;
    HighResTimer timerToHost;
    HighResTimer timerTotal;
    double timerCPU = 0.;
    size_t bytesToGPU = 0;
    size_t bytesToHost = 0;
    uint32_t countToGPU = 0;
    uint32_t countToHost = 0;
  };

  template <class T, int32_t I = 0>
  struct kernelInterfaceArguments {
    using t = T;
    static constexpr int32_t i = I;
  };

  struct krnlExec {
    constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d), step(gpudatatypes::RecoStep::NoRecoStep) {}
    constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, gpudatatypes::RecoStep st) : nBlocks(b), nThreads(t), stream(s), device(GPUReconstruction::krnlDeviceType::Auto), step(st) {}
    constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, GPUReconstruction::krnlDeviceType d, gpudatatypes::RecoStep st) : nBlocks(b), nThreads(t), stream(s), device(d), step(st) {}
    uint32_t nBlocks;
    uint32_t nThreads;
    int32_t stream;
    GPUReconstruction::krnlDeviceType device;
    gpudatatypes::RecoStep step;
  };
  struct krnlRunRange {
    constexpr krnlRunRange() = default;
    constexpr krnlRunRange(uint32_t v) : index(v) {}
    uint32_t index = 0;
  };
  struct krnlEvent {
    constexpr krnlEvent(deviceEvent* e = nullptr, deviceEvent* el = nullptr, int32_t n = 1) : ev(e), evList(el), nEvents(n) {}
    deviceEvent* ev;
    deviceEvent* evList;
    int32_t nEvents;
  };

  struct krnlProperties {
    krnlProperties(int32_t t = 0, int32_t b = 1, int32_t b2 = 0) : nThreads(t), minBlocks(b), forceBlocks(b2) {}
    uint32_t nThreads;
    uint32_t minBlocks;
    uint32_t forceBlocks;
    uint32_t total() { return forceBlocks ? forceBlocks : (nThreads * minBlocks); }
  };

  struct krnlSetup {
    krnlSetup(const krnlExec& xx, const krnlRunRange& yy = {0}, const krnlEvent& zz = {nullptr, nullptr, 0}) : x(xx), y(yy), z(zz) {}
    krnlExec x;
    krnlRunRange y;
    krnlEvent z;
  };

  struct krnlSetupTime : public krnlSetup {
    double& t;
  };

  template <class T, int32_t I = 0, typename... Args>
  struct krnlSetupArgs : public kernelInterfaceArguments<T, I> {
    const krnlSetupTime s;
    std::tuple<typename std::conditional<(sizeof(Args) > sizeof(void*)), const Args&, const Args>::type...> v;
  };

  template <class T, class S>
  class KernelInterface : public S
  {
   public:
    template <typename... Args>
    KernelInterface(const Args&... args) : S(args...)
    {
    }

   protected:
    virtual void runKernelVirtual(const int num, const void* args);
  };

  // Threading
  int32_t getNKernelHostThreads(bool splitCores);
  uint32_t getNActiveThreadsOuterLoop() const { return mNActiveThreadsOuterLoop; }
  void SetNActiveThreadsOuterLoop(uint32_t f) { mNActiveThreadsOuterLoop = f; }
  uint32_t SetAndGetNActiveThreadsOuterLoop(bool condition, uint32_t max);
  void runParallelOuterLoop(bool doGPU, uint32_t nThreads, std::function<void(uint32_t)> lambda);
  void SetNActiveThreads(int32_t n);

  // Interface to query name of a kernel
  template <class T, int32_t I>
  static const char* GetKernelName();
  const std::string& GetKernelName(int32_t i) const { return mKernelNames[i]; }
  template <class T, int32_t I = 0>
  static uint32_t GetKernelNum();

  // Public queries for timers
  auto& getRecoStepTimer(RecoStep step) { return mTimersRecoSteps[getRecoStepNum(step)]; }
  HighResTimer& getGeneralStepTimer(GeneralStep step) { return mTimersGeneralSteps[getGeneralStepNum(step)]; }

  template <class T>
  void AddGPUEvents(T*& events);

  virtual std::unique_ptr<threadContext> GetThreadContext() override;

  const GPUDefParameters& getGPUParameters(bool doGPU) const override { return *(doGPU ? mParDevice : mParCPU); }

 protected:
  GPUReconstructionProcessing(const GPUSettingsDeviceBackend& cfg);

  static const std::vector<std::string> mKernelNames;

  int32_t mActiveHostKernelThreads = 0;  // Number of currently active threads on the host for kernels
  uint32_t mNActiveThreadsOuterLoop = 1; // Number of threads currently running an outer loop

  std::vector<std::vector<deviceEvent>> mEvents;

  // Timer related stuff
  struct timerMeta {
    std::unique_ptr<HighResTimer[]> timer;
    std::string name;
    int32_t num;    // How many parallel instances to sum up (CPU threads / GPU streams)
    int32_t type;   // 0 = kernel, 1 = CPU step, 2 = DMA transfer
    uint32_t count; // How often was the timer queried
    RecoStep step;  // Which RecoStep is this
    size_t memSize; // Memory size for memory bandwidth computation
  };

  HighResTimer mTimersGeneralSteps[gpudatatypes::N_GENERAL_STEPS];

  std::vector<std::unique_ptr<timerMeta>> mTimers;
  RecoStepTimerMeta mTimersRecoSteps[gpudatatypes::N_RECO_STEPS];
  HighResTimer mTimerTotal;
  template <class T, int32_t I = 0>
  HighResTimer& getKernelTimer(RecoStep step, int32_t num = 0, size_t addMemorySize = 0, bool increment = true);
  template <class T, int32_t J = -1>
  HighResTimer& getTimer(const char* name, int32_t num = -1);

  GPUDefParameters* mParCPU = nullptr;
  GPUDefParameters* mParDevice = nullptr;

 private:
  uint32_t getNextTimerId();
  timerMeta* getTimerById(uint32_t id, bool increment = true);
  timerMeta* insertTimer(uint32_t id, std::string&& name, int32_t J, int32_t num, int32_t type, RecoStep step);

  static std::atomic_flag mTimerFlag;
};

template <class T>
inline void GPUReconstructionProcessing::AddGPUEvents(T*& events)
{
  mEvents.emplace_back(std::vector<deviceEvent>(sizeof(T) / sizeof(deviceEvent)));
  events = (T*)mEvents.back().data();
}

template <class T, int32_t I>
HighResTimer& GPUReconstructionProcessing::getKernelTimer(RecoStep step, int32_t num, size_t addMemorySize, bool increment)
{
  static int32_t id = getNextTimerId();
  timerMeta* timer = getTimerById(id, increment);
  if (timer == nullptr) {
    timer = insertTimer(id, GetKernelName<T, I>(), -1, NSECTORS, 0, step);
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
HighResTimer& GPUReconstructionProcessing::getTimer(const char* name, int32_t num)
{
  static int32_t id = getNextTimerId();
  timerMeta* timer = getTimerById(id);
  if (timer == nullptr) {
    int32_t max = std::max<int32_t>({mMaxHostThreads, constants::GPU_MAX_STREAMS});
    timer = insertTimer(id, name, J, max, 1, RecoStep::NoRecoStep);
  }
  if (num == -1) {
    num = getHostThreadIndex();
  }
  if (num < 0 || num >= timer->num) {
    throw std::runtime_error("Invalid timer requested");
  }
  return timer->timer[num];
}

} // namespace o2::gpu

#endif
