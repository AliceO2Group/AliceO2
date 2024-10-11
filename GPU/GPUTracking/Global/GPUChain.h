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

/// \file GPUChain.h
/// \author David Rohr

#ifndef GPUCHAIN_H
#define GPUCHAIN_H

#include "GPUReconstructionCPU.h"
#include "GPUReconstructionHelpers.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUChain
{
  friend class GPUReconstruction;

 public:
  using RecoStep = GPUReconstruction::RecoStep;
  using GeneralStep = GPUReconstruction::GeneralStep;
  using InOutPointerType = GPUReconstruction::InOutPointerType;
  using GeometryType = GPUReconstruction::GeometryType;
  using krnlRunRange = gpu_reconstruction_kernels::krnlRunRange;
  using krnlExec = gpu_reconstruction_kernels::krnlExec;
  using krnlEvent = gpu_reconstruction_kernels::krnlEvent;
  using deviceEvent = gpu_reconstruction_kernels::deviceEvent;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  virtual ~GPUChain() = default;
  virtual void RegisterPermanentMemoryAndProcessors() = 0;
  virtual void RegisterGPUProcessors() = 0;
  virtual int32_t EarlyConfigure() { return 0; };
  virtual int32_t Init() = 0;
  virtual int32_t PrepareEvent() = 0;
  virtual int32_t Finalize() = 0;
  virtual int32_t RunChain() = 0;
  virtual void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) = 0;
  virtual void PrintMemoryStatistics(){};
  virtual int32_t CheckErrorCodes(bool cpuOnly = false, bool forceShowErrors = false, std::vector<std::array<uint32_t, 4>>* fillErrors = nullptr) { return 0; }
  virtual bool SupportsDoublePipeline() { return false; }
  virtual int32_t FinalizePipelinedProcessing() { return 0; }

  constexpr static int32_t NSLICES = GPUReconstruction::NSLICES;

  virtual void DumpSettings(const char* dir = "") {}
  virtual void ReadSettings(const char* dir = "") {}

  const GPUParam& GetParam() const { return mRec->mHostConstantMem->param; }
  const GPUSettingsGRP& GetGRPSettings() const { return mRec->mGRPSettings; }
  const GPUCalibObjectsConst& calib() const { return processors()->calibObjects; }
  GPUReconstruction* rec() { return mRec; }
  const GPUReconstruction* rec() const { return mRec; }
  inline const GPUConstantMem* GetProcessors() { return mRec->processors(); }

  // Make functions from GPUReconstruction*** available
  GPUReconstruction::RecoStepField GetRecoSteps() const { return mRec->GetRecoSteps(); }
  GPUReconstruction::RecoStepField GetRecoStepsGPU() const { return mRec->GetRecoStepsGPU(); }
  GPUReconstruction::InOutTypeField GetRecoStepsInputs() const { return mRec->GetRecoStepsInputs(); }
  GPUReconstruction::InOutTypeField GetRecoStepsOutputs() const { return mRec->GetRecoStepsOutputs(); }
  inline const GPUSettingsDeviceBackend& GetDeviceBackendSettings() const { return mRec->mDeviceBackendSettings; }
  inline const GPUSettingsProcessing& GetProcessingSettings() const { return mRec->mProcessingSettings; }

 protected:
  GPUReconstructionCPU* mRec;
  GPUChain(GPUReconstruction* rec) : mRec((GPUReconstructionCPU*)rec) {}

  int32_t GetThread();
  // Make functions from GPUReconstruction*** available
  inline GPUConstantMem* processors() { return mRec->processors(); }
  inline GPUConstantMem* processorsShadow() { return mRec->mProcessorsShadow; }
  inline GPUConstantMem* processorsDevice() { return mRec->mDeviceConstantMem; }
  inline GPUParam& param() { return mRec->param(); }
  inline const GPUConstantMem* processors() const { return mRec->processors(); }
  inline void SynchronizeStream(int32_t stream) { mRec->SynchronizeStream(stream); }
  inline void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) { mRec->SynchronizeEvents(evList, nEvents); }
  inline void SynchronizeEventAndRelease(deviceEvent& ev, bool doGPU = true)
  {
    if (doGPU) {
      SynchronizeEvents(&ev);
      ReleaseEvent(ev);
    }
  }
  template <class T>
  inline void CondWaitEvent(T& cond, deviceEvent* ev)
  {
    if (cond == true) {
      SynchronizeEvents(ev);
      cond = 2;
    }
  }
  inline bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) { return mRec->IsEventDone(evList, nEvents); }
  inline void RecordMarker(deviceEvent ev, int32_t stream) { mRec->RecordMarker(ev, stream); }
  virtual inline std::unique_ptr<GPUReconstruction::GPUThreadContext> GetThreadContext() { return mRec->GetThreadContext(); }
  inline void SynchronizeGPU() { mRec->SynchronizeGPU(); }
  inline void ReleaseEvent(deviceEvent ev, bool doGPU = true)
  {
    if (doGPU) {
      mRec->ReleaseEvent(ev);
    }
  }
  inline void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) { mRec->StreamWaitForEvents(stream, evList, nEvents); }
  template <class T>
  void RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int32_t count);
  inline void WaitForHelperThreads() { mRec->WaitForHelperThreads(); }
  inline int32_t HelperError(int32_t iThread) const { return mRec->HelperError(iThread); }
  inline int32_t HelperDone(int32_t iThread) const { return mRec->HelperDone(iThread); }
  inline void ResetHelperThreads(int32_t helpers) { mRec->ResetHelperThreads(helpers); }
  inline int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1) { return mRec->GPUDebug(state, stream); }
  // nEvents is forced to 0 if evList ==  nullptr
  inline void TransferMemoryResourceToGPU(RecoStep step, GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourceToGPU, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourceToHost(RecoStep step, GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourceToHost, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourcesToGPU(RecoStep step, GPUProcessor* proc, int32_t stream = -1, bool all = false) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourcesToGPU, proc, stream, all); }
  inline void TransferMemoryResourcesToHost(RecoStep step, GPUProcessor* proc, int32_t stream = -1, bool all = false) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourcesToHost, proc, stream, all); }
  inline void TransferMemoryResourceLinkToGPU(RecoStep step, int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourceLinkToGPU, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourceLinkToHost(RecoStep step, int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourceLinkToHost, res, stream, ev, evList, nEvents); }
  // Todo: retrieve step from proc, move kernelClass->GetStep to retrieve it from GetProcessor
  inline void WriteToConstantMemory(RecoStep step, size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) { timeCpy(step, true, &GPUReconstructionCPU::WriteToConstantMemory, offset, src, size, stream, ev); }
  inline void GPUMemCpy(RecoStep step, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { timeCpy(step, toGPU, &GPUReconstructionCPU::GPUMemCpy, dst, src, size, stream, toGPU, ev, evList, nEvents); }
  inline void GPUMemCpyAlways(RecoStep step, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1)
  {
    if (toGPU == -1) {
      memcpy(dst, src, size);
    } else {
      timeCpy<true>(step, toGPU, &GPUReconstructionCPU::GPUMemCpyAlways, GetRecoStepsGPU() & step, dst, src, size, stream, toGPU, ev, evList, nEvents);
    }
  }

  template <class T>
  inline void AllocateIOMemoryHelper(uint32_t n, const T*& ptr, std::unique_ptr<T[]>& u)
  {
    mRec->AllocateIOMemoryHelper<T>(n, ptr, u);
  }
  template <class T, class S>
  inline uint32_t DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type)
  {
    return mRec->DumpData<T>(fp, entries, num, type);
  }
  template <class T, class S>
  inline size_t ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type, T** nonConstPtrs = nullptr)
  {
    return mRec->ReadData<T>(fp, entries, num, mem, type, nonConstPtrs);
  }
  template <class T>
  inline void DumpFlatObjectToFile(const T* obj, const char* file)
  {
    mRec->DumpFlatObjectToFile<T>(obj, file);
  }
  template <class T>
  inline std::unique_ptr<T> ReadFlatObjectFromFile(const char* file)
  {
    return mRec->ReadFlatObjectFromFile<T>(file);
  }
  template <class T>
  inline void DumpStructToFile(const T* obj, const char* file)
  {
    mRec->DumpStructToFile<T>(obj, file);
  }
  template <class T>
  inline std::unique_ptr<T> ReadStructFromFile(const char* file)
  {
    return mRec->ReadStructFromFile<T>(file);
  }
  template <class T>
  inline void ReadStructFromFile(const char* file, T* obj)
  {
    mRec->ReadStructFromFile<T>(file, obj);
  }
  template <class S, int32_t I = 0, typename... Args>
  inline int32_t runKernel(gpu_reconstruction_kernels::krnlSetup&& setup, Args&&... args)
  {
    return mRec->runKernel<S, I, Args...>(std::forward<gpu_reconstruction_kernels::krnlSetup&&>(setup), std::forward<Args>(args)...);
  }
  template <class S, int32_t I = 0>
  gpu_reconstruction_kernels::krnlProperties getKernelProperties()
  {
    return mRec->getKernelProperties<S, I>();
  }

  template <class T, int32_t I = 0>
  HighResTimer& getKernelTimer(RecoStep step, int32_t num = 0, size_t addMemorySize = 0, bool increment = true)
  {
    return mRec->getKernelTimer<T, I>(step, num, addMemorySize, increment);
  }
  template <class T, int32_t J = -1>
  HighResTimer& getTimer(const char* name, int32_t num = -1)
  {
    return mRec->getTimer<T, J>(name, num);
  }
  // Get GRID with NBLOCKS minimal such that nThreads * NBLOCS >= totalItems
  krnlExec GetGrid(uint32_t totalItems, uint32_t nThreads, int32_t stream, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);
  // Get GRID with NBLOCKS minimal such that ideal number of threads * NBLOCKS >= totalItems
  krnlExec GetGrid(uint32_t totalItems, int32_t stream, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);
  // Get GRID with specified number of blocks, each block with ideal number of threads
  krnlExec GetGridBlk(uint32_t nBlocks, int32_t stream, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);
  krnlExec GetGridBlkStep(uint32_t nBlocks, int32_t stream, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);
  // Get GRID with ideal number of threads / blocks for GPU
  krnlExec GetGridAuto(int32_t stream, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);
  krnlExec GetGridAutoStep(int32_t stream, GPUCA_RECO_STEP st = GPUCA_RECO_STEP::NoRecoStep);

  inline uint32_t BlockCount() const { return mRec->mBlockCount; }
  inline uint32_t WarpSize() const { return mRec->mWarpSize; }
  inline uint32_t ThreadCount() const { return mRec->mThreadCount; }

  inline size_t AllocateRegisteredMemory(GPUProcessor* proc) { return mRec->AllocateRegisteredMemory(proc); }
  inline size_t AllocateRegisteredMemory(int16_t res, GPUOutputControl* control = nullptr) { return mRec->AllocateRegisteredMemory(res, control); }
  template <class T>
  inline void SetupGPUProcessor(T* proc, bool allocate)
  {
    mRec->SetupGPUProcessor<T>(proc, allocate);
  }

  inline GPUChain* GetNextChainInQueue() { return mRec->GetNextChainInQueue(); }

  virtual int32_t PrepareTextures() { return 0; }
  virtual int32_t DoStuckProtection(int32_t stream, deviceEvent event) { return 0; }

  template <class T, class S, typename... Args>
  bool DoDebugAndDump(RecoStep step, int32_t mask, T& processor, S T::*func, Args&&... args)
  {
    return DoDebugAndDump(step, mask, true, processor, func, args...);
  }
  template <class T, class S, typename... Args>
  bool DoDebugAndDump(RecoStep step, int32_t mask, bool transfer, T& processor, S T::*func, Args&&... args);

  template <class T, class S, typename... Args>
  int32_t runRecoStep(RecoStep step, S T::*func, Args... args);

 private:
  template <bool Always = false, class T, class S, typename... Args>
  void timeCpy(RecoStep step, int32_t toGPU, S T::*func, Args... args);
};

template <class T>
inline void GPUChain::RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int32_t count)
{
  mRec->RunHelperThreads((int32_t(GPUReconstructionHelpers::helperDelegateBase::*)(int32_t, int32_t, GPUReconstructionHelpers::helperParam*))function, functionCls, count);
}

template <bool Always, class T, class S, typename... Args>
inline void GPUChain::timeCpy(RecoStep step, int32_t toGPU, S T::*func, Args... args)
{
  if (!Always && step != RecoStep::NoRecoStep && !(GetRecoStepsGPU() & step)) {
    return;
  }
  HighResTimer* timer = nullptr;
  size_t* bytes = nullptr;
  if (mRec->mProcessingSettings.debugLevel >= 1 && toGPU >= 0) { // Todo: time special cases toGPU < 0
    int32_t id = mRec->getRecoStepNum(step, false);
    if (id != -1) {
      auto& tmp = mRec->mTimersRecoSteps[id];
      timer = toGPU ? &tmp.timerToGPU : &tmp.timerToHost;
      bytes = toGPU ? &tmp.bytesToGPU : &tmp.bytesToHost;
      (toGPU ? tmp.countToGPU : tmp.countToHost)++;
      timer->Start();
    }
  }
  size_t n = (mRec->*func)(args...);
  if (timer) {
    SynchronizeGPU();
    timer->Stop();
    *bytes += n;
  }
}

template <class T, class S, typename... Args>
bool GPUChain::DoDebugAndDump(GPUChain::RecoStep step, int32_t mask, bool transfer, T& processor, S T::*func, Args&&... args)
{
  if (GetProcessingSettings().keepAllMemory) {
    if (transfer) {
      TransferMemoryResourcesToHost(step, &processor, -1, true);
    }
    if (GetProcessingSettings().debugLevel >= 6 && (mask == 0 || (GetProcessingSettings().debugMask & mask))) {
      if (func) {
        (processor.*func)(args...);
      }
      return true;
    }
  }
  return false;
}

template <class T, class S, typename... Args>
int32_t GPUChain::runRecoStep(RecoStep step, S T::*func, Args... args)
{
  if (GetRecoSteps().isSet(step)) {
    if (GetProcessingSettings().debugLevel >= 1) {
      mRec->getRecoStepTimer(step).Start();
    }
    int32_t retVal = (reinterpret_cast<T*>(this)->*func)(args...);
    if (GetProcessingSettings().debugLevel >= 1) {
      mRec->getRecoStepTimer(step).Stop();
    }
    return retVal;
  }
  return false;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
