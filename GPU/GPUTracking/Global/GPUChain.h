// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  using InOutPointerType = GPUReconstruction::InOutPointerType;
  using GeometryType = GPUReconstruction::GeometryType;
  using krnlRunRange = GPUReconstruction::krnlRunRange;
  using krnlExec = GPUReconstruction::krnlExec;
  using krnlEvent = GPUReconstruction::krnlEvent;
  using deviceEvent = GPUReconstruction::deviceEvent;
  static constexpr krnlRunRange krnlRunRangeNone{0, -1};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  virtual ~GPUChain() = default;
  virtual void RegisterPermanentMemoryAndProcessors() = 0;
  virtual void RegisterGPUProcessors() = 0;
  virtual int Init() = 0;
  virtual int PrepareEvent() = 0;
  virtual int Finalize() = 0;
  virtual int RunChain() = 0;
  virtual void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) = 0;
  virtual void PrintMemoryStatistics(){};

  constexpr static int NSLICES = GPUReconstruction::NSLICES;

  virtual void DumpSettings(const char* dir = "") {}
  virtual void ReadSettings(const char* dir = "") {}

  const GPUParam& GetParam() const { return mRec->mHostConstantMem->param; }
  const GPUSettingsEvent& GetEventSettings() const { return mRec->mEventSettings; }
  const GPUSettingsProcessing& GetProcessingSettings() const { return mRec->mProcessingSettings; }
  const GPUSettingsDeviceProcessing& GetDeviceProcessingSettings() const { return mRec->mDeviceProcessingSettings; }
  GPUReconstruction* rec() { return mRec; }
  const GPUReconstruction* rec() const { return mRec; }

  GPUReconstruction::RecoStepField GetRecoSteps() const { return mRec->GetRecoSteps(); }
  GPUReconstruction::RecoStepField GetRecoStepsGPU() const { return mRec->GetRecoStepsGPU(); }
  GPUReconstruction::InOutTypeField GetRecoStepsInputs() const { return mRec->GetRecoStepsInputs(); }
  GPUReconstruction::InOutTypeField GetRecoStepsOutputs() const { return mRec->GetRecoStepsOutputs(); }

 protected:
  GPUReconstructionCPU* mRec;
  GPUChain(GPUReconstruction* rec) : mRec((GPUReconstructionCPU*)rec) {}

  int GetThread();

  // Make functions from GPUReconstruction*** available
  inline GPUConstantMem* processors() { return mRec->processors(); }
  inline GPUConstantMem* processorsShadow() { return mRec->mProcessorsShadow; }
  inline GPUConstantMem* processorsDevice() { return mRec->mDeviceConstantMem; }
  inline GPUParam& param() { return mRec->param(); }
  inline const GPUConstantMem* processors() const { return mRec->processors(); }
  inline GPUSettingsDeviceProcessing& DeviceProcessingSettings() { return mRec->mDeviceProcessingSettings; }
  inline void SynchronizeStream(int stream) { mRec->SynchronizeStream(stream); }
  inline void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) { mRec->SynchronizeEvents(evList, nEvents); }
  inline bool IsEventDone(deviceEvent* evList, int nEvents = 1) { return mRec->IsEventDone(evList, nEvents); }
  inline void RecordMarker(deviceEvent* ev, int stream) { mRec->RecordMarker(ev, stream); }
  virtual inline std::unique_ptr<GPUReconstruction::GPUThreadContext> GetThreadContext() { return mRec->GetThreadContext(); }
  inline void SynchronizeGPU() { mRec->SynchronizeGPU(); }
  inline void ReleaseEvent(deviceEvent* ev) { mRec->ReleaseEvent(ev); }
  template <class T>
  void RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int count);
  inline void WaitForHelperThreads() { mRec->WaitForHelperThreads(); }
  inline int HelperError(int iThread) const { return mRec->HelperError(iThread); }
  inline int HelperDone(int iThread) const { return mRec->HelperDone(iThread); }
  inline void ResetHelperThreads(int helpers) { mRec->ResetHelperThreads(helpers); }
  inline int GPUDebug(const char* state = "UNKNOWN", int stream = -1) { return mRec->GPUDebug(state, stream); }
  inline void TransferMemoryResourceToGPU(RecoStep step, GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourceToGPU, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourceToHost(RecoStep step, GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourceToHost, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourcesToGPU(RecoStep step, GPUProcessor* proc, int stream = -1, bool all = false) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourcesToGPU, proc, stream, all); }
  inline void TransferMemoryResourcesToHost(RecoStep step, GPUProcessor* proc, int stream = -1, bool all = false) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourcesToHost, proc, stream, all); }
  inline void TransferMemoryResourceLinkToGPU(RecoStep step, short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy(step, true, &GPUReconstructionCPU::TransferMemoryResourceLinkToGPU, res, stream, ev, evList, nEvents); }
  inline void TransferMemoryResourceLinkToHost(RecoStep step, short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy(step, false, &GPUReconstructionCPU::TransferMemoryResourceLinkToHost, res, stream, ev, evList, nEvents); }
  inline void WriteToConstantMemory(RecoStep step, size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) { timeCpy(step, true, &GPUReconstructionCPU::WriteToConstantMemory, offset, src, size, stream, ev); }
  inline void GPUMemCpy(RecoStep step, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy(step, toGPU, &GPUReconstructionCPU::GPUMemCpy, dst, src, size, stream, toGPU, ev, evList, nEvents); }
  inline void GPUMemCpyAlways(RecoStep step, void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { timeCpy<true>(step, toGPU, &GPUReconstructionCPU::GPUMemCpyAlways, GetRecoStepsGPU() & step, dst, src, size, stream, toGPU, ev, evList, nEvents); }

  template <class T>
  inline void AllocateIOMemoryHelper(unsigned int n, const T*& ptr, std::unique_ptr<T[]>& u)
  {
    mRec->AllocateIOMemoryHelper<T>(n, ptr, u);
  }
  template <class T, class S>
  inline void DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type)
  {
    mRec->DumpData<T>(fp, entries, num, type);
  }
  template <class T, class S>
  inline size_t ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
  {
    return mRec->ReadData<T>(fp, entries, num, mem, type);
  }
  template <class T>
  inline void DumpFlatObjectToFile(const T* obj, const char* file)
  {
    mRec->DumpFlatObjectToFile<T>(obj, file);
  }
  template <class T>
  inline std::unique_ptr<T> ReadFlatObjectFromFile(const char* file)
  {
    return std::move(mRec->ReadFlatObjectFromFile<T>(file));
  }
  template <class T>
  inline void DumpStructToFile(const T* obj, const char* file)
  {
    mRec->DumpStructToFile<T>(obj, file);
  }
  template <class T>
  inline std::unique_ptr<T> ReadStructFromFile(const char* file)
  {
    return std::move(mRec->ReadStructFromFile<T>(file));
  }
  template <class T>
  inline void ReadStructFromFile(const char* file, T* obj)
  {
    mRec->ReadStructFromFile<T>(file, obj);
  }
#ifdef __clang__ // BUG: clang seems broken and does not accept default parameters before parameter pack
  template <class S, int I = 0, int J = -1>
  inline int runKernel(const krnlExec& x, const krnlRunRange& y = krnlRunRangeNone)
  {
    return mRec->runKernel<S, I, J>(x, y);
  }
  template <class S, int I = 0, int J = -1, typename... Args>
  inline int runKernel(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, Args&&... args)
#else
  template <class S, int I = 0, int J = -1, typename... Args>
  inline int runKernel(const krnlExec& x, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEventNone, Args&&... args)
#endif
  {
    return mRec->runKernel<S, I, J, Args...>(x, y, z, std::forward<Args>(args)...);
  }
  template <class T, int I = 0, int J = -1>
  HighResTimer& getKernelTimer(int num = 0)
  {
    return mRec->getKernelTimer<T, I, J>(num);
  }
  template <class T, int J = -1>
  HighResTimer& getTimer(const char* name, int num = -1)
  {
    return mRec->getTimer<T, J>(name, num);
  }
  krnlExec GetGrid(unsigned int totalItems, unsigned int nThreads, int stream);
  inline unsigned int BlockCount() const { return mRec->mBlockCount; }
  inline unsigned int ThreadCount() const { return mRec->mThreadCount; }
  inline unsigned int ConstructorBlockCount() const { return mRec->mConstructorBlockCount; }
  inline unsigned int SelectorBlockCount() const { return mRec->mSelectorBlockCount; }
  inline unsigned int ConstructorThreadCount() const { return mRec->mConstructorThreadCount; }
  inline unsigned int SelectorThreadCount() const { return mRec->mSelectorThreadCount; }
  inline unsigned int FinderThreadCount() const { return mRec->mFinderThreadCount; }
  inline unsigned int ClustererThreadCount() const { return mRec->mClustererThreadCount; }
  inline unsigned int ScanThreadCount() const { return mRec->mScanThreadCount; }
  inline unsigned int TRDThreadCount() const { return mRec->mTRDThreadCount; }
  inline unsigned int ConverterThreadCount() const { return mRec->mConverterThreadCount; }
  inline unsigned int Compression1ThreadCount() const { return mRec->mCompression1ThreadCount; }
  inline unsigned int Compression2ThreadCount() const { return mRec->mCompression2ThreadCount; }
  inline unsigned int CFDecodeThreadCount() const { return mRec->mCFDecodeThreadCount; }
  inline unsigned int FitThreadCount() const { return mRec->mFitThreadCount; }
  inline unsigned int ITSThreadCount() const { return mRec->mITSThreadCount; }
  inline size_t AllocateRegisteredMemory(GPUProcessor* proc) { return mRec->AllocateRegisteredMemory(proc); }
  inline size_t AllocateRegisteredMemory(short res, GPUOutputControl* control = nullptr) { return mRec->AllocateRegisteredMemory(res, control); }
  template <class T>
  inline void SetupGPUProcessor(T* proc, bool allocate)
  {
    mRec->SetupGPUProcessor<T>(proc, allocate);
  }

  virtual int PrepareTextures() { return 0; }
  virtual int DoStuckProtection(int stream, void* event) { return 0; }

  template <class T, class S, typename... Args>
  bool DoDebugAndDump(RecoStep step, int mask, T& processor, S T::*func, Args&... args);

 private:
  template <bool Always = false, class T, class S, typename... Args>
  void timeCpy(RecoStep step, bool toGPU, S T::*func, Args... args);
};

template <class T>
inline void GPUChain::RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
{
  mRec->RunHelperThreads((int (GPUReconstructionHelpers::helperDelegateBase::*)(int, int, GPUReconstructionHelpers::helperParam*))function, functionCls, count);
}

template <bool Always, class T, class S, typename... Args>
inline void GPUChain::timeCpy(RecoStep step, bool toGPU, S T::*func, Args... args)
{
  if (!Always && step != RecoStep::NoRecoStep && !(GetRecoStepsGPU() & step)) {
    return;
  }
  HighResTimer* timer = nullptr;
  size_t* bytes = nullptr;
  if (mRec->mDeviceProcessingSettings.debugLevel >= 1) {
    int id = mRec->getRecoStepNum(step, false);
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
bool GPUChain::DoDebugAndDump(GPUChain::RecoStep step, int mask, T& processor, S T::*func, Args&... args)
{
  if (GetDeviceProcessingSettings().keepAllMemory) {
    TransferMemoryResourcesToHost(step, &processor, -1, true);
    if (GetDeviceProcessingSettings().debugLevel >= 6 && (mask == 0 || (GetDeviceProcessingSettings().debugMask & mask))) {
      (processor.*func)(args...);
      return true;
    }
  }
  return false;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
