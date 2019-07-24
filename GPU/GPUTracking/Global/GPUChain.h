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
  static constexpr krnlRunRange krnlRunRangeNone{ 0, -1 };
  static constexpr krnlEvent krnlEventNone = krnlEvent{ nullptr, nullptr, 0 };

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
  GPUConstantMem* processors() { return mRec->processors(); }
  GPUConstantMem* processorsShadow() { return mRec->mProcessorsShadow; }
  GPUConstantMem* processorsDevice() { return mRec->mDeviceConstantMem; }
  GPUParam& param() { return mRec->param(); }
  const GPUConstantMem* processors() const { return mRec->processors(); }
  GPUSettingsDeviceProcessing& DeviceProcessingSettings() { return mRec->mDeviceProcessingSettings; }
  void SynchronizeStream(int stream) { mRec->SynchronizeStream(stream); }
  void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) { mRec->SynchronizeEvents(evList, nEvents); }
  bool IsEventDone(deviceEvent* evList, int nEvents = 1) { return mRec->IsEventDone(evList, nEvents); }
  void RecordMarker(deviceEvent* ev, int stream) { mRec->RecordMarker(ev, stream); }
  virtual std::unique_ptr<GPUReconstruction::GPUThreadContext> GetThreadContext() { return mRec->GetThreadContext(); }
  void SynchronizeGPU() { mRec->SynchronizeGPU(); }
  void ReleaseEvent(deviceEvent* ev) { mRec->ReleaseEvent(ev); }
  template <class T>
  void RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int count);
  void WaitForHelperThreads() { mRec->WaitForHelperThreads(); }
  int HelperError(int iThread) const { return mRec->HelperError(iThread); }
  int HelperDone(int iThread) const { return mRec->HelperDone(iThread); }
  void ResetHelperThreads(int helpers) { mRec->ResetHelperThreads(helpers); }
  int GPUDebug(const char* state = "UNKNOWN", int stream = -1) { return mRec->GPUDebug(state, stream); }
  void TransferMemoryResourceToGPU(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { mRec->TransferMemoryResourceToGPU(res, stream, ev, evList, nEvents); }
  void TransferMemoryResourceToHost(GPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { mRec->TransferMemoryResourceToHost(res, stream, ev, evList, nEvents); }
  void TransferMemoryResourcesToGPU(GPUProcessor* proc, int stream = -1, bool all = false) { mRec->TransferMemoryResourcesToGPU(proc, stream, all); }
  void TransferMemoryResourcesToHost(GPUProcessor* proc, int stream = -1, bool all = false) { mRec->TransferMemoryResourcesToHost(proc, stream, all); }
  void TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { mRec->TransferMemoryResourceLinkToGPU(res, stream, ev, evList, nEvents); }
  void TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) { mRec->TransferMemoryResourceLinkToHost(res, stream, ev, evList, nEvents); }
  void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) { mRec->WriteToConstantMemory(offset, src, size, stream, ev); }
  template <class T>
  void AllocateIOMemoryHelper(unsigned int n, const T*& ptr, std::unique_ptr<T[]>& u)
  {
    mRec->AllocateIOMemoryHelper<T>(n, ptr, u);
  }
  template <class T>
  void DumpData(FILE* fp, const T* const* entries, const unsigned int* num, InOutPointerType type)
  {
    mRec->DumpData<T>(fp, entries, num, type);
  }
  template <class T>
  size_t ReadData(FILE* fp, const T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
  {
    return mRec->ReadData<T>(fp, entries, num, mem, type);
  }
  template <class T>
  void DumpFlatObjectToFile(const T* obj, const char* file)
  {
    mRec->DumpFlatObjectToFile<T>(obj, file);
  }
  template <class T>
  std::unique_ptr<T> ReadFlatObjectFromFile(const char* file)
  {
    return std::move(mRec->ReadFlatObjectFromFile<T>(file));
  }
  template <class T>
  void DumpStructToFile(const T* obj, const char* file)
  {
    mRec->DumpStructToFile<T>(obj, file);
  }
  template <class T>
  std::unique_ptr<T> ReadStructFromFile(const char* file)
  {
    return std::move(mRec->ReadStructFromFile<T>(file));
  }
  template <class T>
  void ReadStructFromFile(const char* file, T* obj)
  {
    mRec->ReadStructFromFile<T>(file, obj);
  }
#ifdef __clang__ // BUG: clang seems broken and does not accept default parameters before parameter pack
  template <class S, int I = 0>
  inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone)
  {
    return mRec->runKernel<S, I>(x, t, y);
  }
  template <class S, int I = 0, typename... Args>
  inline int runKernel(const krnlExec& x, HighResTimer* t, const krnlRunRange& y, const krnlEvent& z, Args&&... args)
#else
  template <class S, int I = 0, typename... Args>
  inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEventNone, Args&&... args)
#endif
  {
    return mRec->runKernel<S, I, Args...>(x, t, y, z, std::forward<Args>(args)...);
  }
  unsigned int BlockCount() const { return mRec->mBlockCount; }
  unsigned int ThreadCount() const { return mRec->mThreadCount; }
  unsigned int ConstructorBlockCount() const { return mRec->mConstructorBlockCount; }
  unsigned int SelectorBlockCount() const { return mRec->mSelectorBlockCount; }
  unsigned int ConstructorThreadCount() const { return mRec->mConstructorThreadCount; }
  unsigned int SelectorThreadCount() const { return mRec->mSelectorThreadCount; }
  unsigned int FinderThreadCount() const { return mRec->mFinderThreadCount; }
  unsigned int TRDThreadCount() const { return mRec->mTRDThreadCount; }
  size_t AllocateRegisteredMemory(GPUProcessor* proc) { return mRec->AllocateRegisteredMemory(proc); }
  size_t AllocateRegisteredMemory(short res) { return mRec->AllocateRegisteredMemory(res); }
  template <class T>
  void SetupGPUProcessor(T* proc, bool allocate)
  {
    mRec->SetupGPUProcessor<T>(proc, allocate);
  }

  virtual int PrepareTextures() { return 0; }
  virtual int DoStuckProtection(int stream, void* event) { return 0; }
};

template <class T>
inline void GPUChain::RunHelperThreads(T function, GPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
{
  mRec->RunHelperThreads((int (GPUReconstructionHelpers::helperDelegateBase::*)(int, int, GPUReconstructionHelpers::helperParam*))function, functionCls, count);
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
