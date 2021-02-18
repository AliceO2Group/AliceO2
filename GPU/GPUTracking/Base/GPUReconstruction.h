// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstruction.h
/// \author David Rohr

#if !defined(GPURECONSTRUCTION_H) && !defined(__OPENCL__)
#define GPURECONSTRUCTION_H

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "GPUTRDDef.h"
#include "GPUParam.h"
#include "GPUSettings.h"
#include "GPUOutputControl.h"
#include "GPUMemoryResource.h"
#include "GPUConstantMem.h"
#include "GPUTPCSliceOutput.h"
#include "GPUDataTypes.h"
#include "GPULogging.h"

namespace o2
{
namespace its
{
class TrackerTraits;
class VertexerTraits;
} // namespace its
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUChain;
class GPUMemorySizeScalers;

class GPUReconstruction
{
  friend class GPUChain;

 protected:
  class LibraryLoader; // These must be the first members to ensure correct destructor order!
  std::shared_ptr<LibraryLoader> mMyLib = nullptr;
  std::vector<GPUMemoryResource> mMemoryResources;
  std::vector<std::unique_ptr<char[]>> mUnmanagedChunks;
  std::vector<std::unique_ptr<GPUChain>> mChains;

 public:
  virtual ~GPUReconstruction();
  GPUReconstruction(const GPUReconstruction&) = delete;
  GPUReconstruction& operator=(const GPUReconstruction&) = delete;

  // General definitions
  constexpr static unsigned int NSLICES = GPUCA_NSLICES;

  using GeometryType = GPUDataTypes::GeometryType;
  using DeviceType = GPUDataTypes::DeviceType;
  using RecoStep = GPUDataTypes::RecoStep;
  using RecoStepField = GPUDataTypes::RecoStepField;
  using InOutTypeField = GPUDataTypes::InOutTypeField;

  static constexpr const char* const GEOMETRY_TYPE_NAMES[] = {"INVALID", "ALIROOT", "O2"};
#ifdef GPUCA_TPC_GEOMETRY_O2
  static constexpr GeometryType geometryType = GeometryType::O2;
#else
  static constexpr GeometryType geometryType = GeometryType::ALIROOT;
#endif

  static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL", "OCL2"};
  static DeviceType GetDeviceType(const char* type);
  enum InOutPointerType : unsigned int { CLUSTER_DATA = 0,
                                         SLICE_OUT_TRACK = 1,
                                         SLICE_OUT_CLUSTER = 2,
                                         MC_LABEL_TPC = 3,
                                         MC_INFO_TPC = 4,
                                         MERGED_TRACK = 5,
                                         MERGED_TRACK_HIT = 6,
                                         TRD_TRACK = 7,
                                         TRD_TRACKLET = 8,
                                         RAW_CLUSTERS = 9,
                                         CLUSTERS_NATIVE = 10,
                                         TRD_TRACKLET_MC = 11,
                                         TPC_COMPRESSED_CL = 12,
                                         TPC_DIGIT = 13,
                                         TPC_ZS = 14 };
  static constexpr const char* const IOTYPENAMES[] = {"TPC Clusters", "TPC Slice Tracks", "TPC Slice Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets",
                                                      "Raw Clusters", "ClusterNative", "TRD Tracklet MC Labels", "TPC Compressed Clusters", "TPC Digit", "TPC ZS Page"};
  static unsigned int getNIOTypeMultiplicity(InOutPointerType type) { return (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER || type == RAW_CLUSTERS || type == TPC_DIGIT) ? NSLICES : 1; }

  // Functionality to create an instance of GPUReconstruction for the desired device
  static GPUReconstruction* CreateInstance(const GPUSettingsProcessing& cfg);
  static GPUReconstruction* CreateInstance(DeviceType type = DeviceType::CPU, bool forceType = true, GPUReconstruction* master = nullptr);
  static GPUReconstruction* CreateInstance(int type, bool forceType, GPUReconstruction* master = nullptr) { return CreateInstance((DeviceType)type, forceType, master); }
  static GPUReconstruction* CreateInstance(const char* type, bool forceType, GPUReconstruction* master = nullptr);

  // Helpers for kernel launches
  template <class T, int I = 0>
  class classArgument
  {
  };

  typedef void deviceEvent; // We use only pointers anyway, and since cl_event and cudaEvent_t are actually pointers, we can cast them to deviceEvent* this way.

  enum class krnlDeviceType : int { CPU = 0,
                                    Device = 1,
                                    Auto = -1 };
  struct krnlExec {
    constexpr krnlExec(unsigned int b, unsigned int t, int s, krnlDeviceType d = krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d), step(GPUCA_RECO_STEP::NoRecoStep) {}
    constexpr krnlExec(unsigned int b, unsigned int t, int s, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(krnlDeviceType::Auto), step(st) {}
    constexpr krnlExec(unsigned int b, unsigned int t, int s, krnlDeviceType d, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(d), step(st) {}
    unsigned int nBlocks;
    unsigned int nThreads;
    int stream;
    krnlDeviceType device;
    GPUCA_RECO_STEP step;
  };
  struct krnlRunRange {
    constexpr krnlRunRange() = default;
    constexpr krnlRunRange(unsigned int a) : start(a), num(0) {}
    constexpr krnlRunRange(unsigned int s, int n) : start(s), num(n) {}

    unsigned int start = 0;
    int num = 0;
  };
  struct krnlEvent {
    constexpr krnlEvent(deviceEvent* e = nullptr, deviceEvent* el = nullptr, int n = 1) : ev(e), evList(el), nEvents(n) {}
    deviceEvent* ev;
    deviceEvent* evList;
    int nEvents;
  };

  struct krnlProperties {
    krnlProperties(int t = 0, int b = 1) : nThreads(t), minBlocks(b) {}
    unsigned int nThreads;
    unsigned int minBlocks;
    unsigned int total() { return nThreads * minBlocks; }
  };

  struct krnlSetup {
    krnlExec x;
    krnlRunRange y;
    krnlEvent z;
    double t;
  };

  // Global steering functions
  template <class T, typename... Args>
  T* AddChain(Args... args);

  int Init();
  int Finalize();
  int Exit();

  void DumpSettings(const char* dir = "");
  int ReadSettings(const char* dir = "");

  void PrepareEvent();
  virtual int RunChains() = 0;
  unsigned int getNEventsProcessed() { return mNEventsProcessed; }
  unsigned int getNEventsProcessedInStat() { return mStatNEvents; }
  virtual int registerMemoryForGPU(const void* ptr, size_t size) = 0;
  virtual int unregisterMemoryForGPU(const void* ptr) = 0;
  virtual void startGPUProfiling() {}
  virtual void endGPUProfiling() {}

  // Helpers for memory allocation
  GPUMemoryResource& Res(short num) { return mMemoryResources[num]; }
  template <class T>
  short RegisterMemoryAllocation(T* proc, void* (T::*setPtr)(void*), int type, const char* name = "", const GPUMemoryReuse& re = GPUMemoryReuse());
  size_t AllocateMemoryResources();
  size_t AllocateRegisteredMemory(GPUProcessor* proc);
  size_t AllocateRegisteredMemory(short res, GPUOutputControl* control = nullptr);
  void* AllocateUnmanagedMemory(size_t size, int type);
  void FreeRegisteredMemory(GPUProcessor* proc, bool freeCustom = false, bool freePermanent = false);
  void FreeRegisteredMemory(short res);
  void ClearAllocatedMemory(bool clearOutputs = true);
  void ResetRegisteredMemoryPointers(GPUProcessor* proc);
  void ResetRegisteredMemoryPointers(short res);
  void ComputeReuseMax(GPUProcessor* proc);
  void PrintMemoryStatistics();
  void PrintMemoryOverview();
  void SetMemoryExternalInput(short res, void* ptr);
  GPUMemorySizeScalers* MemoryScalers() { return mMemoryScalers.get(); }

  // Helpers to fetch processors from other shared libraries
  virtual void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits);
  bool slavesExist() { return mSlaves.size() || mMaster; }

  // Getters / setters for parameters
  DeviceType GetDeviceType() const { return (DeviceType)mProcessingSettings.deviceType; }
  bool IsGPU() const { return GetDeviceType() != DeviceType::INVALID_DEVICE && GetDeviceType() != DeviceType::CPU; }
  const GPUParam& GetParam() const { return mHostConstantMem->param; }
  const GPUConstantMem& GetConstantMem() const { return *mHostConstantMem; }
  const GPUSettingsEvent& GetEventSettings() const { return mEventSettings; }
  const GPUSettingsProcessing& GetProcessingSettings() { return mProcessingSettings; }
  const GPUSettingsDeviceProcessing& GetDeviceProcessingSettings() const { return mDeviceProcessingSettings; }
  bool IsInitialized() const { return mInitialized; }
  void SetSettings(float solenoidBz);
  void SetSettings(const GPUSettingsEvent* settings, const GPUSettingsRec* rec = nullptr, const GPUSettingsDeviceProcessing* proc = nullptr, const GPURecoStepConfiguration* workflow = nullptr);
  void SetResetTimers(bool reset) { mDeviceProcessingSettings.resetTimers = reset; } // May update also after Init()
  void SetDebugLevelTmp(int level) { mDeviceProcessingSettings.debugLevel = level; } // Temporarily, before calling SetSettings()
  void UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsDeviceProcessing* p = nullptr);
  void SetOutputControl(const GPUOutputControl& v) { mOutputControl = v; }
  void SetOutputControl(void* ptr, size_t size);
  GPUOutputControl& OutputControl() { return mOutputControl; }
  int GetMaxThreads() const { return mMaxThreads; }
  int NStreams() const { return mNStreams; }
  const void* DeviceMemoryBase() const { return mDeviceMemoryBase; }

  RecoStepField GetRecoSteps() const { return mRecoSteps; }
  RecoStepField GetRecoStepsGPU() const { return mRecoStepsGPU; }
  InOutTypeField GetRecoStepsInputs() const { return mRecoStepsInputs; }
  InOutTypeField GetRecoStepsOutputs() const { return mRecoStepsOutputs; }

  // Registration of GPU Processors
  template <class T>
  void RegisterGPUProcessor(T* proc, bool deviceSlave);
  template <class T>
  void SetupGPUProcessor(T* proc, bool allocate);
  void RegisterGPUDeviceProcessor(GPUProcessor* proc, GPUProcessor* slaveProcessor);
  void ConstructGPUProcessor(GPUProcessor* proc);

  // Support / Debugging
  virtual void PrintKernelOccupancies() {}
  double GetStatKernelTime() { return mStatKernelTime; }
  double GetStatWallTime() { return mStatWallTime; }

 protected:
  GPUReconstruction(const GPUSettingsProcessing& cfg); // Constructor
  int InitPhaseBeforeDevice();
  virtual void UpdateSettings() {}
  virtual int InitDevice() = 0;
  int InitPhasePermanentMemory();
  int InitPhaseAfterDevice();
  void WriteConstantParams();
  virtual int ExitDevice() = 0;
  virtual size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) = 0;

  // Management for GPU thread contexts
  class GPUThreadContext
  {
   public:
    GPUThreadContext() = default;
    virtual ~GPUThreadContext() = default;
  };
  virtual std::unique_ptr<GPUThreadContext> GetThreadContext();

  // Private helper functions for memory management
  size_t AllocateRegisteredMemoryHelper(GPUMemoryResource* res, void*& ptr, void*& memorypool, void* memorybase, size_t memorysize, void* (GPUMemoryResource::*SetPointers)(void*));
  size_t AllocateRegisteredPermanentMemory();

  // Private helper functions for reading / writing / allocating IO buffer from/to file
  template <class T, class S>
  void DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type);
  template <class T, class S>
  size_t ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type);
  template <class T>
  void AllocateIOMemoryHelper(unsigned int n, const T*& ptr, std::unique_ptr<T[]>& u);

  // Private helper functions to dump / load flat objects
  template <class T>
  void DumpFlatObjectToFile(const T* obj, const char* file);
  template <class T>
  std::unique_ptr<T> ReadFlatObjectFromFile(const char* file);
  template <class T>
  void DumpStructToFile(const T* obj, const char* file);
  template <class T>
  std::unique_ptr<T> ReadStructFromFile(const char* file);
  template <class T>
  int ReadStructFromFile(const char* file, T* obj);

  // Others
  virtual RecoStepField AvailableRecoSteps() { return RecoStep::AllRecoSteps; }
  virtual bool CanQueryMaxMemory() { return false; }

  // Pointers to tracker classes
  GPUConstantMem* processors() { return mHostConstantMem.get(); }
  const GPUConstantMem* processors() const { return mHostConstantMem.get(); }
  GPUParam& param() { return mHostConstantMem->param; }
  std::unique_ptr<GPUConstantMem> mHostConstantMem;
  GPUConstantMem* mDeviceConstantMem = nullptr;

  // Settings
  GPUSettingsEvent mEventSettings;                       // Event Parameters
  GPUSettingsProcessing mProcessingSettings;             // Processing Parameters (at constructor level)
  GPUSettingsDeviceProcessing mDeviceProcessingSettings; // Processing Parameters (at init level)
  GPUOutputControl mOutputControl;                       // Controls the output of the individual components
  std::unique_ptr<GPUMemorySizeScalers> mMemoryScalers;  // Scalers how much memory will be needed

  RecoStepField mRecoSteps = RecoStep::AllRecoSteps;
  RecoStepField mRecoStepsGPU = RecoStep::AllRecoSteps;
  InOutTypeField mRecoStepsInputs = 0;
  InOutTypeField mRecoStepsOutputs = 0;

  std::string mDeviceName = "CPU";

  // Ptrs to host and device memory;
  void* mHostMemoryBase = nullptr;        // Ptr to begin of large host memory buffer
  void* mHostMemoryPermanent = nullptr;   // Ptr to large host memory buffer offset by permanently allocated memory
  void* mHostMemoryPool = nullptr;        // Ptr to next free location in host memory buffer
  size_t mHostMemorySize = 0;             // Size of host memory buffer
  void* mDeviceMemoryBase = nullptr;      //
  void* mDeviceMemoryPermanent = nullptr; //
  void* mDeviceMemoryPool = nullptr;      //
  size_t mDeviceMemorySize = 0;           //

  GPUReconstruction* mMaster = nullptr;    // Ptr to a GPUReconstruction object serving as master, sharing GPU memory, events, etc.
  std::vector<GPUReconstruction*> mSlaves; // Ptr to slave GPUReconstructions

  // Others
  bool mInitialized = false;
  unsigned int mStatNEvents = 0;
  unsigned int mNEventsProcessed = 0;
  double mStatKernelTime = 0.;
  double mStatWallTime = 0.;

  int mMaxThreads = 0; // Maximum number of threads that may be running, on CPU or GPU
  int mThreadId = -1;  // Thread ID that is valid for the local CUDA context
  int mGPUStuck = 0;   // Marks that the GPU is stuck, skip future events
  int mNStreams = 1;   // Number of parallel GPU streams

  // Management for GPUProcessors
  struct ProcessorData {
    ProcessorData(GPUProcessor* p, void (GPUProcessor::*r)(), void (GPUProcessor::*i)(), void (GPUProcessor::*d)(const GPUTrackingInOutPointers&)) : proc(p), RegisterMemoryAllocation(r), InitializeProcessor(i), SetMaxData(d) {}
    GPUProcessor* proc;
    void (GPUProcessor::*RegisterMemoryAllocation)();
    void (GPUProcessor::*InitializeProcessor)();
    void (GPUProcessor::*SetMaxData)(const GPUTrackingInOutPointers&);
  };
  std::vector<ProcessorData> mProcessors;
  struct MemoryReuseMeta {
    GPUProcessor* proc = nullptr;
    std::vector<unsigned short> res;
  };
  std::unordered_map<GPUMemoryReuse::ID, MemoryReuseMeta> mMemoryReuse1to1;

  // Helpers for loading device library via dlopen
  class LibraryLoader
  {
   public:
    ~LibraryLoader();
    LibraryLoader(const LibraryLoader&) = delete;
    const LibraryLoader& operator=(const LibraryLoader&) = delete;

   private:
    friend class GPUReconstruction;
    LibraryLoader(const char* lib, const char* func);
    int LoadLibrary();
    int CloseLibrary();
    GPUReconstruction* GetPtr(const GPUSettingsProcessing& cfg);

    const char* mLibName;
    const char* mFuncName;
    void* mGPULib;
    void* mGPUEntry;
  };
  static std::shared_ptr<LibraryLoader> sLibCUDA, sLibHIP, sLibOCL, sLibOCL2;

  static GPUReconstruction* GPUReconstruction_Create_CPU(const GPUSettingsProcessing& cfg);
};

template <class T>
inline void GPUReconstruction::AllocateIOMemoryHelper(unsigned int n, const T*& ptr, std::unique_ptr<T[]>& u)
{
  if (n == 0) {
    u.reset(nullptr);
    return;
  }
  u.reset(new GPUCA_NEW_ALIGNMENT T[n]);
  ptr = u.get();
  if (mDeviceProcessingSettings.registerStandaloneInputMemory) {
    registerMemoryForGPU(u.get(), n * sizeof(T));
  }
}

template <class T, typename... Args>
inline T* GPUReconstruction::AddChain(Args... args)
{
  mChains.emplace_back(new T(this, args...));
  return (T*)mChains.back().get();
}

template <class T>
inline short GPUReconstruction::RegisterMemoryAllocation(T* proc, void* (T::*setPtr)(void*), int type, const char* name, const GPUMemoryReuse& re)
{
  if (!(type & (GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU))) {
    if ((type & GPUMemoryResource::MEMORY_SCRATCH) && !mDeviceProcessingSettings.keepDisplayMemory) { // keepAllMemory --> keepDisplayMemory
      type |= (proc->mGPUProcessorType == GPUProcessor::PROCESSOR_TYPE_CPU ? GPUMemoryResource::MEMORY_HOST : GPUMemoryResource::MEMORY_GPU);
    } else {
      type |= GPUMemoryResource::MEMORY_HOST | GPUMemoryResource::MEMORY_GPU;
    }
  }
  if (proc->mGPUProcessorType == GPUProcessor::PROCESSOR_TYPE_CPU) {
    type &= ~GPUMemoryResource::MEMORY_GPU;
  }
  mMemoryResources.emplace_back(proc, static_cast<void* (GPUProcessor::*)(void*)>(setPtr), (GPUMemoryResource::MemoryType)type, name);
  if (mMemoryResources.size() >= 32768) {
    throw std::bad_alloc();
  }
  unsigned short retVal = mMemoryResources.size() - 1;
  if (re.type != GPUMemoryReuse::NONE) {
    const auto& it = mMemoryReuse1to1.find(re.id);
    if (it == mMemoryReuse1to1.end()) {
      mMemoryReuse1to1[re.id] = {proc, {retVal}};
    } else {
      mMemoryResources[retVal].mReuse = it->second.res[0];
      it->second.res.emplace_back(retVal);
    }
  }
  return retVal;
}

template <class T>
inline void GPUReconstruction::RegisterGPUProcessor(T* proc, bool deviceSlave)
{
  mProcessors.emplace_back(proc, static_cast<void (GPUProcessor::*)()>(&T::RegisterMemoryAllocation), static_cast<void (GPUProcessor::*)()>(&T::InitializeProcessor), static_cast<void (GPUProcessor::*)(const GPUTrackingInOutPointers& io)>(&T::SetMaxData));
  GPUProcessor::ProcessorType processorType = deviceSlave ? GPUProcessor::PROCESSOR_TYPE_SLAVE : GPUProcessor::PROCESSOR_TYPE_CPU;
  proc->InitGPUProcessor(this, processorType);
}

template <class T>
inline void GPUReconstruction::SetupGPUProcessor(T* proc, bool allocate)
{
  static_assert(sizeof(T) > sizeof(GPUProcessor), "Need to setup derrived class");
  if (allocate) {
    proc->SetMaxData(mHostConstantMem->ioPtrs);
  }
  if (proc->mDeviceProcessor) {
    std::memcpy((void*)proc->mDeviceProcessor, (const void*)proc, sizeof(*proc));
    proc->mDeviceProcessor->InitGPUProcessor((GPUReconstruction*)this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
  }
  if (allocate) {
    AllocateRegisteredMemory(proc);
  } else {
    ResetRegisteredMemoryPointers(proc);
  }
}

template <class T, class S>
inline void GPUReconstruction::DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type)
{
  int count = getNIOTypeMultiplicity(type);
  unsigned int numTotal = 0;
  for (int i = 0; i < count; i++) {
    numTotal += num[i];
  }
  if (numTotal == 0) {
    return;
  }
  fwrite(&type, sizeof(type), 1, fp);
  for (int i = 0; i < count; i++) {
    fwrite(&num[i], sizeof(num[i]), 1, fp);
    if (num[i]) {
      fwrite(entries[i], sizeof(*entries[i]), num[i], fp);
    }
  }
}

template <class T, class S>
inline size_t GPUReconstruction::ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
{
  if (feof(fp)) {
    return 0;
  }
  InOutPointerType inType;
  size_t r, pos = ftell(fp);
  r = fread(&inType, sizeof(inType), 1, fp);
  if (r != 1 || inType != type) {
    fseek(fp, pos, SEEK_SET);
    return 0;
  }

  int count = getNIOTypeMultiplicity(type);
  size_t numTotal = 0;
  for (int i = 0; i < count; i++) {
    r = fread(&num[i], sizeof(num[i]), 1, fp);
    AllocateIOMemoryHelper(num[i], entries[i], mem[i]);
    if (num[i]) {
      r = fread(mem[i].get(), sizeof(*entries[i]), num[i], fp);
    }
    numTotal += num[i];
  }
  (void)r;
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %d %s", (int)numTotal, IOTYPENAMES[type]);
  }
  return numTotal;
}

template <class T>
inline void GPUReconstruction::DumpFlatObjectToFile(const T* obj, const char* file)
{
  FILE* fp = fopen(file, "w+b");
  if (fp == nullptr) {
    return;
  }
  size_t size[2] = {sizeof(*obj), obj->getFlatBufferSize()};
  fwrite(size, sizeof(size[0]), 2, fp);
  fwrite(obj, 1, size[0], fp);
  fwrite(obj->getFlatBufferPtr(), 1, size[1], fp);
  fclose(fp);
}

template <class T>
inline std::unique_ptr<T> GPUReconstruction::ReadFlatObjectFromFile(const char* file)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return nullptr;
  }
  size_t size[2] = {0}, r;
  r = fread(size, sizeof(size[0]), 2, fp);
  if (r == 0 || size[0] != sizeof(T)) {
    fclose(fp);
    GPUError("ERROR reading %s, invalid size: %lld (%lld expected)", file, (long long int)size[0], (long long int)sizeof(T));
    throw std::runtime_error("invalid size");
  }
  std::unique_ptr<T> retVal(new GPUCA_NEW_ALIGNMENT T);
  char* buf = new GPUCA_NEW_ALIGNMENT char[size[1]]; // Not deleted as ownership is transferred to FlatObject
  r = fread((void*)retVal.get(), 1, size[0], fp);
  r = fread(buf, 1, size[1], fp);
  fclose(fp);
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %d bytes from %s", (int)r, file);
  }
  retVal->clearInternalBufferPtr();
  retVal->setActualBufferAddress(buf);
  retVal->adoptInternalBuffer(buf);
  return retVal;
}

template <class T>
inline void GPUReconstruction::DumpStructToFile(const T* obj, const char* file)
{
  FILE* fp = fopen(file, "w+b");
  if (fp == nullptr) {
    return;
  }
  size_t size = sizeof(*obj);
  fwrite(&size, sizeof(size), 1, fp);
  fwrite(obj, 1, size, fp);
  fclose(fp);
}

template <class T>
inline std::unique_ptr<T> GPUReconstruction::ReadStructFromFile(const char* file)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return nullptr;
  }
  size_t size, r;
  r = fread(&size, sizeof(size), 1, fp);
  if (r == 0 || size != sizeof(T)) {
    fclose(fp);
    GPUError("ERROR reading %s, invalid size: %lld (%lld expected)", file, (long long int)size, (long long int)sizeof(T));
    throw std::runtime_error("invalid size");
  }
  std::unique_ptr<T> newObj(new GPUCA_NEW_ALIGNMENT T);
  r = fread(newObj.get(), 1, size, fp);
  fclose(fp);
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %d bytes from %s", (int)r, file);
  }
  return newObj;
}

template <class T>
inline int GPUReconstruction::ReadStructFromFile(const char* file, T* obj)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return 1;
  }
  size_t size, r;
  r = fread(&size, sizeof(size), 1, fp);
  if (r == 0) {
    fclose(fp);
    return 1;
  }
  r = fread(obj, 1, size, fp);
  fclose(fp);
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %d bytes from %s", (int)r, file);
  }
  return 0;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
