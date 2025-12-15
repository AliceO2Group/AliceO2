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

/// \file GPUReconstruction.h
/// \author David Rohr

#if !defined(GPURECONSTRUCTION_H) && !defined(__OPENCL__)
#define GPURECONSTRUCTION_H

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <iosfwd>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

#include "GPUDataTypes.h"
#include "GPUMemoryResource.h"
#include "GPUOutputControl.h"
#include "GPUParam.h"
#include "GPUConstantMem.h"
#include "GPUDef.h"

namespace o2::its
{
template <int>
class TrackerTraits;
template <int>
class VertexerTraits;
template <int>
class TimeFrame;
} // namespace o2::its

namespace o2::gpu
{
class GPUChain;
struct GPUMemorySizeScalers;
struct GPUReconstructionPipelineContext;
struct GPUReconstructionThreading;
class GPUROOTDumpCore;
class ThrustVolatileAllocator;
struct GPUDefParameters;
class GPUMemoryResource;
struct GPUSettingsDeviceBackend;
struct GPUSettingsGRP;
struct GPUSettingsProcessing;
struct GPUSettingsRec;
struct GPUSettingsRecDynamic;
struct GPUMemoryReuse;

namespace gpu_reconstruction_kernels
{
struct deviceEvent;
class threadContext;
} // namespace gpu_reconstruction_kernels

class GPUReconstruction
{
 protected:
  class LibraryLoader; // These must be the first members to ensure correct destructor order!
  std::shared_ptr<LibraryLoader> mMyLib = nullptr;
  std::vector<GPUMemoryResource> mMemoryResources;
  std::vector<std::unique_ptr<GPUChain>> mChains;

 public:
  virtual ~GPUReconstruction();
  GPUReconstruction(const GPUReconstruction&) = delete;
  GPUReconstruction& operator=(const GPUReconstruction&) = delete;

  // General definitions
  constexpr static uint32_t NSECTORS = GPUCA_NSECTORS;

  using GeometryType = GPUDataTypes::GeometryType;
  using DeviceType = GPUDataTypes::DeviceType;
  using RecoStep = GPUDataTypes::RecoStep;
  using GeneralStep = GPUDataTypes::GeneralStep;
  using RecoStepField = GPUDataTypes::RecoStepField;
  using InOutTypeField = GPUDataTypes::InOutTypeField;

  static constexpr const char* const GEOMETRY_TYPE_NAMES[] = {"INVALID", "ALIROOT", "O2"};
#ifdef GPUCA_TPC_GEOMETRY_O2
  static constexpr GeometryType geometryType = GeometryType::O2;
#else
  static constexpr GeometryType geometryType = GeometryType::ALIROOT;
#endif

  static DeviceType GetDeviceType(const char* type);
  enum InOutPointerType : uint32_t { CLUSTER_DATA = 0,
                                     SECTOR_OUT_TRACK = 1,
                                     SECTOR_OUT_CLUSTER = 2,
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
                                     TPC_ZS = 14,
                                     CLUSTER_NATIVE_MC = 15,
                                     TPC_DIGIT_MC = 16,
                                     TRD_SPACEPOINT = 17,
                                     TRD_TRIGGERRECORDS = 18,
                                     TF_SETTINGS = 19 };
  static constexpr const char* const IOTYPENAMES[] = {"TPC HLT Clusters", "TPC Sector Tracks", "TPC Sector Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets",
                                                      "TPC Raw Clusters", "TPC Native Clusters", "TRD Tracklet MC Labels", "TPC Compressed Clusters", "TPC Digit", "TPC ZS Page", "TPC Native Clusters MC Labels", "TPC Digit MC Labeels",
                                                      "TRD Spacepoints", "TRD Triggerrecords", "TF Settings"};
  static uint32_t getNIOTypeMultiplicity(InOutPointerType type) { return (type == CLUSTER_DATA || type == SECTOR_OUT_TRACK || type == SECTOR_OUT_CLUSTER || type == RAW_CLUSTERS || type == TPC_DIGIT || type == TPC_DIGIT_MC) ? NSECTORS : 1; }

  // Functionality to create an instance of GPUReconstruction for the desired device
  static GPUReconstruction* CreateInstance(const GPUSettingsDeviceBackend& cfg);
  static GPUReconstruction* CreateInstance(DeviceType type = DeviceType::CPU, bool forceType = true, GPUReconstruction* master = nullptr);
  static GPUReconstruction* CreateInstance(int32_t type, bool forceType, GPUReconstruction* master = nullptr) { return CreateInstance((DeviceType)type, forceType, master); }
  static GPUReconstruction* CreateInstance(const char* type, bool forceType, GPUReconstruction* master = nullptr);
  static bool CheckInstanceAvailable(DeviceType type, bool verbose);

  enum class krnlDeviceType : int32_t { CPU = 0,
                                        Device = 1,
                                        Auto = -1 };

  // Global steering functions
  template <class T, typename... Args>
  T* AddChain(Args... args);

  int32_t Init();
  int32_t Finalize();
  int32_t Exit();

  void DumpSettings(const char* dir = "");
  int32_t ReadSettings(const char* dir = "");

  void PrepareEvent();
  virtual int32_t RunChains() = 0;
  uint32_t getNEventsProcessed() { return mNEventsProcessed; }
  uint32_t getNEventsProcessedInStat() { return mStatNEvents; }
  int32_t registerMemoryForGPU(const void* ptr, size_t size);
  int32_t unregisterMemoryForGPU(const void* ptr);
  virtual void* getGPUPointer(void* ptr) { return ptr; }
  virtual void startGPUProfiling() {}
  virtual void endGPUProfiling() {}
  int32_t GPUChkErrA(const int64_t error, const char* file, int32_t line, bool failOnError);
  int32_t CheckErrorCodes(bool cpuOnly = false, bool forceShowErrors = false, std::vector<std::array<uint32_t, 4>>* fillErrors = nullptr);
  void RunPipelineWorker();
  void TerminatePipelineWorker();

  // Helpers for memory allocation
  GPUMemoryResource& Res(int16_t num) { return mMemoryResources[num]; }
  template <class T>
  int16_t RegisterMemoryAllocation(T* proc, void* (T::*setPtr)(void*), int32_t type, const char* name = "", const GPUMemoryReuse& re = GPUMemoryReuse());
  size_t AllocateMemoryResources();
  size_t AllocateRegisteredMemory(GPUProcessor* proc, bool resetCustom = false);

  size_t AllocateRegisteredMemory(int16_t res, GPUOutputControl* control = nullptr);
  void AllocateRegisteredForeignMemory(int16_t res, GPUReconstruction* rec, GPUOutputControl* control = nullptr);
  void* AllocateDirectMemory(size_t size, int32_t type);
  void* AllocateVolatileDeviceMemory(size_t size);
  void* AllocateVolatileMemory(size_t size, bool device);
  void MakeFutureDeviceMemoryAllocationsVolatile();
  void FreeRegisteredMemory(GPUProcessor* proc, bool freeCustom = false, bool freePermanent = false);
  void FreeRegisteredMemory(int16_t res);
  void ClearAllocatedMemory(bool clearOutputs = true);
  void ReturnVolatileDeviceMemory();
  void ReturnVolatileMemory();
  ThrustVolatileAllocator getThrustVolatileDeviceAllocator();
  void PushNonPersistentMemory(uint64_t tag);
  void PopNonPersistentMemory(RecoStep step, uint64_t tag, const GPUProcessor* proc = nullptr);
  void BlockStackedMemory(GPUReconstruction* rec);
  void UnblockStackedMemory();
  void ResetRegisteredMemoryPointers(GPUProcessor* proc);
  void ResetRegisteredMemoryPointers(int16_t res);
  void ComputeReuseMax(GPUProcessor* proc);
  void PrintMemoryStatistics();
  void PrintMemoryOverview();
  void PrintMemoryMax();
  void SetMemoryExternalInput(int16_t res, void* ptr);
  GPUMemorySizeScalers* MemoryScalers() { return mMemoryScalers.get(); }

  // Helpers to fetch processors from other shared libraries
  virtual void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits<7>>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits<7>>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame<7>>* timeFrame);
  bool slavesExist() { return mSlaves.size() || mMaster; }
  int slaveId() { return mSlaveId; }

  // Getters / setters for parameters
  DeviceType GetDeviceType() const;
  bool IsGPU() const { return GetDeviceType() != DeviceType::INVALID_DEVICE && GetDeviceType() != DeviceType::CPU; }
  const GPUParam& GetParam() const;
  const GPUConstantMem& GetConstantMem() const { return *mHostConstantMem; }
  const GPUTrackingInOutPointers GetIOPtrs() const;
  const GPUSettingsGRP& GetGRPSettings() const { return *mGRPSettings; }
  const GPUSettingsDeviceBackend& GetDeviceBackendSettings() const { return *mDeviceBackendSettings; }
  const GPUSettingsProcessing& GetProcessingSettings() const { return *mProcessingSettings; }
  const GPUCalibObjectsConst& GetCalib() const;
  bool IsInitialized() const { return mInitialized; }
  void SetSettings(float solenoidBzNominalGPU, const GPURecoStepConfiguration* workflow = nullptr);
  void SetSettings(const GPUSettingsGRP* grp, const GPUSettingsRec* rec = nullptr, const GPUSettingsProcessing* proc = nullptr, const GPURecoStepConfiguration* workflow = nullptr);
  void SetResetTimers(bool reset);      // May update also after Init()
  void SetDebugLevelTmp(int32_t level); // Temporarily, before calling SetSettings()
  void UpdateSettings(const GPUSettingsGRP* g, const GPUSettingsProcessing* p = nullptr, const GPUSettingsRecDynamic* d = nullptr);
  void UpdateDynamicSettings(const GPUSettingsRecDynamic* d);
  void SetOutputControl(const GPUOutputControl& v) { mOutputControl = v; }
  void SetOutputControl(void* ptr, size_t size);
  void SetInputControl(void* ptr, size_t size);
  GPUOutputControl& OutputControl() { return mOutputControl; }
  uint32_t NStreams() const { return mNStreams; }
  const void* DeviceMemoryBase() const { return mDeviceMemoryBase; }
  virtual const GPUDefParameters& getGPUParameters(bool doGPU) const = 0;

  RecoStepField GetRecoSteps() const { return mRecoSteps.steps; }
  RecoStepField GetRecoStepsGPU() const { return mRecoSteps.stepsGPUMask; }
  InOutTypeField GetRecoStepsInputs() const { return mRecoSteps.inputs; }
  InOutTypeField GetRecoStepsOutputs() const { return mRecoSteps.outputs; }
  int32_t getRecoStepNum(RecoStep step, bool validCheck = true);
  int32_t getGeneralStepNum(GeneralStep step, bool validCheck = true);

  void setErrorCodeOutput(std::vector<std::array<uint32_t, 4>>* v) { mOutputErrorCodes = v; }
  std::vector<std::array<uint32_t, 4>>* getErrorCodeOutput() { return mOutputErrorCodes; }

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
  void setDebugDumpCallback(std::function<void()>&& callback = std::function<void()>(nullptr));
  bool triggerDebugDump();
  std::string getDebugFolder(const std::string& prefix = ""); // empty string = no debug

  // Threading
  std::shared_ptr<GPUReconstructionThreading> mThreading;
  static int32_t getHostThreadIndex();
  int32_t GetMaxBackendThreads() const { return mMaxBackendThreads; }

 protected:
  void AllocateRegisteredMemoryInternal(GPUMemoryResource* res, GPUOutputControl* control, GPUReconstruction* recPool);
  void FreeRegisteredMemory(GPUMemoryResource* res);
  GPUReconstruction(const GPUSettingsDeviceBackend& cfg); // Constructor
  int32_t InitPhaseBeforeDevice();
  virtual int32_t InitDevice() = 0;
  int32_t InitPhasePermanentMemory();
  int32_t InitPhaseAfterDevice();
  void WriteConstantParams();
  virtual int32_t ExitDevice() = 0;
  virtual size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, gpu_reconstruction_kernels::deviceEvent* ev = nullptr) = 0;
  void UpdateMaxMemoryUsed();
  int32_t EnqueuePipeline(bool terminate = false);
  GPUChain* GetNextChainInQueue();
  virtual int32_t GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const { return 0; }

  virtual int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) = 0;
  virtual int32_t unregisterMemoryForGPU_internal(const void* ptr) = 0;

  // Management for GPU thread contexts
  virtual std::unique_ptr<gpu_reconstruction_kernels::threadContext> GetThreadContext() = 0;

  // Private helpers for library loading
  static std::shared_ptr<LibraryLoader>* GetLibraryInstance(DeviceType type, bool verbose);
  static std::string getBackendVersions();

  // Private helper functions for memory management
  size_t AllocateRegisteredMemoryHelper(GPUMemoryResource* res, void*& ptr, void*& memorypool, void* memorybase, size_t memorysize, void* (GPUMemoryResource::*SetPointers)(void*) const, void*& memorypoolend, const char* device);
  size_t AllocateRegisteredPermanentMemory();

  // Private helper functions for reading / writing / allocating IO buffer from/to file
  template <class T, class S>
  uint32_t DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type);
  template <class T, class S>
  size_t ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type, T** nonConstPtrs = nullptr);
  template <class T>
  T* AllocateIOMemoryHelper(size_t n, const T*& ptr, std::unique_ptr<T[]>& u);
  int16_t RegisterMemoryAllocationHelper(GPUProcessor* proc, void* (GPUProcessor::*setPtr)(void*), int32_t type, const char* name, const GPUMemoryReuse& re);

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
  int32_t ReadStructFromFile(const char* file, T* obj);

  // Others
  virtual RecoStepField AvailableGPURecoSteps() { return RecoStep::AllRecoSteps; }
  virtual bool CanQueryMaxMemory() { return false; }

  // Pointers to tracker classes
  GPUConstantMem* processors() { return mHostConstantMem.get(); }
  const GPUConstantMem* processors() const { return mHostConstantMem.get(); }
  GPUParam& param();
  std::unique_ptr<GPUConstantMem> mHostConstantMem;
  GPUConstantMem* mDeviceConstantMem = nullptr;

  // Settings
  std::unique_ptr<GPUSettingsGRP> mGRPSettings;                     // Global Run Parameters
  std::unique_ptr<GPUSettingsDeviceBackend> mDeviceBackendSettings; // Processing Parameters (at constructor level)
  std::unique_ptr<GPUSettingsProcessing> mProcessingSettings;       // Processing Parameters (at init level)
  GPUOutputControl mOutputControl;                                  // Controls the output of the individual components
  GPUOutputControl mInputControl;                                   // Prefefined input memory location for reading standalone dumps
  std::unique_ptr<GPUMemorySizeScalers> mMemoryScalers;             // Scalers how much memory will be needed

  GPURecoStepConfiguration mRecoSteps;

  std::string mDeviceName = "CPU";

  // Ptrs to host and device memory;
  void* mHostMemoryBase = nullptr;          // Ptr to begin of large host memory buffer
  void* mHostMemoryPermanent = nullptr;     // Ptr to large host memory buffer offset by permanently allocated memory
  void* mHostMemoryPool = nullptr;          // Ptr to next free location in host memory buffer
  void* mHostMemoryPoolEnd = nullptr;       // Ptr to end of pool
  void* mHostMemoryPoolBlocked = nullptr;   // Ptr to end of pool
  size_t mHostMemorySize = 0;               // Size of host memory buffer
  size_t mHostMemoryUsedMax = 0;            // Maximum host memory size used over time
  void* mDeviceMemoryBase = nullptr;        // Same for device ...
  void* mDeviceMemoryPermanent = nullptr;   // ...
  void* mDeviceMemoryPool = nullptr;        // ...
  void* mDeviceMemoryPoolEnd = nullptr;     // ...
  void* mDeviceMemoryPoolBlocked = nullptr; // ...
  size_t mDeviceMemorySize = 0;             // ...
  size_t mDeviceMemoryUsedMax = 0;          // ...
  void* mVolatileMemoryStart = nullptr;     // Ptr to beginning of temporary volatile memory allocation, nullptr if uninitialized
  bool mDeviceMemoryAsVolatile = false;     // Make device memory allocations volatile

  std::unordered_set<const void*> mRegisteredMemoryPtrs; // List of pointers registered for GPU

  GPUReconstruction* mMaster = nullptr;    // Ptr to a GPUReconstruction object serving as master, sharing GPU memory, events, etc.
  std::vector<GPUReconstruction*> mSlaves; // Ptr to slave GPUReconstructions
  int mSlaveId = -1;                       // Id of this slave (-1 for master)

  // Others
  bool mInitialized = false;
  bool mInErrorHandling = false;
  uint32_t mStatNEvents = 0;
  uint32_t mNEventsProcessed = 0;
  double mStatKernelTime = 0.;
  double mStatWallTime = 0.;
  double mStatCPUTime = 0.;
  std::shared_ptr<GPUROOTDumpCore> mROOTDump;
  std::vector<std::array<uint32_t, 4>>* mOutputErrorCodes = nullptr;

  int32_t mMaxBackendThreads = 0; // Maximum number of threads that may be running, on CPU or GPU
  int32_t mGPUStuck = 0;          // Marks that the GPU is stuck, skip future events
  int32_t mNStreams = 1;          // Number of parallel GPU streams
  int32_t mMaxHostThreads = 0;    // Maximum number of OMP threads

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
    MemoryReuseMeta() = default;
    MemoryReuseMeta(GPUProcessor* p, uint16_t r) : proc(p), res{r} {}
    GPUProcessor* proc = nullptr;
    std::vector<uint16_t> res;
  };
  struct alignedDeleter {
    void operator()(void* ptr) { ::operator delete[](ptr, std::align_val_t(GPUCA_BUFFER_ALIGNMENT)); };
  };
  std::unordered_map<GPUMemoryReuse::ID, MemoryReuseMeta> mMemoryReuse1to1;
  std::vector<std::tuple<void*, void*, size_t, size_t, uint64_t>> mNonPersistentMemoryStack; // hostPoolAddress, devicePoolAddress, individualAllocationCount, directIndividualAllocationCound, tag
  std::vector<GPUMemoryResource*> mNonPersistentIndividualAllocations;
  std::vector<std::unique_ptr<char[], alignedDeleter>> mNonPersistentIndividualDirectAllocations;
  std::vector<std::unique_ptr<char[], alignedDeleter>> mDirectMemoryChunks;
  std::vector<std::unique_ptr<char[], alignedDeleter>> mVolatileChunks;
  std::atomic_flag mMemoryMutex = ATOMIC_FLAG_INIT;

  std::unique_ptr<GPUReconstructionPipelineContext> mPipelineContext;

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
    int32_t LoadLibrary();
    int32_t CloseLibrary();
    GPUReconstruction* GetPtr(const GPUSettingsDeviceBackend& cfg);

    const char* mLibName;
    const char* mFuncName;
    void* mGPULib;
    void* mGPUEntry;
  };
  static std::shared_ptr<LibraryLoader> sLibCUDA, sLibHIP, sLibOCL;

  // Debugging
  struct debugInternal;
  static std::unique_ptr<debugInternal> mDebugData;
  bool mDebugEnabled = false;
  void debugInit();
  void debugExit();

  static GPUReconstruction* GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
};

template <class T, typename... Args>
inline T* GPUReconstruction::AddChain(Args... args)
{
  mChains.emplace_back(new T(this, args...));
  return (T*)mChains.back().get();
}

template <class T>
inline int16_t GPUReconstruction::RegisterMemoryAllocation(T* proc, void* (T::*setPtr)(void*), int32_t type, const char* name, const GPUMemoryReuse& re)
{
  return RegisterMemoryAllocationHelper(proc, static_cast<void* (GPUProcessor::*)(void*)>(setPtr), type, name, re);
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
  static_assert(sizeof(T) > sizeof(GPUProcessor), "Need to setup derived class");
  if (allocate) {
    proc->SetMaxData(GetIOPtrs());
  }
  if (proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && proc->mLinkedProcessor) {
    std::memcpy((void*)proc->mLinkedProcessor, (const void*)proc, sizeof(*proc));
    proc->mLinkedProcessor->InitGPUProcessor((GPUReconstruction*)this, GPUProcessor::PROCESSOR_TYPE_DEVICE, proc);
  }
  if (allocate) {
    AllocateRegisteredMemory(proc, true);
  } else {
    ResetRegisteredMemoryPointers(proc);
  }
}

} // namespace o2::gpu

#endif
