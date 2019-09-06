// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstruction.cxx
/// \author David Rohr

#include <cstring>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <map>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#include <conio.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

#include "GPUReconstruction.h"
#include "GPUReconstructionIncludes.h"

#include "GPUMemoryResource.h"
#include "GPUChain.h"
#include "GPUMemorySizeScalers.h"

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

using namespace GPUCA_NAMESPACE::gpu;

constexpr const char* const GPUReconstruction::DEVICE_TYPE_NAMES[];
constexpr const char* const GPUReconstruction::GEOMETRY_TYPE_NAMES[];
constexpr const char* const GPUReconstruction::IOTYPENAMES[];
constexpr GPUReconstruction::GeometryType GPUReconstruction::geometryType;

GPUReconstruction::GPUReconstruction(const GPUSettingsProcessing& cfg) : mHostConstantMem(new GPUConstantMem), mProcessingSettings(cfg)
{
  mDeviceProcessingSettings.SetDefaults();
  mEventSettings.SetDefaults();
  param().SetDefaults(&mEventSettings);
  mMemoryScalers.reset(new GPUMemorySizeScalers);
}

GPUReconstruction::~GPUReconstruction()
{
  if (mInitialized) {
    GPUError("GPU Reconstruction not properly deinitialized!");
  }
}

void GPUReconstruction::GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits)
{
  if (trackerTraits) {
    trackerTraits->reset(new o2::its::TrackerTraitsCPU);
  }
  if (vertexerTraits) {
    vertexerTraits->reset(new o2::its::VertexerTraits);
  }
}

int GPUReconstruction::Init()
{
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_AUTO) {
    mDeviceProcessingSettings.memoryAllocationStrategy = IsGPU() ? GPUMemoryResource::ALLOCATION_GLOBAL : GPUMemoryResource::ALLOCATION_INDIVIDUAL;
  }
  if (mDeviceProcessingSettings.eventDisplay) {
    mDeviceProcessingSettings.keepAllMemory = true;
  }
  if (mDeviceProcessingSettings.debugLevel >= 4) {
    mDeviceProcessingSettings.keepAllMemory = true;
  }
  if (mDeviceProcessingSettings.debugLevel < 6) {
    mDeviceProcessingSettings.debugMask = 0;
  }

#ifndef HAVE_O2HEADERS
  mRecoSteps.setBits(RecoStep::ITSTracking, false);
  mRecoSteps.setBits(RecoStep::TRDTracking, false);
  mRecoSteps.setBits(RecoStep::TPCConversion, false);
  mRecoSteps.setBits(RecoStep::TPCCompression, false);
  mRecoSteps.setBits(RecoStep::TPCdEdx, false);
#endif
  mRecoStepsGPU &= mRecoSteps;
  mRecoStepsGPU &= AvailableRecoSteps();
  if (!IsGPU()) {
    mRecoStepsGPU.set((unsigned char)0);
  }
  if (!IsGPU()) {
    mDeviceProcessingSettings.trackletConstructorInPipeline = mDeviceProcessingSettings.trackletSelectorInPipeline = false;
  }
  if (param().rec.NonConsecutiveIDs) {
    param().rec.DisableRefitAttachment = 0xFF;
  }
  if (!mDeviceProcessingSettings.trackletConstructorInPipeline) {
    mDeviceProcessingSettings.trackletSelectorInPipeline = false;
  }

#ifdef GPUCA_HAVE_OPENMP
  if (mDeviceProcessingSettings.nThreads <= 0) {
    mDeviceProcessingSettings.nThreads = omp_get_max_threads();
  } else {
    omp_set_num_threads(mDeviceProcessingSettings.nThreads);
  }
#else
  mDeviceProcessingSettings.nThreads = 1;
#endif

  mDeviceMemorySize = mHostMemorySize = 0;
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->RegisterPermanentMemoryAndProcessors();
    size_t memGpu, memHost;
    mChains[i]->MemorySize(memGpu, memHost);
    mDeviceMemorySize += memGpu;
    mHostMemorySize += memHost;
  }
  if (mDeviceProcessingSettings.forceMemoryPoolSize) {
    mDeviceMemorySize = mHostMemorySize = mDeviceProcessingSettings.forceMemoryPoolSize;
  }

  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].RegisterMemoryAllocation))();
  }

  if (InitDevice()) {
    return 1;
  }
  if (IsGPU()) {
    for (unsigned int i = 0; i < mChains.size(); i++) {
      mChains[i]->RegisterGPUProcessors();
    }
  }
  AllocateRegisteredPermanentMemory();

  for (unsigned int i = 0; i < mChains.size(); i++) {
    if (mChains[i]->Init()) {
      return 1;
    }
  }
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    (mProcessors[i].proc->*(mProcessors[i].InitializeProcessor))();
  }

  if (IsGPU()) {
    const auto threadContext = GetThreadContext();
    WriteToConstantMemory((char*)&processors()->param - (char*)processors(), &param(), sizeof(GPUParam), -1);
  }

  mInitialized = true;
  return 0;
}

int GPUReconstruction::Finalize()
{
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->Finalize();
  }
  return 0;
}

int GPUReconstruction::Exit()
{
  mChains.clear();          // Make sure we destroy a possible ITS GPU tracker before we call the destructors
  mHostConstantMem.reset(); // Reset these explicitly before the destruction of other members unloads the library
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
      operator delete(mMemoryResources[i].mPtrDevice);
      mMemoryResources[i].mPtr = mMemoryResources[i].mPtrDevice = nullptr;
    }
  }
  mMemoryResources.clear();
  if (mInitialized) {
    ExitDevice();
  }
  mInitialized = false;
  return 0;
}

void GPUReconstruction::RegisterGPUDeviceProcessor(GPUProcessor* proc, GPUProcessor* slaveProcessor) { proc->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE, slaveProcessor); }

size_t GPUReconstruction::AllocateRegisteredMemory(GPUProcessor* proc)
{
  if (mDeviceProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating memory %p", (void*)proc);
  }
  size_t total = 0;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if ((proc == nullptr ? !mMemoryResources[i].mProcessor->mAllocateAndInitializeLate : mMemoryResources[i].mProcessor == proc) && !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) {
      total += AllocateRegisteredMemory(i);
    }
  }
  if (mDeviceProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating memory done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredPermanentMemory()
{
  if (mDeviceProcessingSettings.debugLevel >= 5) {
    GPUInfo("Allocating Permanent Memory");
  }
  int total = 0;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if ((mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && mMemoryResources[i].mPtr == nullptr) {
      total += AllocateRegisteredMemory(i);
    }
  }
  mHostMemoryPermanent = mHostMemoryPool;
  mDeviceMemoryPermanent = mDeviceMemoryPool;
  if (mDeviceProcessingSettings.debugLevel >= 5) {
    GPUInfo("Permanent Memory Done");
  }
  return total;
}

size_t GPUReconstruction::AllocateRegisteredMemoryHelper(GPUMemoryResource* res, void*& ptr, void*& memorypool, void* memorybase, size_t memorysize, void* (GPUMemoryResource::*setPtr)(void*))
{
  if (memorypool == nullptr) {
    GPUInfo("Memory pool uninitialized");
    throw std::bad_alloc();
  }
  ptr = memorypool;
  memorypool = (char*)((res->*setPtr)(memorypool));
  size_t retVal = (char*)memorypool - (char*)ptr;
  if (IsGPU() && retVal == 0) { // Transferring 0 bytes might break some GPU backends, but we cannot simply skip the transfer, or we will break event dependencies
    GPUProcessor::getPointerWithAlignment<GPUProcessor::MIN_ALIGNMENT, char>(memorypool, retVal = GPUProcessor::MIN_ALIGNMENT);
  }
  if ((size_t)((char*)memorypool - (char*)memorybase) > memorysize) {
    std::cout << "Memory pool size exceeded (" << res->mName << ": " << (char*)memorypool - (char*)memorybase << " < " << memorysize << "\n";
    throw std::bad_alloc();
  }
  memorypool = (void*)((char*)memorypool + GPUProcessor::getAlignment<GPUCA_MEMALIGN>(memorypool));
  if (mDeviceProcessingSettings.debugLevel >= 5) {
    std::cout << "Allocated " << res->mName << ": " << retVal << " - available: " << memorysize - ((char*)memorypool - (char*)memorybase) << "\n";
  }
  return (retVal);
}

size_t GPUReconstruction::AllocateRegisteredMemory(short ires)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if ((res->mType & GPUMemoryResource::MEMORY_PERMANENT) && res->mPtr != nullptr) {
    ResetRegisteredMemoryPointers(ires);
  } else if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
      if (res->mPtrDevice) {
        operator delete(res->mPtrDevice);
      }
      res->mSize = (size_t)res->SetPointers((void*)1) - 1;
      res->mPtrDevice = operator new(res->mSize + GPUProcessor::MIN_ALIGNMENT);
      res->mPtr = GPUProcessor::alignPointer<GPUProcessor::MIN_ALIGNMENT>(res->mPtrDevice);
      res->SetPointers(res->mPtr);
      if (mDeviceProcessingSettings.debugLevel >= 5) {
        std::cout << "Allocated " << res->mName << ": " << res->mSize << "\n";
      }
    }
  } else {
    if (res->mPtr != nullptr) {
      GPUError("Double allocation! (%s)", res->mName);
      throw std::bad_alloc();
    }
    if ((!IsGPU() || (res->mType & GPUMemoryResource::MEMORY_HOST) || mDeviceProcessingSettings.keepAllMemory) && !(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
      res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, mHostMemoryPool, mHostMemoryBase, mHostMemorySize, &GPUMemoryResource::SetPointers);
    }
    if (IsGPU() && (res->mType & GPUMemoryResource::MEMORY_GPU)) {
      if (res->mProcessor->mDeviceProcessor == nullptr) {
        GPUError("Device Processor not set (%s)", res->mName);
        throw std::bad_alloc();
      }
      size_t size = AllocateRegisteredMemoryHelper(res, res->mPtrDevice, mDeviceMemoryPool, mDeviceMemoryBase, mDeviceMemorySize, &GPUMemoryResource::SetDevicePointers);

      if (!(res->mType & GPUMemoryResource::MEMORY_HOST) || (res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
        res->mSize = size;
      } else if (size != res->mSize) {
        GPUError("Inconsistent device memory allocation (%s)", res->mName);
        throw std::bad_alloc();
      }
    }
  }
  return res->mSize;
}

void* GPUReconstruction::AllocateUnmanagedMemory(size_t size, int type)
{
  if (type != GPUMemoryResource::MEMORY_HOST && (!IsGPU() || type != GPUMemoryResource::MEMORY_GPU)) {
    throw std::bad_alloc();
  }
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mUnmanagedChunks.emplace_back(new char[size + GPUProcessor::MIN_ALIGNMENT]);
    return GPUProcessor::alignPointer<GPUProcessor::MIN_ALIGNMENT>(mUnmanagedChunks.back().get());
  } else {
    void* pool = type == GPUMemoryResource::MEMORY_GPU ? mDeviceMemoryPool : mHostMemoryPool;
    void* base = type == GPUMemoryResource::MEMORY_GPU ? mDeviceMemoryBase : mHostMemoryBase;
    size_t poolsize = type == GPUMemoryResource::MEMORY_GPU ? mDeviceMemorySize : mHostMemorySize;
    char* retVal;
    GPUProcessor::computePointerWithAlignment(pool, retVal, size);
    if ((size_t)((char*)pool - (char*)base) > poolsize) {
      throw std::bad_alloc();
    }
    return retVal;
  }
}

void GPUReconstruction::ResetRegisteredMemoryPointers(GPUProcessor* proc)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if (proc == nullptr || mMemoryResources[i].mProcessor == proc) {
      ResetRegisteredMemoryPointers(i);
    }
  }
}

void GPUReconstruction::ResetRegisteredMemoryPointers(short ires)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if (!(res->mType & GPUMemoryResource::MEMORY_EXTERNAL)) {
    res->SetPointers(res->mPtr);
  }
  if (IsGPU() && (res->mType & GPUMemoryResource::MEMORY_GPU)) {
    res->SetDevicePointers(res->mPtrDevice);
  }
}

void GPUReconstruction::FreeRegisteredMemory(GPUProcessor* proc, bool freeCustom, bool freePermanent)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if ((proc == nullptr || mMemoryResources[i].mProcessor == proc) && (freeCustom || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_CUSTOM)) && (freePermanent || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT))) {
      FreeRegisteredMemory(i);
    }
  }
}

void GPUReconstruction::FreeRegisteredMemory(short ires)
{
  GPUMemoryResource* res = &mMemoryResources[ires];
  if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    operator delete(res->mPtrDevice);
  }
  res->mPtr = nullptr;
  res->mPtrDevice = nullptr;
}

void GPUReconstruction::ClearAllocatedMemory(bool clearOutputs)
{
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    if (!(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_PERMANENT) && (clearOutputs || !(mMemoryResources[i].mType & GPUMemoryResource::MEMORY_OUTPUT))) {
      FreeRegisteredMemory(i);
    }
  }
  mHostMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mHostMemoryPermanent);
  mDeviceMemoryPool = GPUProcessor::alignPointer<GPUCA_MEMALIGN>(mDeviceMemoryPermanent);
  mUnmanagedChunks.clear();
}

static long long int ptrDiff(void* a, void* b) { return (long long int)((char*)a - (char*)b); }

void GPUReconstruction::PrintMemoryStatistics()
{
  std::map<std::string, std::array<size_t, 3>> sizes;
  for (unsigned int i = 0; i < mMemoryResources.size(); i++) {
    auto& res = mMemoryResources[i];
    auto& x = sizes[res.mName];
    if (res.mPtr) {
      x[0] += res.mSize;
    }
    if (res.mPtrDevice) {
      x[1] += res.mSize;
    }
    if (res.mType & GPUMemoryResource::MemoryType::MEMORY_PERMANENT) {
      x[2] = 1;
    }
  }
  for (auto it = sizes.begin(); it != sizes.end(); it++) {
    printf("Allocation %30s %s: Size %'13lld / %'13lld\n", it->first.c_str(), it->second[2] ? "P" : " ", (long long int)it->second[0], (long long int)it->second[1]);
  }
  if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    printf("Memory Allocation: Host %'lld / %'lld (Permanent %'lld), Device %'lld / %'lld, (Permanent %'lld) %d chunks\n",
           ptrDiff(mHostMemoryPool, mHostMemoryBase), (long long int)mHostMemorySize, ptrDiff(mHostMemoryPermanent, mHostMemoryBase),
           ptrDiff(mDeviceMemoryPool, mDeviceMemoryBase), (long long int)mDeviceMemorySize, ptrDiff(mDeviceMemoryPermanent, mDeviceMemoryBase), (int)mMemoryResources.size());
  }
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->PrintMemoryStatistics();
  }
}

void GPUReconstruction::PrepareEvent()
{
  ClearAllocatedMemory();
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->PrepareEvent();
  }
  for (unsigned int i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mAllocateAndInitializeLate) {
      continue;
    }
    (mProcessors[i].proc->*(mProcessors[i].SetMaxData))();
    if (mProcessors[i].proc->mDeviceProcessor) {
      (mProcessors[i].proc->mDeviceProcessor->*(mProcessors[i].SetMaxData))();
    }
  }
  AllocateRegisteredMemory(nullptr);
}

void GPUReconstruction::DumpSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  DumpStructToFile(&mEventSettings, f.c_str());
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->DumpSettings(dir);
  }
}

void GPUReconstruction::UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsDeviceProcessing* p)
{
  param().UpdateEventSettings(e, p);
}

void GPUReconstruction::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "settings.dump";
  mEventSettings.SetDefaults();
  ReadStructFromFile(f.c_str(), &mEventSettings);
  param().UpdateEventSettings(&mEventSettings);
  for (unsigned int i = 0; i < mChains.size(); i++) {
    mChains[i]->ReadSettings(dir);
  }
}

void GPUReconstruction::SetSettings(float solenoidBz)
{
  GPUSettingsEvent ev;
  ev.SetDefaults();
  ev.solenoidBz = solenoidBz;
  SetSettings(&ev, nullptr, nullptr);
}

void GPUReconstruction::SetSettings(const GPUSettingsEvent* settings, const GPUSettingsRec* rec, const GPUSettingsDeviceProcessing* proc, const GPURecoStepConfiguration* workflow)
{
  if (mInitialized) {
    GPUError("Cannot update settings while initialized");
    throw std::runtime_error("Settings updated while initialized");
  }
  mEventSettings = *settings;
  if (proc) {
    mDeviceProcessingSettings = *proc;
  }
  if (workflow) {
    mRecoSteps = workflow->steps;
    mRecoStepsGPU &= workflow->stepsGPUMask;
    mRecoStepsInputs = workflow->inputs;
    mRecoStepsOutputs = workflow->outputs;
  }
  param().SetDefaults(&mEventSettings, rec, proc, workflow);
}

void GPUReconstruction::SetOutputControl(void* ptr, size_t size)
{
  GPUOutputControl outputControl;
  outputControl.OutputType = GPUOutputControl::UseExternalBuffer;
  outputControl.OutputPtr = (char*)ptr;
  outputControl.OutputMaxSize = size;
  SetOutputControl(outputControl);
}

int GPUReconstruction::GetMaxThreads() { return mDeviceProcessingSettings.nThreads; }

std::unique_ptr<GPUReconstruction::GPUThreadContext> GPUReconstruction::GetThreadContext() { return std::unique_ptr<GPUReconstruction::GPUThreadContext>(new GPUThreadContext); }

GPUReconstruction* GPUReconstruction::CreateInstance(DeviceType type, bool forceType)
{
  GPUSettingsProcessing cfg;
  cfg.SetDefaults();
  cfg.deviceType = type;
  cfg.forceDeviceType = forceType;
  return CreateInstance(cfg);
}

GPUReconstruction* GPUReconstruction::CreateInstance(const GPUSettingsProcessing& cfg)
{
  GPUReconstruction* retVal = nullptr;
  unsigned int type = cfg.deviceType;
  if (type == DeviceType::CPU) {
    retVal = GPUReconstruction_Create_CPU(cfg);
  } else if (type == DeviceType::CUDA) {
    if ((retVal = sLibCUDA->GetPtr(cfg))) {
      retVal->mMyLib = sLibCUDA;
    }
  } else if (type == DeviceType::HIP) {
    if ((retVal = sLibHIP->GetPtr(cfg))) {
      retVal->mMyLib = sLibHIP;
    }
  } else if (type == DeviceType::OCL) {
    if ((retVal = sLibOCL->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL;
    }
  } else if (type == DeviceType::OCL2) {
    if ((retVal = sLibOCL2->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL2;
    }
  } else {
    GPUError("Error: Invalid device type %u", type);
    return nullptr;
  }

  if (retVal == nullptr) {
    if (cfg.forceDeviceType) {
      GPUError("Error: Could not load GPUReconstruction for specified device: %s (%u)", DEVICE_TYPE_NAMES[type], type);
    } else {
      GPUError("Could not load GPUReconstruction for device type %s (%u), falling back to CPU version", DEVICE_TYPE_NAMES[type], type);
      GPUSettingsProcessing cfg2 = cfg;
      cfg2.deviceType = DeviceType::CPU;
      retVal = CreateInstance(cfg2);
    }
  } else {
    GPUInfo("Created GPUReconstruction instance for device type %s (%u)", DEVICE_TYPE_NAMES[type], type);
  }

  return retVal;
}

GPUReconstruction* GPUReconstruction::CreateInstance(const char* type, bool forceType)
{
  DeviceType t = GetDeviceType(type);
  if (t == DeviceType::INVALID_DEVICE) {
    GPUError("Invalid device type: %s", type);
    return nullptr;
  }
  return CreateInstance(t, forceType);
}

GPUReconstruction::DeviceType GPUReconstruction::GetDeviceType(const char* type)
{
  for (unsigned int i = 1; i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]); i++) {
    if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0) {
      return (DeviceType)i;
    }
  }
  return DeviceType::INVALID_DEVICE;
}

#ifdef _WIN32
#define LIBRARY_EXTENSION ".dll"
#define LIBRARY_TYPE HMODULE
#define LIBRARY_LOAD(name) LoadLibraryEx(name, nullptr, nullptr)
#define LIBRARY_CLOSE FreeLibrary
#define LIBRARY_FUNCTION GetProcAddress
#else
#define LIBRARY_EXTENSION ".so"
#define LIBRARY_TYPE void*
#define LIBRARY_LOAD(name) dlopen(name, RTLD_NOW)
#define LIBRARY_CLOSE dlclose
#define LIBRARY_FUNCTION dlsym
#endif

#if defined(GPUCA_ALIROOT_LIB)
#define LIBRARY_PREFIX "Ali"
#elif defined(GPUCA_O2_LIB)
#define LIBRARY_PREFIX "O2"
#else
#define LIBRARY_PREFIX ""
#endif

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibCUDA(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "CUDA" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "CUDA"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibHIP(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "HIP" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "HIP"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "OCL" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "OCL"));

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL2(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "OCL2" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "OCL2"));

GPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr) {}

GPUReconstruction::LibraryLoader::~LibraryLoader() { CloseLibrary(); }

int GPUReconstruction::LibraryLoader::LoadLibrary()
{
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);

  if (mGPUEntry) {
    return 0;
  }

  LIBRARY_TYPE hGPULib;
  hGPULib = LIBRARY_LOAD(mLibName);
  if (hGPULib == nullptr) {
#ifndef _WIN32
    GPUImportant("The following error occured during dlopen: %s", dlerror());
#endif
    GPUError("Error Opening cagpu library for GPU Tracker (%s)", mLibName);
    return 1;
  } else {
    void* createFunc = LIBRARY_FUNCTION(hGPULib, mFuncName);
    if (createFunc == nullptr) {
      GPUError("Error fetching entry function in GPU library\n");
      LIBRARY_CLOSE(hGPULib);
      return 1;
    } else {
      mGPULib = (void*)(size_t)hGPULib;
      mGPUEntry = createFunc;
      GPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
    }
  }
  return 0;
}

GPUReconstruction* GPUReconstruction::LibraryLoader::GetPtr(const GPUSettingsProcessing& cfg)
{
  if (LoadLibrary()) {
    return nullptr;
  }
  if (mGPUEntry == nullptr) {
    return nullptr;
  }
  GPUReconstruction* (*tmp)(const GPUSettingsProcessing& cfg) = (GPUReconstruction * (*)(const GPUSettingsProcessing& cfg)) mGPUEntry;
  return tmp(cfg);
}

int GPUReconstruction::LibraryLoader::CloseLibrary()
{
  if (mGPUEntry == nullptr) {
    return 1;
  }
  LIBRARY_CLOSE((LIBRARY_TYPE)(size_t)mGPULib);
  mGPULib = nullptr;
  mGPUEntry = nullptr;
  return 0;
}
