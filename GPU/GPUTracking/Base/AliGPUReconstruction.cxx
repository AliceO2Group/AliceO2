#include <cstring>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>

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

#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionIncludes.h"

#include "AliGPUMemoryResource.h"
#include "AliGPUChain.h"

#define GPUCA_LOGGING_PRINTF
#include "AliGPULogging.h"

constexpr const char* const AliGPUReconstruction::DEVICE_TYPE_NAMES[];
constexpr const char* const AliGPUReconstruction::GEOMETRY_TYPE_NAMES[];
constexpr const char* const AliGPUReconstruction::IOTYPENAMES[];
constexpr AliGPUReconstruction::GeometryType AliGPUReconstruction::geometryType;

AliGPUReconstruction::AliGPUReconstruction(const AliGPUSettingsProcessing& cfg) : mHostConstantMem(new AliGPUConstantMem)
{
	mProcessingSettings = cfg;
	mDeviceProcessingSettings.SetDefaults();
	mEventSettings.SetDefaults();
	param().SetDefaults(&mEventSettings);
}

AliGPUReconstruction::~AliGPUReconstruction()
{
	//Reset these explicitly before the destruction of other members unloads the library
	mHostConstantMem.reset();
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		for (unsigned int i = 0;i < mMemoryResources.size();i++)
		{
			operator delete(mMemoryResources[i].mPtrDevice);
			mMemoryResources[i].mPtr = mMemoryResources[i].mPtrDevice = nullptr;
		}
	}
}

void AliGPUReconstruction::GetITSTraits(std::unique_ptr<o2::ITS::TrackerTraits>& trackerTraits, std::unique_ptr<o2::ITS::VertexerTraits>& vertexerTraits)
{
	trackerTraits.reset(new o2::ITS::TrackerTraitsCPU);
	vertexerTraits.reset(new o2::ITS::VertexerTraits);
}

int AliGPUReconstruction::Init()
{
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_AUTO) mDeviceProcessingSettings.memoryAllocationStrategy = IsGPU() ? AliGPUMemoryResource::ALLOCATION_GLOBAL : AliGPUMemoryResource::ALLOCATION_INDIVIDUAL;
	if (mDeviceProcessingSettings.eventDisplay) mDeviceProcessingSettings.keepAllMemory = true;
	if (mDeviceProcessingSettings.debugLevel >= 4) mDeviceProcessingSettings.keepAllMemory = true;
	if (mDeviceProcessingSettings.debugLevel < 6) mDeviceProcessingSettings.debugMask = 0;
#ifndef HAVE_O2HEADERS
	mRecoSteps.setBits(RecoStep::ITSTracking, false);
	mRecoSteps.setBits(RecoStep::TRDTracking, false);
#endif
	mRecoStepsGPU &= mRecoSteps;
	mRecoStepsGPU &= AvailableRecoSteps();
	if (!IsGPU()) mRecoStepsGPU.set((unsigned char) 0);
	if (!IsGPU()) mDeviceProcessingSettings.trackletConstructorInPipeline = mDeviceProcessingSettings.trackletSelectorInPipeline = false;
	if (param().rec.NonConsecutiveIDs) param().rec.DisableRefitAttachment = 0xFF;
	if (!mDeviceProcessingSettings.trackletConstructorInPipeline) mDeviceProcessingSettings.trackletSelectorInPipeline = false;
		
#ifdef GPUCA_HAVE_OPENMP
	if (mDeviceProcessingSettings.nThreads <= 0) mDeviceProcessingSettings.nThreads = omp_get_max_threads();
	else omp_set_num_threads(mDeviceProcessingSettings.nThreads);
#else
	mDeviceProcessingSettings.nThreads = 1;
#endif
	
	for (unsigned int i = 0;i < mChains.size();i++)
	{
		mChains[i]->RegisterPermanentMemoryAndProcessors();
	}

	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		(mProcessors[i].proc->*(mProcessors[i].RegisterMemoryAllocation))();
	}

	if (InitDevice()) return 1;
	AllocateRegisteredPermanentMemory();
	if (InitializeProcessors()) return 1;
	
	for (unsigned int i = 0;i < mChains.size();i++)
	{
		if (mChains[i]->Init()) return 1;
	}
	
	mInitialized = true;
	return 0;
}

int AliGPUReconstruction::Finalize()
{
	ExitDevice();
	mInitialized = false;
	return 0;
}

int AliGPUReconstruction::InitializeProcessors()
{
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		workers()->tpcTrackers[i].SetSlice(i);
	}
	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		(mProcessors[i].proc->*(mProcessors[i].InitializeProcessor))();
	}
	return 0;
}

void AliGPUReconstruction::RegisterGPUDeviceProcessor(AliGPUProcessor* proc, AliGPUProcessor* slaveProcessor)
{
	proc->InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE, slaveProcessor);
}

size_t AliGPUReconstruction::AllocateRegisteredMemory(AliGPUProcessor* proc)
{
	if (mDeviceProcessingSettings.debugLevel >= 5) printf("Allocating memory %p\n", proc);
	size_t total = 0;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		if ((proc == nullptr ? !mMemoryResources[i].mProcessor->mAllocateAndInitializeLate :  mMemoryResources[i].mProcessor == proc) && !(mMemoryResources[i].mType & AliGPUMemoryResource::MEMORY_CUSTOM)) total += AllocateRegisteredMemory(i);
	}
	if (mDeviceProcessingSettings.debugLevel >= 5) printf("Allocating memory done\n");
	return total;
}

size_t AliGPUReconstruction::AllocateRegisteredPermanentMemory()
{
	if (mDeviceProcessingSettings.debugLevel >= 5) printf("Allocating Permanent Memory\n");
	int total = 0;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		if ((mMemoryResources[i].mType & AliGPUMemoryResource::MEMORY_PERMANENT) && mMemoryResources[i].mPtr == nullptr) total += AllocateRegisteredMemory(i);
	}
	mHostMemoryPermanent = mHostMemoryPool;
	mDeviceMemoryPermanent = mDeviceMemoryPool;
	if (mDeviceProcessingSettings.debugLevel >= 5) printf("Permanent Memory Done\n");
	return total;
}

size_t AliGPUReconstruction::AllocateRegisteredMemoryHelper(AliGPUMemoryResource* res, void* &ptr, void* &memorypool, void* memorybase, size_t memorysize, void* (AliGPUMemoryResource::*setPtr)(void*))
{
	if (memorypool == nullptr) {printf("Memory pool uninitialized\n");throw std::bad_alloc();}
	ptr = memorypool;
	memorypool = (char*) ((res->*setPtr)(memorypool));
	size_t retVal = (char*) memorypool - (char*) ptr;
	if (IsGPU() && retVal == 0) //Transferring 0 bytes might break some GPU backends, but we cannot simply skip the transfer, or we will break event dependencies
	{
		AliGPUProcessor::getPointerWithAlignment<AliGPUProcessor::MIN_ALIGNMENT, char>(memorypool, retVal = AliGPUProcessor::MIN_ALIGNMENT);
	}
	if ((size_t) ((char*) memorypool - (char*) memorybase) > memorysize) {std::cout << "Memory pool size exceeded (" << res->mName << ": " << (char*) memorypool - (char*) memorybase << " < " << memorysize << "\n"; throw std::bad_alloc();}
	memorypool = (void*) ((char*) memorypool + AliGPUProcessor::getAlignment<GPUCA_MEMALIGN>(memorypool));
	if (mDeviceProcessingSettings.debugLevel >= 5) std::cout << "Allocated " << res->mName << ": " << retVal << " - available: " << memorysize - ((char*) memorypool - (char*) memorybase) << "\n";
	return(retVal);
}

size_t AliGPUReconstruction::AllocateRegisteredMemory(short ires)
{
	AliGPUMemoryResource* res = &mMemoryResources[ires];
	if ((res->mType & AliGPUMemoryResource::MEMORY_PERMANENT) && res->mPtr != nullptr)
	{
		ResetRegisteredMemoryPointers(ires);
	}
	else if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		if (res->mPtrDevice) operator delete(res->mPtrDevice);
		res->mSize = (size_t) res->SetPointers((void*) 1) - 1;
		res->mPtrDevice = operator new(res->mSize + AliGPUProcessor::MIN_ALIGNMENT);
		res->mPtr = AliGPUProcessor::alignPointer<AliGPUProcessor::MIN_ALIGNMENT>(res->mPtrDevice);
		res->SetPointers(res->mPtr);
		if (mDeviceProcessingSettings.debugLevel >= 5) std::cout << "Allocated " << res->mName << ": " << res->mSize << "\n";
	}
	else
	{
		if (res->mPtr != nullptr) {printf("Double allocation! (%s)\n", res->mName); throw std::bad_alloc();}
		if (!IsGPU() || (res->mType & AliGPUMemoryResource::MEMORY_HOST) || mDeviceProcessingSettings.keepAllMemory)
		{
			res->mSize = AllocateRegisteredMemoryHelper(res, res->mPtr, mHostMemoryPool, mHostMemoryBase, mHostMemorySize, &AliGPUMemoryResource::SetPointers);
		}
		if (IsGPU() && (res->mType & AliGPUMemoryResource::MEMORY_GPU))
		{
			if (res->mProcessor->mDeviceProcessor == nullptr) {printf("Device Processor not set (%s)\n", res->mName); throw std::bad_alloc();}
			size_t size = AllocateRegisteredMemoryHelper(res, res->mPtrDevice, mDeviceMemoryPool, mDeviceMemoryBase, mDeviceMemorySize, &AliGPUMemoryResource::SetDevicePointers);
			
			if (!(res->mType & AliGPUMemoryResource::MEMORY_HOST))
			{
				res->mSize = size;
			}
			else if (size != res->mSize)
			{
				printf("Inconsistent device memory allocation (%s)\n", res->mName);
				throw std::bad_alloc();
			}
		}
	}
	return res->mSize;
}

void* AliGPUReconstruction::AllocateUnmanagedMemory(size_t size, int type)
{
	if (type != AliGPUMemoryResource::MEMORY_HOST && (!IsGPU() || type != AliGPUMemoryResource::MEMORY_GPU)) throw std::bad_alloc();
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		mUnmanagedChunks.emplace_back(new char[size + AliGPUProcessor::MIN_ALIGNMENT]);
		return AliGPUProcessor::alignPointer<AliGPUProcessor::MIN_ALIGNMENT>(mUnmanagedChunks.back().get());
	}
	else
	{
		void* pool = type == AliGPUMemoryResource::MEMORY_GPU ? mDeviceMemoryPool : mHostMemoryPool;
		void* base = type == AliGPUMemoryResource::MEMORY_GPU ? mDeviceMemoryBase : mHostMemoryBase;
		size_t poolsize = type == AliGPUMemoryResource::MEMORY_GPU ? mDeviceMemorySize : mHostMemorySize;
		char* retVal;
		AliGPUProcessor::computePointerWithAlignment(pool, retVal, size);
		if ((size_t) ((char*) pool - (char*) base) > poolsize) throw std::bad_alloc();
		return retVal;
	}
}

void AliGPUReconstruction::ResetRegisteredMemoryPointers(AliGPUProcessor* proc)
{
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		if (proc == nullptr || mMemoryResources[i].mProcessor == proc) ResetRegisteredMemoryPointers(i);
	}
}

void AliGPUReconstruction::ResetRegisteredMemoryPointers(short ires)
{
	AliGPUMemoryResource* res = &mMemoryResources[ires];
	res->SetPointers(res->mPtr);
	if (IsGPU() && (res->mType & AliGPUMemoryResource::MEMORY_GPU)) res->SetDevicePointers(res->mPtrDevice);
}

void AliGPUReconstruction::FreeRegisteredMemory(AliGPUProcessor* proc, bool freeCustom)
{
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		if ((proc == nullptr || mMemoryResources[i].mProcessor == proc) && (freeCustom || !(mMemoryResources[i].mType & AliGPUMemoryResource::MEMORY_CUSTOM))) FreeRegisteredMemory(i);
	}
}

void AliGPUReconstruction::FreeRegisteredMemory(short ires)
{
	AliGPUMemoryResource* res = &mMemoryResources[ires];
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL) operator delete(res->mPtrDevice);
	res->mPtr = nullptr;
	res->mPtrDevice = nullptr;
}

void AliGPUReconstruction::ClearAllocatedMemory()
{
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		if (!(mMemoryResources[i].mType & AliGPUMemoryResource::MEMORY_PERMANENT)) FreeRegisteredMemory(i);
	}
	mHostMemoryPool = AliGPUProcessor::alignPointer<GPUCA_MEMALIGN>(mHostMemoryPermanent);
	mDeviceMemoryPool = AliGPUProcessor::alignPointer<GPUCA_MEMALIGN>(mDeviceMemoryPermanent);
	mUnmanagedChunks.clear();
}

void AliGPUReconstruction::PrepareEvent()
{
	ClearAllocatedMemory();
	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		if (mProcessors[i].proc->mAllocateAndInitializeLate) continue;
		(mProcessors[i].proc->*(mProcessors[i].SetMaxData))();
		if (mProcessors[i].proc->mDeviceProcessor) (mProcessors[i].proc->mDeviceProcessor->*(mProcessors[i].SetMaxData))();
	}
	AllocateRegisteredMemory(nullptr);
}

void AliGPUReconstruction::DumpSettings(const char* dir)
{
	std::string f;
	f = dir;
	f += "settings.dump";
	DumpStructToFile(&mEventSettings, f.c_str());
	for (unsigned int i = 0;i < mChains.size();i++)
	{
		mChains[i]->DumpSettings(dir);
	}
}

void AliGPUReconstruction::ReadSettings(const char* dir)
{
	std::string f;
	f = dir;
	f += "settings.dump";
	mEventSettings.SetDefaults();
	ReadStructFromFile(f.c_str(), &mEventSettings);
	param().UpdateEventSettings(&mEventSettings);
	for (unsigned int i = 0;i < mChains.size();i++)
	{
		mChains[i]->ReadSettings(dir);
	}
}

void AliGPUReconstruction::SetSettings(float solenoidBz)
{
	AliGPUSettingsEvent ev;
	ev.SetDefaults();
	ev.solenoidBz = solenoidBz;
	SetSettings(&ev, nullptr, nullptr);
}

void AliGPUReconstruction::SetSettings(const AliGPUSettingsEvent* settings, const AliGPUSettingsRec* rec, const AliGPUSettingsDeviceProcessing* proc)
{
	mEventSettings = *settings;
	if (proc) mDeviceProcessingSettings = *proc;
	param().SetDefaults(&mEventSettings, rec, proc);
}

void AliGPUReconstruction::SetOutputControl(void* ptr, size_t size)
{
	AliGPUOutputControl outputControl;
	outputControl.OutputType = AliGPUOutputControl::UseExternalBuffer;
	outputControl.OutputPtr = (char*) ptr;
	outputControl.OutputMaxSize = size;
	SetOutputControl(outputControl);
}

int AliGPUReconstruction::GetMaxThreads()
{
	return mDeviceProcessingSettings.nThreads;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(DeviceType type, bool forceType)
{
	AliGPUSettingsProcessing cfg;
	cfg.SetDefaults();
	cfg.deviceType = type;
	cfg.forceDeviceType = forceType;
	return CreateInstance(cfg);
}
	
AliGPUReconstruction* AliGPUReconstruction::CreateInstance(const AliGPUSettingsProcessing& cfg)
{
	AliGPUReconstruction* retVal = nullptr;
	unsigned int type = cfg.deviceType;
	if (type == DeviceType::CPU)
	{
		retVal = AliGPUReconstruction_Create_CPU(cfg);
	}
	else if (type == DeviceType::CUDA)
	{
		if ((retVal = sLibCUDA->GetPtr(cfg))) retVal->mMyLib = sLibCUDA;
	}
	else if (type == DeviceType::HIP)
	{
		if((retVal = sLibHIP->GetPtr(cfg))) retVal->mMyLib = sLibHIP;
	}
	else if (type == DeviceType::OCL)
	{
		if((retVal = sLibOCL->GetPtr(cfg))) retVal->mMyLib = sLibOCL;
	}
	else
	{
		printf("Error: Invalid device type %d\n", type);
		return nullptr;
	}
	
	if (retVal == 0)
	{
		if (cfg.forceDeviceType)
		{
			printf("Error: Could not load AliGPUReconstruction for specified device: %s (%d)\n", DEVICE_TYPE_NAMES[type], type);
		}
		else
		{
			printf("Could not load AliGPUReconstruction for device type %s (%d), falling back to CPU version\n", DEVICE_TYPE_NAMES[type], type);
			AliGPUSettingsProcessing cfg2 = cfg;
			cfg2.deviceType = DeviceType::CPU;
			retVal = CreateInstance(cfg2);
		}
	}
	else
	{
		printf("Created AliGPUReconstruction instance for device type %s (%d)\n", DEVICE_TYPE_NAMES[type], type);
	}
	
	return retVal;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(const char* type, bool forceType)
{
	DeviceType t = GetDeviceType(type);
	if (t == DeviceType::INVALID_DEVICE)
	{
		printf("Invalid device type: %s\n", type);
		return nullptr;
	}
	return CreateInstance(t, forceType);
}

AliGPUReconstruction::DeviceType AliGPUReconstruction::GetDeviceType(const char* type)
{
	for (unsigned int i = 1;i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]);i++)
	{
		if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0)
		{
			return (DeviceType) i;
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

std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibCUDA(new AliGPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking" "CUDA" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "CUDA"));
std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibHIP(new AliGPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking" "HIP" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "HIP"));
std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibOCL(new AliGPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking" "OCL" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "OCL"));

AliGPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr)
{}

AliGPUReconstruction::LibraryLoader::~LibraryLoader()
{
	CloseLibrary();
}

int AliGPUReconstruction::LibraryLoader::LoadLibrary()
{
	static std::mutex mut;
	std::lock_guard<std::mutex> lock(mut);
	
	if (mGPUEntry) return 0;
	
	LIBRARY_TYPE hGPULib;
	hGPULib = LIBRARY_LOAD(mLibName);
	if (hGPULib == nullptr)
	{
		#ifndef _WIN32
			GPUImportant("The following error occured during dlopen: %s", dlerror());
		#endif
		GPUError("Error Opening cagpu library for GPU Tracker (%s)", mLibName);
		return 1;
	}
	else
	{
		void* createFunc = LIBRARY_FUNCTION(hGPULib, mFuncName);
		if (createFunc == nullptr)
		{
			GPUError("Error fetching entry function in GPU library\n");
			LIBRARY_CLOSE(hGPULib);
			return 1;
		}
		else
		{
			mGPULib = (void*) (size_t) hGPULib;
			mGPUEntry = createFunc;
			GPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
		}
	}
	return 0;
}

AliGPUReconstruction* AliGPUReconstruction::LibraryLoader::GetPtr(const AliGPUSettingsProcessing& cfg)
{
	if (LoadLibrary()) return nullptr;
	if (mGPUEntry == nullptr) return nullptr;
	AliGPUReconstruction* (*tmp)(const AliGPUSettingsProcessing& cfg) = (AliGPUReconstruction* (*)(const AliGPUSettingsProcessing& cfg)) mGPUEntry;
	return tmp(cfg);
}

int AliGPUReconstruction::LibraryLoader::CloseLibrary()
{
	if (mGPUEntry == nullptr) return 1;
	LIBRARY_CLOSE((LIBRARY_TYPE) (size_t) mGPULib);
	mGPULib = nullptr;
	mGPUEntry = nullptr;
	return 0;
}
