#if !defined(ALIGPURECONSTRUCTION_H) && !defined(__OPENCL__)
#define ALIGPURECONSTRUCTION_H

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>
#include <vector>

#include "AliGPUTRDDef.h"
#include "AliGPUParam.h"
#include "AliGPUSettings.h"
#include "AliGPUOutputControl.h"
#include "AliGPUMemoryResource.h"
#include "AliGPUConstantMem.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUDataTypes.h"

#include "utils/bitfield.h"

class AliGPUChain;

namespace o2 { namespace ITS { class TrackerTraits; class VertexerTraits; }}

class AliGPUReconstruction
{
	friend class AliGPUChain;
protected:
	class LibraryLoader; //These must be the first members to ensure correct destructor order!
	std::shared_ptr<LibraryLoader> mMyLib = nullptr;
	std::vector<AliGPUMemoryResource> mMemoryResources;
	std::vector<std::unique_ptr<char[]>> mUnmanagedChunks;
	std::vector<std::unique_ptr<AliGPUChain>> mChains;
	
public:
	virtual ~AliGPUReconstruction();
	AliGPUReconstruction(const AliGPUReconstruction&) = delete;
	AliGPUReconstruction& operator=(const AliGPUReconstruction&) = delete;
	
	//General definitions
	constexpr static unsigned int NSLICES = GPUCA_NSLICES;

	using GeometryType = AliGPUDataTypes::GeometryType;
	using DeviceType = AliGPUDataTypes::DeviceType;
	using RecoStep = AliGPUDataTypes::RecoStep;
	
	static constexpr const char* const GEOMETRY_TYPE_NAMES[] = {"INVALID", "ALIROOT", "O2"};
#ifdef GPUCA_TPC_GEOMETRY_O2
	static constexpr GeometryType geometryType = GeometryType::O2;
#else
	static constexpr GeometryType geometryType = GeometryType::ALIROOT;
#endif
	
	static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL"};
	static DeviceType GetDeviceType(const char* type);
	enum InOutPointerType : unsigned int {CLUSTER_DATA = 0, SLICE_OUT_TRACK = 1, SLICE_OUT_CLUSTER = 2, MC_LABEL_TPC = 3, MC_INFO_TPC = 4, MERGED_TRACK = 5, MERGED_TRACK_HIT = 6, TRD_TRACK = 7, TRD_TRACKLET = 8, RAW_CLUSTERS = 9, CLUSTERS_NATIVE = 10, TRD_TRACKLET_MC = 11};
	static constexpr const char* const IOTYPENAMES[] = {"TPC Clusters", "TPC Slice Tracks", "TPC Slice Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets", "Raw Clusters", "ClusterNative", "TRD Tracklet MC Labels"};
	typedef bitfield<RecoStep, unsigned int> RecoStepField;

	//Functionality to create an instance of AliGPUReconstruction for the desired device
	static AliGPUReconstruction* CreateInstance(const AliGPUSettingsProcessing& cfg);
	static AliGPUReconstruction* CreateInstance(DeviceType type = DeviceType::CPU, bool forceType = true);
	static AliGPUReconstruction* CreateInstance(int type, bool forceType) {return CreateInstance((DeviceType) type, forceType);}
	static AliGPUReconstruction* CreateInstance(const char* type, bool forceType);
	
	template <class T> T* AddChain();
	
	int Init();
	int Finalize();
	
	void DumpSettings(const char* dir = "");
	void ReadSettings(const char* dir = "");
	
	virtual int RunStandalone() = 0;
	
	//Helpers for memory allocation
	AliGPUMemoryResource& Res(short num) {return mMemoryResources[num];}
	template <class T> short RegisterMemoryAllocation(T* proc, void* (T::* setPtr)(void*), int type, const char* name = "");
	size_t AllocateMemoryResources();
	size_t AllocateRegisteredMemory(AliGPUProcessor* proc);
	size_t AllocateRegisteredMemory(short res);
	void* AllocateUnmanagedMemory(size_t size, int type);
	void FreeRegisteredMemory(AliGPUProcessor* proc, bool freeCustom = false);
	void FreeRegisteredMemory(short res);
	void ClearAllocatedMemory();
	void ResetRegisteredMemoryPointers(AliGPUProcessor* proc);
	void ResetRegisteredMemoryPointers(short res);
	void PrepareEvent();
	
	//Helpers to fetch processors from other shared libraries
	virtual void GetITSTraits(std::unique_ptr<o2::ITS::TrackerTraits>& trackerTraits, std::unique_ptr<o2::ITS::VertexerTraits>& vertexerTraits);
	
	//Getters / setters for parameters
	DeviceType GetDeviceType() const {return (DeviceType) mProcessingSettings.deviceType;}
	bool IsGPU() const {return GetDeviceType() != DeviceType::INVALID_DEVICE && GetDeviceType() != DeviceType::CPU;}
	const AliGPUParam& GetParam() const {return mHostConstantMem->param;}
	const AliGPUSettingsEvent& GetEventSettings() const {return mEventSettings;}
	const AliGPUSettingsProcessing& GetProcessingSettings() {return mProcessingSettings;}
	const AliGPUSettingsDeviceProcessing& GetDeviceProcessingSettings() const {return mDeviceProcessingSettings;}
	bool IsInitialized() const {return mInitialized;}
	void SetSettings(float solenoidBz);
	void SetSettings(const AliGPUSettingsEvent* settings, const AliGPUSettingsRec* rec = nullptr, const AliGPUSettingsDeviceProcessing* proc = nullptr);
	void SetResetTimers(bool reset) {mDeviceProcessingSettings.resetTimers = reset;}
	void SetOutputControl(const AliGPUOutputControl& v) {mOutputControl = v;}
	void SetOutputControl(void* ptr, size_t size);
	AliGPUOutputControl& OutputControl() {return mOutputControl;}
	virtual int GetMaxThreads();
	const void* DeviceMemoryBase() const {return mDeviceMemoryBase;}
	
	RecoStepField& RecoSteps() {if (mInitialized) throw std::runtime_error("Cannot change reco steps once initialized"); return mRecoSteps;}
	RecoStepField& RecoStepsGPU() {if (mInitialized) throw std::runtime_error("Cannot change reco steps once initialized"); return mRecoStepsGPU;}
	RecoStepField GetRecoSteps() const {return mRecoSteps;}
	RecoStepField GetRecoStepsGPU() const {return mRecoStepsGPU;}
	
	//Registration of GPU Processors
	template <class T> void RegisterGPUProcessor(T* proc, bool deviceSlave);
	template <class T> void SetupGPUProcessor(T* proc, bool allocate);
	void RegisterGPUDeviceProcessor(AliGPUProcessor* proc, AliGPUProcessor* slaveProcessor);
	
protected:
	AliGPUReconstruction(const AliGPUSettingsProcessing& cfg);				//Constructor
	virtual int InitDevice() = 0;
	virtual int ExitDevice() = 0;
	
	//Private helper functions for memory management
	size_t AllocateRegisteredMemoryHelper(AliGPUMemoryResource* res, void* &ptr, void* &memorypool, void* memorybase, size_t memorysize, void* (AliGPUMemoryResource::*SetPointers)(void*));
	size_t AllocateRegisteredPermanentMemory();
	
	//Private helper functions for reading / writing / allocating IO buffer from/to file
	template <class T> void DumpData(FILE* fp, const T* const* entries, const unsigned int* num, InOutPointerType type);
	template <class T> size_t ReadData(FILE* fp, const T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type);
	template <class T> void AllocateIOMemoryHelper(unsigned int n, const T* &ptr, std::unique_ptr<T[]> &u);
	
	//Private helper functions to dump / load flat objects
	template <class T> void DumpFlatObjectToFile(const T* obj, const char* file);
	template <class T> std::unique_ptr<T> ReadFlatObjectFromFile(const char* file);
	template <class T> void DumpStructToFile(const T* obj, const char* file);
	template <class T> std::unique_ptr<T> ReadStructFromFile(const char* file);
	template <class T> void ReadStructFromFile(const char* file, T* obj);
	
	//Others
	virtual RecoStepField AvailableRecoSteps() {return RecoStep::AllRecoSteps;}
	
	//Pointers to tracker classes
	AliGPUConstantMem* workers() {return mHostConstantMem.get();}
	const AliGPUConstantMem* workers() const {return mHostConstantMem.get();}
	AliGPUParam& param() {return mHostConstantMem->param;}
	std::unique_ptr<AliGPUConstantMem> mHostConstantMem;

	//Settings
	AliGPUSettingsEvent mEventSettings;										//Event Parameters
	AliGPUSettingsProcessing mProcessingSettings;								//Processing Parameters (at constructor level)
	AliGPUSettingsDeviceProcessing mDeviceProcessingSettings;					//Processing Parameters (at init level)
	AliGPUOutputControl mOutputControl;										//Controls the output of the individual components
	
	RecoStepField mRecoSteps = (unsigned char) -1;
	RecoStepField mRecoStepsGPU = (unsigned char) -1;

	//Ptrs to host and device memory;
	void* mHostMemoryBase = nullptr;
	void* mHostMemoryPermanent = nullptr;
	void* mHostMemoryPool = nullptr;
	size_t mHostMemorySize = 0;
	void* mDeviceMemoryBase = nullptr;
	void* mDeviceMemoryPermanent = nullptr;
	void* mDeviceMemoryPool = nullptr;
	size_t mDeviceMemorySize = 0;
	
	//Others
	bool mInitialized = false;
	int mStatNEvents = 0;

	//Management for AliGPUProcessors
	struct ProcessorData
	{
		ProcessorData(AliGPUProcessor* p, void (AliGPUProcessor::* r)(), void (AliGPUProcessor::* i)(), void (AliGPUProcessor::* d)()) : proc(p), RegisterMemoryAllocation(r), InitializeProcessor(i), SetMaxData(d) {}
		AliGPUProcessor* proc;
		void (AliGPUProcessor::* RegisterMemoryAllocation)();
		void (AliGPUProcessor::* InitializeProcessor)();
		void (AliGPUProcessor::* SetMaxData)();
	};
	std::vector<ProcessorData> mProcessors;

	//Helpers for loading device library via dlopen
	class LibraryLoader
	{
	public:
		~LibraryLoader();

	private:
		friend class AliGPUReconstruction;
		LibraryLoader(const char* lib, const char* func);
		LibraryLoader(const LibraryLoader&) CON_DELETE;
		const LibraryLoader& operator= (const LibraryLoader&) CON_DELETE;
		int LoadLibrary();
		int CloseLibrary();
		AliGPUReconstruction* GetPtr(const AliGPUSettingsProcessing& cfg);
		
		const char* mLibName;
		const char* mFuncName;
		void* mGPULib;
		void* mGPUEntry;
	};
	static std::shared_ptr<LibraryLoader> sLibCUDA, sLibHIP, sLibOCL;
	
private:
	static AliGPUReconstruction* AliGPUReconstruction_Create_CPU(const AliGPUSettingsProcessing& cfg);
};

template <class T> inline void AliGPUReconstruction::AllocateIOMemoryHelper(unsigned int n, const T* &ptr, std::unique_ptr<T[]> &u)
{
	if (n == 0)
	{
		u.reset(nullptr);
		return;
	}
	u.reset(new T[n]);
	ptr = u.get();
}

template <class T> inline T* AliGPUReconstruction::AddChain()
{
	mChains.emplace_back(new T(this));
	return (T*) mChains.back().get();
}

template <class T> inline short AliGPUReconstruction::RegisterMemoryAllocation(T* proc, void* (T::* setPtr)(void*), int type, const char* name)
{
	if (!(type & (AliGPUMemoryResource::MEMORY_HOST | AliGPUMemoryResource::MEMORY_GPU)))
	{
		if ((type & AliGPUMemoryResource::MEMORY_SCRATCH) && !mDeviceProcessingSettings.keepAllMemory)
		{
			type |= (proc->mGPUProcessorType == AliGPUProcessor::PROCESSOR_TYPE_CPU ? AliGPUMemoryResource::MEMORY_HOST : AliGPUMemoryResource::MEMORY_GPU);
		}
		else
		{
			type |= AliGPUMemoryResource::MEMORY_HOST | AliGPUMemoryResource::MEMORY_GPU;
		}

	}
	if (proc->mGPUProcessorType == AliGPUProcessor::PROCESSOR_TYPE_CPU) type &= ~AliGPUMemoryResource::MEMORY_GPU;
	mMemoryResources.emplace_back(proc, static_cast<void* (AliGPUProcessor::*)(void*)>(setPtr), (AliGPUMemoryResource::MemoryType) type, name);
	if (mMemoryResources.size() == 32768) throw std::bad_alloc();
	return mMemoryResources.size() - 1;
}

template <class T> inline void AliGPUReconstruction::RegisterGPUProcessor(T* proc, bool deviceSlave)
{
	mProcessors.emplace_back(proc, static_cast<void (AliGPUProcessor::*)()>(&T::RegisterMemoryAllocation), static_cast<void (AliGPUProcessor::*)()>(&T::InitializeProcessor), static_cast<void (AliGPUProcessor::*)()>(&T::SetMaxData));
	AliGPUProcessor::ProcessorType processorType = deviceSlave ? AliGPUProcessor::PROCESSOR_TYPE_SLAVE : AliGPUProcessor::PROCESSOR_TYPE_CPU;
	proc->InitGPUProcessor(this, processorType);
}

template <class T> inline void AliGPUReconstruction::SetupGPUProcessor(T* proc, bool allocate)
{
	static_assert(sizeof(T) > sizeof(AliGPUProcessor), "Need to setup derrived class");
	if (allocate) proc->SetMaxData();
	if (proc->mDeviceProcessor)
	{
		std::memcpy((void*) proc->mDeviceProcessor, (const void*) proc, sizeof(*proc));
		proc->mDeviceProcessor->InitGPUProcessor((AliGPUReconstruction*) this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
	}
	if (allocate) AllocateRegisteredMemory(proc);
	else ResetRegisteredMemoryPointers(proc);
}

template <class T> inline void AliGPUReconstruction::DumpData(FILE* fp, const T* const* entries, const unsigned int* num, InOutPointerType type)
{
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER || type == RAW_CLUSTERS) count = NSLICES;
	else if (type == CLUSTERS_NATIVE) count = NSLICES * GPUCA_ROW_COUNT;
	else count = 1;
	unsigned int numTotal = 0;
	for (int i = 0;i < count;i++) numTotal += num[i];
	if (numTotal == 0) return;
	fwrite(&type, sizeof(type), 1, fp);
	for (int i = 0;i < count;i++)
	{
		fwrite(&num[i], sizeof(num[i]), 1, fp);
		if (num[i])
		{
			fwrite(entries[i], sizeof(*entries[i]), num[i], fp);
		}
	}
}

template <class T> inline size_t AliGPUReconstruction::ReadData(FILE* fp, const T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
{
	if (feof(fp)) return 0;
	InOutPointerType inType;
	size_t r, pos = ftell(fp);
	r = fread(&inType, sizeof(inType), 1, fp);
	if (r != 1 || inType != type)
	{
		fseek(fp, pos, SEEK_SET);
		return 0;
	}
	
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER || type == RAW_CLUSTERS) count = NSLICES;
	else if (type == CLUSTERS_NATIVE) count = NSLICES * GPUCA_ROW_COUNT;
	else count = 1;
	size_t numTotal = 0;
	for (int i = 0;i < count;i++)
	{
		r = fread(&num[i], sizeof(num[i]), 1, fp);
		AllocateIOMemoryHelper(num[i], entries[i], mem[i]);
		if (num[i]) r = fread(mem[i].get(), sizeof(*entries[i]), num[i], fp);
		numTotal += num[i];
	}
	(void) r;
	if (mDeviceProcessingSettings.debugLevel >= 2) printf("Read %d %s\n", (int) numTotal, IOTYPENAMES[type]);
	return numTotal;
}

template <class T> inline void AliGPUReconstruction::DumpFlatObjectToFile(const T* obj, const char* file)
{
	FILE* fp = fopen(file, "w+b");
	if (fp == nullptr) return;
	size_t size[2] = {sizeof(*obj), obj->getFlatBufferSize()};
	fwrite(size, sizeof(size[0]), 2, fp);
	fwrite(obj, 1, size[0], fp);
	fwrite(obj->getFlatBufferPtr(), 1, size[1], fp);
	fclose(fp);
}

template <class T> inline std::unique_ptr<T> AliGPUReconstruction::ReadFlatObjectFromFile(const char* file)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return nullptr;
	size_t size[2], r;
	r = fread(size, sizeof(size[0]), 2, fp);
	if (r == 0 || size[0] != sizeof(T)) {fclose(fp); return nullptr;}
	std::unique_ptr<T> retVal(new T);
	char* buf = new char[size[1]]; //Not deleted as ownership is transferred to FlatObject
	r = fread((void*) retVal.get(), 1, size[0], fp);
	r = fread(buf, 1, size[1], fp);
	fclose(fp);
	if (mDeviceProcessingSettings.debugLevel >= 2) printf("Read %d bytes from %s\n", (int) r, file);
	retVal->clearInternalBufferPtr();
	retVal->setActualBufferAddress(buf);
	retVal->adoptInternalBuffer(buf);
	return std::move(retVal);
}

template <class T> inline void AliGPUReconstruction::DumpStructToFile(const T* obj, const char* file)
{
	FILE* fp = fopen(file, "w+b");
	if (fp == nullptr) return;
	size_t size = sizeof(*obj);
	fwrite(&size, sizeof(size), 1, fp);
	fwrite(obj, 1, size, fp);
	fclose(fp);
}

template <class T> inline std::unique_ptr<T> AliGPUReconstruction::ReadStructFromFile(const char* file)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return nullptr;
	size_t size, r;
	r = fread(&size, sizeof(size), 1, fp);
	if (r == 0 || size != sizeof(T)) {fclose(fp); return nullptr;}
	std::unique_ptr<T> newObj(new T);
	r = fread(newObj.get(), 1, size, fp);
	fclose(fp);
	if (mDeviceProcessingSettings.debugLevel >= 2) printf("Read %d bytes from %s\n", (int) r, file);
	return std::move(newObj);
}

template <class T> inline void AliGPUReconstruction::ReadStructFromFile(const char* file, T* obj)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return;
	size_t size, r;
	r = fread(&size, sizeof(size), 1, fp);
	if (r == 0) {fclose(fp); return;}
	r = fread(obj, 1, size, fp);
	fclose(fp);
	if (mDeviceProcessingSettings.debugLevel >= 2) printf("Read %d bytes from %s\n", (int) r, file);
}

#endif
