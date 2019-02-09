#if !defined(ALIGPURECONSTRUCTION_H) && !defined(__OPENCL__)
#define ALIGPURECONSTRUCTION_H

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>
#include <vector>

#include "AliGPUTRDDef.h"
#include "AliGPUCAParam.h"
#include "AliGPUCASettings.h"
#include "AliGPUCAOutputControl.h"
#include "AliGPUMemoryResource.h"
#include "AliGPUCADataTypes.h"
#include "AliGPUTPCSliceOutput.h"

class AliGPUTPCSliceOutput;
class AliGPUTPCSliceOutTrack;
class AliGPUTPCSliceOutCluster;
class AliGPUTPCGMMergedTrack;
struct AliGPUTPCGMMergedTrackHit;
class AliGPUTRDTrackletWord;
class AliHLTTPCClusterMCLabel;
class AliGPUTPCMCInfo;
class AliGPUTRDTracker;
class AliGPUTPCGPUTracker;
struct AliGPUTPCClusterData;
struct AliHLTTPCRawCluster;
struct ClusterNativeAccessExt;
struct AliGPUTRDTrackletLabels;
class AliGPUCADisplay;
class AliGPUCAQA;
class AliGPUTRDGeometry;

namespace o2 { namespace ITS { class TrackerTraits; class VertexerTraits; }}
namespace o2 { namespace trd { class TRDGeometryFlat; }}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative; }}
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class AliGPUReconstruction
{
protected:
	class LibraryLoader; //These must be the first members to ensure correct destructor order!
	std::shared_ptr<LibraryLoader> mMyLib = nullptr;
	std::vector<AliGPUMemoryResource> mMemoryResources;
	std::vector<std::unique_ptr<char[]>> mUnmanagedChunks;
	
public:
	virtual ~AliGPUReconstruction();
	AliGPUReconstruction(const AliGPUReconstruction&) = delete;
	AliGPUReconstruction& operator=(const AliGPUReconstruction&) = delete;
	
	//General definitions
	constexpr static unsigned int NSLICES = GPUCA_NSLICES;

	//Definition of the Geometry: Are we AliRoot or O2
	enum GeometryType : unsigned int {RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2};
	static constexpr const char* const GEOMETRY_TYPE_NAMES[] = {"INVALID", "ALIROOT", "O2"};
#ifdef GPUCA_TPC_GEOMETRY_O2
	static constexpr GeometryType geometryType = O2;
#else
	static constexpr GeometryType geometryType = ALIROOT;
#endif
	
	enum DeviceType : unsigned int {INVALID_DEVICE = 0, CPU = 1, CUDA = 2, HIP = 3, OCL = 4};
	static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL"};
	static DeviceType GetDeviceType(const char* type);

	//Functionality to create an instance of AliGPUReconstruction for the desired device
	static AliGPUReconstruction* CreateInstance(const AliGPUCASettingsProcessing& cfg);
	static AliGPUReconstruction* CreateInstance(DeviceType type = CPU, bool forceType = true);
	static AliGPUReconstruction* CreateInstance(int type, bool forceType) {return CreateInstance((DeviceType) type, forceType);}
	static AliGPUReconstruction* CreateInstance(const char* type, bool forceType);
	
	int Init();
	int Finalize();
	
	//Structures for input and output data
	struct InOutPointers
	{
		InOutPointers() : mcLabelsTPC(nullptr), nMCLabelsTPC(0), mcInfosTPC(nullptr), nMCInfosTPC(0),
			mergedTracks(nullptr), nMergedTracks(0), mergedTrackHits(nullptr), nMergedTrackHits(0),
			trdTracks(nullptr), nTRDTracks(0), trdTracklets(nullptr), nTRDTracklets(0), trdTrackletsMC(nullptr),
			nTRDTrackletsMC(0)
		{}
		InOutPointers(const InOutPointers&) = default;
		
		const AliGPUTPCClusterData* clusterData[NSLICES];
		unsigned int nClusterData[NSLICES];
		const AliHLTTPCRawCluster* rawClusters[NSLICES];
		unsigned int nRawClusters[NSLICES];
		const o2::TPC::ClusterNativeAccessFullTPC* clustersNative;
		const AliGPUTPCSliceOutTrack* sliceOutTracks[NSLICES];
		unsigned int nSliceOutTracks[NSLICES];
		const AliGPUTPCSliceOutCluster* sliceOutClusters[NSLICES];
		unsigned int nSliceOutClusters[NSLICES];
		const AliHLTTPCClusterMCLabel* mcLabelsTPC;
		unsigned int nMCLabelsTPC;
		const AliGPUTPCMCInfo* mcInfosTPC;
		unsigned int nMCInfosTPC;
		const AliGPUTPCGMMergedTrack* mergedTracks;
		unsigned int nMergedTracks;
		const AliGPUTPCGMMergedTrackHit* mergedTrackHits;
		unsigned int nMergedTrackHits;
		const GPUTRDTrack* trdTracks;
		unsigned int nTRDTracks;
		const AliGPUTRDTrackletWord* trdTracklets;
		unsigned int nTRDTracklets;
		const AliGPUTRDTrackletLabels* trdTrackletsMC;
		unsigned int nTRDTrackletsMC;
	} mIOPtrs;

	struct InOutMemory
	{
		InOutMemory();
		~InOutMemory();
		InOutMemory(AliGPUReconstruction::InOutMemory&&);
		InOutMemory& operator=(InOutMemory&&);
		
		std::unique_ptr<AliGPUTPCClusterData[]> clusterData[NSLICES];
		std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
		std::unique_ptr<o2::TPC::ClusterNative[]> clustersNative[NSLICES * GPUCA_ROW_COUNT];
		std::unique_ptr<AliGPUTPCSliceOutTrack[]> sliceOutTracks[NSLICES];
		std::unique_ptr<AliGPUTPCSliceOutCluster[]> sliceOutClusters[NSLICES];
		std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
		std::unique_ptr<AliGPUTPCMCInfo[]> mcInfosTPC;
		std::unique_ptr<AliGPUTPCGMMergedTrack[]> mergedTracks;
		std::unique_ptr<AliGPUTPCGMMergedTrackHit[]> mergedTrackHits;
		std::unique_ptr<GPUTRDTrack[]> trdTracks;
		std::unique_ptr<AliGPUTRDTrackletWord[]> trdTracklets;
		std::unique_ptr<AliGPUTRDTrackletLabels[]> trdTrackletsMC;
	} mIOMem;
	
	//Functionality to dump and read input / output data
	enum InOutPointerType : unsigned int {CLUSTER_DATA = 0, SLICE_OUT_TRACK = 1, SLICE_OUT_CLUSTER = 2, MC_LABEL_TPC = 3, MC_INFO_TPC = 4, MERGED_TRACK = 5, MERGED_TRACK_HIT = 6, TRD_TRACK = 7, TRD_TRACKLET = 8, RAW_CLUSTERS = 9, CLUSTERS_NATIVE = 10, TRD_TRACKLET_MC = 11};
	static constexpr const char* const IOTYPENAMES[] = {"TPC Clusters", "TPC Slice Tracks", "TPC Slice Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets", "Raw Clusters", "ClusterNative", "TRD Tracklet MC Labels"};

	void ClearIOPointers();
	void AllocateIOMemory();
	void DumpData(const char* filename);
	int ReadData(const char* filename);
	void DumpSettings(const char* dir = "");
	void ReadSettings(const char* dir = "");
	
	//Helpers for memory allocation
	AliGPUMemoryResource& Res(short num) {return mMemoryResources[num];}
	template <class T> short RegisterMemoryAllocation(T* proc, void* (T::* setPtr)(void*), int type, const char* name = "")
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
	
	//Converter functions
	void ConvertNativeToClusterData();
	
	//Getters for external usage of tracker classes
	AliGPUTRDTracker* GetTRDTracker() {return &workers()->trdTracker;}
	o2::ITS::TrackerTraits* GetITSTrackerTraits() {return mITSTrackerTraits.get();}
	o2::ITS::VertexerTraits* GetITSVertexerTraits() {return mITSVertexerTraits.get();}
	AliGPUTPCTracker* GetTPCSliceTrackers() {return workers()->tpcTrackers;}
	const AliGPUTPCTracker* GetTPCSliceTrackers() const {return workers()->tpcTrackers;}
	const AliGPUTPCGMMerger& GetTPCMerger() const {return workers()->tpcMerger;}
	AliGPUTPCGMMerger& GetTPCMerger() {return workers()->tpcMerger;}
	AliGPUCADisplay* GetEventDisplay() {return mEventDisplay.get();}
	const AliGPUCAQA* GetQA() const {return mQA.get();}
	AliGPUCAQA* GetQA() {return mQA.get();}
	
	//Processing functions
	virtual int RunStandalone() = 0;
	virtual int RunTPCTrackingSlices() = 0;
	virtual int RunTPCTrackingMerger() = 0;
	virtual int RunTRDTracking() = 0;
	virtual int DoTRDGPUTracking() { printf("Does only work on GPU\n"); exit(1); }
	
	//Getters / setters for parameters
	DeviceType GetDeviceType() const {return (DeviceType) mProcessingSettings.deviceType;}
	bool IsGPU() const {return GetDeviceType() != INVALID_DEVICE && GetDeviceType() != CPU;}
	const AliGPUCAParam& GetParam() const {return mHostConstantMem->param;}
	const TPCFastTransform* GetTPCTransform() const {return mTPCFastTransform.get();}
	const AliGPUTRDGeometry* GetTRDGeometry() const {return (AliGPUTRDGeometry*) mTRDGeometry.get();}
	const ClusterNativeAccessExt* GetClusterNativeAccessExt() const {return mClusterNativeAccess.get();}
	const AliGPUCASettingsEvent& GetEventSettings() const {return mEventSettings;}
	const AliGPUCASettingsProcessing& GetProcessingSettings() {return mProcessingSettings;}
	const AliGPUCASettingsDeviceProcessing& GetDeviceProcessingSettings() const {return mDeviceProcessingSettings;}
	bool IsInitialized() const {return mInitialized;}
	void SetSettings(float solenoidBz);
	void SetSettings(const AliGPUCASettingsEvent* settings, const AliGPUCASettingsRec* rec = nullptr, const AliGPUCASettingsDeviceProcessing* proc = nullptr);
	void SetTPCFastTransform(std::unique_ptr<TPCFastTransform> tpcFastTransform);
	void SetTRDGeometry(const o2::trd::TRDGeometryFlat& geo);
	void LoadClusterErrors();
	void SetResetTimers(bool reset) {mDeviceProcessingSettings.resetTimers = reset;}
	void SetOutputControl(const AliGPUCAOutputControl& v) {mOutputControl = v;}
	void SetOutputControl(void* ptr, size_t size);
	AliGPUCAOutputControl& OutputControl() {return mOutputControl;}
	const AliGPUTPCSliceOutput** SliceOutput() const {return (const AliGPUTPCSliceOutput**) &mSliceOutput;}
	virtual int GetMaxThreads();
	
	const void* mConfigDisplay = nullptr;										//Abstract pointer to Standalone Display Configuration Structure
	const void* mConfigQA = nullptr;											//Abstract pointer to Standalone QA Configuration Structure
	
	//Registration of GPU Processors
	template <class T> void RegisterGPUProcessor(T* proc, bool deviceSlave)
	{
		mProcessors.emplace_back(proc, static_cast<void (AliGPUProcessor::*)()>(&T::RegisterMemoryAllocation), static_cast<void (AliGPUProcessor::*)()>(&T::InitializeProcessor), static_cast<void (AliGPUProcessor::*)()>(&T::SetMaxData));
		AliGPUProcessor::ProcessorType processorType = deviceSlave ? AliGPUProcessor::PROCESSOR_TYPE_SLAVE : AliGPUProcessor::PROCESSOR_TYPE_CPU;
		proc->InitGPUProcessor(this, processorType);
	}
	template <class T> void SetupGPUProcessor(T* proc, bool allocate)
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
	void RegisterGPUDeviceProcessor(AliGPUProcessor* proc, AliGPUProcessor* slaveProcessor);
	
protected:
	AliGPUReconstruction(const AliGPUCASettingsProcessing& cfg);				//Constructor
	virtual int InitDevice();
	virtual int ExitDevice();
	int InitializeProcessors();
	
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
	
	//Pointers to tracker classes
	AliGPUCAWorkers* workers() {return mHostConstantMem.get();}
	const AliGPUCAWorkers* workers() const {return mHostConstantMem.get();}
	AliGPUCAParam& param() {return mHostConstantMem->param;}
	std::unique_ptr<AliGPUCAConstantMem> mHostConstantMem;
	std::unique_ptr<o2::ITS::TrackerTraits> mITSTrackerTraits;
	std::unique_ptr<o2::ITS::VertexerTraits> mITSVertexerTraits;
	AliGPUTPCSliceOutput* mSliceOutput[NSLICES];
	
	AliGPUCASettingsEvent mEventSettings;										//Event Parameters
	AliGPUCASettingsProcessing mProcessingSettings;								//Processing Parameters (at constructor level)
	AliGPUCASettingsDeviceProcessing mDeviceProcessingSettings;					//Processing Parameters (at init level)
	AliGPUCAOutputControl mOutputControl;										//Controls the output of the individual components
	
	std::unique_ptr<AliGPUCADisplay> mEventDisplay;
	bool mDisplayRunning = false;
	std::unique_ptr<AliGPUCAQA> mQA;
	bool mQAInitialized = false;
	
	std::unique_ptr<ClusterNativeAccessExt> mClusterNativeAccess;				//Internal memory for clusterNativeAccess
	std::unique_ptr<TPCFastTransform> mTPCFastTransform;						//Global TPC fast transformation object
	std::unique_ptr<o2::trd::TRDGeometryFlat> mTRDGeometry;						//TRD Geometry
	
	bool mInitialized = false;
	std::ofstream mDebugFile;
	
	int mStatNEvents = 0;
	
	void* mHostMemoryBase = nullptr;
	void* mHostMemoryPermanent = nullptr;
	void* mHostMemoryPool = nullptr;
	size_t mHostMemorySize = 0;
	void* mDeviceMemoryBase = nullptr;
	void* mDeviceMemoryPermanent = nullptr;
	void* mDeviceMemoryPool = nullptr;
	size_t mDeviceMemorySize = 0;
	
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
		AliGPUReconstruction* GetPtr(const AliGPUCASettingsProcessing& cfg);
		
		const char* mLibName;
		const char* mFuncName;
		void* mGPULib;
		void* mGPUEntry;
	};
	static std::shared_ptr<LibraryLoader> sLibCUDA, sLibHIP, sLibOCL;
	
private:
	static AliGPUReconstruction* AliGPUReconstruction_Create_CPU(const AliGPUCASettingsProcessing& cfg);
};

#endif
