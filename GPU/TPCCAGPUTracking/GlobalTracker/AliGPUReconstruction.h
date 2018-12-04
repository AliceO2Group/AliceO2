#ifndef ALIGPURECONSTRUCTION_H
#define ALIGPURECONSTRUCTION_H

#include <cstddef>
#include <stdio.h>
#include <memory>

#include "AliHLTTPCCAClusterData.h"
class AliHLTTPCCASliceOutput;
class AliHLTTPCCASliceOutTrack;
class AliHLTTPCCASliceOutCluster;
class AliHLTTPCGMMergedTrack;
class AliHLTTPCGMMergedTrackHit;
class AliHLTTRDTrackletWord;
class AliHLTTPCClusterMCLabel;
class AliHLTTPCCAMCInfo;
class AliHLTTRDTracker;
class AliHLTTPCCAGPUTracker;
#include "AliHLTTRDDef.h"
#include "AliGPUCAParam.h"
struct hltca_event_dump_settings;
struct AliHLTTPCRawCluster;

struct ClusterNativeAccessExt;
namespace o2 { namespace ITS { class TrackerTraits; }}
namespace o2 { namespace trd { class TRDGeometryFlat; }}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative; }}
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstruction();
	
	//General definitions
	constexpr static size_t MIN_ALIGNMENT = 64;
	constexpr static unsigned int NSLICES = 36;

	//Definition of the Geometry: Are we AliRoot or O2
	enum GeometryType : unsigned int {RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2};
	static constexpr const char* const GEOMETRY_TYPE_NAMES[] = {"INVALID", "ALIROOT", "O2"};
	#ifdef HLTCA_TPC_GEOMETRY_O2
		static constexpr GeometryType geometryType = O2;
	#else
		static constexpr GeometryType geometryType = ALIROOT;
	#endif
	
	//Functionality to create an instance of AliGPUReconstruction for the desired device
	enum DeviceType : unsigned int {RESERVED_DEVICE = 0, CPU = 1, CUDA = 2, HIP = 3, OCL = 4};
	static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL"};
	static AliGPUReconstruction* CreateInstance(DeviceType type = CPU, bool forceType = true);
	static AliGPUReconstruction* CreateInstance(int type, bool forceType) {return CreateInstance((DeviceType) type, forceType);}
	static AliGPUReconstruction* CreateInstance(const char* type, bool forceType);
	
	virtual int Init();
	
	//Structures for input and output data
	struct InOutPointers
	{
		InOutPointers() : mcLabelsTPC(nullptr), nMCLabelsTPC(0), mcInfosTPC(nullptr), nMCInfosTPC(0),
			mergedTracks(nullptr), nMergedTracks(0), mergedTrackHits(nullptr), nMergedTrackHits(0),
			trdTracks(nullptr), nTRDTracks(0), trdTracklets(nullptr), nTRDTracklets(0)
		{}
		
		const AliHLTTPCCAClusterData::Data* clusterData[NSLICES];
		unsigned int nClusterData[NSLICES];
		const AliHLTTPCRawCluster* rawClusters[NSLICES];
		unsigned int nRawClusters[NSLICES];
		const o2::TPC::ClusterNativeAccessFullTPC* clustersNative;
		const AliHLTTPCCASliceOutTrack* sliceOutTracks[NSLICES];
		unsigned int nSliceOutTracks[NSLICES];
		const AliHLTTPCCASliceOutCluster* sliceOutClusters[NSLICES];
		unsigned int nSliceOutClusters[NSLICES];
		const AliHLTTPCClusterMCLabel* mcLabelsTPC;
		unsigned int nMCLabelsTPC;
		const AliHLTTPCCAMCInfo* mcInfosTPC;
		unsigned int nMCInfosTPC;
		const AliHLTTPCGMMergedTrack* mergedTracks;
		unsigned int nMergedTracks;
		const AliHLTTPCGMMergedTrackHit* mergedTrackHits;
		unsigned int nMergedTrackHits;
		const HLTTRDTrack* trdTracks;
		unsigned int nTRDTracks;
		const AliHLTTRDTrackletWord* trdTracklets;
		unsigned int nTRDTracklets;
	} mIOPtrs;

	struct InOutMemory
	{
		InOutMemory();
		~InOutMemory();
		std::unique_ptr<AliHLTTPCCAClusterData::Data[]> clusterData[NSLICES];
		std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
		std::unique_ptr<o2::TPC::ClusterNative[]> clustersNative[NSLICES * HLTCA_ROW_COUNT];
		std::unique_ptr<AliHLTTPCCASliceOutTrack[]> sliceOutTracks[NSLICES];
		std::unique_ptr<AliHLTTPCCASliceOutCluster[]> sliceOutClusters[NSLICES];
		std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
		std::unique_ptr<AliHLTTPCCAMCInfo[]> mcInfosTPC;
		std::unique_ptr<AliHLTTPCGMMergedTrack[]> mergedTracks;
		std::unique_ptr<AliHLTTPCGMMergedTrackHit[]> mergedTrackHits;
		std::unique_ptr<HLTTRDTrack[]> trdTracks;
		std::unique_ptr<AliHLTTRDTrackletWord[]> trdTracklets;
	} mIOMem;
	
	//Functionality to dump and read input / output data
	enum InOutPointerType : unsigned int {CLUSTER_DATA = 0, SLICE_OUT_TRACK = 1, SLICE_OUT_CLUSTER = 2, MC_LABEL_TPC = 3, MC_INFO_TPC = 4, MERGED_TRACK = 5, MERGED_TRACK_HIT = 6, TRD_TRACK = 7, TRD_TRACKLET = 8, RAW_CLUSTERS = 9, CLUSTERS_NATIVE = 10};
	static constexpr const char* const IOTYPENAMES[] = {"TPC Clusters", "TPC Slice Tracks", "TPC Slice Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets", "Raw Clusters", "ClusterNative"};

	void ClearIOPointers();
	void AllocateIOMemory();
	void DumpData(const char* filename);
	int ReadData(const char* filename);
	void DumpSettings(const char* dir = "");
	void ReadSettings(const char* dir = "");
	
	//Helpers for memory allocation
	static inline size_t getAlignment(size_t addr, size_t alignment = MIN_ALIGNMENT)
	{
		if (alignment <= 1) return 0;
		size_t mod = addr & (alignment - 1);
		if (mod == 0) return 0;
		return (alignment - mod);
	}
	static inline size_t getAlignment(void* addr, size_t alignment = MIN_ALIGNMENT)
	{
		return(getAlignment(reinterpret_cast<size_t>(addr), alignment));
	}
	template <class S> static inline S* getPointerWithAlignment(size_t& basePtr, size_t nEntries = 1)
	{
		basePtr += getAlignment(basePtr, alignof(S));
		S* retVal = (S*) (basePtr);
		basePtr += nEntries * sizeof(S);
		return retVal;
	}
	template <class S> static inline S* getPointerWithAlignment(void*& basePtr, size_t nEntries = 1)
	{
		return getPointerWithAlignment<S>(reinterpret_cast<size_t&>(basePtr), nEntries);
	}
	template <class T, class S> static inline void computePointerWithAlignment(T*& basePtr, S*& objPtr, size_t nEntries = 1)
	{
		objPtr = getPointerWithAlignment<S>(reinterpret_cast<size_t&>(basePtr), nEntries);
	}
	
	//Converter functions
	void ConvertNativeToClusterData();
	
	//Getters for external usage of tracker classes
	AliHLTTRDTracker* GetTRDTracker() {return mTRDTracker.get();}
	AliHLTTPCCAGPUTracker* GetTPCTracker() {return mTPCTracker.get();}
	o2::ITS::TrackerTraits* GetITSTrackerTraits() {return mITSTrackerTraits.get();}
	
	//Processing functions
	int RunTRDTracking();
	
	//Getters / setters for parameters
	DeviceType GetDeviceType() const {return mDeviceType;}
	void SetParam(const AliGPUCAParam& param) {mParam = param;}
	const AliGPUCAParam& GetParam() const {return mParam;}
	const TPCFastTransform* GetTPCTransform() const {return mTPCFastTransform.get();}
	const ClusterNativeAccessExt* GetClusterNativeAccessExt() const {return mClusterNativeAccess.get();}
	AliGPUCAParam& GetParam() {return mParam;}
	hltca_event_dump_settings& GetEventSettings() {return *mEventDumpSettings;}
	void SetSettingsStandalone(float solenoidBz);
	void SetSettingsStandalone(const hltca_event_dump_settings& settings);
	void SetTPCFastTransform(std::unique_ptr<TPCFastTransform> tpcFastTransform);
	void SetTRDGeometry(const o2::trd::TRDGeometryFlat& geo);
	
protected:
	AliGPUReconstruction(DeviceType type);								//Constructor
	
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
	std::unique_ptr<AliHLTTRDTracker> mTRDTracker;
	std::unique_ptr<AliHLTTPCCAGPUTracker> mTPCTracker;
	std::unique_ptr<o2::ITS::TrackerTraits> mITSTrackerTraits;
	
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
		AliGPUReconstruction* GetPtr();
		
		const char* mLibName;
		const char* mFuncName;
		void* mGPULib;
		void* mGPUEntry;
	};
	static std::shared_ptr<LibraryLoader> sLibCUDA, sLibHIP, sLibOCL;
	std::shared_ptr<LibraryLoader> mMyLib = nullptr;
	DeviceType mDeviceType;
	
	AliGPUCAParam mParam;														//Reconstruction parameters
	std::unique_ptr<TPCFastTransform> mTPCFastTransform;						//Global TPC fast transformation object
	std::unique_ptr<hltca_event_dump_settings> mEventDumpSettings;				//Standalone event dump settings
	std::unique_ptr<ClusterNativeAccessExt> mClusterNativeAccess;	//Internal memory for clusterNativeAccess
	std::unique_ptr<o2::trd::TRDGeometryFlat> mTRDGeometry;						//TRD Geometry
};

#endif
