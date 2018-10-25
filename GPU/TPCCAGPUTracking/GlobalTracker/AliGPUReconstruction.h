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
#include "AliHLTTRDDef.h"

class AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstruction();
	
	static AliGPUReconstruction* CreateInstance() {return new AliGPUReconstruction;}
	
	constexpr static size_t MIN_ALIGNMENT = 64;
	constexpr static unsigned int NSLICES = 36;
	struct InOutPointers
	{
		InOutPointers() : mcLabelsTPC(nullptr), nMCLabelsTPC(0), mcInfosTPC(nullptr), nMCInfosTPC(0),
			mergedTracks(nullptr), nMergedTracks(0), mergedTrackHits(nullptr), nMergedTrackHits(0),
			trdTracks(nullptr), nTRDTracks(0), trdTracklets(nullptr), nTRDTracklets(0)
		{}
		AliHLTTPCCAClusterData::Data* clusterData[NSLICES];
		unsigned int nClusterData[NSLICES];
		AliHLTTPCCASliceOutTrack* sliceOutTracks[NSLICES];
		unsigned int nSliceOutTracks[NSLICES];
		AliHLTTPCCASliceOutCluster* sliceOutClusters[NSLICES];
		unsigned int nSliceOutClusters[NSLICES];
		AliHLTTPCClusterMCLabel* mcLabelsTPC;
		unsigned int nMCLabelsTPC;
		AliHLTTPCCAMCInfo* mcInfosTPC;
		unsigned int nMCInfosTPC;
		AliHLTTPCGMMergedTrack* mergedTracks;
		unsigned int nMergedTracks;
		AliHLTTPCGMMergedTrackHit* mergedTrackHits;
		unsigned int nMergedTrackHits;
		HLTTRDTrack* trdTracks;
		unsigned int nTRDTracks;
		AliHLTTRDTrackletWord* trdTracklets;
		unsigned int nTRDTracklets;
	} mIOPtrs;
	enum InOutPointerType : unsigned int {CLUSTER_DATA = 0, SLICE_OUT_TRACK = 1, SLICE_OUT_CLUSTER = 2, MC_LABEL_TPC = 3, MC_INFO_TPC = 4, MERGED_TRACK = 5, MERGED_TRACK_HIT = 6, TRD_TRACK = 7, TRD_TRACKLET = 8};
	static constexpr const char* IOTYPENAMES[] = {"TPC Clusters", "TPC Slice Tracks", "TPC Slice Track Clusters", "TPC Cluster MC Labels", "TPC Track MC Informations", "TPC Tracks", "TPC Track Clusters", "TRD Tracks", "TRD Tracklets"};
	
	struct InOutMemory
	{
		InOutMemory();
		~InOutMemory();
		std::unique_ptr<AliHLTTPCCAClusterData::Data[]> clusterData[NSLICES];
		std::unique_ptr<AliHLTTPCCASliceOutTrack[]> sliceOutTracks[NSLICES];
		std::unique_ptr<AliHLTTPCCASliceOutCluster[]> sliceOutClusters[NSLICES];
		std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
		std::unique_ptr<AliHLTTPCCAMCInfo[]> mcInfosTPC;
		std::unique_ptr<AliHLTTPCGMMergedTrack[]> mergedTracks;
		std::unique_ptr<AliHLTTPCGMMergedTrackHit[]> mergedTrackHits;
		std::unique_ptr<HLTTRDTrack[]> trdTracks;
		std::unique_ptr<AliHLTTRDTrackletWord[]> trdTracklets;
	} mIOMem;
	
	void ClearIOPointers();
	void AllocateIOMemory();
	
	void DumpData(const char* filename);
	int ReadData(const char* filename);
		
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
	
protected:
	AliGPUReconstruction() : mIOPtrs(), mIOMem() {}
	
	template <class T> void DumpData(FILE* fp, T** entries, unsigned int* num, InOutPointerType type);
	template <class T> void ReadData(FILE* fp, T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type);
	template <class T> void AllocateIOMemoryHelper(unsigned int n, T* &ptr, std::unique_ptr<T[]> &u);
};

#endif
