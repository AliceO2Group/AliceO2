#ifndef GPUITSFITTER_H
#define GPUITSFITTER_H

#include "AliGPUProcessor.h"
#include "GPUITSTrack.h"

namespace o2 { namespace ITS { class Road; struct TrackingFrameInfo; class GPUITSTrack; struct Cluster; class Cell; }}

class GPUITSFitter : public AliGPUProcessor
{
public:
#ifndef GPUCA_GPUCODE
	void InitializeProcessor();
	void RegisterMemoryAllocation();
	void SetMaxData();
	
	void* SetPointersInput(void* mem);
	void* SetPointersTracks(void* mem);
	void* SetPointersMemory(void* mem);
#endif

	GPUd() o2::ITS::Road* roads() {return mRoads;}
	GPUd() void SetNumberOfRoads(int v) {mNumberOfRoads = v;}
	GPUd() int NumberOfRoads() {return mNumberOfRoads;}
	GPUd() o2::ITS::GPUITSTrack* tracks() {return mTracks;}
	GPUd() int& NumberOfTracks() {return mMemory->mNumberOfTracks;}
	GPUd() void SetNumberTF(int i, int v) {mNTF[i] = v;}
	GPUd() o2::ITS::TrackingFrameInfo** trackingFrame() {return mTF;}
	GPUd() const o2::ITS::Cluster** clusters() {return mClusterPtrs;}
	GPUd() const o2::ITS::Cell** cells() {return mCellPtrs;}
	
	void clearMemory();
	
	struct Memory
	{
		int mNumberOfTracks = 0;
	};
	
protected:
	int mNumberOfRoads = 0;
	int mNMaxTracks = 0;
	int mNTF[7] = {};
	Memory* mMemory = nullptr;
	o2::ITS::Road* mRoads = nullptr;
	o2::ITS::TrackingFrameInfo* mTF[7] = {};
	o2::ITS::GPUITSTrack* mTracks = nullptr;
	
	const o2::ITS::Cluster* mClusterPtrs[7];
	const o2::ITS::Cell* mCellPtrs[5];
	
	short mMemoryResInput = -1;
	short mMemoryResTracks = -1;
	short mMemoryResMemory = -1;
};

#endif
