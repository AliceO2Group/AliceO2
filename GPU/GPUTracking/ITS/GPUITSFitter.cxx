#include "GPUITSFitter.h"

#include "ITStracking/Road.h"
#include "ITStracking/Cluster.h"
#include "GPUITSTrack.h"
#include "GPUReconstruction.h"

#ifndef GPUCA_GPUCODE
void GPUITSFitter::InitializeProcessor()
{
	
}

void* GPUITSFitter::SetPointersInput(void* mem)
{
	computePointerWithAlignment(mem, mRoads, mNumberOfRoads);
	for (int i = 0;i < 7;i++)
	{
		computePointerWithAlignment(mem, mTF[i], mNTF[i]);
	}
	return mem;
}

void* GPUITSFitter::SetPointersTracks(void* mem)
{
	computePointerWithAlignment(mem, mTracks, mNMaxTracks);
	return mem;
}

void* GPUITSFitter::SetPointersMemory(void* mem)
{
	computePointerWithAlignment(mem, mMemory, 1);
	return mem;
}

void GPUITSFitter::RegisterMemoryAllocation()
{
	AllocateAndInitializeLate();
	mMemoryResInput = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersInput, GPUMemoryResource::MEMORY_INPUT, "ITSInput");
	mMemoryResTracks = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersTracks, GPUMemoryResource::MEMORY_OUTPUT, "ITSTracks");
	mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "ITSMemory");
}

void GPUITSFitter::SetMaxData()
{
	mNMaxTracks = mNumberOfRoads;
}
#endif

void GPUITSFitter::clearMemory()
{
	new(mMemory) Memory;
}
