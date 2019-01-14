#define __OPENCL__
#define GPUCA_GPUTYPE_RADEON

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#include "AliGPUTPCTrackParam.cxx"
#include "AliGPUTPCTrack.cxx"

#include "AliGPUTPCHitArea.cxx"
#include "AliGPUTPCGrid.cxx"
#include "AliGPUTPCRow.cxx"
#include "AliGPUCAParam.cxx"
#include "AliGPUTPCTracker.cxx"

#include "AliGPUTPCTrackletSelector.cxx"
#include "AliGPUTPCNeighboursFinder.cxx"
#include "AliGPUTPCNeighboursCleaner.cxx"
#include "AliGPUTPCStartHitsFinder.cxx"
#include "AliGPUTPCStartHitsSorter.cxx"
#include "AliGPUTPCTrackletConstructor.cxx"

__kernel void PreInitRowBlocks(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;

	//Initialize GPU RowBlocks and HitWeights
	const int nSliceDataHits = pTracker.Data().NumberOfHitsPlusAlign();
	__global int4* SliceDataHitWeights4 = (__global int4*) pTracker.Data().HitWeights();

	const int stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	for (int i = get_global_id(0);i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		SliceDataHitWeights4[i] = i0;
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursFinder(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCNeighboursFinder::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCNeighboursFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursCleaner(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursCleaner::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCNeighboursCleaner::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCNeighboursCleaner::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsFinder(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCStartHitsFinder::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCStartHitsFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsSorter(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsSorter::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCStartHitsSorter::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCStartHitsSorter::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcessMulti_AliGPUTPCTrackletSelector(__global char* gpu_mem, GPUconstant() void* pTrackerTmp, int firstSlice, int nSliceCount)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = (( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp)[firstSlice + iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCTrackletSelector::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCTrackletSelector::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCTrackletSelector::Thread( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCTrackletConstructorGPU(__global char* gpu_mem, GPUconstant() void* pTrackerTmp)
{
	//GPU Wrapper for AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker = ( GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) * ) pTrackerTmp ;
	if (gpu_mem != pTracker[0].GPUParametersConst()->fGPUMem) return;
	GPUshared() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) sMem;
	AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU(pTracker, sMem);
}
