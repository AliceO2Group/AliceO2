#define __OPENCL__
#define GPUCA_GPUTYPE_RADEON

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#define DEVICE_KERNELS_PREA GPUglobalref() char* gpu_mem, GPUconstant() MEM_CONSTANT(AliGPUCAConstantMem)* pConstant
#define DEVICE_KERNELS_PRE DEVICE_KERNELS_PREA,
#include "AliGPUDeviceKernels.h"

#include "AliGPUCADataTypes.h"

GPUg() void KernelMemClean16(DEVICE_KERNELS_PRE unsigned long ptr, unsigned long size)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[0];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUMemClean16::AliGPUTPCSharedMemory smem;
	AliGPUMemClean16::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), 0/*SYNC*/, smem, pTracker, (GPUglobalref() void*) ptr, size );
}

GPUg() void PreInitRowBlocks(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;

	//Initialize GPU RowBlocks and HitWeights
	const int nSliceDataHits = pTracker.Data().NumberOfHitsPlusAlign();
	GPUglobalref() int4* SliceDataHitWeights4 = (GPUglobalref() int4*) pTracker.Data().HitWeights();

	const int stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	for (int i = get_global_id(0);i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		SliceDataHitWeights4[i] = i0;
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursFinder(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCNeighboursFinder::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCNeighboursFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursCleaner(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursCleaner::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCNeighboursCleaner::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCNeighboursCleaner::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsFinder(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCStartHitsFinder::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCStartHitsFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsSorter(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsSorter::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCStartHitsSorter::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCStartHitsSorter::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCProcessMulti_AliGPUTPCTrackletSelector(DEVICE_KERNELS_PRE int firstSlice, int nSliceCount)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[firstSlice + iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCTrackletSelector::MEM_LOCAL(AliGPUTPCSharedMemory) smem;

	for( int iSync=0; iSync<=AliGPUTPCTrackletSelector::NThreadSyncPoints(); iSync++){
		GPUsync();
		AliGPUTPCTrackletSelector::Thread( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), iSync, smem, pTracker  );
	}
}

GPUg() void AliGPUTPCTrackletConstructorGPU(DEVICE_KERNELS_PREA)
{
	//GPU Wrapper for AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker = pConstant->tpcTrackers;
	if (gpu_mem != pTracker[0].GPUParametersConst()->fGPUMem) return;
	GPUshared() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) sMem;
	AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU(pTracker, sMem);
}
