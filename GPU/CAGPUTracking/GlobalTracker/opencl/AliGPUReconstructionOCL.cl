#define __OPENCL__
#define GPUCA_GPUTYPE_RADEON
#ifdef __OPENCLCPP__
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#include <opencl_def>
#include <opencl_common>
#include <opencl_math>
#include <opencl_atomic>
#include <opencl_memory>
#include <opencl_work_item>
#include <opencl_synchronization>
#include <opencl_printf>
#include <opencl_integer>
#define M_PI 3.1415926535f
using namespace cl;
#endif

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#define DEVICE_KERNELS_PREA GPUglobal() char* gpu_mem, GPUconstant() MEM_CONSTANT(AliGPUCAConstantMem)* pConstant
#define DEVICE_KERNELS_PRE DEVICE_KERNELS_PREA,
#include "AliGPUDeviceKernels.h"

#include "AliGPUCADataTypes.h"

GPUg() void KernelMemClean16(DEVICE_KERNELS_PRE unsigned long ptr, unsigned long size)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[0];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUMemClean16::AliGPUTPCSharedMemory smem;
	AliGPUMemClean16::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker, (GPUglobalref() void*) (void*) ptr, size);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursFinder(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCNeighboursFinder::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursCleaner(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCNeighboursCleaner::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCNeighboursCleaner::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsFinder(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsFinder::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCStartHitsFinder::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsSorter(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCStartHitsSorter::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCStartHitsSorter::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
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
	AliGPUTPCTrackletSelector::template Thread<0>( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, pTracker);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCTrackletConstructor0(DEVICE_KERNELS_PRE int iSlice)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[iSlice];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCTrackletConstructor::template Thread<0>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCTrackletConstructor1(DEVICE_KERNELS_PREA)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &pTracker = pConstant->tpcTrackers[0];
	if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return;
	GPUshared() typename AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) smem;
	AliGPUTPCTrackletConstructor::template Thread<1>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, pTracker);
}
