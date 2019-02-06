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

#include "AliGPUDeviceKernels.h"
#include "AliGPUCADataTypes.h"

#define OCL_DEVICE_KERNELS_PRE GPUglobal() char* gpu_mem, GPUconstant() MEM_CONSTANT(AliGPUCAConstantMem)* pConstant
#define OCL_CALL_KERNEL(T, I, num) \
	GPUshared() typename T::MEM_LOCAL(AliGPUTPCSharedMemory) smem; \
	T::template Thread<I>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Worker(*pConstant)[num]);
	
#define OCL_CALL_KERNEL_MULTI(T, I) \
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0); \
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount; \
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset; \
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount; \
	GPUshared() typename T::MEM_LOCAL(AliGPUTPCSharedMemory) smem; \
	T::template Thread<I>( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Worker(*pConstant)[firstSlice + iSlice]);

#define OCL_CALL_KERNEL_ARGS(T, I, ...) \
	GPUshared() typename T::AliGPUTPCSharedMemory smem; \
	T::template Thread<I>( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Worker(*pConstant)[0], __VA_ARGS__);

//if (gpu_mem != pTracker.GPUParametersConst()->fGPUMem) return; //TODO!

GPUg() void KernelMemClean16(OCL_DEVICE_KERNELS_PRE, unsigned long ptr, unsigned long size)
{
	OCL_CALL_KERNEL_ARGS(AliGPUMemClean16, 0, (GPUglobalref() void*) (void*) ptr, size);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursFinder(OCL_DEVICE_KERNELS_PRE, int iSlice)
{
	OCL_CALL_KERNEL(AliGPUTPCNeighboursFinder, 0, iSlice);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCNeighboursCleaner(OCL_DEVICE_KERNELS_PRE, int iSlice)
{
	OCL_CALL_KERNEL(AliGPUTPCNeighboursCleaner, 0, iSlice);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsFinder(OCL_DEVICE_KERNELS_PRE, int iSlice)
{
	OCL_CALL_KERNEL(AliGPUTPCStartHitsFinder, 0, iSlice);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCStartHitsSorter(OCL_DEVICE_KERNELS_PRE, int iSlice)
{
	OCL_CALL_KERNEL(AliGPUTPCStartHitsSorter, 0, iSlice);
}

GPUg() void AliGPUTPCProcessMulti_AliGPUTPCTrackletSelector(OCL_DEVICE_KERNELS_PRE, int firstSlice, int nSliceCount)
{
	OCL_CALL_KERNEL_MULTI(AliGPUTPCTrackletSelector, 0);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCTrackletConstructor0(OCL_DEVICE_KERNELS_PRE, int iSlice)
{
	OCL_CALL_KERNEL(AliGPUTPCTrackletConstructor, 0, iSlice);
}

GPUg() void AliGPUTPCProcess_AliGPUTPCTrackletConstructor1(OCL_DEVICE_KERNELS_PRE)
{
	OCL_CALL_KERNEL(AliGPUTPCTrackletConstructor, 1, 0);
}
