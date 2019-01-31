#include "AliGPUGeneralKernels.h"

template <> GPUd() void AliGPUMemClean16::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUglobalref() void* ptr, unsigned long size)
{
	const unsigned long stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	int4* ptra = (int4*) ptr;
	unsigned long len = (size + sizeof(int4) - 1) / sizeof(int4);
	for (unsigned long i = get_global_id(0);i < len;i += stride)
	{
		ptra[i] = i0;
	}
}
