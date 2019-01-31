#ifndef ALIGPUGENERALKERNELS_H
#define ALIGPUGENERALKERNELS_H

#include "AliGPUTPCDef.h"

MEM_CLASS_PRE()
class AliGPUTPCTracker;

//Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class AliGPUMemClean16
{
  public:
	class AliGPUTPCSharedMemory
	{
	};

	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUglobalref() void* ptr, unsigned long size);
};

#endif
