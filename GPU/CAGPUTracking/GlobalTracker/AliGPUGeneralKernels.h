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

	GPUd() static int NThreadSyncPoints() { return 1; }
	GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, int iSync, GPUsharedref() AliGPUTPCSharedMemory &smem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUglobalref() void* ptr, unsigned long size);
};

#endif
