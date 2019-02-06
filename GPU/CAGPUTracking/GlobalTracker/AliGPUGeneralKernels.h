#ifndef ALIGPUGENERALKERNELS_H
#define ALIGPUGENERALKERNELS_H

#include "AliGPUTPCDef.h"
MEM_CLASS_PRE()
struct AliGPUCAConstantMem;

//Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class AliGPUMemClean16
{
  public:
	class AliGPUTPCSharedMemory
	{
	};

	typedef GPUconstant() MEM_CONSTANT(AliGPUCAConstantMem) workerType;
	MEM_TEMPLATE() GPUhdi() static workerType *Worker(MEM_TYPE(AliGPUCAConstantMem) &workers) {return &workers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers, GPUglobalref() void* ptr, unsigned long size);
};

#endif
