#ifndef ALIGPUGENERALKERNELS_H
#define ALIGPUGENERALKERNELS_H

#include "AliGPUTPCDef.h"
#include "AliGPUCADataTypes.h"

class AliGPUCAKernelTemplate
{
public:
	class AliGPUTPCSharedMemory
	{
	};

	typedef GPUconstant() MEM_CONSTANT(AliGPUCAConstantMem) workerType;
	MEM_TEMPLATE() GPUhdi() static workerType *Worker(MEM_TYPE(AliGPUCAConstantMem) &workers) {return &workers;}
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
	template <int iKernel, typename... Args> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers, Args... args) {}
#else
	template <int iKernel> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers) {}
#endif
};

//Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class AliGPUMemClean16 : public AliGPUCAKernelTemplate
{
public:
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers, GPUglobalref() void* ptr, unsigned long size);
};

#endif
