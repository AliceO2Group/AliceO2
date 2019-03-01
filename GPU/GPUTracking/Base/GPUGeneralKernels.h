#ifndef GPUGENERALKERNELS_H
#define GPUGENERALKERNELS_H

#include "GPUTPCDef.h"
#include "GPUDataTypes.h"
MEM_CLASS_PRE() struct GPUConstantMem;

class GPUKernelTemplate
{
public:
	class GPUTPCSharedMemory
	{
	};

	typedef GPUconstantref() MEM_CONSTANT(GPUConstantMem) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::AllRecoSteps;}
	MEM_TEMPLATE() GPUhdi() static workerType *Worker(MEM_TYPE(GPUConstantMem) &workers) {return &workers;}
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
	template <int iKernel, typename... Args> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &workers, Args... args) {}
#else
	template <int iKernel> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &workers) {}
#endif
};

//Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class GPUMemClean16 : public GPUKernelTemplate
{
public:
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::NoRecoStep;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &workers, GPUglobalref() void* ptr, unsigned long size);
};

#endif
