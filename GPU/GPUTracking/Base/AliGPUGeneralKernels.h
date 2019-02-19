#ifndef ALIGPUGENERALKERNELS_H
#define ALIGPUGENERALKERNELS_H

#include "AliGPUTPCDef.h"
#include "AliGPUDataTypes.h"
MEM_CLASS_PRE() struct AliGPUConstantMem;

class AliGPUKernelTemplate
{
public:
	class AliGPUTPCSharedMemory
	{
	};

	typedef GPUconstantref() MEM_CONSTANT(AliGPUConstantMem) workerType;
	GPUhdi() static AliGPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::AllRecoSteps;}
	MEM_TEMPLATE() GPUhdi() static workerType *Worker(MEM_TYPE(AliGPUConstantMem) &workers) {return &workers;}
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
	template <int iKernel, typename... Args> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers, Args... args) {}
#else
	template <int iKernel> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers) {}
#endif
};

//Clean memory, ptr multiple of 16, size will be extended to multiple of 16
class AliGPUMemClean16 : public AliGPUKernelTemplate
{
public:
	GPUhdi() static AliGPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::NoRecoStep;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers, GPUglobalref() void* ptr, unsigned long size);
};

#endif
