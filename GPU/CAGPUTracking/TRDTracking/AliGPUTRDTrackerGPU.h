#ifndef ALIGPUTRDTRACKERGPU_H
#define ALIGPUTRDTRACKERGPU_H

#include "AliGPUGeneralKernels.h"

class AliGPUTRDTrackerGPU : public AliGPUCAKernelTemplate
{
  public:
	typedef AliGPUCAConstantMem workerType;
	GPUhdi() static workerType *Worker(AliGPUCAConstantMem &workers) {return &workers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &merger);
};

#endif
