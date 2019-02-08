#ifndef ALIGPUTPCGMMERGERGPU_H
#define ALIGPUTPCGMMERGERGPU_H

#include "AliGPUGeneralKernels.h"

class AliGPUTPCGMMergerTrackFit : public AliGPUCAKernelTemplate
{
  public:
#if defined(GPUCA_BUILD_MERGER) || !defined(GPUCA_GPUCODE)
	typedef AliGPUTPCGMMerger workerType;
	GPUhdi() static workerType *Worker(AliGPUCAConstantMem &workers) {return &workers.tpcMerger;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &merger);
#endif
};

#endif
