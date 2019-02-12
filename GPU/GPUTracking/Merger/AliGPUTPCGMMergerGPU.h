#ifndef ALIGPUTPCGMMERGERGPUCA_H
#define ALIGPUTPCGMMERGERGPUCA_H

#include "AliGPUGeneralKernels.h"

class AliGPUTPCGMMergerTrackFit : public AliGPUKernelTemplate
{
  public:
#if defined(GPUCA_BUILD_MERGER) || !defined(GPUCA_GPUCODE)
	typedef AliGPUTPCGMMerger workerType;
	GPUhdi() static workerType *Worker(AliGPUConstantMem &workers) {return &workers.tpcMerger;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &merger);
#endif
};

#endif
