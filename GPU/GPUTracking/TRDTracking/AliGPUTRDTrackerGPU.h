#ifndef ALIGPUTRDTRACKERGPUCA_H
#define ALIGPUTRDTRACKERGPUCA_H

#include "AliGPUGeneralKernels.h"

class AliGPUTRDTrackerGPU : public AliGPUKernelTemplate
{
  public:
#if defined(GPUCA_BUILD_TRD) || !defined(GPUCA_GPUCODE)
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers);
#endif
};

#endif
