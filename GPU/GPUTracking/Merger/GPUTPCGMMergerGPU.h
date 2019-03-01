#ifndef GPUTPCGMMERGERGPUCA_H
#define GPUTPCGMMERGERGPUCA_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

class GPUTPCGMMergerTrackFit : public GPUKernelTemplate
{
public:
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUDataTypes::RecoStep::TPCMerging;}
#if defined(GPUCA_BUILD_MERGER) || !defined(GPUCA_GPUCODE)
	typedef GPUTPCGMMerger workerType;
	GPUhdi() static workerType *Worker(GPUConstantMem &workers) {return &workers.tpcMerger;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &merger);
#endif
};

#endif
