#ifndef GPUTRDTRACKERGPUCA_H
#define GPUTRDTRACKERGPUCA_H

#include "GPUGeneralKernels.h"

namespace o2
{
namespace gpu
{

class GPUTRDTrackerGPU : public GPUKernelTemplate
{
 public:
  GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TRDTracking; }
#if defined(GPUCA_BUILD_TRD) || !defined(GPUCA_GPUCODE)
  template <int iKernel = 0>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, workerType& workers);
#endif
};
} // namespace gpu
} // namespace o2

#endif
