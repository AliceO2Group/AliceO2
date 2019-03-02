#include "GPUTRDTrackerGPU.h"
#include "GPUTRDGeometry.h"
#include "GPUConstantMem.h"
#if defined(GPUCA_HAVE_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace o2::gpu;

template <>
GPUd() void GPUTRDTrackerGPU::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, workerType& workers)
{
#if defined(GPUCA_HAVE_OPENMP) && !defined(GPUCA_GPUCODE)
#pragma omp parallel for num_threads(workers.trdTracker.GetRec().GetDeviceProcessingSettings().nThreads)
#endif
  for (int i = get_global_id(0); i < workers.trdTracker.NTracks(); i += get_global_size(0)) {
    workers.trdTracker.DoTrackingThread(i, &workers.tpcMerger, get_global_id(0));
  }
}
