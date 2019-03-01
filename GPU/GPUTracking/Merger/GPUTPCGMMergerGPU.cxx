#include "GPUTPCGMMergerGPU.h"
#include "GPUConstantMem.h"
#if defined(GPUCA_HAVE_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

template <> GPUd() void GPUTPCGMMergerTrackFit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &merger)
{
#if defined(GPUCA_HAVE_OPENMP) && !defined(GPUCA_GPUCODE)
#pragma omp parallel for num_threads(merger.GetRec().GetDeviceProcessingSettings().nThreads)
#endif
	for (int i = get_global_id(0);i < merger.NOutputTracks();i += get_global_size(0))
	{
		GPUTPCGMTrackParam::RefitTrack(merger.OutputTracks()[i], i, &merger, merger.Clusters());
	}
}
