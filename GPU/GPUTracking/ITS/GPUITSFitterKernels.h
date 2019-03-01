#ifndef GPUITSFITTERKERNELS_H
#define GPUITSFITTERKERNELS_H

class GPUTPCGMPropagator;
class GPUITSFitter;

#include "GPUGeneralKernels.h"
namespace o2 { namespace ITS { class GPUITSTrack; struct TrackingFrameInfo;}}

class GPUITSFitterKernel : public GPUKernelTemplate
{
public:
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUDataTypes::RecoStep::ITSTracking;}
#if defined(GPUCA_BUILD_ITS)
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory &smem, workerType &workers);

protected:
	GPUd() static bool fitTrack(GPUITSFitter& Fitter, GPUTPCGMPropagator& prop, o2::ITS::GPUITSTrack& track, int start, int end, int step);
#endif
};

#endif
