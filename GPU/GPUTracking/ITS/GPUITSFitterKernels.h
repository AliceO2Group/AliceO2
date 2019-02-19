#ifndef GPUITSFITTERKERNELS_H
#define GPUITSFITTERKERNELS_H

class AliGPUTPCGMPropagator;
class GPUITSFitter;

#include "AliGPUGeneralKernels.h"
namespace o2 { namespace ITS { class GPUITSTrack; struct TrackingFrameInfo;}}

class GPUITSFitterKernel : public AliGPUKernelTemplate
{
public:
	GPUhdi() static AliGPUDataTypes::RecoStep GetRecoStep() {return AliGPUDataTypes::RecoStep::ITSTracking;}
#if defined(GPUCA_BUILD_ITS)
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers);

protected:
	GPUd() static bool fitTrack(GPUITSFitter& Fitter, AliGPUTPCGMPropagator& prop, o2::ITS::GPUITSTrack& track, int start, int end, int step);
#endif
};

#endif
