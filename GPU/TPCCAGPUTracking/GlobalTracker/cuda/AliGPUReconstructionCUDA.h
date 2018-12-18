#ifndef ALIGPURECONSTRUCTIONCUDA_H
#define ALIGPURECONSTRUCTIONCUDA_H

#include "AliGPUReconstructionDeviceBase.h"
class AliGPUReconstructionCUDAInternals;

#ifdef WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
#endif

class AliGPUReconstructionCUDA : public AliGPUReconstructionDeviceBase
{
public:
	virtual ~AliGPUReconstructionCUDA();

	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers) const;
    
protected:
	friend AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
	AliGPUReconstructionCUDA(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime();
	virtual int RunTPCTrackingSlices();
	virtual int ExitDevice_Runtime();
	virtual int GPUMergerAvailable();

	virtual void ActivateThreadContext();
	virtual void ReleaseThreadContext();
	virtual void SynchronizeGPU();
	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0);

private:
	AliGPUReconstructionCUDAInternals* mInternals;
	bool GPUFailedMsgA(const long long int error, const char* file, int line) const;
};

#endif
