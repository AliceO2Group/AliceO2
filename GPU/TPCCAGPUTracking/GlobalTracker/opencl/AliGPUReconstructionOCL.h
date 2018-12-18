#ifndef ALIGPURECONSTRUCTIONOCL_H
#define ALIGPURECONSTRUCTIONOCL_H

#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUReconstructionOCLInternals.h"

#ifdef WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_OCLconst AliGPUCASettingsProcessing& cfg);
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
#endif

class AliGPUReconstructionOCL : public AliGPUReconstructionDeviceBase
{
public:
	virtual ~AliGPUReconstructionOCL();
    
	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers) const;

protected:
	friend AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
	AliGPUReconstructionOCL(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime();
	virtual int RunTPCTrackingSlices();
	virtual int ExitDevice_Runtime();

	virtual void ActivateThreadContext();
	virtual void ReleaseThreadContext();
	virtual void SynchronizeGPU();
	virtual int GPUSync(const char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0);

private:
	bool GPUFailedMsgA(int, const char* file, int line);
	AliGPUReconstructionOCLInternals* ocl;

};

#endif
