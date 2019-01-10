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
    
	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers) const override;

protected:
	friend AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
	AliGPUReconstructionOCL(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int RunTPCTrackingSlices() override;
	virtual int ExitDevice_Runtime() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual void SynchronizeGPU() override;
	virtual int GPUSync(const char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0) override;

private:
	bool GPUFailedMsgA(int, const char* file, int line);
	AliGPUReconstructionOCLInternals* ocl;

};

#endif
