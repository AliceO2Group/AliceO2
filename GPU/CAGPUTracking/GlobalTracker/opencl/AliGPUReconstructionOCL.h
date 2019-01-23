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
    
	virtual int RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers) override;

protected:
	friend AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
	AliGPUReconstructionOCL(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int RunTPCTrackingSlices() override;
	int RunTPCTrackingSlices_internal();
	virtual int ExitDevice_Runtime() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual int SynchronizeGPU() override;
	virtual int DoStuckProtection(int stream, void* event) override;
	virtual int GPUSync(const char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0) override;

	virtual int TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;
	virtual int TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;

private:
	AliGPUReconstructionOCLInternals* mInternals;
};

#endif
