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

	virtual int RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers) override;
    
protected:
	friend AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
	AliGPUReconstructionCUDA(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int RunTPCTrackingSlices() override;
	virtual int ExitDevice_Runtime() override;
	virtual int GPUMergerAvailable() const override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual int SynchronizeGPU() override;
	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0) override;
	
	virtual int TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;
	virtual int TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;

private:
	AliGPUReconstructionCUDAInternals* mInternals;
	int GPUFailedMsgA(const long long int error, const char* file, int line) const;
};

#endif
