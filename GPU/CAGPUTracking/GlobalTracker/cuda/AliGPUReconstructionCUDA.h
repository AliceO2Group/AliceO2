#ifndef ALIGPURECONSTRUCTIONCUDA_H
#define ALIGPURECONSTRUCTIONCUDA_H

#include "AliGPUReconstructionDeviceBase.h"
class AliGPUReconstructionCUDAInternals;

#ifdef _WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg);
#endif

class AliGPUReconstructionCUDABackend : public AliGPUReconstructionDeviceBase
{
public:
	virtual ~AliGPUReconstructionCUDABackend();
	virtual int DoTRDGPUTracking() override;
	virtual int RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers) override;
	virtual int GPUMergerAvailable() const override;
    
protected:
	AliGPUReconstructionCUDABackend(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int RunTPCTrackingSlices() override;
	int RunTPCTrackingSlices_internal();
	virtual int ExitDevice_Runtime() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual int SynchronizeGPU() override;
	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0) override;
	virtual int PrepareTextures() override;
	virtual int PrepareProfile() override;
	virtual int DoProfile() override;
	
	virtual int TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;
	virtual int TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) override;
	
	template <class T, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const Args&... args);

private:
	AliGPUReconstructionCUDAInternals* mInternals;
};

using AliGPUReconstructionCUDA = AliGPUReconstructionImpl<AliGPUReconstructionCUDABackend>;

#endif
