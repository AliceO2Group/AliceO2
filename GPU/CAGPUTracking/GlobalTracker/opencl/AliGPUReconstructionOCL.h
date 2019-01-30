#ifndef ALIGPURECONSTRUCTIONOCL_H
#define ALIGPURECONSTRUCTIONOCL_H

#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUReconstructionOCLInternals.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_OCLconst AliGPUCASettingsProcessing& cfg);
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
#endif

class AliGPUReconstructionOCLBackend : public AliGPUReconstructionDeviceBase
{
public:
	virtual ~AliGPUReconstructionOCLBackend();
    
	virtual int RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers) override;

protected:
	AliGPUReconstructionOCLBackend(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int RunTPCTrackingSlices() override;
	int RunTPCTrackingSlices_internal();
	virtual int ExitDevice_Runtime() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual int SynchronizeGPU() override;
	virtual int DoStuckProtection(int stream, void* event) override;
	virtual int GPUSync(const char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0) override;

	virtual int TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
	virtual int TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
	
	template <class T, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);

private:
	template <class T, class S> S& getKernelObject();
	
	AliGPUReconstructionOCLInternals* mInternals;
};

using AliGPUReconstructionOCL = AliGPUReconstructionImpl<AliGPUReconstructionOCLBackend>;

#endif
