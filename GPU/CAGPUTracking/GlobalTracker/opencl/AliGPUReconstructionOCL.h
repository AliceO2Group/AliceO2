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
	virtual void SynchronizeGPU() override;
	virtual int DoStuckProtection(int stream, void* event) override;
	virtual int GPUDebug(const char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0) override;
	virtual void SynchronizeStream(int stream) override;
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
	virtual int IsEventDone(deviceEvent* evList, int nEvents = 1) override;

	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
	virtual void TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
	virtual void TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
	virtual void ReleaseEvent(deviceEvent* ev) override;
	
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);

private:
	template <class T, int I = 0, class S = cl_event> S& getKernelObject();
	
	AliGPUReconstructionOCLInternals* mInternals;
};

using AliGPUReconstructionOCL = AliGPUReconstructionImpl<AliGPUReconstructionOCLBackend>;

#endif
