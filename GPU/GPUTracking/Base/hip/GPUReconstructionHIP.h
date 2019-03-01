#ifndef GPURECONSTRUCTIONHIP_H
#define GPURECONSTRUCTIONHIP_H

#include "GPUReconstructionDeviceBase.h"
struct GPUReconstructionHIPInternals;

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUReconstruction* GPUReconstruction_Create_HIP(const GPUSettingsProcessing& cfg);
#else
extern "C" GPUReconstruction* GPUReconstruction_Create_HIP(const GPUSettingsProcessing& cfg);
#endif

class GPUReconstructionHIPBackend : public GPUReconstructionDeviceBase
{
public:
	virtual ~GPUReconstructionHIPBackend();
    
protected:
	GPUReconstructionHIPBackend(const GPUSettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int ExitDevice_Runtime() override;
	virtual void SetThreadCounts() override;

	virtual void SynchronizeGPU() override;
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override;
	virtual void SynchronizeStream(int stream) override;
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
	virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;
	
	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
	virtual void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) override;
	virtual void ReleaseEvent(deviceEvent* ev) override;
	virtual void RecordMarker(deviceEvent* ev, int stream) override;
	
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, Args... args);

private:
	GPUReconstructionHIPInternals* mInternals;
	int mCoreCount = 0;
};

using GPUReconstructionHIP = GPUReconstructionKernels<GPUReconstructionHIPBackend>;

#endif
