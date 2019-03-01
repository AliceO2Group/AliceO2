#ifndef GPURECONSTRUCTIONCUDA_H
#define GPURECONSTRUCTIONCUDA_H

#include "GPUReconstructionDeviceBase.h"
struct GPUReconstructionCUDAInternals;

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsProcessing& cfg);
#else
extern "C" GPUReconstruction* GPUReconstruction_Create_CUDA(const GPUSettingsProcessing& cfg);
#endif

class GPUReconstructionCUDABackend : public GPUReconstructionDeviceBase
{
public:
	virtual ~GPUReconstructionCUDABackend();
    
protected:
	GPUReconstructionCUDABackend(const GPUSettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int ExitDevice_Runtime() override;
	virtual void SetThreadCounts() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual void SynchronizeGPU() override;
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override;
	virtual void SynchronizeStream(int stream) override;
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
	virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;
	
	virtual int PrepareTextures() override;
	
	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
	virtual void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) override;
	virtual void ReleaseEvent(deviceEvent* ev) override;
	virtual void RecordMarker(deviceEvent* ev, int stream) override;
	
	virtual void GetITSTraits(std::unique_ptr<o2::ITS::TrackerTraits>& trackerTraits, std::unique_ptr<o2::ITS::VertexerTraits>& vertexerTraits) override;
	
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);

private:
	GPUReconstructionCUDAInternals* mInternals;
	int mCoreCount = 0;
};

using GPUReconstructionCUDA = GPUReconstructionKernels<GPUReconstructionCUDABackend>;

#endif
