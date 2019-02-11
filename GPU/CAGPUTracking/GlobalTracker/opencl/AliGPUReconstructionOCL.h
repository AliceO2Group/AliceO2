#ifndef ALIGPURECONSTRUCTIONOCL_H
#define ALIGPURECONSTRUCTIONOCL_H

#include "AliGPUReconstructionDeviceBase.h"
struct AliGPUReconstructionOCLInternals;

#ifdef _WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_OCLconst AliGPUCASettingsProcessing& cfg);
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg);
#endif

class AliGPUReconstructionOCLBackend : public AliGPUReconstructionDeviceBase
{
public:
	virtual ~AliGPUReconstructionOCLBackend();

protected:
	AliGPUReconstructionOCLBackend(const AliGPUCASettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int ExitDevice_Runtime() override;

	virtual void ActivateThreadContext() override;
	virtual void ReleaseThreadContext() override;
	virtual void SynchronizeGPU() override;
	virtual int DoStuckProtection(int stream, void* event) override;
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override;
	virtual void SynchronizeStream(int stream) override;
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
	virtual int IsEventDone(deviceEvent* evList, int nEvents = 1) override;

	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
	virtual void TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) override;
	virtual void ReleaseEvent(deviceEvent* ev) override;
	virtual void RecordMarker(deviceEvent* ev, int stream) override;
	
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);
	
	virtual bitfield<RecoStep, unsigned char> AvailableRecoSteps() {return (RecoStep::TPCSliceTracking);}

private:
	template <class S, class T, int I = 0> S& getKernelObject(int num);
	template <class T, int I = 0> int AddKernel(bool multi = false);
	template <class T, int I = 0> int FindKernel(int num);
	
	AliGPUReconstructionOCLInternals* mInternals;
};

using AliGPUReconstructionOCL = AliGPUReconstructionKernels<AliGPUReconstructionOCLBackend>;

#endif
