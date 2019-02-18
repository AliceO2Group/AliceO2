#ifndef ALIGPURECONSTRUCTIONIMPL_H
#define ALIGPURECONSTRUCTIONIMPL_H

#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionHelpers.h"
#include "AliGPUConstantMem.h"
#include <stdexcept>
#include "utils/timer.h"
#include <vector>

#include "AliGPUGeneralKernels.h"
#include "AliGPUTPCNeighboursFinder.h"
#include "AliGPUTPCNeighboursCleaner.h"
#include "AliGPUTPCStartHitsFinder.h"
#include "AliGPUTPCStartHitsSorter.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCTrackletSelector.h"
#include "AliGPUTPCGMMergerGPU.h"
#include "AliGPUTRDTrackerGPU.h"

namespace AliGPUReconstruction_krnlHelpers {
template <class T, int I = 0> class classArgument {};

typedef void deviceEvent; //We use only pointers anyway, and since cl_event and cudaEvent_t are actually pointers, we can cast them to deviceEvent* this way.

enum class krnlDeviceType : int {CPU = 0, Device = 1, Auto = -1};
struct krnlExec
{
	krnlExec(unsigned int b, unsigned int t, int s, krnlDeviceType d = krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d) {}
	unsigned int nBlocks;
	unsigned int nThreads;
	int stream;
	krnlDeviceType device;
};
struct krnlRunRange
{
	krnlRunRange() : start(0), num(0) {}
	krnlRunRange(unsigned int a) : start(a), num(0) {}
	krnlRunRange(unsigned int s, int n) : start(s), num(n) {}
	
	unsigned int start;
	int num;
};
static const krnlRunRange krnlRunRangeNone(0, -1);
struct krnlEvent
{
	krnlEvent(deviceEvent* e = nullptr, deviceEvent* el = nullptr, int n = 1) : ev(e), evList(el), nEvents(n) {}
	deviceEvent* ev;
	deviceEvent* evList;
	int nEvents;
};
} //End Namespace

using namespace AliGPUReconstruction_krnlHelpers;

class AliGPUReconstructionCPUBackend : public AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstructionCPUBackend() = default;
	
protected:
	AliGPUReconstructionCPUBackend(const AliGPUSettingsProcessing& cfg) : AliGPUReconstruction(cfg) {}
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);
};

#include "AliGPUReconstructionKernels.h"
#ifndef GPUCA_ALIGPURECONSTRUCTIONCPU_IMPLEMENTATION
	#define GPUCA_ALIGPURECONSTRUCTIONCPU_DECLONLY
	#undef ALIGPURECONSTRUCTIONKERNELS_H
	#include "AliGPUReconstructionKernels.h"
#endif

class AliGPUReconstructionCPU : public AliGPUReconstructionKernels<AliGPUReconstructionCPUBackend>
{
	friend class AliGPUReconstruction;
	friend class AliGPUChain;
	
public:
	virtual ~AliGPUReconstructionCPU() = default;

#ifdef __APPLE__ //MacOS compiler BUG: clang seems broken and does not accept default parameters before parameter pack
	template <class S, int I = 0> inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone)
	{
		return runKernel<S, I>(x, t, y, krnlEvent());
	}
	template <class S, int I = 0, typename... Args> inline int runKernel(const krnlExec& x, HighResTimer* t, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
#else
	template <class S, int I = 0, typename... Args> inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEvent(), const Args&... args)
#endif
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) printf("Running %s Stream %d (Range %d/%d)\n", typeid(S).name(), x.stream, y.start, y.num);
		if (t && mDeviceProcessingSettings.debugLevel) t->Start();
		if (runKernelImpl(classArgument<S, I>(), x, y, z, args...)) return 1;
		if (mDeviceProcessingSettings.debugLevel)
		{
			if (GPUDebug(typeid(S).name(), x.stream)) throw std::runtime_error("kernel failure");
			if (t) t->Stop();
		}
		return 0;
	}
	
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1);
	void TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice());}
	void TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr());}
	void TransferMemoryResourcesToGPU(AliGPUProcessor* proc, int stream = -1, bool all = false) {TransferMemoryResourcesHelper(proc, stream, all, true);}
	void TransferMemoryResourcesToHost(AliGPUProcessor* proc, int stream = -1, bool all = false) {TransferMemoryResourcesHelper(proc, stream, all, false);}
	void TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents);}
	void TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents);}
	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr);
	int GPUStuck() {return mGPUStuck;}
	int NStreams() {return mNStreams;}
	void SetThreadCounts(RecoStep step);
	void ResetDeviceProcessorTypes();
	template <class T> void AddGPUEvents(T& events)
	{
		mEvents.emplace_back((void*) &events, sizeof(T) / sizeof(deviceEvent*));
	}

	virtual int RunStandalone() override;
	
protected:
	struct AliGPUProcessorWorkers : public AliGPUProcessor
	{
		AliGPUConstantMem* mWorkersProc = nullptr;
		void* SetPointersDeviceProcessor(void* mem);
		short mMemoryResWorkers = -1;
	};

	AliGPUReconstructionCPU(const AliGPUSettingsProcessing& cfg) : AliGPUReconstructionKernels(cfg) {}

	virtual void SynchronizeStream(int stream) {}
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {}
	virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) {return true;}
	virtual void RecordMarker(deviceEvent* ev, int stream) {}
	virtual void ActivateThreadContext() {}
	virtual void ReleaseThreadContext() {}
	virtual void SynchronizeGPU() {}
	virtual void ReleaseEvent(deviceEvent* ev) {}
	virtual int StartHelperThreads() {return 0;}
	virtual int StopHelperThreads() {return 0;}
	virtual void RunHelperThreads(int (AliGPUReconstructionHelpers::helperDelegateBase::* function)(int, int, AliGPUReconstructionHelpers::helperParam*), AliGPUReconstructionHelpers::helperDelegateBase* functionCls, int count) {}
	virtual void WaitForHelperThreads() {}
	virtual int HelperError(int iThread) const {return 0;}
	virtual int HelperDone(int iThread) const {return 0;}
	virtual void ResetHelperThreads(int helpers) {}
	virtual void TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst);
	
	virtual void SetThreadCounts();
	
	virtual int InitDevice() override;
	virtual int ExitDevice() override;
	int GetThread();
	
	virtual int PrepareTextures() {return 0;}
	virtual int DoStuckProtection(int stream, void* event) {return 0;}
	
	//Pointers to tracker classes
	AliGPUProcessorWorkers mProcShadow; //Host copy of tracker objects that will be used on the GPU
	AliGPUConstantMem* &mWorkersShadow = mProcShadow.mWorkersProc;
	
	unsigned int fBlockCount = 0;                 //Default GPU block count
	unsigned int fThreadCount = 0;                //Default GPU thread count
	unsigned int fConstructorBlockCount = 0;      //GPU blocks used in Tracklet Constructor
	unsigned int fSelectorBlockCount = 0;         //GPU blocks used in Tracklet Selector
	unsigned int fConstructorThreadCount = 0;
	unsigned int fSelectorThreadCount = 0;
	unsigned int fFinderThreadCount = 0;
	unsigned int fTRDThreadCount = 0;

	int mThreadId = -1; //Thread ID that is valid for the local CUDA context
	int mGPUStuck = 0;		//Marks that the GPU is stuck, skip future events
	int mNStreams = 1;
	
	AliGPUConstantMem* mDeviceConstantMem = nullptr;
	std::vector<std::pair<deviceEvent*, size_t>> mEvents;
	
private:
	void TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU);
};

#endif
