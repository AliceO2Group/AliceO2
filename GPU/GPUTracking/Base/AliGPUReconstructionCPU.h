#ifndef ALIGPURECONSTRUCTIONIMPL_H
#define ALIGPURECONSTRUCTIONIMPL_H

#include "AliGPUReconstruction.h"
#include "AliGPUConstantMem.h"
#include <stdexcept>
#include <atomic>
#include <array>
#include "utils/timer.h"

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
	
	int RunTPCTrackingSlices();
	virtual int RunTPCTrackingMerger();
	virtual int RefitMergedTracks(bool resetTimers);
	virtual int RunTRDTracking();
	virtual int RunStandalone();
	
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1);
	
	void TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice());}
	void TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr());}
	virtual void TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst);
	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr);

	void TransferMemoryResourcesToGPU(AliGPUProcessor* proc, int stream = -1, bool all = false) {TransferMemoryResourcesHelper(proc, stream, all, true);}
	void TransferMemoryResourcesToHost(AliGPUProcessor* proc, int stream = -1, bool all = false) {TransferMemoryResourcesHelper(proc, stream, all, false);}
	void TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents);}
	void TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents);}
	
	HighResTimer timerTPCtracking[NSLICES][10];
	
protected:
	template <class T> struct eventStruct
	{
		T selector[NSLICES];
		T stream[GPUCA_GPUCA_MAX_STREAMS];
		T init;
		T constructor;
	};
	
	struct AliGPUProcessorWorkers : public AliGPUProcessor
	{
		AliGPUWorkers* mWorkersProc = nullptr;
		TPCFastTransform* fTpcTransform = nullptr;
		char* fTpcTransformBuffer = nullptr;
		o2::trd::TRDGeometryFlat* fTrdGeometry = nullptr;
		void* SetPointersDeviceProcessor(void* mem);
		void* SetPointersFlatObjects(void* mem);
		short mMemoryResWorkers = -1;
		short mMemoryResFlat = -1;
	};
	
	virtual void SynchronizeStream(int stream) {}
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {}
	virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) {return true;}
	virtual void RecordMarker(deviceEvent* ev, int stream) {}

	AliGPUReconstructionCPU(const AliGPUSettingsProcessing& cfg) : AliGPUReconstructionKernels(cfg) {}
	virtual void ActivateThreadContext() {}
	virtual void ReleaseThreadContext() {}
	virtual void SynchronizeGPU() {}
	virtual void ReleaseEvent(deviceEvent* ev) {}
	virtual int StartHelperThreads() {return 0;}
	virtual int StopHelperThreads() {return 0;}
	virtual void RunHelperThreads(int phase) {}
	virtual void WaitForHelperThreads() {}
	virtual int HelperError(int iThread) const {return 0;}
	virtual int HelperDone(int iThread) const {return 0;}
	virtual void ResetHelperThreads(int helpers) {}
	
	virtual void SetThreadCounts();
	void SetThreadCounts(RecoStep step);
	int ReadEvent(int iSlice, int threadId);
	void WriteOutput(int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId);
	
	virtual int InitDevice();
	virtual int ExitDevice();
	int GetThread();
	
	virtual int PrepareTextures() {return 0;}
	virtual int DoStuckProtection(int stream, void* event) {return 0;}
	virtual int PrepareProfile() {return 0;}
	virtual int DoProfile() {return 0;}
	
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
	eventStruct<void*> mEvents;
	
	AliGPUConstantMem* mDeviceConstantMem = nullptr;
	AliGPUProcessorWorkers mProcShadow; //Host copy of tracker objects that will be used on the GPU
	AliGPUProcessorWorkers mProcDevice; //tracker objects that will be used on the GPU
	AliGPUWorkers* &mWorkersShadow = mProcShadow.mWorkersProc;
	AliGPUWorkers* &mWorkersDevice = mProcDevice.mWorkersProc;
	
#ifdef __ROOT__ //ROOT5 BUG: cint doesn't do volatile
#define volatile
#endif
	volatile int fSliceOutputReady;
	volatile char fSliceLeftGlobalReady[NSLICES];
	volatile char fSliceRightGlobalReady[NSLICES];
#ifdef __ROOT__
#undef volatile
#endif
	std::array<char, NSLICES> fGlobalTrackingDone;
	std::array<char, NSLICES> fWriteOutputDone;

private:
	void TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU);
	int RunTPCTrackingSlices_internal();
	std::atomic_flag mLockAtomic = ATOMIC_FLAG_INIT;
};

#endif
