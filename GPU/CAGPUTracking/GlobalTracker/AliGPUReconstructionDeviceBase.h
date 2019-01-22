#ifndef ALIGPURECONSTRUCTIONDEVICEBASE_H
#define ALIGPURECONSTRUCTIONDEVICEBASE_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"

class AliGPUReconstructionDeviceBase : public AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstructionDeviceBase() override = default;

	char* MergerHostMemory() const {return((char*) fGPUMergerHostMemory);}
	const AliGPUCAParam* DeviceParam() const {return mDeviceParam;}
	virtual int RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers) override = 0;

protected:
	typedef void deviceEvent;
	
	AliGPUReconstructionDeviceBase(const AliGPUCASettingsProcessing& cfg);
	AliGPUCAConstantMem mGPUReconstructors;
	void* fGPUMergerHostMemory = nullptr;
	AliGPUCAParam* mDeviceParam = nullptr;
    
#ifdef GPUCA_ENABLE_GPU_TRACKER
	virtual int RunTPCTrackingSlices() override = 0;

	virtual int InitDevice() override;
	virtual int InitDevice_Runtime() = 0;
	virtual int ExitDevice() override;
	virtual int ExitDevice_Runtime() = 0;

	virtual const AliGPUTPCTracker* CPUTracker(int iSlice);

	virtual void ActivateThreadContext() = 0;
	virtual void ReleaseThreadContext() = 0;
	virtual int SynchronizeGPU() = 0;
	
	virtual int TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) = 0;
	virtual int TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr) = 0;
	int TransferMemoryResourcesToGPU(AliGPUProcessor* proc, int stream = -1, bool all = false);
	int TransferMemoryResourcesToHost(AliGPUProcessor* proc, int stream = -1, bool all = false);
	int TransferMemoryResourceLinkToGPU(short res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr);
	int TransferMemoryResourceLinkToHost(short res, int stream = -1, int nEvents = 0, deviceEvent* evList = nullptr, deviceEvent* ev = nullptr);

	struct helperParam
	{
		void* fThreadId;
		AliGPUReconstructionDeviceBase* fCls;
		int fNum;
		void* fMutex;
		char fTerminate;
		int fPhase;
		volatile int fDone;
		volatile char fError;
		volatile char fReset;
	};
	
	struct AliGPUProcessorWorkers : public AliGPUProcessor
	{
		AliGPUTPCTracker *fGpuTracker = nullptr; //Tracker Objects that will be used on the GPU
		AliGPUTPCGMMerger* fGpuMerger = nullptr;
		void* SetPointersDeviceProcessor(void* mem);
		short mMemoryResWorkers;
	};

	int Reconstruct_Base_Init();
	int Reconstruct_Base_SliceInit(unsigned int iSlice);
	int Reconstruct_Base_StartGlobal();
	int Reconstruct_Base_FinishSlices(unsigned int iSlice);
	int Reconstruct_Base_Finalize();

	int ReadEvent(int iSlice, int threadId);
	void WriteOutput(int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId, helperParam* hParam);

	int StartHelperThreads();
	int StopHelperThreads();
	void ResetHelperThreads(int helpers);
	void ResetThisHelperThread(helperParam* par);

	int GetThread();
	void ReleaseGlobalLock(void* sem);

	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0) = 0;

	static void* helperWrapper(void*);
	
	AliGPUProcessorWorkers workers;
	AliGPUProcessorWorkers workersDevice;
	AliGPUTPCTracker* &fGpuTracker = workers.fGpuTracker;
	AliGPUTPCGMMerger* &fGpuMerger = workers.fGpuMerger;

	int fThreadId = -1; //Thread ID that is valid for the local CUDA context
    int fDeviceId = -1; //Device ID used by backend

	int fConstructorBlockCount = 0; //GPU blocks used in Tracklet Constructor
	int fSelectorBlockCount = 0; //GPU blocks used in Tracklet Selector
	int fConstructorThreadCount = 0;

#ifdef GPUCA_GPU_TIME_PROFILE
	unsigned long long int fProfTimeC, fProfTimeD; //Timing
#endif

	helperParam* fHelperParams = nullptr; //Control Struct for helper threads
	void* fHelperMemMutex = nullptr;

#ifdef __ROOT__
#define volatile
#endif
	volatile int fSliceOutputReady;
	volatile char fSliceLeftGlobalReady[NSLICES];
	volatile char fSliceRightGlobalReady[NSLICES];
#ifdef __ROOT__
#undef volatile
#endif
	void* fSliceGlobalMutexes = nullptr;
	char fGlobalTrackingDone[NSLICES];
	char fWriteOutputDone[NSLICES];

	int fNSlaveThreads = 0;	//Number of slave threads currently active

	int fGPUStuck = 0;		//Marks that the GPU is stuck, skip future events
#endif
};

#endif
