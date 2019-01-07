#ifndef ALIGPURECONSTRUCTIONDEVICEBASE_H
#define ALIGPURECONSTRUCTIONDEVICEBASE_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)

class AliGPUReconstructionDeviceBase : public AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstructionDeviceBase() override = default;

	char* MergerHostMemory() const {return((char*) fGPUMergerHostMemory);}
	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers) const = 0;

protected:
	AliGPUReconstructionDeviceBase(const AliGPUCASettingsProcessing& cfg);
	AliGPUCAConstantMem mGPUReconstructors;
    
#ifdef GPUCA_ENABLE_GPU_TRACKER
	virtual int RunTPCTrackingSlices() override = 0;

	virtual int InitDevice() override;
	virtual int InitDevice_Runtime() = 0;
	virtual int ExitDevice() override;
	virtual int ExitDevice_Runtime() = 0;

	virtual const AliHLTTPCCATracker* CPUTracker(int iSlice);

	virtual void ActivateThreadContext() = 0;
	virtual void ReleaseThreadContext() = 0;
	virtual void SynchronizeGPU() = 0;

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

	static void* RowMemory(void* const BaseMemory, int iSlice)       { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (GPUCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* const BaseMemory, int iSlice)    { return( ((char*) BaseMemory) + GPUCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + GPUCA_GPU_ROWS_MEMORY + GPUCA_GPU_COMMON_MEMORY + iSlice * GPUCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* const BaseMemory, int iSlice) const     { return( ((char*) BaseMemory) + GPUCA_GPU_ROWS_MEMORY + GPUCA_GPU_COMMON_MEMORY + NSLICES * (GPUCA_GPU_SLICE_DATA_MEMORY) + iSlice * GPUCA_GPU_GLOBAL_MEMORY ); } //in GPU memory, not host memory!!!
	void* TracksMemory(void* const BaseMemory, int iSlice) const     { return( ((char*) BaseMemory) + GPUCA_GPU_ROWS_MEMORY + GPUCA_GPU_COMMON_MEMORY + NSLICES * (GPUCA_GPU_SLICE_DATA_MEMORY) + iSlice * GPUCA_GPU_TRACKS_MEMORY ); } //in host memory, not GPU memory!!!
	void* TrackerMemory(void* const BaseMemory, int iSlice) const    { return( ((char*) BaseMemory) + GPUCA_GPU_ROWS_MEMORY + GPUCA_GPU_COMMON_MEMORY + NSLICES * (GPUCA_GPU_SLICE_DATA_MEMORY + GPUCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }
    
	int Reconstruct_Base_Init();
	int Reconstruct_Base_SliceInit(unsigned int iSlice);
	int Reconstruct_Base_StartGlobal(char*& tmpMemoryGlobalTracking);
	int Reconstruct_Base_FinishSlices(unsigned int iSlice);
	int Reconstruct_Base_Finalize(char*& tmpMemoryGlobalTracking);

	int ReadEvent(int iSlice, int threadId);
	void WriteOutput(int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId, helperParam* hParam);

	int StartHelperThreads();
	int StopHelperThreads();
	void ResetHelperThreads(int helpers);
	void ResetThisHelperThread(helperParam* par);

	int GetThread();
	void ReleaseGlobalLock(void* sem);
	int CheckMemorySizes(int sliceCount);

	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0) = 0;

	static void* helperWrapper(void*);

	AliHLTTPCCATracker *fGpuTracker = nullptr; //Tracker Objects that will be used on the GPU
	void* fGPUMemory = nullptr; //Pointer to GPU Memory Base Adress
	void* fHostLockedMemory = nullptr; //Pointer to Base Adress of Page Locked Host Memory for DMA Transfer

	void* fGPUMergerMemory = nullptr;
	void* fGPUMergerHostMemory = nullptr;
	unsigned long long int fGPUMergerMaxMemory = 0;

	unsigned long long int fGPUMemSize = 0;	//Memory Size to allocate on GPU
	unsigned long long int fHostMemSize = 0; //Memory Size to allocate on Host

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
