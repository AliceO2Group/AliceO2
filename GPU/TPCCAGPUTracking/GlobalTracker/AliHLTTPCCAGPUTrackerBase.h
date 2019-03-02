//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliHLTTPCCAGPUTrackerBase.h
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 

#ifndef ALIHLTTPCCAGPUTRACKERBASE_H
#define ALIHLTTPCCAGPUTRACKERBASE_H

#define HLTCA_GPU_DEFAULT_MAX_SLICE_COUNT 36

#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTLogging.h"
#include "AliHLTTPCCASliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(R__WIN32)
#include "../cmodules/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#include <errno.h>
#endif

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

MEM_CLASS_PRE() class AliHLTTPCCARow;

class AliHLTTPCCAGPUTrackerBase : public AliHLTTPCCAGPUTracker, public AliHLTLogging
{
	friend void* helperWrapper(void*);
public:
	AliHLTTPCCAGPUTrackerBase();
	virtual ~AliHLTTPCCAGPUTrackerBase();

	virtual int InitGPU(int sliceCount = -1, int forceDeviceID = -1);
	virtual int InitGPU_Runtime(int sliceCount = -1, int forceDeviceID = -1) = 0;
	virtual int IsInitialized();
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1) = 0;
	int SelfHealReconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU();
	virtual int ExitGPU_Runtime() = 0;

	virtual void SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile = NULL);
	virtual int SetGPUTrackerOption(const char* OptionName, int OptionValue);

	virtual double GetTimer(int iSlice, unsigned int i);
	virtual void ResetTimer(int iSlice, unsigned int i);

	virtual int InitializeSliceParam(int iSlice, const AliHLTTPCCAParam &param);
	virtual void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);

	virtual const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const;
	virtual int GetSliceCount() const;

	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers) = 0;
	virtual char* MergerHostMemory() {return((char*) fGPUMergerHostMemory);}

protected:
	virtual void ActivateThreadContext() = 0;
	virtual void ReleaseThreadContext() = 0;
	virtual void SynchronizeGPU() = 0;

	struct helperParam
	{
		void* fThreadId;
		AliHLTTPCCAGPUTrackerBase* fCls;
		int fNum;
		int fSliceCount;
		AliHLTTPCCAClusterData* pClusterData;
		AliHLTTPCCASliceOutput** pOutput;
		int fFirstSlice;
		void* fMutex;
		char fTerminate;
		int fPhase;
		int CPUTracker;
		volatile int fDone;
		volatile char fError;
		volatile char fReset;
	};

	static void* RowMemory(void* const BaseMemory, int iSlice)       { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (HLTCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* const BaseMemory, int iSlice)    { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + iSlice * HLTCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* const BaseMemory, int iSlice) const     { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_GLOBAL_MEMORY ); } //in GPU memory, not host memory!!!
	void* TracksMemory(void* const BaseMemory, int iSlice) const     { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_TRACKS_MEMORY ); } //in host memory, not GPU memory!!!
	void* TrackerMemory(void* const BaseMemory, int iSlice) const    { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }

	int Reconstruct_Base_Init(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int& firstSlice, int& sliceCountLocal);
	int Reconstruct_Base_SliceInit(AliHLTTPCCAClusterData* pClusterData, int& iSlice, int& firstSlice);
	int Reconstruct_Base_StartGlobal(AliHLTTPCCASliceOutput** pOutput, char*& tmpMemoryGlobalTracking);
	int Reconstruct_Base_FinishSlices(AliHLTTPCCASliceOutput** pOutput, int& iSlice, int& firstSlice);
	int Reconstruct_Base_Finalize(AliHLTTPCCASliceOutput** pOutput, char*& tmpMemoryGlobalTracking, int& firstSlice);
	
	int ReadEvent(AliHLTTPCCAClusterData* pClusterData, int firstSlice, int iSlice, int threadId);
	void WriteOutput(AliHLTTPCCASliceOutput** pOutput, int firstSlice, int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId, helperParam* hParam);

	int StartHelperThreads();
	int StopHelperThreads();
	void ResetHelperThreads(int helpers);
	void ResetThisHelperThread(AliHLTTPCCAGPUTrackerBase::helperParam* par);

	int GetThread();
	void ReleaseGlobalLock(void* sem);
	int CheckMemorySizes(int sliceCount);

	virtual int GPUSync(const char* state = "UNKNOWN", int stream = -1, int slice = 0) = 0;
	template <class T> T* alignPointer(T* ptr, int alignment);
	void StandalonePerfTime(int iSlice, int i);
#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
	
	static void* helperWrapper(void*);

	AliHLTTPCCATracker *fGpuTracker; //Tracker Objects that will be used on the GPU
	void* fGPUMemory; //Pointer to GPU Memory Base Adress
	void* fHostLockedMemory; //Pointer to Base Adress of Page Locked Host Memory for DMA Transfer

	void* fGPUMergerMemory;
	void* fGPUMergerHostMemory;
	int fGPUMergerMaxMemory;

	int fDebugLevel;			//Debug Level for GPU Tracker
	unsigned int fDebugMask;	//Mask which Debug Data is written to file
	std::ostream* fOutFile;		//Debug Output Stream Pointer
	unsigned long long int fGPUMemSize;	//Memory Size to allocate on GPU
	unsigned long long int fHostMemSize; //Memory Size to allocate on Host

	int fSliceCount; //Maximum Number of Slices this GPU tracker can process in parallel
	int fCudaDevice; //CUDA device used by GPU tracker

	static const int fgkNSlices = 36; //Number of Slices in Alice
	AliHLTTPCCATracker fSlaveTrackers[fgkNSlices]; //CPU Slave Trackers for Initialization and Output

	AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl; //Output Control Structure
	
	int fThreadId; //Thread ID that is valid for the local CUDA context
	int fCudaInitialized; //Flag if CUDA is initialized

	int fSelfheal; //Reinitialize GPU on failure

	int fConstructorBlockCount; //GPU blocks used in Tracklet Constructor
	int fSelectorBlockCount; //GPU blocks used in Tracklet Selector
	int fConstructorThreadCount;
	
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int fProfTimeC, fProfTimeD; //Timing
#endif

	int fNHelperThreads; //Number of helper threads for post/preprocessing
	helperParam* fHelperParams; //Control Struct for helper threads
	void* fHelperMemMutex;
	
#ifdef __ROOT__
#define volatile
#endif
	volatile int fSliceOutputReady;
	volatile char fSliceLeftGlobalReady[fgkNSlices];
	volatile char fSliceRightGlobalReady[fgkNSlices];
#ifdef __ROOT__
#undef volatile
#endif
	void* fSliceGlobalMutexes;
	char fGlobalTrackingDone[fgkNSlices];
	char fWriteOutputDone[fgkNSlices];

	int fNCPUTrackers; //Number of CPU trackers to use
	int fNSlicesPerCPUTracker; //Number of slices processed by each CPU tracker

	int fGlobalTracking; //Use Global Tracking
	int fUseGlobalTracking; 

	int fNSlaveThreads;	//Number of slave threads currently active
	
	int fStuckProtection;   //Protection from GPU stuck, set maximum time to wait for GPU in usec!
	int fGPUStuck;		//Marks that the GPU is stuck, skip future events

	// disable copy
	AliHLTTPCCAGPUTrackerBase( const AliHLTTPCCAGPUTrackerBase& );
	AliHLTTPCCAGPUTrackerBase &operator=( const AliHLTTPCCAGPUTrackerBase& );

	ClassDef( AliHLTTPCCAGPUTrackerBase, 0 )
};

template <class T> inline T* AliHLTTPCCAGPUTrackerBase::alignPointer(T* ptr, int alignment)
{
	//Macro to align Pointers.
	//Will align to start at 1 MB segments, this should be consistent with every alignment in the tracker
	//(As long as every single data structure is <= 1 MB)

	size_t adr = (size_t) ptr;
	if (adr % alignment)
	{
		adr += alignment - (adr % alignment);
	}
	return((T*) adr);
}

#endif
