//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliHLTTPCCAGPUTrackerNVCC.h
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 


#ifndef ALIHLTTPCCAGPUTRACKERNVCC_H
#define ALIHLTTPCCAGPUTRACKERNVCC_H

#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTLogging.h"
#include "AliHLTTPCCASliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(R__WIN32)
#include "pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#endif

class AliHLTTPCCARow;

class AliHLTTPCCAGPUTrackerNVCC : public AliHLTTPCCAGPUTracker, public AliHLTLogging
{
	friend void* helperWrapper(void*);
public:
	AliHLTTPCCAGPUTrackerNVCC();
	virtual ~AliHLTTPCCAGPUTrackerNVCC();

	virtual int InitGPU(int sliceCount = -1, int forceDeviceID = -1);
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	int ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	int SelfHealReconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU();

	virtual void SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile = NULL);
	virtual int SetGPUTrackerOption(char* OptionName, int OptionValue);

	virtual unsigned long long int* PerfTimer(int iSlice, unsigned int i);

	virtual int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);
	virtual void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);

	virtual const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const;
	virtual int GetSliceCount() const;

	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger);
	virtual char* MergerBaseMemory();

private:
	static void* RowMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (HLTCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + iSlice * HLTCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_GLOBAL_MEMORY ); }
	void* TracksMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_TRACKS_MEMORY ); }
	void* TrackerMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }
	
	void ReadEvent(AliHLTTPCCAClusterData* pClusterData, int firstSlice, int iSlice, int threadId);
	void WriteOutput(AliHLTTPCCASliceOutput** pOutput, int firstSlice, int iSlice, int threadId);

	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);
	int GetThread();
	void ReleaseGlobalLock(void* sem);
	int CheckMemorySizes(int sliceCount);

	int CUDASync(char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0);
	template <class T> T* alignPointer(T* ptr, int alignment);
	void StandalonePerfTime(int iSlice, int i);
	bool CudaFailedMsg(cudaError_t error);
	
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

	void* fpCudaStreams; //Pointer to array of CUDA Streams
	int fSliceCount; //Maximum Number of Slices this GPU tracker can process in parallel
	int fCudaDevice; //CUDA device used by GPU tracker

	static const int fgkNSlices = 36; //Number of Slices in Alice
	AliHLTTPCCATracker fSlaveTrackers[fgkNSlices]; //CPU Slave Trackers for Initialization and Output

	AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl; //Output Control Structure
	
	int fThreadId; //Thread ID that is valid for the local CUDA context
	int fCudaInitialized; //Flag if CUDA is initialized

	int fPPMode; //Flag if GPU tracker runs in PP Mode
	int fSelfheal; //Reinitialize GPU on failure

	int fConstructorBlockCount; //GPU blocks used in Tracklet Constructor
	int selectorBlockCount; //GPU blocks used in Tracklet Selector
	
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int fProfTimeC, fProfTimeD; //Timing
#endif

	void* fCudaContext; //Pointer to CUDA context
	
	struct helperParam
	{
		AliHLTTPCCAGPUTrackerNVCC* fCls;
		int fNum;
		int fSliceCount;
		AliHLTTPCCAClusterData* pClusterData;
		AliHLTTPCCASliceOutput** pOutput;
		int fFirstSlice;
		void* fMutex;
		bool fTerminate;
		int fPhase;
		volatile int fDone;
	};
	static const int fNHelperThreads = HLTCA_GPU_HELPER_THREADS; //Number of helper threads for post/preprocessing
	helperParam fHelperParams[fNHelperThreads]; //Control Struct for helper threads

	// disable copy
	AliHLTTPCCAGPUTrackerNVCC( const AliHLTTPCCAGPUTrackerNVCC& );
	AliHLTTPCCAGPUTrackerNVCC &operator=( const AliHLTTPCCAGPUTrackerNVCC& );

	ClassDef( AliHLTTPCCAGPUTrackerNVCC, 0 )
};

#ifdef R__WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" DLL_EXPORT AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate();
extern "C" DLL_EXPORT void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr);

#endif //ALIHLTTPCCAGPUTRACKER_H
