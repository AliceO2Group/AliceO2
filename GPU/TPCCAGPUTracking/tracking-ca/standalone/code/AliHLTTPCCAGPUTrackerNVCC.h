// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAGPUTRACKERNVCC_H
#define ALIHLTTPCCAGPUTRACKERNVCC_H

#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTLogging.h"
#include "AliHLTTPCCASliceOutput.h"

class AliHLTTPCCARow;

class AliHLTTPCCAGPUTrackerNVCC : public AliHLTTPCCAGPUTracker, public AliHLTLogging
{
public:
	AliHLTTPCCAGPUTrackerNVCC();
	~AliHLTTPCCAGPUTrackerNVCC();

	virtual int InitGPU(int sliceCount = 12, int forceDeviceID = -1);
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU();

	virtual void SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile = NULL);
	virtual int SetGPUTrackerOption(char* OptionName, int OptionValue);

	virtual unsigned long long int* PerfTimer(int iSlice, unsigned int i);

	virtual int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);
	virtual void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);

	virtual const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const;
	virtual int GetSliceCount() const;

private:
	static void* RowMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (HLTCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + iSlice * HLTCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_GLOBAL_MEMORY ); }
	void* TracksMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_TRACKS_MEMORY ); }
	void* TrackerMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }

	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);
	int GetThread();
	void ReleaseGlobalLock(void* sem);
	int CheckMemorySizes(int sliceCount);

	AliHLTTPCCATracker *fGpuTracker;
	void* fGPUMemory;
	void* fHostLockedMemory;

	int CUDASync(char* state = "UNKNOWN");
	template <class T> T* alignPointer(T* ptr, int alignment);

	void StandalonePerfTime(int iSlice, int i);

	int fDebugLevel;			//Debug Level for GPU Tracker
	std::ostream* fOutFile;		//Debug Output Stream Pointer
	unsigned long long int fGPUMemSize;	//Memory Size to allocate on GPU

	void* fpCudaStreams;

	int fSliceCount;

	static const int fgkNSlices = 36;
	AliHLTTPCCATracker fSlaveTrackers[fgkNSlices];
#ifdef HLTCA_GPUCODE
	bool CudaFailedMsg(cudaError_t error);
#endif //HLTCA_GPUCODE

	AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl;
	
	static bool fgGPUUsed;
	int fThreadId;
	int fCudaInitialized;

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
