// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAGPUTRACKER_H
#define ALIHLTTPCCAGPUTRACKER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTLogging.h"
#include "AliHLTTPCCASliceOutput.h"

class AliHLTTPCCARow;

class AliHLTTPCCAGPUTracker : AliHLTLogging
{
public:
	AliHLTTPCCAGPUTracker() :
	  fGpuTracker(NULL),
	  fGPUMemory(NULL),
	  fHostLockedMemory(NULL),
	  fDebugLevel(0),
	  fOutFile(NULL),
	  fGPUMemSize(0),
	  fpCudaStreams(NULL),
	  fSliceCount(0),
	  fOutputControl(NULL),
	  fThreadId(0),
	  fCudaInitialized(0)
	  {};
	  ~AliHLTTPCCAGPUTracker() {};

	int InitGPU(int sliceCount = 12, int forceDeviceID = -1);
	int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	int ExitGPU();

	void SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile = NULL);
	int SetGPUTrackerOption(char* OptionName, int OptionValue);

	unsigned long long int* PerfTimer(int iSlice, unsigned int i) {return(fSlaveTrackers ? fSlaveTrackers[iSlice].PerfTimer(i) : NULL); }

	int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);

	const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const { return fOutputControl; }
	void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);
	
	int GetSliceCount() const { return(fSliceCount); }

private:
	static void* RowMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (HLTCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* const BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + iSlice * HLTCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_GLOBAL_MEMORY ); }
	void* TracksMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_TRACKS_MEMORY ); }
	void* TrackerMemory(void* const BaseMemory, int iSlice) const { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }

	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);
	int GetThread();

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
#endif

	AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl;
	
	static bool fgGPUUsed;
	int fThreadId;
	int fCudaInitialized;

	// disable copy
	AliHLTTPCCAGPUTracker( const AliHLTTPCCAGPUTracker& );
	AliHLTTPCCAGPUTracker &operator=( const AliHLTTPCCAGPUTracker& );

	ClassDef( AliHLTTPCCAGPUTracker, 0 )
};

#endif
