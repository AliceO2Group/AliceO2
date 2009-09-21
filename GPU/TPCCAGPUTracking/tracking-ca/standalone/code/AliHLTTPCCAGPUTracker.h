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
#include "AliHLTTPCCARow.h"

class AliHLTTPCCAGPUTracker
{
public:
	AliHLTTPCCAGPUTracker() :
	  fGpuTracker(NULL),
	  fGPUMemory(NULL),
	  fHostLockedMemory(NULL),
	  fDebugLevel(0),
	  fOutFile(NULL),
	  fGPUMemSize(0),
	  fOptionSimpleSched(0),
	  pCudaStreams(NULL),
	  fSliceCount(0)
	  {};
	  ~AliHLTTPCCAGPUTracker() {};

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1);
	int Reconstruct(AliHLTTPCCASliceOutput* pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	int ExitGPU();

	void SetDebugLevel(int dwLevel, std::ostream *NewOutFile = NULL);
	int SetGPUTrackerOption(char* OptionName, int OptionValue);

	unsigned long long int* PerfTimer(int iSlice, unsigned int i) {return(fSlaveTrackers ? fSlaveTrackers[iSlice].PerfTimer(i) : NULL); }

	int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);

private:
	static void* RowMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + iSlice * sizeof(AliHLTTPCCARow) * (HLTCA_ROW_COUNT + 1) ); }
	static void* CommonMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + iSlice * AliHLTTPCCATracker::CommonMemorySize() ); }
	static void* SliceDataMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + iSlice * HLTCA_GPU_SLICE_DATA_MEMORY ); }
	void* GlobalMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_GLOBAL_MEMORY ); }
	void* TracksMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY) + iSlice * HLTCA_GPU_TRACKS_MEMORY ); }
	void* TrackerMemory(void* BaseMemory, int iSlice) { return( ((char*) BaseMemory) + HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + fSliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + iSlice * sizeof(AliHLTTPCCATracker) ); }

	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);

	AliHLTTPCCATracker *fGpuTracker;
	void* fGPUMemory;
	void* fHostLockedMemory;

	int CUDASync(char* state = "UNKNOWN");
	template <class T> T* alignPointer(T* ptr, int alignment);

	void StandalonePerfTime(int iSlice, int i);

	int fDebugLevel;			//Debug Level for GPU Tracker
	std::ostream *fOutFile;		//Debug Output Stream Pointer
	unsigned long long int fGPUMemSize;	//Memory Size to allocate on GPU

	int fOptionSimpleSched;		//Simple scheduler not row based

	void* pCudaStreams;

	int fSliceCount;

	static const int fgkNSlices = 36;
	AliHLTTPCCATracker fSlaveTrackers[fgkNSlices];
#ifdef HLTCA_GPUCODE
	bool CUDA_FAILED_MSG(cudaError_t error);
#endif

	// disable copy
	AliHLTTPCCAGPUTracker( const AliHLTTPCCAGPUTracker& );
	AliHLTTPCCAGPUTracker &operator=( const AliHLTTPCCAGPUTracker& );

};

#endif