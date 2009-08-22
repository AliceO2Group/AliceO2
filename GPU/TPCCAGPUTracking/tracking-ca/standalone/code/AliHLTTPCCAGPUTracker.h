// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"

class AliHLTTPCCAGPUTracker
{
public:
	AliHLTTPCCAGPUTracker() :
	  fGpuTracker(NULL),
	  fGPUMemory(NULL),
	  fDebugLevel(0),
	  fOutFile(NULL),
	  fGPUMemSize(0),
	  fOptionSingleBlock(0),
	  fOptionAdaptiveSched(1),
	  fSliceCount(0)
	  {};
	  ~AliHLTTPCCAGPUTracker() {};

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1);
	int Reconstruct(AliHLTTPCCATracker* tracker, int fSliceCount = -1);
	int ExitGPU();

	void SetDebugLevel(int dwLevel, std::ostream *NewOutFile = NULL);
	int SetGPUTrackerOption(char* OptionName, int OptionValue);

	unsigned long long int* PerfTimer(unsigned int i) {return(fGpuTracker ? fGpuTracker[0].PerfTimer(i) : NULL); }

private:
	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);

	AliHLTTPCCATracker *fGpuTracker;
	void* fGPUMemory;

	int CUDASync(char* state = "UNKNOWN");
	template <class T> T* alignPointer(T* ptr, int alignment);

	void StandalonePerfTime(int i);

	int fDebugLevel;			//Debug Level for GPU Tracker
	std::ostream *fOutFile;		//Debug Output Stream Pointer
	long long int fGPUMemSize;	//Memory Size to allocate on GPU

	int fOptionSingleBlock;		//Use only one single Multiprocessor on GPU to check for problems related to multi processing
	int fOptionAdaptiveSched;	//Adaptive shedule track/thread assignments

	int fSliceCount;
#ifdef HLTCA_GPUCODE
	bool CUDA_FAILED_MSG(cudaError_t error);
#endif

	// disable copy
	AliHLTTPCCAGPUTracker( const AliHLTTPCCAGPUTracker& );
	AliHLTTPCCAGPUTracker &operator=( const AliHLTTPCCAGPUTracker& );

};
