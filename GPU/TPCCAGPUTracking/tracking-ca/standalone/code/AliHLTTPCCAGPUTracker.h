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
	AliHLTTPCCAGPUTracker();
	~AliHLTTPCCAGPUTracker();

	int InitGPU();
	int Reconstruct(AliHLTTPCCATracker* tracker);
	int ExitGPU();

	void SetDebugLevel(int dwLevel, std::ostream *NewOutFile = NULL);
	int SetGPUTrackerOption(char* OptionName, int OptionValue);

private:
	AliHLTTPCCATracker gpuTracker;
	void* GPUMemory;

	int CUDASync();
	template <class T> T* alignPointer(T* ptr, int alignment);

	int fDebugLevel;			//Debug Level for GPU Tracker
	std::ostream *fOutFile;		//Debug Output Stream Pointer
	int fGPUMemSize;			//Memory Size to allocate on GPU

	int fOptionSingleBlock;		//Use only one single Multiprocessor on GPU to check for problems related to multi processing
#ifdef HLTCA_GPUCODE
	bool CUDA_FAILED_MSG(cudaError_t error);
#endif
	// disable copy
	AliHLTTPCCAGPUTracker( const AliHLTTPCCAGPUTracker& );
	AliHLTTPCCAGPUTracker &operator=( const AliHLTTPCCAGPUTracker& );

};
