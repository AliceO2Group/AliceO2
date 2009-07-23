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

private:
	AliHLTTPCCATracker gpuTracker;
	void* GPUMemory;

	int CUDASync();
	template <class T> T* alignPointer(T* ptr, int alignment);

	int DebugLevel;
	std::ostream *OutFile;
	int GPUMemSize;
#ifdef HLTCA_GPUCODE
	bool CUDA_FAILED_MSG(cudaError_t error);
#endif
};