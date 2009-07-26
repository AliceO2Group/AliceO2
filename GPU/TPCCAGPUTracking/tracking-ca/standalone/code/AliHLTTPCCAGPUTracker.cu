// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//					David Rohr <drohr@kip.uni-heidelberg.de>				*
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include <cutil.h>
#include <cutil_inline_runtime.h>
#include <sm_11_atomic_functions.h>
#include <sm_12_atomic_functions.h>

#include <iostream>

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#include "AliHLTTPCCAGPUTracker.h"

#ifdef BUILD_GPU

//Include CXX Files, GPUd() macro will then produce CUDA device code out of the tracker source code
#include "AliHLTTPCCATrackParam.cxx"
#include "AliHLTTPCCATrack.cxx" 

#include "AliHLTTPCCATrackletSelector.cxx"

#include "AliHLTTPCCAHitArea.cxx"
#include "AliHLTTPCCAGrid.cxx"
#include "AliHLTTPCCARow.cxx"
#include "AliHLTTPCCAParam.cxx"
#include "AliHLTTPCCATracker.cxx"

#include "AliHLTTPCCAOutTrack.cxx"

#include "AliHLTTPCCAProcess.h"

#include "AliHLTTPCCANeighboursFinder.cxx"

#include "AliHLTTPCCANeighboursCleaner.cxx"
#include "AliHLTTPCCAStartHitsFinder.cxx"
#include "AliHLTTPCCATrackletConstructor.cxx"
#include "AliHLTTPCCASliceOutput.cxx"

#endif

AliHLTTPCCAGPUTracker::AliHLTTPCCAGPUTracker() : gpuTracker(), fDebugLevel(0) {}
AliHLTTPCCAGPUTracker::~AliHLTTPCCAGPUTracker() {}

//Find best CUDA device, initialize and allocate memory
int AliHLTTPCCAGPUTracker::InitGPU()
{
#ifdef BUILD_GPU
	int cudaDevice = cutGetMaxGflopsDeviceId();
	cudaSetDevice(cudaDevice);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop ,0 ); 
  std::cout<<"CUDA Device Properties: "<<std::endl;
  std::cout<<"name = "<<prop.name<<std::endl;
  std::cout<<"totalGlobalMem = "<<prop.totalGlobalMem<<std::endl;
  std::cout<<"sharedMemPerBlock = "<<prop.sharedMemPerBlock<<std::endl;
  std::cout<<"regsPerBlock = "<<prop.regsPerBlock<<std::endl;
  std::cout<<"warpSize = "<<prop.warpSize<<std::endl;
  std::cout<<"memPitch = "<<prop.memPitch<<std::endl;
  std::cout<<"maxThreadsPerBlock = "<<prop.maxThreadsPerBlock<<std::endl;
  std::cout<<"maxThreadsDim = "<<prop.maxThreadsDim[0]<<" "<<prop.maxThreadsDim[1]<<" "<<prop.maxThreadsDim[2]<<std::endl;
  std::cout<<"maxGridSize = "  <<prop.maxGridSize[0]<<" "<<prop.maxGridSize[1]<<" "<<prop.maxGridSize[2]<<std::endl;
  std::cout<<"totalConstMem = "<<prop.totalConstMem<<std::endl;
  std::cout<<"major = "<<prop.major<<std::endl;
  std::cout<<"minor = "<<prop.minor<<std::endl;
  std::cout<<"clockRate = "<<prop.clockRate<<std::endl;
  std::cout<<"textureAlignment = "<<prop.textureAlignment<<std::endl;

  fGPUMemSize = (int) prop.totalGlobalMem - 400 * 1024 * 1024;
  if (CUDA_FAILED_MSG(cudaMalloc(&GPUMemory, (size_t) fGPUMemSize)))
  {
	  std::cout << "CUDA Memory Allocation Error\n";
	  return(1);
  }
  std::cout << "CUDA Initialisation successfull\n";
#endif

	return(0);
}

//Macro to align Pointers.
//Will align to start at 1 MB segments, this should be consistent with every alignment in the tracker
//(As long as every single data structure is <= 1 MB)
template <class T> inline T* AliHLTTPCCAGPUTracker::alignPointer(T* ptr, int alignment)
{
	size_t adr = (size_t) ptr;
	if (adr % alignment)
	{
		adr += alignment - (adr % alignment);
	}
	return((T*) adr);
}

//Check for CUDA Error and in the case of an error display the corresponding error string
bool AliHLTTPCCAGPUTracker::CUDA_FAILED_MSG(cudaError_t error)
{
	if (error == cudaSuccess) return(false);
	printf("CUDA Error: %d / %s\n", error, cudaGetErrorString(error));
	return(true);
}

//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards
int AliHLTTPCCAGPUTracker::CUDASync()
{
	if (fDebugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		printf("Cuda Error %s while invoking kernel\n", cudaGetErrorString(cuErr));
		return(1);
	}
	if (CUDA_FAILED_MSG(cudaThreadSynchronize()))
	{
		printf("CUDA Error while synchronizing\n");
		return(1);
	}
	if (fDebugLevel >= 4) printf("CUDA Sync Done\n");
	return(0);
}

void AliHLTTPCCAGPUTracker::SetDebugLevel(int dwLevel, std::ostream *NewOutFile)
{
	fDebugLevel = dwLevel;
	if (NewOutFile) fOutFile = NewOutFile;
}

int AliHLTTPCCAGPUTracker::SetGPUTrackerOption(char* OptionName, int OptionValue)
{
	if (strcmp(OptionName, "SingleBlock") == 0)
	{
		fOptionSingleBlock = OptionValue;
	}
	else
	{
		printf("Unknown Option: %s\n", OptionName);
		return(1);
	}
	return(0);
}

//Primary reconstruction function
int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCATracker* tracker)
{
    int nThreads;
    int nBlocks;
	int size;

	if (tracker->CheckEmptySlice())
	{
		if (fDebugLevel >= 4) printf("Slice Empty, not running GPU Tracker\n");
		return(0);
	}

	if (fDebugLevel >= 3)
	{
		*fOutFile << endl << endl << "Slice: " << tracker->Param().ISlice() << endl;
	}

	if (fDebugLevel >= 4) printf("\n\nInitialising GPU Tracker\n");
	memcpy(&gpuTracker, tracker, sizeof(AliHLTTPCCATracker));
	char* tmpMem = alignPointer((char*) GPUMemory, 1024 * 1024);
	gpuTracker.SetGPUTracker();

	if (fDebugLevel >= 4) printf("Initialising GPU Common Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerCommonMemory(tmpMem);
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (fDebugLevel >= 4) printf("Initialising GPU Hits Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerHitsMemory(tmpMem, tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (fDebugLevel >= 4) printf("Initialising GPU Slice Data Memory\n");
	tmpMem = gpuTracker.fData.SetGPUSliceDataMemory(tmpMem, gpuTracker.fClusterData);
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) GPUMemory > fGPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}
	
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fCommonMemory, tracker->fCommonMemory, tracker->fCommonMemorySize, cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fData.fMemory, tracker->fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &gpuTracker, sizeof(AliHLTTPCCATracker)));

	if (fDebugLevel >= 4) printf("Running GPU Neighbours Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<gpuTracker.Param().NRows(), 256>>>();
	if (CUDASync()) return 1;

	if (fDebugLevel >= 3)
	{
		*fOutFile << "Neighbours Finder:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fData.fMemory, gpuTracker.fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*fOutFile);
    }

	if (fDebugLevel >= 4) printf("Running GPU Neighbours Cleaner\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<gpuTracker.Param().NRows()-2, 256>>>();
	if (CUDASync()) return 1;

	if (fDebugLevel >= 3)
	{
		*fOutFile << "Neighbours Cleaner:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fData.fMemory, gpuTracker.fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*fOutFile);
    }

	if (fDebugLevel >= 4) printf("Running GPU Start Hits Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<gpuTracker.Param().NRows()-4, 256>>>();
	if (CUDASync()) return 1;

	if (fDebugLevel >= 4) printf("Obtaining Number of Start Hits from GPU: ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->fCommonMemory, gpuTracker.fCommonMemory, tracker->fCommonMemorySize, cudaMemcpyDeviceToHost));
	if (fDebugLevel >= 4) printf("%d\n", *tracker->NTracklets());
	else if (fDebugLevel >= 2) printf("%3d ", *tracker->NTracklets());

	if (fDebugLevel >= 3)
	{
		*fOutFile << "Start Hits: (" << *tracker->NTracklets() << ")" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fHitMemory, gpuTracker.fHitMemory, tracker->fHitMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpStartHits(*fOutFile);
    }

	/*tracker->RunNeighboursFinder();
	tracker->RunNeighboursCleaner();
	tracker->RunStartHitsFinder();*/

	if (fDebugLevel >= 4) printf("Initialising GPU Track Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerTracksMemory(tmpMem, *tracker->NTracklets(), tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) GPUMemory > fGPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}

	tracker->fData.ClearHitWeights();
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fData.fHitWeights, tracker->fData.fHitWeights, tracker->fData.fNumberOfHits * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &gpuTracker, sizeof(AliHLTTPCCATracker)));

	if (fDebugLevel >= 4) printf("Initialising Slice Tracker (CPU) Track Memory\n");
	tracker->fTrackMemory = reinterpret_cast<char*> ( new uint4 [ gpuTracker.fTrackMemorySize/sizeof( uint4 ) + 100] );
    tracker->SetPointersTracks( *tracker->NTracklets(), tracker->NHitsTotal() );

/*	tracker->RunTrackletConstructor();
	if (fDebugLevel >= 3)
	{
		*fOutFile << "Tracklet Hits:" << endl;
		tracker->DumpTrackletHits(*fOutFile);
	}*/

	int nMemThreads = TRACKLET_CONSTRUCTOR_NMEMTHREDS;
    nThreads = 256;//96;
    nBlocks = *tracker->NTracklets()/nThreads + 1;
    if( nBlocks<30 ){
		nBlocks = 30;
		nThreads = (*tracker->NTracklets())/nBlocks+1;
		if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
	}
	if (fDebugLevel >= 4) printf("Running GPU Tracklet Constructor\n");

	if (!fOptionSingleBlock)
	{
		AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<nBlocks, nMemThreads+nThreads>>>(); 
	}
	else
	{
		AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<1, TRACKLET_CONSTRUCTOR_NMEMTHREDS + *tracker->fNTracklets>>>();
	}
	if (CUDASync()) return 1;

	if (fDebugLevel >= 3)
	{
		*fOutFile << "Tracklet Hits:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fNTracklets, gpuTracker.fNTracklets, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fTracklets, gpuTracker.fTracklets, gpuTracker.fTrackMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpTrackletHits(*fOutFile);
    }

	//tracker->RunTrackletSelector();
	

	nThreads = 128;
	nBlocks = *tracker->NTracklets()/nThreads + 1;
	if( nBlocks<30 ){
	  nBlocks = 30;  
	  nThreads = *tracker->NTracklets()/nBlocks+1;
	  nThreads = (nThreads/32+1)*32;
	}
	if (fDebugLevel >= 4) printf("Running GPU Tracklet Selector\n");
	if (!fOptionSingleBlock)
	{
		AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<nBlocks, nThreads>>>();
	}
	else
	{
		AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<1, *tracker->fNTracklets>>>();
	}
	if (CUDASync()) return 1;

	if (fDebugLevel >= 4) printf("Transfering Tracks from GPU to Host ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->NTracks(), gpuTracker.NTracks(), sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_FAILED_MSG(cudaMemcpy(tracker->NTrackHits(), gpuTracker.NTrackHits(), sizeof(int), cudaMemcpyDeviceToHost));
	if (fDebugLevel >= 4) printf("%d / %d\n", *tracker->fNTracks, *tracker->fNTrackHits);
	size = sizeof(AliHLTTPCCATrack) * *tracker->NTracks();
	CUDA_FAILED_MSG(cudaMemcpy(tracker->Tracks(), gpuTracker.Tracks(), size, cudaMemcpyDeviceToHost));
	size = sizeof(AliHLTTPCCAHitId) * *tracker->NTrackHits();
	if (CUDA_FAILED_MSG(cudaMemcpy(tracker->TrackHits(), gpuTracker.TrackHits(), size, cudaMemcpyDeviceToHost)))
	{
		printf("CUDA Error during Reconstruction\n");
		return(1);
	}

	if (fDebugLevel >= 3)
	{
		*fOutFile << "Track Hits: (" << *tracker->NTracks() << ")" << endl;
		tracker->DumpTrackHits(*fOutFile);
    }

	if (fDebugLevel >= 4) printf("Running WriteOutput\n");
	tracker->WriteOutput();

	if (fDebugLevel >= 4) printf("GPU Reconstruction finished\n");
	
	return(0);
}

int AliHLTTPCCAGPUTracker::ExitGPU()
{
	cudaFree(GPUMemory);
	return(0);
}
