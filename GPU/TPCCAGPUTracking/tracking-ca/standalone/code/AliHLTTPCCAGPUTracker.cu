#include <cutil.h>
#include <cutil_inline_runtime.h>
#include <sm_11_atomic_functions.h>
#include <sm_12_atomic_functions.h>

#include <iostream>

#ifdef assert
#undef assert
#endif
#define assert(param)

#include "AliHLTTPCCAGPUTracker.h"

#ifdef BUILD_GPU

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

AliHLTTPCCAGPUTracker::AliHLTTPCCAGPUTracker() : gpuTracker(), DebugLevel(0)
{
}

AliHLTTPCCAGPUTracker::~AliHLTTPCCAGPUTracker()
{
}

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

  GPUMemSize = (int) prop.totalGlobalMem - 400 * 1024 * 1024;
  if (CUDA_FAILED_MSG(cudaMalloc(&GPUMemory, (size_t) GPUMemSize)))
  {
	  std::cout << "CUDA Memory Allocation Error\n";
	  return(1);
  }
  std::cout << "CUDA Initialisation successfull\n";
#endif

	return(0);
}

template <class T> inline T* AliHLTTPCCAGPUTracker::alignPointer(T* ptr, int alignment)
{
	size_t adr = (size_t) ptr;
	if (adr % alignment)
	{
		adr += alignment - (adr % alignment);
	}
	return((T*) adr);
}

bool AliHLTTPCCAGPUTracker::CUDA_FAILED_MSG(cudaError_t error)
{
	if (error == cudaSuccess) return(false);
	printf("CUDA Error: %d / %s\n", error, cudaGetErrorString(error));
	return(true);
}

int AliHLTTPCCAGPUTracker::CUDASync()
{
	if (DebugLevel == 0) return(0);
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
	if (DebugLevel >= 4) printf("CUDA Sync Done\n");
	return(0);
}

void AliHLTTPCCAGPUTracker::SetDebugLevel(int dwLevel, std::ostream *NewOutFile)
{
	DebugLevel = dwLevel;
	if (NewOutFile) OutFile = NewOutFile;
}

int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCATracker* tracker)
{
    int nThreads;
    int nBlocks;
	int size;

	if (tracker->CheckEmptySlice())
	{
		if (DebugLevel >= 4) printf("Slice Empty, not running GPU Tracker\n");
		return(0);
	}

	if (DebugLevel >= 3)
	{
		*OutFile << endl << endl << "Slice: " << tracker->Param().ISlice() << endl;
	}

	if (DebugLevel >= 4) printf("\n\nInitialising GPU Tracker\n");
	memcpy(&gpuTracker, tracker, sizeof(AliHLTTPCCATracker));
	char* tmpMem = alignPointer((char*) GPUMemory, 1024 * 1024);
	gpuTracker.SetGPUTracker();

	if (DebugLevel >= 4) printf("Initialising GPU Common Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerCommonMemory(tmpMem);
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (DebugLevel >= 4) printf("Initialising GPU Hits Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerHitsMemory(tmpMem, tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (DebugLevel >= 4) printf("Initialising GPU Slice Data Memory\n");
	tmpMem = gpuTracker.fData.SetGPUSliceDataMemory(tmpMem, gpuTracker.fClusterData);
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) GPUMemory > GPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}
	
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fCommonMemory, tracker->fCommonMemory, tracker->fCommonMemorySize, cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fData.fMemory, tracker->fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &gpuTracker, sizeof(AliHLTTPCCATracker)));

	if (DebugLevel >= 4) printf("Running GPU Neighbours Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<gpuTracker.Param().NRows(), 256>>>();
	if (CUDASync()) return 1;

	if (DebugLevel >= 3)
	{
		*OutFile << "Neighbours Finder:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fData.fMemory, gpuTracker.fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*OutFile);
    }

	if (DebugLevel >= 4) printf("Running GPU Neighbours Cleaner\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<gpuTracker.Param().NRows()-2, 256>>>();
	if (CUDASync()) return 1;

	if (DebugLevel >= 3)
	{
		*OutFile << "Neighbours Cleaner:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fData.fMemory, gpuTracker.fData.fMemory, tracker->fData.fMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*OutFile);
    }

	if (DebugLevel >= 4) printf("Running GPU Start Hits Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<gpuTracker.Param().NRows()-4, 256>>>();
	if (CUDASync()) return 1;

	if (DebugLevel >= 4) printf("Obtaining Number of Start Hits from GPU: ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->fCommonMemory, gpuTracker.fCommonMemory, tracker->fCommonMemorySize, cudaMemcpyDeviceToHost));
	if (DebugLevel >= 4) printf("%d\n", *tracker->NTracklets());
	else if (DebugLevel >= 2) printf("%3d ", *tracker->NTracklets());

	if (DebugLevel >= 3)
	{
		*OutFile << "Start Hits: (" << *tracker->NTracklets() << ")" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fHitMemory, gpuTracker.fHitMemory, tracker->fHitMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpStartHits(*OutFile);
    }

	/*tracker->RunNeighboursFinder();
	tracker->RunNeighboursCleaner();
	tracker->RunStartHitsFinder();*/

	if (DebugLevel >= 4) printf("Initialising GPU Track Memory\n");
	tmpMem = gpuTracker.SetGPUTrackerTracksMemory(tmpMem, *tracker->NTracklets(), tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) GPUMemory > GPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}

	tracker->fData.ClearHitWeights();
	CUDA_FAILED_MSG(cudaMemcpy(gpuTracker.fData.fHitWeights, tracker->fData.fHitWeights, tracker->fData.fNumberOfHits * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &gpuTracker, sizeof(AliHLTTPCCATracker)));

	if (DebugLevel >= 4) printf("Initialising Slice Tracker (CPU) Track Memory\n");
	tracker->fTrackMemory = reinterpret_cast<char*> ( new uint4 [ gpuTracker.fTrackMemorySize/sizeof( uint4 ) + 100] );
    tracker->SetPointersTracks( *tracker->NTracklets(), tracker->NHitsTotal() );

/*	tracker->RunTrackletConstructor();
	if (DebugLevel >= 3)
	{
		*OutFile << "Tracklet Hits:" << endl;
		tracker->DumpTrackletHits(*OutFile);
	}*/

	int nMemThreads = TRACKLET_CONSTRUCTOR_NMEMTHREDS;
    nThreads = 256;//96;
    nBlocks = *tracker->NTracklets()/nThreads + 1;
    if( nBlocks<30 ){
		nBlocks = 30;
		nThreads = (*tracker->NTracklets())/nBlocks+1;
		if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
	}
	if (DebugLevel >= 4) printf("Running GPU Tracklet Constructor\n");

	//AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<nBlocks, nMemThreads+nThreads>>>(); 
	AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<1, TRACKLET_CONSTRUCTOR_NMEMTHREDS + *tracker->fNTracklets>>>();
	if (CUDASync()) return 1;

	if (DebugLevel >= 3)
	{
		*OutFile << "Tracklet Hits:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fNTracklets, gpuTracker.fNTracklets, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker->fTracklets, gpuTracker.fTracklets, gpuTracker.fTrackMemorySize, cudaMemcpyDeviceToHost));
		tracker->DumpTrackletHits(*OutFile);
    }

	//tracker->RunTrackletSelector();
	

	nThreads = 128;
	nBlocks = *tracker->NTracklets()/nThreads + 1;
	if( nBlocks<30 ){
	  nBlocks = 30;  
	  nThreads = *tracker->NTracklets()/nBlocks+1;
	  nThreads = (nThreads/32+1)*32;
	}
	if (DebugLevel >= 4) printf("Running GPU Tracklet Selector\n");
	AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<nBlocks, nThreads>>>();
	//AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<1, *tracker->fNTracklets>>>();
	if (CUDASync()) return 1;

	if (DebugLevel >= 4) printf("Transfering Tracks from GPU to Host ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->NTracks(), gpuTracker.NTracks(), sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_FAILED_MSG(cudaMemcpy(tracker->NTrackHits(), gpuTracker.NTrackHits(), sizeof(int), cudaMemcpyDeviceToHost));
	if (DebugLevel >= 4) printf("%d / %d\n", *tracker->fNTracks, *tracker->fNTrackHits);
	size = sizeof(AliHLTTPCCATrack) * *tracker->NTracks();
	CUDA_FAILED_MSG(cudaMemcpy(tracker->Tracks(), gpuTracker.Tracks(), size, cudaMemcpyDeviceToHost));
	size = sizeof(AliHLTTPCCAHitId) * *tracker->NTrackHits();
	if (CUDA_FAILED_MSG(cudaMemcpy(tracker->TrackHits(), gpuTracker.TrackHits(), size, cudaMemcpyDeviceToHost)))
	{
		printf("CUDA Error during Reconstruction\n");
		return(1);
	}

	if (DebugLevel >= 3)
	{
		*OutFile << "Track Hits: (" << *tracker->NTracks() << ")" << endl;
		tracker->DumpTrackHits(*OutFile);
    }

	if (DebugLevel >= 4) printf("Running WriteOutput\n");
	tracker->WriteOutput();

	if (DebugLevel >= 4) printf("GPU Reconstruction finished\n");
	
	return(0);
}

int AliHLTTPCCAGPUTracker::ExitGPU()
{
	cudaFree(GPUMemory);
	return(0);
}
