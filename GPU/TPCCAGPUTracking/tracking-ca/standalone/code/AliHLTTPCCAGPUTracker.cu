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

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"

//#include <cutil.h>
#ifndef CUDA_DEVICE_EMULATION
//#include <cutil_inline_runtime.h>
#else
//#include <cutil_inline.h>
#endif
#include <sm_11_atomic_functions.h>
#include <sm_12_atomic_functions.h>

#include <iostream>

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#include "AliHLTTPCCAGPUTracker.h"

#include "AliHLTTPCCAHit.h"

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
#include "AliHLTTPCCAStartHitsSorter.cxx"
#include "AliHLTTPCCATrackletConstructor.cxx"
#include "AliHLTTPCCASliceOutput.cxx"

#include "MemoryAssignmentHelpers.h"

//Find best CUDA device, initialize and allocate memory
int AliHLTTPCCAGPUTracker::InitGPU(int forceDeviceID)
{
	cudaDeviceProp fCudaDeviceProp;


	int count, bestDevice, bestDeviceSpeed = 0;
	cudaGetDeviceCount(&count);
	if (fDebugLevel >= 2) std::cout << "Available CUDA devices: ";
	for (int i = 0;i < count;i++)
	{
		cudaGetDeviceProperties(&fCudaDeviceProp, i);
		if (fDebugLevel >= 2) std::cout << fCudaDeviceProp.name << " (" << i << ")     ";
		if (fCudaDeviceProp.multiProcessorCount * fCudaDeviceProp.clockRate > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = fCudaDeviceProp.multiProcessorCount * fCudaDeviceProp.clockRate;
		}
	}
	if (fDebugLevel >= 2) std::cout << std::endl;

  int cudaDevice;
  if (forceDeviceID == -1)
	  cudaDevice = bestDevice;
  else
	  cudaDevice = forceDeviceID;
  cudaSetDevice(cudaDevice);

  cudaGetDeviceProperties(&fCudaDeviceProp ,cudaDevice ); 

  if (fDebugLevel >= 1)
  {
	  std::cout<<"CUDA Device Properties: "<<std::endl;
	  std::cout<<"name = "<<fCudaDeviceProp.name<<std::endl;
	  std::cout<<"totalGlobalMem = "<<fCudaDeviceProp.totalGlobalMem<<std::endl;
	  std::cout<<"sharedMemPerBlock = "<<fCudaDeviceProp.sharedMemPerBlock<<std::endl;
	  std::cout<<"regsPerBlock = "<<fCudaDeviceProp.regsPerBlock<<std::endl;
	  std::cout<<"warpSize = "<<fCudaDeviceProp.warpSize<<std::endl;
	  std::cout<<"memPitch = "<<fCudaDeviceProp.memPitch<<std::endl;
	  std::cout<<"maxThreadsPerBlock = "<<fCudaDeviceProp.maxThreadsPerBlock<<std::endl;
	  std::cout<<"maxThreadsDim = "<<fCudaDeviceProp.maxThreadsDim[0]<<" "<<fCudaDeviceProp.maxThreadsDim[1]<<" "<<fCudaDeviceProp.maxThreadsDim[2]<<std::endl;
	  std::cout<<"maxGridSize = "  <<fCudaDeviceProp.maxGridSize[0]<<" "<<fCudaDeviceProp.maxGridSize[1]<<" "<<fCudaDeviceProp.maxGridSize[2]<<std::endl;
	  std::cout<<"totalConstMem = "<<fCudaDeviceProp.totalConstMem<<std::endl;
	  std::cout<<"major = "<<fCudaDeviceProp.major<<std::endl;
	  std::cout<<"minor = "<<fCudaDeviceProp.minor<<std::endl;
	  std::cout<<"clockRate = "<<fCudaDeviceProp.clockRate<<std::endl;
	  std::cout<<"textureAlignment = "<<fCudaDeviceProp.textureAlignment<<std::endl;
  }

  if (fCudaDeviceProp.major < 1 || (fCudaDeviceProp.major == 1 && fCudaDeviceProp.minor < 2))
  {
	  std::cout << "Unsupported CUDA Device\n";
	  return(1);
  }

  fGPUMemSize = (long long int) fCudaDeviceProp.totalGlobalMem - 400 * 1024 * 1024;
  if (fGPUMemSize > 1024 * 1024 * 1024) fGPUMemSize = 1024 * 1024 * 1024;
  if (CUDA_FAILED_MSG(cudaMalloc(&fGPUMemory, (size_t) fGPUMemSize)))
  {
	  std::cout << "CUDA Memory Allocation Error\n";
	  return(1);
  }
  if (fDebugLevel >= 1)
  {
	  CUDA_FAILED_MSG(cudaMemset(fGPUMemory, 255, (size_t) fGPUMemSize));
  }
  std::cout << "CUDA Initialisation successfull\n";

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
int AliHLTTPCCAGPUTracker::CUDASync(char* state)
{
	if (fDebugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		printf("Cuda Error %s while invoking kernel (%s)\n", cudaGetErrorString(cuErr), state);
		return(1);
	}
	if (CUDA_FAILED_MSG(cudaThreadSynchronize()))
	{
		printf("CUDA Error while synchronizing (%s)\n", state);
		return(1);
	}
	if (fDebugLevel >= 5) printf("CUDA Sync Done\n");
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
	else if (strcmp(OptionName, "AdaptSched") == 0)
	{
		fOptionAdaptiveSched = OptionValue;
	}
	else
	{
		printf("Unknown Option: %s\n", OptionName);
		return(1);
	}
	return(0);
}

void AliHLTTPCCAGPUTracker::StandalonePerfTime(int i)
{
#ifdef HLTCA_STANDALONE
  if (fDebugLevel >= 1)
  {
	  fGpuTracker.StandaloneQueryTime( fGpuTracker.PerfTimer(i));
  }
#endif
}

void AliHLTTPCCAGPUTracker::DumpRowBlocks(AliHLTTPCCATracker* tracker, bool check)
{
	if (fDebugLevel >= 4)
	{
		*fOutFile << "RowBlock Tracklets" << std::endl;
	
		int3* RowBlockPos = (int3*) malloc(sizeof(int3) * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2);
		int* RowBlockTracklets = (int*) malloc(sizeof(int) * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2);
		uint2* BlockStartingTracklet = (uint2*) malloc(sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT);
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockPos, fGpuTracker.RowBlockPos(), sizeof(int3) * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockTracklets, fGpuTracker.RowBlockTracklets(), sizeof(int) * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(BlockStartingTracklet, fGpuTracker.BlockStartingTracklet(), sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker->CommonMemory(), fGpuTracker.CommonMemory(), tracker->CommonMemorySize(), cudaMemcpyDeviceToHost));

		int k = tracker->GPUParameters()->fScheduleFirstDynamicTracklet;
		for (int i = 0; i < tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1;i++)
		{
			*fOutFile << "Rowblock: " << i << ", up " << RowBlockPos[i].y << "/" << RowBlockPos[i].x << ", down " << 
				RowBlockPos[tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].y << "/" << RowBlockPos[tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x << endl << "Phase 1: ";
			for (int j = 0;j < RowBlockPos[i].x;j++)
			{
				//Use Tracker Object to calculate Offset instead of fGpuTracker, since *fNTracklets of fGpuTracker points to GPU Mem!
				*fOutFile << RowBlockTracklets[(tracker->RowBlockTracklets(0, i) - tracker->RowBlockTracklets(0, 0)) + j] << ", ";
				if (check && RowBlockTracklets[(tracker->RowBlockTracklets(0, i) - tracker->RowBlockTracklets(0, 0)) + j] != k)
				{
					printf("Wrong starting Row Block %d, entry %d, is %d, should be %d\n", i, j, RowBlockTracklets[(tracker->RowBlockTracklets(0, i) - tracker->RowBlockTracklets(0, 0)) + j], k);
				}
				k++;
				if (RowBlockTracklets[(tracker->RowBlockTracklets(0, i) - tracker->RowBlockTracklets(0, 0)) + j] == -1)
				{
					printf("Error, -1 Tracklet found\n");
				}
			}
			*fOutFile << endl << "Phase 2: ";
			for (int j = 0;j < RowBlockPos[tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x;j++)
			{
				*fOutFile << RowBlockTracklets[(tracker->RowBlockTracklets(1, i) - tracker->RowBlockTracklets(0, 0)) + j] << ", ";
			}
			*fOutFile << endl;
		}

		if (check)
		{
			*fOutFile << "Starting Threads: (First Dynamic: " << tracker->GPUParameters()->fScheduleFirstDynamicTracklet << ")" << std::endl;
			for (int i = 0;i < HLTCA_GPU_BLOCK_COUNT;i++)
			{
				*fOutFile << i << ": " << BlockStartingTracklet[i].x << " - " << BlockStartingTracklet[i].y << std::endl;
			}
		}

		free(RowBlockPos);
		free(RowBlockTracklets);
		free(BlockStartingTracklet);
	}
}

//Primary reconstruction function
int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCATracker* tracker)
{
    int nThreads;
    int nBlocks;
	int size;

	int cudaDevice;
	cudaDeviceProp fCudaDeviceProp;
	cudaGetDevice(&cudaDevice);
	cudaGetDeviceProperties(&fCudaDeviceProp, cudaDevice);
	
	if (tracker->CheckEmptySlice())
	{
		if (fDebugLevel >= 5) printf("Slice Empty, not running GPU Tracker\n");
		return(0);
	}

	if (fDebugLevel >= 4)
	{
		*fOutFile << endl << endl << "Slice: " << tracker->Param().ISlice() << endl;
	}

	if (fDebugLevel >= 5) printf("\n\nInitialising GPU Tracker\n");
	memcpy(&fGpuTracker, tracker, sizeof(AliHLTTPCCATracker));

	StandalonePerfTime(0);

	char* tmpMem = alignPointer((char*) fGPUMemory, 1024 * 1024);
	fGpuTracker.SetGPUTracker();

	if (fDebugLevel >= 5) printf("Initialising GPU Common Memory\n");
	tmpMem = fGpuTracker.SetGPUTrackerCommonMemory(tmpMem);
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (fDebugLevel >= 5) printf("Initialising GPU Hits Memory\n");
	tmpMem = fGpuTracker.SetGPUTrackerHitsMemory(tmpMem, tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);

	if (fDebugLevel >= 5) printf("Initialising GPU Slice Data Memory\n");
	tmpMem = fGpuTracker.SetGPUSliceDataMemory(tmpMem, fGpuTracker.ClusterData());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) fGPUMemory > fGPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}

#ifdef HLTCA_STANDALONE
	if (fDebugLevel >= 2)
	{
		if (CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker.fGPUDebugMem, 100 * 1024 * 1024)))
		{
			printf("Out of CUDA Memory\n");
			return(1);
		}
		CUDA_FAILED_MSG(cudaMemset(fGpuTracker.fGPUDebugMem, 0, 100 * 1024 * 1024));
	}
#endif

	if (fDebugLevel >= 5) printf("Initialising GPU Track Memory\n");
	tmpMem = fGpuTracker.SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS /**tracker->NTracklets()*/, tracker->NHitsTotal());
	tmpMem = alignPointer(tmpMem, 1024 * 1024);
	if (tmpMem - (char*) fGPUMemory > fGPUMemSize)
	{
		printf("Out of CUDA Memory\n");
		return(1);
	}

	if (fDebugLevel >= 1)
	{
		static int showMemInfo = true;
		if (showMemInfo)
			printf("GPU Memory used: %d bytes\n", (int) (tmpMem - (char*) fGPUMemory));
		showMemInfo = false;
	}

	*tracker->NTracklets() = 0;
	tracker->GPUParameters()->fStaticStartingTracklets = 1;
	tracker->GPUParameters()->fGPUError = 0;
	tracker->GPUParameters()->fGPUSchedCollisions = 0;

	CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker.CommonMemory(), tracker->CommonMemory(), tracker->CommonMemorySize(), cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker.SliceDataMemory(), tracker->SliceDataMemory(), tracker->SliceDataMemorySize(), cudaMemcpyHostToDevice));
	CUDA_FAILED_MSG(cudaMemset(fGpuTracker.RowBlockPos(), 0, sizeof(int3) * 2 * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1)));
	CUDA_FAILED_MSG(cudaMemset(fGpuTracker.RowBlockTracklets(), -1, sizeof(int) * (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &fGpuTracker, sizeof(AliHLTTPCCATracker)));

	StandalonePerfTime(1);

	if (fDebugLevel >= 5) printf("Running GPU Neighbours Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fGpuTracker.Param().NRows(), 256>>>();
	if (CUDASync("Neighbours finder")) return 1;

	StandalonePerfTime(2);

	if (fDebugLevel >= 4)
	{
		*fOutFile << "Neighbours Finder:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->SliceDataMemory(), fGpuTracker.SliceDataMemory(), tracker->SliceDataMemorySize(), cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*fOutFile);
    }

	if (fDebugLevel >= 5) printf("Running GPU Neighbours Cleaner\n");
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fGpuTracker.Param().NRows()-2, 256>>>();
	if (CUDASync("Neighbours Cleaner")) return 1;

	StandalonePerfTime(3);

	if (fDebugLevel >= 4)
	{
		*fOutFile << "Neighbours Cleaner:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->SliceDataMemory(), fGpuTracker.SliceDataMemory(), tracker->SliceDataMemorySize(), cudaMemcpyDeviceToHost));
		tracker->DumpLinks(*fOutFile);
    }

	if (fDebugLevel >= 5) printf("Running GPU Start Hits Finder\n");
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fGpuTracker.Param().NRows()-4, 256>>>();
	if (CUDASync("Start Hits Finder")) return 1;

	StandalonePerfTime(4);

#ifdef HLTCA_GPU_SORT_STARTHITS
	if (fDebugLevel >= 5) printf("Running GPU Start Hits Sorter\n");
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<30, 256>>>();
	if (CUDASync("Start Hits Sorter")) return 1;
#endif

	StandalonePerfTime(5);

	if (fDebugLevel >= 5) printf("Obtaining Number of Start Hits from GPU: ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->CommonMemory(), fGpuTracker.CommonMemory(), tracker->CommonMemorySize(), cudaMemcpyDeviceToHost));
	if (fDebugLevel >= 5) printf("%d\n", *tracker->NTracklets());
	else if (fDebugLevel >= 2) printf("%3d ", *tracker->NTracklets());

#ifdef HLTCA_GPU_SORT_STARTHITS
	if (fDebugLevel >= 4)
	{
		*fOutFile << "Start Hits Tmp: (" << *tracker->NTracklets() << ")" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->TrackletStartHits(), fGpuTracker.TrackletTmpStartHits(), tracker->NHitsTotal() * sizeof(AliHLTTPCCAHit), cudaMemcpyDeviceToHost));
		tracker->DumpStartHits(*fOutFile);
		uint3* tmpMem = (uint3*) malloc(sizeof(uint3) * tracker->Param().NRows());
		CUDA_FAILED_MSG(cudaMemcpy(tmpMem, fGpuTracker.RowStartHitCountOffset(), tracker->Param().NRows() * sizeof(uint3), cudaMemcpyDeviceToHost));
		*fOutFile << "Start Hits Sort Vector:" << std::endl;
		for (int i = 0;i < tracker->Param().NRows();i++)
		{
			*fOutFile << "Row: " << i << ", Len: " << tmpMem[i].x << ", Offset: " << tmpMem[i].y << ", New Offset: " << tmpMem[i].z << std::endl;
		}
		free(tmpMem);
    }
#endif

	if (fDebugLevel >= 4)
	{
		*fOutFile << "Start Hits: (" << *tracker->NTracklets() << ")" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->HitMemory(), fGpuTracker.HitMemory(), tracker->HitMemorySize(), cudaMemcpyDeviceToHost));
		tracker->DumpStartHits(*fOutFile);
    }

	/*tracker->RunNeighboursFinder();
	tracker->RunNeighboursCleaner();
	tracker->RunStartHitsFinder();*/

	if (*tracker->NTracklets() > HLTCA_GPU_MAX_TRACKLETS)
	{
		printf("HLTCA_GPU_MAX_TRACKLETS constant insuffisant\n");
		return(1);
	}

	CUDA_FAILED_MSG(cudaMemset(fGpuTracker.SliceDataHitWeights(), 0, tracker->NHitsTotal() * sizeof(int)));
	//tracker->ClearSliceDataHitWeights();
	//CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker.SliceDataHitWeights(), tracker->SliceDataHitWeights(), tracker->NHitsTotal() * sizeof(int), cudaMemcpyHostToDevice));

	if (fDebugLevel >= 5) printf("Initialising Slice Tracker (CPU) Track Memory\n");
	tracker->TrackMemory() = reinterpret_cast<char*> ( new uint4 [ fGpuTracker.TrackMemorySize()/sizeof( uint4 ) + 100] );
    tracker->SetPointersTracks( *tracker->NTracklets(), tracker->NHitsTotal() );

/*	tracker->RunTrackletConstructor();
	if (fDebugLevel >= 4)
	{
		*fOutFile << "Tracklet Hits:" << endl;
		tracker->DumpTrackletHits(*fOutFile);
	}*/

	StandalonePerfTime(6);

#ifdef HLTCA_GPU_PREFETCHDATA
	if (tracker->Data().GPUSharedDataReq() * sizeof(ushort_v) > ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4))
	{
		printf("Insufficiant GPU shared Memory, required: %d, available %d\n", tracker->Data().GPUSharedDataReq() * sizeof(ushort_v), ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4));
		return(1);
	}
#endif

	int nMemThreads = TRACKLET_CONSTRUCTOR_NMEMTHREDS;
    nThreads = HLTCA_GPU_THREAD_COUNT - nMemThreads;//96;
    nBlocks = *tracker->NTracklets()/nThreads + 1;
    if( nBlocks<30 ){
		nBlocks = HLTCA_GPU_BLOCK_COUNT;
		nThreads = (*tracker->NTracklets())/nBlocks+1;
		if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
	}
	if (nThreads + nMemThreads > fCudaDeviceProp.maxThreadsPerBlock || (nThreads + nMemThreads) * HLTCA_GPU_REGS > fCudaDeviceProp.regsPerBlock)
	{
		printf("Invalid CUDA Kernel Configuration %d blocks %d threads %d memthreads\n", nBlocks, nThreads, nMemThreads);
		return(1);
	}

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker.fStageAtSync, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int))) ||
		CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker.fThreadTimes, 30 * 256 * sizeof(int))))
	{
		return(1);
	}
	CUDA_FAILED_MSG(cudaMemset(fGpuTracker.fStageAtSync, 0, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int)));
	int* StageAtSync = (int*) malloc(nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int));
	int* ThreadTimes = (int*) malloc(30 * 256 * sizeof(int));
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, &fGpuTracker, sizeof(AliHLTTPCCATracker)));
#endif

	if (fDebugLevel >= 5) printf("Running GPU Tracklet Constructor\n");

	if (fOptionAdaptiveSched)
	{
		AliHLTTPCCATrackletConstructorInit<<<*tracker->NTracklets() / HLTCA_GPU_THREAD_COUNT + 1, HLTCA_GPU_THREAD_COUNT>>>();
		if (CUDASync("Tracklet Initializer")) return 1;
		StandalonePerfTime(7);

		DumpRowBlocks(tracker);
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
		for (int i = 0;i < (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2;i++)
		{
			if (fDebugLevel >= 4) *fOutFile << "Scheduled Tracklet Constructor Iteration " << i << std::endl;
			AliHLTTPCCATrackletConstructorNew<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
			if (CUDASync("Tracklet Constructor (new)")) return 1;
			AliHLTTPCCATrackletConstructorUpdateRowBlockPos<<<HLTCA_GPU_BLOCK_COUNT, (tracker->Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2 / HLTCA_GPU_BLOCK_COUNT + 1>>>();
			if (CUDASync("Tracklet Constructor (update)")) return 1;
			DumpRowBlocks(tracker, false);
		}
#else
		AliHLTTPCCATrackletConstructorNew<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
		DumpRowBlocks(tracker, false);
#endif
	}
	else if (!fOptionSingleBlock)
	{
		StandalonePerfTime(7);
		//AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<nBlocks, nMemThreads+nThreads>>>(); 
	}
	else
	{
		StandalonePerfTime(7);
		//AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<1, TRACKLET_CONSTRUCTOR_NMEMTHREDS + *tracker->NTracklets()>>>();
	}
	if (CUDASync("Tracklet Constructor")) return 1;

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	printf("Saving Profile\n");
	CUDA_FAILED_MSG(cudaMemcpy(StageAtSync, fGpuTracker.fStageAtSync, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_FAILED_MSG(cudaMemcpy(ThreadTimes, fGpuTracker.fThreadTimes, 30 * 256 * sizeof(int), cudaMemcpyDeviceToHost));

	FILE *fp = fopen("profile.txt", "w+"), *fp2 = fopen("profile.bmp", "w+b"), *fp3 = fopen("times.txt", "w+");
	if (fp == NULL || fp2 == NULL || fp3 == NULL)
	{
		printf("Error opening Profile File\n");
		return(1);
	}
	BITMAPFILEHEADER bmpFH;
	BITMAPINFOHEADER bmpIH;
	ZeroMemory(&bmpFH, sizeof(bmpFH));
	ZeroMemory(&bmpIH, sizeof(bmpIH));
	
	bmpFH.bfType = 19778; //"BM"
	bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * 4;
	bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

	bmpIH.biSize = sizeof(bmpIH);
	bmpIH.biWidth = nBlocks * nThreads + nBlocks * nThreads / 32;
	if (nBlocks * nThreads % 32 == 0) bmpIH.biWidth--;
	bmpIH.biHeight = 4 + 159*2 + 1 + 1 + 1;
	bmpIH.biPlanes = 1;
	bmpIH.biBitCount = 32;

	fwrite(&bmpFH, 1, sizeof(bmpFH), fp2);
	fwrite(&bmpIH, 1, sizeof(bmpIH), fp2);

	for (int i = 0;i < 4 + 159*2 + 1 + 1 + 1;i++)
	{
		for (int j = 0;j < nBlocks * nThreads;j++)
		{
			fprintf(fp, "%d\t", StageAtSync[i * nBlocks * nThreads + j]);
			int color = 0;
			if (StageAtSync[i * nBlocks * nThreads + j] == 1) color = RGB(255, 0, 0);
			if (StageAtSync[i * nBlocks * nThreads + j] == 2) color = RGB(0, 255, 0);
			if (StageAtSync[i * nBlocks * nThreads + j] == 3) color = RGB(0, 0, 255);
			if (StageAtSync[i * nBlocks * nThreads + j] == 4) color = RGB(255, 255, 0);
			fwrite(&color, 1, 4, fp2);
			if (j > 0 && j % 32 == 0)
			{
				color = RGB(255, 255, 255);
				fwrite(&color, 1, 4, fp2);
			}
		}
		fprintf(fp, "\n");
	}

	for (int i = 0;i < 30;i++)
	{
		for (int j = 0;j < 256;j++)
		{
			fprintf(fp3, "%d\t", ThreadTimes[i * 256 + j]);
		}
		fprintf(fp3, "\n");
	}
	fclose(fp);
	fclose(fp2);
	fclose(fp3);

	cudaFree(fGpuTracker.fStageAtSync);
	cudaFree(fGpuTracker.fThreadTimes);
	free(StageAtSync);
	free(ThreadTimes);
#endif

	StandalonePerfTime(8);

	if (fDebugLevel >= 4)
	{
		*fOutFile << "Tracklet Hits:" << endl;
		CUDA_FAILED_MSG(cudaMemcpy(tracker->NTracklets(), fGpuTracker.NTracklets(), sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker->Tracklets(), fGpuTracker.Tracklets(), fGpuTracker.TrackMemorySize(), cudaMemcpyDeviceToHost));
		tracker->DumpTrackletHits(*fOutFile);
    }

	//tracker->RunTrackletSelector();
	

	nThreads = HLTCA_GPU_THREAD_COUNT;
	nBlocks = *tracker->NTracklets()/nThreads + 1;
	if( nBlocks<30 ){
	  nBlocks = HLTCA_GPU_BLOCK_COUNT;  
	  nThreads = *tracker->NTracklets()/nBlocks+1;
	  nThreads = (nThreads/32+1)*32;
	}
	if (nThreads > fCudaDeviceProp.maxThreadsPerBlock || (nThreads) * HLTCA_GPU_REGS > fCudaDeviceProp.regsPerBlock)
	{
		printf("Invalid CUDA Kernel Configuration %d blocks %d threads\n", nBlocks, nThreads);
		return(1);
	}

	if (fDebugLevel >= 5) printf("Running GPU Tracklet Selector\n");
	if (!fOptionSingleBlock)
	{
		AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<nBlocks, nThreads>>>();
	}
	else
	{
		AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<1, *tracker->NTracklets()>>>();
	}
	if (CUDASync("Tracklet Selector")) return 1;

	StandalonePerfTime(9);

	if (fDebugLevel >= 5) printf("Transfering Tracks from GPU to Host ");
	CUDA_FAILED_MSG(cudaMemcpy(tracker->CommonMemory(), fGpuTracker.CommonMemory(), tracker->CommonMemorySize(), cudaMemcpyDeviceToHost));
	if (tracker->GPUParameters()->fGPUError)
	{
		printf("GPU Tracker returned Error Code %d\n", tracker->GPUParameters()->fGPUError);
		return(1);
	}
	if (tracker->GPUParameters()->fGPUSchedCollisions)
		printf("Collisions: %d\n", tracker->GPUParameters()->fGPUSchedCollisions);
	if (fDebugLevel >= 5) printf("%d / %d\n", *tracker->NTracks(), *tracker->NTrackHits());
	size = sizeof(AliHLTTPCCATrack) * *tracker->NTracks();
	CUDA_FAILED_MSG(cudaMemcpy(tracker->Tracks(), fGpuTracker.Tracks(), size, cudaMemcpyDeviceToHost));
	size = sizeof(AliHLTTPCCAHitId) * *tracker->NTrackHits();
	if (CUDA_FAILED_MSG(cudaMemcpy(tracker->TrackHits(), fGpuTracker.TrackHits(), size, cudaMemcpyDeviceToHost)))
	{
		printf("CUDA Error during Reconstruction\n");
		return(1);
	}

	if (fDebugLevel >= 4)
	{
		*fOutFile << "Track Hits: (" << *tracker->NTracks() << ")" << endl;
		tracker->DumpTrackHits(*fOutFile);
    }

	if (fDebugLevel >= 5) printf("Running WriteOutput\n");
	tracker->WriteOutput();

	StandalonePerfTime(10);

	if (fDebugLevel >= 5) printf("GPU Reconstruction finished\n");

#ifdef HLTCA_STANDALONE
	if (fDebugLevel >= 2)
	{
		std::ofstream tmpout("tmpdebug.out");
		int* GPUDebug = (int*) malloc(100 * 1024 * 1024);
		CUDA_FAILED_MSG(cudaMemcpy(GPUDebug, fGpuTracker.fGPUDebugMem, 100 * 1024 * 1024, cudaMemcpyDeviceToHost));
		free(GPUDebug);
		cudaFree(fGpuTracker.fGPUDebugMem);
		tmpout.close();
	}
#endif

	
	return(0);
}

int AliHLTTPCCAGPUTracker::ExitGPU()
{
	cudaFree(fGPUMemory);
	return(0);
}
