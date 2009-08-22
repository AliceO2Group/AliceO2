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

__constant__ float4 gAliHLTTPCCATracker[HLTCA_GPU_TRACKER_CONSTANT_MEM / sizeof( float4 )];

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
int AliHLTTPCCAGPUTracker::InitGPU(int sliceCount, int forceDeviceID)
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

  if ((fGpuTracker = new AliHLTTPCCATracker[sliceCount]) == NULL)
  {
	printf("Error Creating GpuTrackers\n");
	return(1);
  }
  fSliceCount = sliceCount;
  
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
	  fGpuTracker[0].StandaloneQueryTime( fGpuTracker[0].PerfTimer(i));
  }
#endif
}

void AliHLTTPCCAGPUTracker::DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check)
{
	if (fDebugLevel >= 4)
	{
		*fOutFile << "RowBlock Tracklets" << std::endl;
	
		int3* RowBlockPos = (int3*) malloc(sizeof(int3) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2);
		int* RowBlockTracklets = (int*) malloc(sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2);
		uint2* BlockStartingTracklet = (uint2*) malloc(sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT);
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockPos, fGpuTracker[iSlice].RowBlockPos(), sizeof(int3) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockTracklets, fGpuTracker[iSlice].RowBlockTracklets(), sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(BlockStartingTracklet, fGpuTracker[iSlice].BlockStartingTracklet(), sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));

		int k = tracker[iSlice].GPUParameters()->fScheduleFirstDynamicTracklet;
		for (int i = 0; i < tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1;i++)
		{
			*fOutFile << "Rowblock: " << i << ", up " << RowBlockPos[i].y << "/" << RowBlockPos[i].x << ", down " << 
				RowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].y << "/" << RowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x << endl << "Phase 1: ";
			for (int j = 0;j < RowBlockPos[i].x;j++)
			{
				//Use Tracker Object to calculate Offset instead of fGpuTracker, since *fNTracklets of fGpuTracker points to GPU Mem!
				*fOutFile << RowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] << ", ";
				if (check && RowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] != k)
				{
					printf("Wrong starting Row Block %d, entry %d, is %d, should be %d\n", i, j, RowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j], k);
				}
				k++;
				if (RowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] == -1)
				{
					printf("Error, -1 Tracklet found\n");
				}
			}
			*fOutFile << endl << "Phase 2: ";
			for (int j = 0;j < RowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x;j++)
			{
				*fOutFile << RowBlockTracklets[(tracker[iSlice].RowBlockTracklets(1, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] << ", ";
			}
			*fOutFile << endl;
		}

		if (check)
		{
			*fOutFile << "Starting Threads: (First Dynamic: " << tracker[iSlice].GPUParameters()->fScheduleFirstDynamicTracklet << ")" << std::endl;
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
int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCATracker* tracker, int fSliceCount)
{
    int nThreads;
    int nBlocks;
	int size;

	if (fSliceCount == -1) fSliceCount = this->fSliceCount;

	if (fSliceCount * sizeof(AliHLTTPCCATracker) > HLTCA_GPU_TRACKER_CONSTANT_MEM)
	{
		printf("Insuffissant constant memory (Required %d, Available %d, Tracker %d, Param %d, SliceData %d)\n", fSliceCount * sizeof(AliHLTTPCCATracker), HLTCA_GPU_TRACKER_CONSTANT_MEM, sizeof(AliHLTTPCCATracker), sizeof(AliHLTTPCCAParam), sizeof(AliHLTTPCCASliceData));
		return(1);
	}

	int cudaDevice;
	cudaDeviceProp fCudaDeviceProp;
	cudaGetDevice(&cudaDevice);
	cudaGetDeviceProperties(&fCudaDeviceProp, cudaDevice);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (tracker[iSlice].Param().NRows() != HLTCA_ROW_COUNT)
		{
			printf("Error, Slice Tracker %d Row Count of %d exceeds Constant of %d\n", iSlice, tracker[iSlice].Param().NRows(), HLTCA_ROW_COUNT);
			return(1);
		}
		if (tracker[iSlice].CheckEmptySlice())
		{
			if (fDebugLevel >= 5) printf("Slice Empty, not running GPU Tracker\n");
			if (fSliceCount == 1)
				return(0);
		}

		if (fDebugLevel >= 4)
		{
			*fOutFile << endl << endl << "Slice: " << tracker[iSlice].Param().ISlice() << endl;
		}
	}

	if (fDebugLevel >= 5) printf("\n\nInitialising GPU Tracker\n");
	memcpy(fGpuTracker, tracker, sizeof(AliHLTTPCCATracker) * fSliceCount);

	StandalonePerfTime(0);

	char* tmpMem = alignPointer((char*) fGPUMemory, 1024 * 1024);


	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		fGpuTracker[iSlice].SetGPUTracker();

		if (fDebugLevel >= 5) printf("Initialising GPU Common Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerCommonMemory(tmpMem);
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 5) printf("Initialising GPU Hits Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerHitsMemory(tmpMem, tracker[iSlice].NHitsTotal());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 5) printf("Initialising GPU Slice Data Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUSliceDataMemory(tmpMem, fGpuTracker[iSlice].ClusterData());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);
		if (tmpMem - (char*) fGPUMemory > fGPUMemSize)
		{
			printf("Out of CUDA Memory\n");
			return(1);
		}

	#ifdef HLTCA_STANDALONE
		if (fDebugLevel >= 6)
		{
			if (CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker[iSlice].fGPUDebugMem, 100 * 1024 * 1024)))
			{
				printf("Out of CUDA Memory\n");
				return(1);
			}
			CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].fGPUDebugMem, 0, 100 * 1024 * 1024));
		}
	#endif

		if (fDebugLevel >= 5) printf("Initialising GPU Track Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS /**tracker[iSlice].NTracklets()*/, tracker[iSlice].NHitsTotal());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);
		if (tmpMem - (char*) fGPUMemory > fGPUMemSize)
		{
			printf("Out of CUDA Memory\n");
			return(1);
		}

		*tracker[iSlice].NTracklets() = 0;
		tracker[iSlice].GPUParameters()->fStaticStartingTracklets = 1;
		tracker[iSlice].GPUParameters()->fGPUError = 0;
		tracker[iSlice].GPUParameters()->fGPUSchedCollisions = 0;
		if (HLTCA_GPU_BLOCK_COUNT % fSliceCount == 0)
			fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount = HLTCA_GPU_BLOCK_COUNT / fSliceCount;
		else
			fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount = HLTCA_GPU_BLOCK_COUNT * (iSlice + 1) / fSliceCount - HLTCA_GPU_BLOCK_COUNT * (iSlice) / fSliceCount;
		if (fDebugLevel >= 5) printf("Blocks for Slice %d: %d\n", iSlice, fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUnSlices = fSliceCount;

		CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemorySize(), cudaMemcpyHostToDevice));
		CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker[iSlice].SliceDataMemory(), tracker[iSlice].SliceDataMemory(), tracker[iSlice].SliceDataMemorySize(), cudaMemcpyHostToDevice));
		CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].RowBlockPos(), 0, sizeof(int3) * 2 * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1)));
		CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].RowBlockTracklets(), -1, sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2));
	}
	CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker) * fSliceCount));
	if (fDebugLevel >= 1)
	{
		static int showMemInfo = true;
		if (showMemInfo)
			printf("GPU Memory used: %d bytes\n", (int) (tmpMem - (char*) fGPUMemory));
		showMemInfo = false;
	}

	StandalonePerfTime(1);

	if (fDebugLevel >= 5) printf("Running GPU Neighbours Finder\n");
	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fGpuTracker[iSlice].Param().NRows(), 256>>>(iSlice);
		if (CUDASync("Neighbours finder")) return 1;
	}

	StandalonePerfTime(2);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 4)
		{
			*fOutFile << "Neighbours Finder:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].SliceDataMemory(), fGpuTracker[iSlice].SliceDataMemory(), tracker[iSlice].SliceDataMemorySize(), cudaMemcpyDeviceToHost));
			tracker[iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 5) printf("Running GPU Neighbours Cleaner\n");
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fGpuTracker[iSlice].Param().NRows()-2, 256>>>(iSlice);
		if (CUDASync("Neighbours Cleaner")) return 1;
	}

	StandalonePerfTime(3);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 4)
		{
			*fOutFile << "Neighbours Cleaner:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].SliceDataMemory(), fGpuTracker[iSlice].SliceDataMemory(), tracker[iSlice].SliceDataMemorySize(), cudaMemcpyDeviceToHost));
			tracker[iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 5) printf("Running GPU Start Hits Finder\n");
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fGpuTracker[iSlice].Param().NRows()-4, 256>>>(iSlice);
		if (CUDASync("Start Hits Finder")) return 1;
	}

	StandalonePerfTime(4);

#ifdef HLTCA_GPU_SORT_STARTHITS
	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 5) printf("Running GPU Start Hits Sorter\n");
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<30, 256>>>(iSlice);
		if (CUDASync("Start Hits Sorter")) return 1;
	}
#endif

	StandalonePerfTime(5);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 5) printf("Obtaining Number of Start Hits from GPU: ");
		CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
		if (fDebugLevel >= 5) printf("%d\n", *tracker[iSlice].NTracklets());
		else if (fDebugLevel >= 2) printf("%3d ", *tracker[iSlice].NTracklets());

#ifdef HLTCA_GPU_SORT_STARTHITS
		if (fDebugLevel >= 4)
		{
			*fOutFile << "Start Hits Tmp: (" << *tracker[iSlice].NTracklets() << ")" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].TrackletStartHits(), fGpuTracker[iSlice].TrackletTmpStartHits(), tracker[iSlice].NHitsTotal() * sizeof(AliHLTTPCCAHit), cudaMemcpyDeviceToHost));
			tracker[iSlice].DumpStartHits(*fOutFile);
			uint3* tmpMem = (uint3*) malloc(sizeof(uint3) * tracker[iSlice].Param().NRows());
			CUDA_FAILED_MSG(cudaMemcpy(tmpMem, fGpuTracker[iSlice].RowStartHitCountOffset(), tracker[iSlice].Param().NRows() * sizeof(uint3), cudaMemcpyDeviceToHost));
			*fOutFile << "Start Hits Sort Vector:" << std::endl;
			for (int i = 0;i < tracker[iSlice].Param().NRows();i++)
			{
				*fOutFile << "Row: " << i << ", Len: " << tmpMem[i].x << ", Offset: " << tmpMem[i].y << ", New Offset: " << tmpMem[i].z << std::endl;
			}
			free(tmpMem);
		}
#endif

		if (fDebugLevel >= 4)
		{
			*fOutFile << "Start Hits: (" << *tracker[iSlice].NTracklets() << ")" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), tracker[iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			tracker[iSlice].DumpStartHits(*fOutFile);
		}

		/*tracker[iSlice].RunNeighboursFinder();
		tracker[iSlice].RunNeighboursCleaner();
		tracker[iSlice].RunStartHitsFinder();*/

		if (*tracker[iSlice].NTracklets() > HLTCA_GPU_MAX_TRACKLETS)
		{
			printf("HLTCA_GPU_MAX_TRACKLETS constant insuffisant\n");
			return(1);
		}

		CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].SliceDataHitWeights(), 0, tracker[iSlice].NHitsTotal() * sizeof(int)));
		//tracker[iSlice].ClearSliceDataHitWeights();
		//CUDA_FAILED_MSG(cudaMemcpy(fGpuTracker[iSlice].SliceDataHitWeights(), tracker[iSlice].SliceDataHitWeights(), tracker[iSlice].NHitsTotal() * sizeof(int), cudaMemcpyHostToDevice));

		if (fDebugLevel >= 5) printf("Initialising Slice Tracker (CPU) Track Memory\n");
		tracker[iSlice].TrackMemory() = reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackMemorySize()/sizeof( uint4 ) + 100] );
		tracker[iSlice].SetPointersTracks( *tracker[iSlice].NTracklets(), tracker[iSlice].NHitsTotal() );

		/*tracker[iSlice].RunTrackletConstructor();
		if (fDebugLevel >= 4)
		{
			*fOutFile << "Tracklet Hits:" << endl;
			tracker[iSlice].DumpTrackletHits(*fOutFile);
		}*/
	}

	StandalonePerfTime(6);

#ifdef HLTCA_GPU_PREFETCHDATA
	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (tracker[iSlice].Data().GPUSharedDataReq() * sizeof(ushort_v) > ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4))
		{
			printf("Insufficiant GPU shared Memory, required: %d, available %d\n", tracker[iSlice].Data().GPUSharedDataReq() * sizeof(ushort_v), ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4));
			return(1);
		}
	}
#endif

	if (fDebugLevel >= 5) printf("Running GPU Tracklet Constructor\n");

	if (fOptionAdaptiveSched)
	{
		for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
		{
			AliHLTTPCCATrackletConstructorInit<<<*tracker[iSlice].NTracklets() / HLTCA_GPU_THREAD_COUNT + 1, HLTCA_GPU_THREAD_COUNT>>>(iSlice);
			if (CUDASync("Tracklet Initializer")) return 1;
			DumpRowBlocks(tracker, iSlice);
		}
		StandalonePerfTime(7);

#ifdef HLTCA_GPU_SCHED_HOST_SYNC
		for (int i = 0;i < (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2;i++)
		{
			if (fDebugLevel >= 4) *fOutFile << "Scheduled Tracklet Constructor Iteration " << i << std::endl;
			AliHLTTPCCATrackletConstructorNew<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
			if (CUDASync("Tracklet Constructor (new)")) return 1;
			for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
			{
				AliHLTTPCCATrackletConstructorUpdateRowBlockPos<<<HLTCA_GPU_BLOCK_COUNT, (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2 / HLTCA_GPU_BLOCK_COUNT + 1>>>(iSlice);
				if (CUDASync("Tracklet Constructor (update)")) return 1;
				DumpRowBlocks(tracker, iSlice, false);
			}
		}
#else
		AliHLTTPCCATrackletConstructorNew<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
		if (CUDASync("Tracklet Constructor (new)")) return 1;
		for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
		{
			DumpRowBlocks(tracker, iSlice, false);
		}
#endif
	}
	else
	{
		StandalonePerfTime(7);
		for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
		{
			int nMemThreads = TRACKLET_CONSTRUCTOR_NMEMTHREDS;
			nThreads = HLTCA_GPU_THREAD_COUNT - nMemThreads;//96;
			nBlocks = *tracker[iSlice].NTracklets()/nThreads + 1;
			if( nBlocks<30 ){
				nBlocks = HLTCA_GPU_BLOCK_COUNT;
				nThreads = (*tracker[iSlice].NTracklets())/nBlocks+1;
				if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
			}
			if (nThreads + nMemThreads > fCudaDeviceProp.maxThreadsPerBlock || (nThreads + nMemThreads) * HLTCA_GPU_REGS > fCudaDeviceProp.regsPerBlock)
			{
				printf("Invalid CUDA Kernel Configuration %d blocks %d threads %d memthreads\n", nBlocks, nThreads, nMemThreads);
				return(1);
			}

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
			if (CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker[iSlice].fStageAtSync, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int))) ||
				CUDA_FAILED_MSG(cudaMalloc((void**) &fGpuTracker[iSlice].fThreadTimes, 30 * 256 * sizeof(int))))
			{
				return(1);
			}
			CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].fStageAtSync, 0, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int)));
			int* StageAtSync = (int*) malloc(nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int));
			int* ThreadTimes = (int*) malloc(30 * 256 * sizeof(int));
			CUDA_FAILED_MSG(cudaMemcpyToSymbol(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker * fSliceCount)));
#endif

			if (!fOptionSingleBlock)
			{
				//AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<nBlocks, nMemThreads+nThreads>>>(iSlice); 
			}
			else
			{
				//AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor> <<<1, TRACKLET_CONSTRUCTOR_NMEMTHREDS + *tracker[iSlice].NTracklets()>>>(iSlice);
			}
			if (CUDASync("Tracklet Constructor")) return 1;

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
			printf("Saving Profile\n");
			CUDA_FAILED_MSG(cudaMemcpy(StageAtSync, fGpuTracker[iSlice].fStageAtSync, nBlocks * nThreads * (4 + 159*2 + 1 + 1 + 1) * sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_FAILED_MSG(cudaMemcpy(ThreadTimes, fGpuTracker[iSlice].fThreadTimes, 30 * 256 * sizeof(int), cudaMemcpyDeviceToHost));

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

			cudaFree(fGpuTracker[iSlice].fStageAtSync);
			cudaFree(fGpuTracker[iSlice].fThreadTimes);
			free(StageAtSync);
			free(ThreadTimes);
#endif
		}
	}
	
	StandalonePerfTime(8);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 4)
		{
			*fOutFile << "Tracklet Hits:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5)
			{
				printf("Obtained %d tracklets\n", *tracker[iSlice].NTracklets());
			}
			CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].Tracklets(), fGpuTracker[iSlice].Tracklets(), fGpuTracker[iSlice].TrackMemorySize(), cudaMemcpyDeviceToHost));
			tracker[iSlice].DumpTrackletHits(*fOutFile);
		}

		//tracker[iSlice].RunTrackletSelector();
		

		nThreads = HLTCA_GPU_THREAD_COUNT;
		nBlocks = *tracker[iSlice].NTracklets()/nThreads + 1;
		if( nBlocks<30 ){
		  nBlocks = HLTCA_GPU_BLOCK_COUNT;  
		  nThreads = *tracker[iSlice].NTracklets()/nBlocks+1;
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
			AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<nBlocks, nThreads>>>(iSlice);
		}
		else
		{
			AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector><<<1, *tracker[iSlice].NTracklets()>>>(iSlice);
		}
		if (CUDASync("Tracklet Selector")) return 1;
	}

	StandalonePerfTime(9);

	for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
	{
		if (fDebugLevel >= 5) printf("Transfering Tracks from GPU to Host ");
		CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), tracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
		if (tracker[iSlice].GPUParameters()->fGPUError)
		{
			printf("GPU Tracker returned Error Code %d\n", tracker[iSlice].GPUParameters()->fGPUError);
			return(1);
		}
		if (tracker[iSlice].GPUParameters()->fGPUSchedCollisions)
			printf("Collisions: %d\n", tracker[iSlice].GPUParameters()->fGPUSchedCollisions);
		if (fDebugLevel >= 5) printf("%d / %d\n", *tracker[iSlice].NTracks(), *tracker[iSlice].NTrackHits());
		size = sizeof(AliHLTTPCCATrack) * *tracker[iSlice].NTracks();
		CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].Tracks(), fGpuTracker[iSlice].Tracks(), size, cudaMemcpyDeviceToHost));
		size = sizeof(AliHLTTPCCAHitId) * *tracker[iSlice].NTrackHits();
		if (CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].TrackHits(), fGpuTracker[iSlice].TrackHits(), size, cudaMemcpyDeviceToHost)))
		{
			printf("CUDA Error during Reconstruction\n");
			return(1);
		}

		if (fDebugLevel >= 4)
		{
			*fOutFile << "Track Hits: (" << *tracker[iSlice].NTracks() << ")" << endl;
			tracker[iSlice].DumpTrackHits(*fOutFile);
		}

		if (fDebugLevel >= 5) printf("Running WriteOutput\n");
		tracker[iSlice].WriteOutput();
	}

	StandalonePerfTime(10);

	if (fDebugLevel >= 5) printf("GPU Reconstruction finished\n");

#ifdef HLTCA_STANDALONE
	if (fDebugLevel >= 6)
	{
		for (int iSlice = 0;iSlice < fSliceCount;iSlice++)
		{
			std::ofstream tmpout("tmpdebug.out");
			int* GPUDebug = (int*) malloc(100 * 1024 * 1024);
			CUDA_FAILED_MSG(cudaMemcpy(GPUDebug, fGpuTracker[iSlice].fGPUDebugMem, 100 * 1024 * 1024, cudaMemcpyDeviceToHost));
			free(GPUDebug);
			cudaFree(fGpuTracker[iSlice].fGPUDebugMem);
			tmpout.close();
		}
	}
#endif
	
	return(0);
}

int AliHLTTPCCAGPUTracker::ExitGPU()
{
	cudaFree(fGPUMemory);
#ifndef CUDA_DEVICE_EMULATION
	delete[] fGpuTracker;
#endif
	fGpuTracker = NULL;
	return(0);
}
