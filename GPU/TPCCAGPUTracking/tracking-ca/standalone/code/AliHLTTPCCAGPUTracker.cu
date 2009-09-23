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
#ifdef HLTCA_GPU_TEXTURE_FETCH
texture<ushort2, 1, cudaReadModeElementType> texRefu2;
texture<unsigned short, 1, cudaReadModeElementType> texRefu;
texture<signed short, 1, cudaReadModeElementType> texRefs;
#endif

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

#ifndef CUDA_DEVICE_EMULATION
	int count, bestDevice = -1, bestDeviceSpeed = 0;
	if (CUDA_FAILED_MSG(cudaGetDeviceCount(&count)))
	{
		std::cout << "Error getting CUDA Device Count" << std::endl;
		return(1);
	}
	if (fDebugLevel >= 2) std::cout << "Available CUDA devices: ";
	for (int i = 0;i < count;i++)
	{
		cudaGetDeviceProperties(&fCudaDeviceProp, i);
		if (fDebugLevel >= 2) std::cout << fCudaDeviceProp.name << " (" << i << ")     ";
		if (fCudaDeviceProp.major < 9 && !(fCudaDeviceProp.major < 1 || (fCudaDeviceProp.major == 1 && fCudaDeviceProp.minor < 2)) && fCudaDeviceProp.multiProcessorCount * fCudaDeviceProp.clockRate > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = fCudaDeviceProp.multiProcessorCount * fCudaDeviceProp.clockRate;
		}
	}
	if (fDebugLevel >= 2) std::cout << std::endl;

	if (bestDevice == -1)
	{
		std::cout << "No CUDA Device available, aborting CUDA Initialisation" << std::endl;
		return(1);
	}

  int cudaDevice;
  if (forceDeviceID == -1)
	  cudaDevice = bestDevice;
  else
	  cudaDevice = forceDeviceID;
#else
	int cudaDevice = 0;
#endif

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

  if (CUDA_FAILED_MSG(cudaSetDevice(cudaDevice)))
  {
	  std::cout << "Could not set CUDA Device!\n";
	  return(1);
  }

  if (fgkNSlices * AliHLTTPCCATracker::CommonMemorySize() > HLTCA_GPU_COMMON_MEMORY)
  {
	  std::cout << "Insufficiant Common Memory\n";
	  cudaThreadExit();
	  return(1);
  }

  if (fgkNSlices * (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow) > HLTCA_GPU_ROWS_MEMORY)
  {
	  std::cout << "Insufficiant Row Memory\n";
	  cudaThreadExit();
	  return(1);
  }

  fGPUMemSize = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + sliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_GLOBAL_MEMORY);
  if (fGPUMemSize > fCudaDeviceProp.totalGlobalMem || CUDA_FAILED_MSG(cudaMalloc(&fGPUMemory, (size_t) fGPUMemSize)))
  {
	  std::cout << "CUDA Memory Allocation Error\n";
	  cudaThreadExit();
	  return(1);
  }
  if (fDebugLevel >= 1) std::cout << "GPU Memory used: " << fGPUMemSize << std::endl;
  int HostMemSize = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + sliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + HLTCA_GPU_TRACKER_OBJECT_MEMORY;
  if (CUDA_FAILED_MSG(cudaMallocHost(&fHostLockedMemory, HostMemSize)))
  {
	  cudaFree(fGPUMemory);
	  cudaThreadExit();
	  std::cout << "Error allocating Page Locked Host Memory";
	  return(1);
  }
  if (fDebugLevel >= 1) std::cout << "Host Memory used: " << HostMemSize << std::endl;

  if (fDebugLevel >= 1)
  {
	  CUDA_FAILED_MSG(cudaMemset(fGPUMemory, 143, (size_t) fGPUMemSize));
  }
  std::cout << "CUDA Initialisation successfull\n";

  //Don't run constructor / destructor here, this will be just local memcopy of Tracker in GPU Memory
  if (sizeof(AliHLTTPCCATracker) * sliceCount > HLTCA_GPU_TRACKER_OBJECT_MEMORY)
  {
	  std::cout << "Insufficiant Tracker Object Memory\n";
	  return(1);
  }
  fSliceCount = sliceCount;
  fGpuTracker = (AliHLTTPCCATracker*) TrackerMemory(fHostLockedMemory, 0);

  for (int i = 0;i < fgkNSlices;i++)
  {
    fSlaveTrackers[i].SetGPUTracker();
	fSlaveTrackers[i].SetGPUTrackerCommonMemory((char*) CommonMemory(fHostLockedMemory, i));
	fSlaveTrackers[i].pData()->SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, i), RowMemory(fHostLockedMemory, i));
  }

  pCudaStreams = malloc(CAMath::Max(3, fSliceCount) * sizeof(cudaStream_t));
  cudaStream_t* const cudaStreams = (cudaStream_t*) pCudaStreams;
  for (int i = 0;i < CAMath::Max(3, fSliceCount);i++)
  {
	if (CUDA_FAILED_MSG(cudaStreamCreate(&cudaStreams[i])))
	{
		std::cout << "Error creating CUDA Stream" << std::endl;
		return(1);
	}
  }

#if defined(HLTCA_STANDALONE) & !defined(CUDA_DEVICE_EMULATION)
  if (fDebugLevel < 2)
  {
	  //Do one initial run for Benchmark reasons
	  const int useDebugLevel = fDebugLevel;
	  fDebugLevel = 0;
	  AliHLTTPCCAClusterData tmpCluster;
	  AliHLTTPCCASliceOutput tmpOutput;
	  AliHLTTPCCAParam tmpParam;
	  tmpParam.SetNRows(HLTCA_ROW_COUNT);
	  fSlaveTrackers[0].SetParam(tmpParam);
	  Reconstruct(&tmpOutput, &tmpCluster, 0, 1);
	  fDebugLevel = useDebugLevel;
  }
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
	if (strcmp(OptionName, "SimpleSched") == 0)
	{
		fOptionSimpleSched = OptionValue;
	}
	else
	{
		printf("Unknown Option: %s\n", OptionName);
		return(1);
	}
	return(0);
}

#ifdef HLTCA_STANDALONE
void AliHLTTPCCAGPUTracker::StandalonePerfTime(int iSlice, int i)
{
  if (fDebugLevel >= 1)
  {
	  AliHLTTPCCAStandaloneFramework::StandaloneQueryTime( fSlaveTrackers[iSlice].PerfTimer(i));
  }
}
#else
void AliHLTTPCCAGPUTracker::StandalonePerfTime(int /*iSlice*/, int /*i*/) {}
#endif

void AliHLTTPCCAGPUTracker::DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check)
{
	if (fDebugLevel >= 4)
	{
		*fOutFile << "RowBlock Tracklets" << std::endl;
	
		int4* RowBlockPos = (int4*) malloc(sizeof(int4) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2);
		int* RowBlockTracklets = (int*) malloc(sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2);
		uint2* BlockStartingTracklet = (uint2*) malloc(sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT);
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockPos, fGpuTracker[iSlice].RowBlockPos(), sizeof(int4) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(RowBlockTracklets, fGpuTracker[iSlice].RowBlockTracklets(), sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(BlockStartingTracklet, fGpuTracker[iSlice].BlockStartingTracklet(), sizeof(uint2) * HLTCA_GPU_BLOCK_COUNT, cudaMemcpyDeviceToHost));
		CUDA_FAILED_MSG(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));

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

__global__ void PreInitRowBlocks(int4* const RowBlockPos, int* const RowBlockTracklets, int* const SliceDataHitWeights, int nSliceDataHits)
{
	int4* const RowBlockTracklets4 = (int4*) RowBlockTracklets;
	int4* const SliceDataHitWeights4 = (int4*) SliceDataHitWeights;
	const int stride = blockDim.x * gridDim.x;
	int4 i0, i1;
	i0.x = i0.y = i0.z = i0.w = 0;
	i1.x = i1.y = i1.z = i1.w = -1;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < sizeof(int4) * 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1) / sizeof(int4);i += stride)
		RowBlockPos[i] = i0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < sizeof(int) * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2 / sizeof(int4);i += stride)
		RowBlockTracklets4[i] = i1;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		SliceDataHitWeights4[i] = i0;
}

//Primary reconstruction function
int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCASliceOutput* pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	cudaStream_t* const cudaStreams = (cudaStream_t*) pCudaStreams;

	if (sliceCountLocal == -1) sliceCountLocal = this->fSliceCount;

	if (sliceCountLocal * sizeof(AliHLTTPCCATracker) > HLTCA_GPU_TRACKER_CONSTANT_MEM)
	{
		printf("Insuffissant constant memory (Required %d, Available %d, Tracker %d, Param %d, SliceData %d)\n", sliceCountLocal * (int) sizeof(AliHLTTPCCATracker), (int) HLTCA_GPU_TRACKER_CONSTANT_MEM, (int) sizeof(AliHLTTPCCATracker), (int) sizeof(AliHLTTPCCAParam), (int) sizeof(AliHLTTPCCASliceData));
		return(1);
	}

	if (fDebugLevel >= 4)
	{
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			*fOutFile << endl << endl << "Slice: " << fSlaveTrackers[firstSlice + iSlice].Param().ISlice() << endl;
		}
	}

	memcpy(fGpuTracker, &fSlaveTrackers[firstSlice], sizeof(AliHLTTPCCATracker) * sliceCountLocal);

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (fDebugLevel >= 2) printf("\n\nInitialising GPU Tracker (Slice %d)\n", fSlaveTrackers[firstSlice + iSlice].Param().ISlice());

		//Make this a GPU Tracker
		fGpuTracker[iSlice].SetGPUTracker();
		fGpuTracker[iSlice].SetGPUTrackerCommonMemory((char*) CommonMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].pData()->SetGPUSliceDataMemory(SliceDataMemory(fGPUMemory, iSlice), RowMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].pData()->SetPointers(&pClusterData[iSlice], false);

		//Set Pointers to GPU Memory
		char* tmpMem = (char*) GlobalMemory(fGPUMemory, iSlice);

		if (fDebugLevel >= 5) printf("Initialising GPU Hits Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerHitsMemory(tmpMem, pClusterData[iSlice].NumberOfClusters(), fOptionSimpleSched);
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 5) printf("Initialising GPU Tracklet Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTrackletsMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS /* *fSlaveTrackers[firstSlice + iSlice].NTracklets()*/, fOptionSimpleSched);
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 5) printf("Initialising GPU Track Memory\n");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKS /* *fSlaveTrackers[firstSlice + iSlice].NTracklets()*/, pClusterData[iSlice].NumberOfClusters());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fGpuTracker[iSlice].TrackMemorySize() >= HLTCA_GPU_TRACKS_MEMORY)
		{
			printf("Insufficiant Track Memory\n");
			return(1);
		}

		if (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice) > HLTCA_GPU_GLOBAL_MEMORY)
		{
			printf("Insufficiant Global Memory\n");
			return(1);
		}

		//Initialize Startup Constants
		*fSlaveTrackers[firstSlice + iSlice].NTracklets() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTracks() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTrackHits() = 0;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount = HLTCA_GPU_BLOCK_COUNT * (iSlice + 1) / sliceCountLocal - HLTCA_GPU_BLOCK_COUNT * (iSlice) / sliceCountLocal;
		if (fDebugLevel >= 5) printf("Blocks for Slice %d: %d\n", iSlice, fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUnSlices = sliceCountLocal;
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError = 0;
#ifdef HLTCA_GPU_SCHED_FIXED_START
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fNextTracklet = fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount * HLTCA_GPU_THREAD_COUNT;
#else
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fNextTracklet = 0;
#endif

		fGpuTracker[iSlice].pData()->GPUTextureBase() = fGpuTracker[0].SliceDataMemory();
	}

#ifdef HLTCA_GPU_TEXTURE_FETCH
		cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<ushort2>();
		size_t offset;
//		if (pClusterData[iSlice].NumberOfClusters() && (CUDA_FAILED_MSG(cudaBindTexture(&offset, &texRef, fGpuTracker[iSlice].pData()->HitData(), &channelDesc, fGpuTracker[iSlice].Data().NumberOfHitsPlusAlign() * sizeof(ushort2))) || offset))
		if (CUDA_FAILED_MSG(cudaBindTexture(&offset, &texRefu2, fGpuTracker[0].SliceDataMemory(), &channelDescu2, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
		{
			printf("Error binding CUDA Texture (Offset %d)\n", (size_t) offset);
			return(1);
		}
		cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<unsigned short>();
		if (CUDA_FAILED_MSG(cudaBindTexture(&offset, &texRefu, fGpuTracker[0].SliceDataMemory(), &channelDescu, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
		{
			printf("Error binding CUDA Texture (Offset %d)\n", (size_t) offset);
			return(1);
		}
		cudaChannelFormatDesc channelDescs = cudaCreateChannelDesc<signed short>();
		if (CUDA_FAILED_MSG(cudaBindTexture(&offset, &texRefs, fGpuTracker[0].SliceDataMemory(), &channelDescs, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
		{
			printf("Error binding CUDA Texture (Offset %d)\n", (size_t) offset);
			return(1);
		}
#endif

	//Copy Tracker Object to GPU Memory
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (CUDA_FAILED_MSG(cudaMalloc(&fGpuTracker[0].fStageAtSync, 100000000))) return(1);
	CUDA_FAILED_MSG(cudaMemset(fGpuTracker[0].fStageAtSync, 0, 100000000));
#endif
	CUDA_FAILED_MSG(cudaMemcpyToSymbolAsync(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker) * sliceCountLocal, 0, cudaMemcpyHostToDevice, cudaStreams[0]));

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		StandalonePerfTime(firstSlice + iSlice, 0);

		if (!fOptionSimpleSched)
		{
			PreInitRowBlocks<<<30, 256, 0, cudaStreams[2]>>>(fGpuTracker[iSlice].RowBlockPos(), fGpuTracker[iSlice].RowBlockTracklets(), fGpuTracker[iSlice].SliceDataHitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign());
		}
		else
		{
			CUDA_FAILED_MSG(cudaMemset(fGpuTracker[iSlice].SliceDataHitWeights(), 0, fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign() * sizeof(int)));
		}

		//Initialize GPU Slave Tracker
		fSlaveTrackers[firstSlice + iSlice].pData()->SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, iSlice), RowMemory(fHostLockedMemory, firstSlice + iSlice));
		fSlaveTrackers[firstSlice + iSlice].ReadEvent(&pClusterData[iSlice]);
		if (fSlaveTrackers[firstSlice + iSlice].Data().MemorySize() > HLTCA_GPU_SLICE_DATA_MEMORY)
		{
			printf("Insufficiant Slice Data Memory\n");
			return(1);
		}

		/*if (fSlaveTrackers[firstSlice + iSlice].CheckEmptySlice())
		{
			if (fDebugLevel >= 5) printf("Slice Empty, not running GPU Tracker\n");
			if (sliceCountLocal == 1)
				return(0);
		}*/

		if (fDebugLevel >= 5) printf("Initialising Slice Tracker (CPU) Output Memory\n");

		if (fDebugLevel >= 4)
		{
			fSlaveTrackers[firstSlice + iSlice].TrackletMemory() = reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] );
			fSlaveTrackers[firstSlice + iSlice].SetPointersTracklets( HLTCA_GPU_MAX_TRACKLETS );
			fSlaveTrackers[firstSlice + iSlice].HitMemory() = reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100] );
			fSlaveTrackers[firstSlice + iSlice].SetPointersHits( pClusterData[iSlice].NumberOfClusters() );
		}

		//Copy Data to GPU Global Memory
		CUDA_FAILED_MSG(cudaMemcpyAsync(fGpuTracker[iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
		CUDA_FAILED_MSG(cudaMemcpyAsync(fGpuTracker[iSlice].SliceDataMemory(), fSlaveTrackers[firstSlice + iSlice].SliceDataMemory(), fSlaveTrackers[firstSlice + iSlice].SliceDataMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));

#ifdef SLICE_DATA_EXTERN_ROWS
		CUDA_FAILED_MSG(cudaMemcpyAsync(fGpuTracker[iSlice].SliceDataRows(), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
#endif

		if (CUDASync("Initialization")) return(1);
		StandalonePerfTime(firstSlice + iSlice, 1);

		if (fDebugLevel >= 5) printf("Running GPU Neighbours Finder\n");
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows(), 256, 0, cudaStreams[iSlice & 1]>>>(iSlice);

		if (CUDASync("Neighbours finder")) return 1;

		StandalonePerfTime(firstSlice + iSlice, 2);

		if (fDebugLevel >= 4)
		{
			*fOutFile << "Neighbours Finder:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].SliceDataMemory(), fGpuTracker[iSlice].SliceDataMemory(), fSlaveTrackers[firstSlice + iSlice].SliceDataMemorySize(), cudaMemcpyDeviceToHost));
			fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 5) printf("Running GPU Neighbours Cleaner\n");
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-2, 256, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (CUDASync("Neighbours Cleaner")) return 1;

		StandalonePerfTime(firstSlice + iSlice, 3);

		if (fDebugLevel >= 4)
		{
			*fOutFile << "Neighbours Cleaner:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].SliceDataMemory(), fGpuTracker[iSlice].SliceDataMemory(), fSlaveTrackers[firstSlice + iSlice].SliceDataMemorySize(), cudaMemcpyDeviceToHost));
			fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 5) printf("Running GPU Start Hits Finder\n");
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-6, 256, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (CUDASync("Start Hits Finder")) return 1;

		StandalonePerfTime(firstSlice + iSlice, 4);

		if (!fOptionSimpleSched)
		{
			if (fDebugLevel >= 5) printf("Running GPU Start Hits Sorter\n");
			AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<30, 256, 0, cudaStreams[iSlice & 1]>>>(iSlice);
			if (CUDASync("Start Hits Sorter")) return 1;
		}

		StandalonePerfTime(firstSlice + iSlice, 5);

		if (fDebugLevel >= 2)
		{
			if (fDebugLevel >= 5) printf("Obtaining Number of Start Hits from GPU: ");
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5) printf("%d\n", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			else if (fDebugLevel >= 2) printf("%3d ", *fSlaveTrackers[firstSlice + iSlice].NTracklets());

			if (*fSlaveTrackers[firstSlice + iSlice].NTracklets() > HLTCA_GPU_MAX_TRACKLETS)
			{
				printf("HLTCA_GPU_MAX_TRACKLETS constant insuffisant\n");
				return(1);
			}
		}

		if (!fOptionSimpleSched && fDebugLevel >= 4)
		{
			*fOutFile << "Start Hits Tmp: (" << *fSlaveTrackers[firstSlice + iSlice].NTracklets() << ")" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletStartHits(), fGpuTracker[iSlice].TrackletTmpStartHits(), pClusterData[iSlice].NumberOfClusters() * sizeof(AliHLTTPCCAHitId), cudaMemcpyDeviceToHost));
			fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
			uint3* tmpMemory = (uint3*) malloc(sizeof(uint3) * fSlaveTrackers[firstSlice + iSlice].Param().NRows());
			CUDA_FAILED_MSG(cudaMemcpy(tmpMemory, fGpuTracker[iSlice].RowStartHitCountOffset(), fSlaveTrackers[firstSlice + iSlice].Param().NRows() * sizeof(uint3), cudaMemcpyDeviceToHost));
			*fOutFile << "Start Hits Sort Vector:" << std::endl;
			for (int i = 0;i < fSlaveTrackers[firstSlice + iSlice].Param().NRows();i++)
			{
				*fOutFile << "Row: " << i << ", Len: " << tmpMemory[i].x << ", Offset: " << tmpMemory[i].y << ", New Offset: " << tmpMemory[i].z << std::endl;
			}
			free(tmpMemory);
		}

		if (fDebugLevel >= 4)
		{
			*fOutFile << "Start Hits: (" << *fSlaveTrackers[firstSlice + iSlice].NTracklets() << ")" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fSlaveTrackers[firstSlice + iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
		}

		StandalonePerfTime(firstSlice + iSlice, 6);
		
		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
	}

	StandalonePerfTime(firstSlice, 7);
	if (fOptionSimpleSched)
	{
		AliHLTTPCCATrackletConstructorNewGPUSimple<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
		if (CUDASync("Tracklet Constructor Simple Sched")) return 1;
	}
	else
	{
#ifdef HLTCA_GPU_PREFETCHDATA
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			if (fSlaveTrackers[firstSlice + iSlice].Data().GPUSharedDataReq() * sizeof(ushort_v) > ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4))
			{
				printf("Insufficiant GPU shared Memory, required: %d, available %d\n", fSlaveTrackers[firstSlice + iSlice].Data().GPUSharedDataReq() * sizeof(ushort_v), ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4 * sizeof(uint4));
				return(1);
			}
			if (fDebugLevel >= 1)
			{
				static int infoShown = 0;
				if (!infoShown)
				{
					printf("GPU Shared Memory Cache Size: %d\n", 2 * fSlaveTrackers[firstSlice + iSlice].Data().GPUSharedDataReq() * sizeof(ushort_v));
					infoShown = 1;
				}
			}
		}
#endif

		if (fDebugLevel >= 5) printf("Running GPU Tracklet Constructor\n");

		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			AliHLTTPCCATrackletConstructorInit<<<HLTCA_GPU_MAX_TRACKLETS /* *fSlaveTrackers[firstSlice + iSlice].NTracklets() */ / HLTCA_GPU_THREAD_COUNT + 1, HLTCA_GPU_THREAD_COUNT>>>(iSlice);
			if (CUDASync("Tracklet Initializer")) return 1;
			DumpRowBlocks(fSlaveTrackers, iSlice);
		}

		AliHLTTPCCATrackletConstructorNewGPU<<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT>>>();
		if (CUDASync("Tracklet Constructor (new)")) return 1;
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			DumpRowBlocks(&fSlaveTrackers[firstSlice], iSlice, false);
		}
	}
	
	StandalonePerfTime(firstSlice, 8);

	if (fDebugLevel >= 4)
	{
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			*fOutFile << "Tracklet Hits:" << endl;
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5)
			{
				printf("Obtained %d tracklets\n", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			}
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemorySize(), cudaMemcpyDeviceToHost));
			CUDA_FAILED_MSG(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fGpuTracker[iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			fSlaveTrackers[firstSlice + iSlice].DumpTrackletHits(*fOutFile);
		}
	}

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT)
	{
		AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<HLTCA_GPU_BLOCK_COUNT, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice]>>>(iSlice, CAMath::Min(HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT, sliceCountLocal - iSlice));
	}
	if (CUDASync("Tracklet Selector")) return 1;
	StandalonePerfTime(firstSlice, 9);

	CUDA_FAILED_MSG(cudaMemcpyAsync(fSlaveTrackers[firstSlice + 0].CommonMemory(), fGpuTracker[0].CommonMemory(), fGpuTracker[0].CommonMemorySize(), cudaMemcpyDeviceToHost, cudaStreams[0]));
	for (int iSliceTmp = 0;iSliceTmp <= sliceCountLocal;iSliceTmp++)
	{
		if (iSliceTmp < sliceCountLocal)
		{
			int iSlice = iSliceTmp;
			if (fDebugLevel >= 5) printf("Transfering Tracks from GPU to Host ");
			cudaStreamSynchronize(cudaStreams[iSlice]);
			CUDA_FAILED_MSG(cudaMemcpyAsync(fSlaveTrackers[firstSlice + iSlice].Tracks(), fGpuTracker[iSlice].Tracks(), sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + iSlice].NTracks(), cudaMemcpyDeviceToHost, cudaStreams[iSlice]));
			CUDA_FAILED_MSG(cudaMemcpyAsync(fSlaveTrackers[firstSlice + iSlice].TrackHits(), fGpuTracker[iSlice].TrackHits(), sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + iSlice].NTrackHits(), cudaMemcpyDeviceToHost, cudaStreams[iSlice]));
			if (iSlice + 1 < sliceCountLocal)
				CUDA_FAILED_MSG(cudaMemcpyAsync(fSlaveTrackers[firstSlice + iSlice + 1].CommonMemory(), fGpuTracker[iSlice + 1].CommonMemory(), fGpuTracker[iSlice + 1].CommonMemorySize(), cudaMemcpyDeviceToHost, cudaStreams[iSlice + 1]));
		}

		if (iSliceTmp)
		{
			int iSlice = iSliceTmp - 1;
			cudaStreamSynchronize(cudaStreams[iSlice]);

			if (fDebugLevel >= 4)
			{
				*fOutFile << "Track Hits: (" << *fSlaveTrackers[firstSlice + iSlice].NTracks() << ")" << endl;
				fSlaveTrackers[firstSlice + iSlice].DumpTrackHits(*fOutFile);
			}

			if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError)
			{
				printf("GPU Tracker returned Error Code %d\n", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError);
				return(1);
			}
			if (fDebugLevel >= 5) printf("%d / %d\n", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

			fSlaveTrackers[firstSlice + iSlice].SetOutput(&pOutput[iSlice]);
			fSlaveTrackers[firstSlice + iSlice].WriteOutput();

			if (fDebugLevel >= 4)
			{
				delete[] fSlaveTrackers[firstSlice + iSlice].HitMemory();
				delete[] fSlaveTrackers[firstSlice + iSlice].TrackletMemory();
			}
		}
	}

	StandalonePerfTime(firstSlice, 10);

	if (fDebugLevel >= 5) printf("GPU Reconstruction finished\n");

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* stageAtSync = (char*) malloc(100000000);
	CUDA_FAILED_MSG(cudaMemcpy(stageAtSync, fGpuTracker[0].fStageAtSync, 100 * 1000 * 1000, cudaMemcpyDeviceToHost));
	cudaFree(fGpuTracker[0].fStageAtSync);

	FILE* fp = fopen("profile.txt", "w+");
	FILE* fp2 = fopen("profile.bmp", "w+b");
	int nEmptySync = 0, fEmpty;

	const int bmpheight = 1000;
	BITMAPFILEHEADER bmpFH;
	BITMAPINFOHEADER bmpIH;
	ZeroMemory(&bmpFH, sizeof(bmpFH));
	ZeroMemory(&bmpIH, sizeof(bmpIH));
	
	bmpFH.bfType = 19778; //"BM"
	bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + (HLTCA_GPU_BLOCK_COUNT * HLTCA_GPU_THREAD_COUNT / 32 * 33 - 1) * bmpheight ;
	bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

	bmpIH.biSize = sizeof(bmpIH);
	bmpIH.biWidth = HLTCA_GPU_BLOCK_COUNT * HLTCA_GPU_THREAD_COUNT / 32 * 33 - 1;
	bmpIH.biHeight = bmpheight;
	bmpIH.biPlanes = 1;
	bmpIH.biBitCount = 32;

	fwrite(&bmpFH, 1, sizeof(bmpFH), fp2);
	fwrite(&bmpIH, 1, sizeof(bmpIH), fp2); 	

	for (int i = 0;i < bmpheight * HLTCA_GPU_BLOCK_COUNT * HLTCA_GPU_THREAD_COUNT;i += HLTCA_GPU_BLOCK_COUNT * HLTCA_GPU_THREAD_COUNT)
	{
		fEmpty = 1;
		for (int j = 0;j < HLTCA_GPU_BLOCK_COUNT * HLTCA_GPU_THREAD_COUNT;j++)
		{
			fprintf(fp, "%d\t", stageAtSync[i + j]);
			int color = 0;
			if (stageAtSync[i + j] == 1) color = RGB(255, 0, 0);
			if (stageAtSync[i + j] == 2) color = RGB(0, 255, 0);
			if (stageAtSync[i + j] == 3) color = RGB(0, 0, 255);
			if (stageAtSync[i + j] == 4) color = RGB(255, 255, 0);
			fwrite(&color, 1, sizeof(int), fp2);
			if (j > 0 && j % 32 == 0)
			{
				color = RGB(255, 255, 255);
				fwrite(&color, 1, 4, fp2);
			}
			if (stageAtSync[i + j]) fEmpty = 0;
		}
		fprintf(fp, "\n");
		if (fEmpty) nEmptySync++;
		else nEmptySync = 0;
		//if (nEmptySync == HLTCA_GPU_SCHED_ROW_STEP + 2) break;
	}

	fclose(fp);
	fclose(fp2);
	free(stageAtSync);
#endif 

	return(0);
}

int AliHLTTPCCAGPUTracker::InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param)
{
	fSlaveTrackers[iSlice].Initialize(param);
	if (fSlaveTrackers[iSlice].Param().NRows() != HLTCA_ROW_COUNT)
	{
		printf("Error, Slice Tracker %d Row Count of %d exceeds Constant of %d\n", iSlice, fSlaveTrackers[iSlice].Param().NRows(), HLTCA_ROW_COUNT);
		return(1);
	}
	return(0);
}

int AliHLTTPCCAGPUTracker::ExitGPU()
{
	cudaThreadSynchronize();
	if (fGPUMemory)
	{
		cudaFree(fGPUMemory);
		fGPUMemory = NULL;
	}
	if (fHostLockedMemory)
	{
		for (int i = 0;i < CAMath::Max(3, fSliceCount);i++)
		{
			cudaStreamDestroy(((cudaStream_t*) pCudaStreams)[i]);
		}
		free(pCudaStreams);
		fGpuTracker = NULL;
		cudaFreeHost(fHostLockedMemory);
	}

	if (CUDA_FAILED_MSG(cudaThreadExit()))
	{
		printf("Could not uninitialize GPU\n");
		return(1);
	}
	printf("CUDA Uninitialized\n");
	return(0);
}
