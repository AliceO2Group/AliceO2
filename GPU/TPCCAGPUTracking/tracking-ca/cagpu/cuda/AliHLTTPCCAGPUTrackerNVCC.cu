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

#define FERMI
#include "AliHLTTPCCAGPUTrackerNVCC.h"
#include "AliHLTTPCCAGPUTrackerCommon.h"
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size(dim) (blockDim.x * gridDim.x)
#define get_num_groups(dim) (gridDim.x)
#define get_local_id(dim) (threadIdx.x)
#define get_local_size(dim) (blockDim.x)
#define get_group_id(dim) (blockIdx.x)

#include <cuda.h>
#include <sm_11_atomic_functions.h>
#include <sm_12_atomic_functions.h>

__constant__ float4 gAliHLTTPCCATracker[HLTCA_GPU_TRACKER_CONSTANT_MEM / sizeof( float4 )];
#ifdef HLTCA_GPU_TEXTURE_FETCH
texture<ushort2, 1, cudaReadModeElementType> gAliTexRefu2;
texture<unsigned short, 1, cudaReadModeElementType> gAliTexRefu;
texture<signed short, 1, cudaReadModeElementType> gAliTexRefs;
#endif

//Include CXX Files, GPUd() macro will then produce CUDA device code out of the tracker source code
#include "AliHLTTPCCATrackParam.cxx"
#include "AliHLTTPCCATrack.cxx" 

#include "AliHLTTPCCAHitArea.cxx"
#include "AliHLTTPCCAGrid.cxx"
#include "AliHLTTPCCARow.cxx"
#include "AliHLTTPCCAParam.cxx"
#include "AliHLTTPCCATracker.cxx"

#include "AliHLTTPCCAProcess.h"

#include "AliHLTTPCCATrackletSelector.cxx"
#include "AliHLTTPCCANeighboursFinder.cxx"
#include "AliHLTTPCCANeighboursCleaner.cxx"
#include "AliHLTTPCCAStartHitsFinder.cxx"
#include "AliHLTTPCCAStartHitsSorter.cxx"
#include "AliHLTTPCCATrackletConstructor.cxx"

#ifdef HLTCA_GPU_MERGER
#include "AliHLTTPCGMMerger.h"
#include "AliHLTTPCGMTrackParam.cxx"
#endif

ClassImp( AliHLTTPCCAGPUTrackerNVCC )

AliHLTTPCCAGPUTrackerNVCC::AliHLTTPCCAGPUTrackerNVCC() : fpCudaStreams(NULL)
{
	fCudaContext = (void*) new CUcontext;
};

AliHLTTPCCAGPUTrackerNVCC::~AliHLTTPCCAGPUTrackerNVCC()
{
	delete (CUcontext*) fCudaContext;
};

int AliHLTTPCCAGPUTrackerNVCC::InitGPU_Runtime(int sliceCount, int forceDeviceID)
{
	//Find best CUDA device, initialize and allocate memory

	cudaDeviceProp fCudaDeviceProp;

#ifndef CUDA_DEVICE_EMULATION
	int count, bestDevice = -1;
	long long int bestDeviceSpeed = 0, deviceSpeed;
	if (GPUFailedMsg(cudaGetDeviceCount(&count)))
	{
		HLTError("Error getting CUDA Device Count");
		return(1);
	}
	if (fDebugLevel >= 2) HLTInfo("Available CUDA devices:");
#if defined(FERMI) || defined(KEPLER)
	const int reqVerMaj = 2;
	const int reqVerMin = 0;
#else
	const int reqVerMaj = 1;
	const int reqVerMin = 2;
#endif
	for (int i = 0;i < count;i++)
	{
		if (fDebugLevel >= 4) printf("Examining device %d\n", i);
#if CUDA_VERSION > 3010
		size_t free, total;
#else
		unsigned int free, total;
#endif
		cuInit(0);
		CUdevice tmpDevice;
		cuDeviceGet(&tmpDevice, i);
		CUcontext tmpContext;
		cuCtxCreate(&tmpContext, 0, tmpDevice);
		if(cuMemGetInfo(&free, &total)) std::cout << "Error\n";
		cuCtxDestroy(tmpContext);
		if (fDebugLevel >= 4) printf("Obtained current memory usage for device %d\n", i);
		if (GPUFailedMsg(cudaGetDeviceProperties(&fCudaDeviceProp, i))) continue;
		if (fDebugLevel >= 4) printf("Obtained device properties for device %d\n", i);
		int deviceOK = fCudaDeviceProp.major < 9 && !(fCudaDeviceProp.major < reqVerMaj || (fCudaDeviceProp.major == reqVerMaj && fCudaDeviceProp.minor < reqVerMin)) && free >= fGPUMemSize + 100 * 1024 + 1024;
#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
		//if (sliceCount > fCudaDeviceProp.multiProcessorCount * HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER) deviceOK = 0;
#endif

		if (fDebugLevel >= 2) HLTInfo("%s%2d: %s (Rev: %d.%d - Mem Avail %lld / %lld)%s", deviceOK ? " " : "[", i, fCudaDeviceProp.name, fCudaDeviceProp.major, fCudaDeviceProp.minor, (long long int) free, (long long int) fCudaDeviceProp.totalGlobalMem, deviceOK ? "" : " ]");
		deviceSpeed = (long long int) fCudaDeviceProp.multiProcessorCount * (long long int) fCudaDeviceProp.clockRate * (long long int) fCudaDeviceProp.warpSize * (long long int) free * (long long int) fCudaDeviceProp.major * (long long int) fCudaDeviceProp.major;
		if (deviceOK && deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
	}
	if (bestDevice == -1)
	{
		HLTWarning("No %sCUDA Device available, aborting CUDA Initialisation", count ? "appropriate " : "");
		HLTInfo("Requiring Revision %d.%d, Mem: %lld, Multiprocessors: %d", reqVerMaj, reqVerMin, fGPUMemSize + 100 * 1024 * 1024, sliceCount);
		return(1);
	}

	if (forceDeviceID == -1)
		fCudaDevice = bestDevice;
	else
		fCudaDevice = forceDeviceID;
#else
	fCudaDevice = 0;
#endif

	cudaGetDeviceProperties(&fCudaDeviceProp ,fCudaDevice ); 

	if (fDebugLevel >= 1)
	{
		HLTInfo("Using CUDA Device %s with Properties:", fCudaDeviceProp.name);
		HLTInfo("totalGlobalMem = %lld", (unsigned long long int) fCudaDeviceProp.totalGlobalMem);
		HLTInfo("sharedMemPerBlock = %lld", (unsigned long long int) fCudaDeviceProp.sharedMemPerBlock);
		HLTInfo("regsPerBlock = %d", fCudaDeviceProp.regsPerBlock);
		HLTInfo("warpSize = %d", fCudaDeviceProp.warpSize);
		HLTInfo("memPitch = %lld", (unsigned long long int) fCudaDeviceProp.memPitch);
		HLTInfo("maxThreadsPerBlock = %d", fCudaDeviceProp.maxThreadsPerBlock);
		HLTInfo("maxThreadsDim = %d %d %d", fCudaDeviceProp.maxThreadsDim[0], fCudaDeviceProp.maxThreadsDim[1], fCudaDeviceProp.maxThreadsDim[2]);
		HLTInfo("maxGridSize = %d %d %d", fCudaDeviceProp.maxGridSize[0], fCudaDeviceProp.maxGridSize[1], fCudaDeviceProp.maxGridSize[2]);
		HLTInfo("totalConstMem = %lld", (unsigned long long int) fCudaDeviceProp.totalConstMem);
		HLTInfo("major = %d", fCudaDeviceProp.major);
		HLTInfo("minor = %d", fCudaDeviceProp.minor);
		HLTInfo("clockRate = %d", fCudaDeviceProp.clockRate);
		HLTInfo("memoryClockRate = %d", fCudaDeviceProp.memoryClockRate);
		HLTInfo("multiProcessorCount = %d", fCudaDeviceProp.multiProcessorCount);
		HLTInfo("textureAlignment = %lld", (unsigned long long int) fCudaDeviceProp.textureAlignment);
	}
	fConstructorBlockCount = fCudaDeviceProp.multiProcessorCount * HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
	selectorBlockCount = fCudaDeviceProp.multiProcessorCount * HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;

	if (fCudaDeviceProp.major < 1 || (fCudaDeviceProp.major == 1 && fCudaDeviceProp.minor < 2))
	{
		HLTError( "Unsupported CUDA Device" );
		return(1);
	}

	if (cuCtxCreate((CUcontext*) fCudaContext, CU_CTX_SCHED_AUTO, fCudaDevice) != CUDA_SUCCESS)
	{
		HLTError("Could not set CUDA Device!");
		return(1);
	}

	if (fGPUMemSize > fCudaDeviceProp.totalGlobalMem || GPUFailedMsg(cudaMalloc(&fGPUMemory, (size_t) fGPUMemSize)))
	{
		HLTError("CUDA Memory Allocation Error");
		cudaThreadExit();
		return(1);
	}
	fGPUMergerMemory = ((char*) fGPUMemory) + fGPUMemSize - fGPUMergerMaxMemory;
	if (fDebugLevel >= 1) HLTInfo("GPU Memory used: %d", (int) fGPUMemSize);
	int hostMemSize = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + sliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + HLTCA_GPU_TRACKER_OBJECT_MEMORY;
#ifdef HLTCA_GPU_MERGER
	hostMemSize += fGPUMergerMaxMemory;
#endif
	if (GPUFailedMsg(cudaMallocHost(&fHostLockedMemory, hostMemSize)))
	{
		cudaFree(fGPUMemory);
		cudaThreadExit();
		HLTError("Error allocating Page Locked Host Memory");
		return(1);
	}
	if (fDebugLevel >= 1) HLTInfo("Host Memory used: %d", hostMemSize);
	fGPUMergerHostMemory = ((char*) fHostLockedMemory) + hostMemSize - fGPUMergerMaxMemory;

	if (fDebugLevel >= 1)
	{
		GPUFailedMsg(cudaMemset(fGPUMemory, 143, (size_t) fGPUMemSize));
	}

	fpCudaStreams = malloc(CAMath::Max(3, fSliceCount) * sizeof(cudaStream_t));
	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;
	for (int i = 0;i < CAMath::Max(3, fSliceCount);i++)
	{
		if (GPUFailedMsg(cudaStreamCreate(&cudaStreams[i])))
		{
			cudaFree(fGPUMemory);
			cudaFreeHost(fHostLockedMemory);
			cudaThreadExit();
			HLTError("Error creating CUDA Stream");
			return(1);
		}
	}

	cuCtxPopCurrent((CUcontext*) fCudaContext);
	HLTImportant("CUDA Initialisation successfull (Device %d: %s, Thread %d, Max slices: %d)", fCudaDevice, fCudaDeviceProp.name, fThreadId, fSliceCount);

	return(0);
}

bool AliHLTTPCCAGPUTrackerNVCC::GPUFailedMsgA(cudaError_t error, const char* file, int line)
{
	//Check for CUDA Error and in the case of an error display the corresponding error string
	if (error == cudaSuccess) return(false);
	HLTWarning("CUDA Error: %d / %s (%s:%d)", error, cudaGetErrorString(error), file, line);
	return(true);
}

int AliHLTTPCCAGPUTrackerNVCC::GPUSync(char* state, int stream, int slice)
{
	//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards

	if (fDebugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		HLTError("Cuda Error %s while running kernel (%s) (Stream %d; %d/%d)", cudaGetErrorString(cuErr), state, stream, slice, fgkNSlices);
		return(1);
	}
	if (GPUFailedMsg(cudaThreadSynchronize()))
	{
		HLTError("CUDA Error while synchronizing (%s) (Stream %d; %d/%d)", state, stream, slice, fgkNSlices);
		return(1);
	}
	if (fDebugLevel >= 3) HLTInfo("CUDA Sync Done");
	return(0);
}

#if defined(BITWISE_COMPATIBLE_DEBUG_OUTPUT) || defined(HLTCA_GPU_ALTERNATIVE_SCHEDULER)
void AliHLTTPCCAGPUTrackerNVCC::DumpRowBlocks(AliHLTTPCCATracker*, int, bool) {}
#else
void AliHLTTPCCAGPUTrackerNVCC::DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check)
{
	//Dump Rowblocks to File
	if (fDebugLevel >= 4)
	{
		*fOutFile << "RowBlock Tracklets (Slice" << tracker[iSlice].Param().ISlice() << " (" << iSlice << " of reco))";
		*fOutFile << " after Tracklet Reconstruction";
		*fOutFile << std::endl;

		int4* rowBlockPos = (int4*) malloc(sizeof(int4) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2);
		int* rowBlockTracklets = (int*) malloc(sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2);
		uint2* blockStartingTracklet = (uint2*) malloc(sizeof(uint2) * fConstructorBlockCount);
		GPUFailedMsg(cudaMemcpy(rowBlockPos, fGpuTracker[iSlice].RowBlockPos(), sizeof(int4) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2, cudaMemcpyDeviceToHost));
		GPUFailedMsg(cudaMemcpy(rowBlockTracklets, fGpuTracker[iSlice].RowBlockTracklets(), sizeof(int) * (tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2, cudaMemcpyDeviceToHost));
		GPUFailedMsg(cudaMemcpy(blockStartingTracklet, fGpuTracker[iSlice].BlockStartingTracklet(), sizeof(uint2) * fConstructorBlockCount, cudaMemcpyDeviceToHost));
		GPUFailedMsg(cudaMemcpy(tracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));

		int k = tracker[iSlice].GPUParameters()->fScheduleFirstDynamicTracklet;
		for (int i = 0; i < tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1;i++)
		{
			*fOutFile << "Rowblock: " << i << ", up " << rowBlockPos[i].y << "/" << rowBlockPos[i].x << ", down " << 
				rowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].y << "/" << rowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x << std::endl << "Phase 1: ";
			for (int j = 0;j < rowBlockPos[i].x;j++)
			{
				//Use Tracker Object to calculate Offset instead of fGpuTracker, since *fNTracklets of fGpuTracker points to GPU Mem!
				*fOutFile << rowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] << ", ";
#ifdef HLTCA_GPU_SCHED_FIXED_START
				if (check && rowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] != k)
				{
					HLTError("Wrong starting Row Block %d, entry %d, is %d, should be %d", i, j, rowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j], k);
				}
#endif //HLTCA_GPU_SCHED_FIXED_START
				k++;
				if (rowBlockTracklets[(tracker[iSlice].RowBlockTracklets(0, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] == -1)
				{
					HLTError("Error, -1 Tracklet found");
				}
			}
			*fOutFile << std::endl << "Phase 2: ";
			for (int j = 0;j < rowBlockPos[tracker[iSlice].Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1 + i].x;j++)
			{
				*fOutFile << rowBlockTracklets[(tracker[iSlice].RowBlockTracklets(1, i) - tracker[iSlice].RowBlockTracklets(0, 0)) + j] << ", ";
			}
			*fOutFile << std::endl;
		}

		if (check)
		{
			*fOutFile << "Starting Threads: (Slice" << tracker[iSlice].Param().ISlice() << ", First Dynamic: " << tracker[iSlice].GPUParameters()->fScheduleFirstDynamicTracklet << ")" << std::endl;
			for (int i = 0;i < fConstructorBlockCount;i++)
			{
				*fOutFile << i << ": " << blockStartingTracklet[i].x << " - " << blockStartingTracklet[i].y << std::endl;
			}
		}

		free(rowBlockPos);
		free(rowBlockTracklets);
		free(blockStartingTracklet);
	}
}
#endif

__global__ void PreInitRowBlocks(int4* const RowBlockPos, int* const RowBlockTracklets, int* const SliceDataHitWeights, int nSliceDataHits)
{
	//Initialize GPU RowBlocks and HitWeights
	int4* const sliceDataHitWeights4 = (int4*) SliceDataHitWeights;
	const int stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
	int4* const rowBlockTracklets4 = (int4*) RowBlockTracklets;
	int4 i1;
	i1.x = i1.y = i1.z = i1.w = -1;
	for (int i = get_global_id(0);i < sizeof(int4) * 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1) / sizeof(int4);i += stride)
		RowBlockPos[i] = i0;
	for (int i = get_global_id(0);i < sizeof(int) * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_MAX_TRACKLETS * 2 / sizeof(int4);i += stride)
		rowBlockTracklets4[i] = i1;
#endif
	for (int i = get_global_id(0);i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		sliceDataHitWeights4[i] = i0;
}

int AliHLTTPCCAGPUTrackerNVCC::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function

	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;

	if (Reconstruct_Base_Init(pOutput, pClusterData, firstSlice, sliceCountLocal)) return(1);

#ifdef HLTCA_GPU_TEXTURE_FETCH
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<ushort2>();
	size_t offset;
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, fGpuTracker[0].Data().Memory(), &channelDescu2, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		HLTError("Error binding CUDA Texture ushort2 (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<unsigned short>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, fGpuTracker[0].Data().Memory(), &channelDescu, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		HLTError("Error binding CUDA Texture ushort (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
	cudaChannelFormatDesc channelDescs = cudaCreateChannelDesc<signed short>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefs, fGpuTracker[0].Data().Memory(), &channelDescs, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		HLTError("Error binding CUDA Texture short (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
#endif

	//Copy Tracker Object to GPU Memory
	if (fDebugLevel >= 3) HLTInfo("Copying Tracker objects to GPU");
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* tmpMem;
	if (GPUFailedMsg(cudaMalloc(&tmpMem, 100000000)))
	{
		HLTError("Error allocating CUDA profile memory");
		ResetHelperThreads(0);
		return(1);
	}
	fGpuTracker[0].fStageAtSync = tmpMem;
	GPUFailedMsg(cudaMemset(fGpuTracker[0].StageAtSync(), 0, 100000000));
#endif
	GPUFailedMsg(cudaMemcpyToSymbolAsync(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker) * sliceCountLocal, 0, cudaMemcpyHostToDevice, cudaStreams[0]));
	if (GPUSync("Initialization (1)", 0, firstSlice) RANDOM_ERROR)
	{
		ResetHelperThreads(0);
		return(1);
	}

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(pClusterData, iSlice, firstSlice)) return(1);

		//Initialize temporary memory where needed
		if (fDebugLevel >= 3) HLTInfo("Copying Slice Data to GPU and initializing temporary memory");		
		PreInitRowBlocks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[2]>>>(fGpuTracker[iSlice].RowBlockPos(), fGpuTracker[iSlice].RowBlockTracklets(), fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign());
		if (GPUSync("Initialization (2)", 2, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].SliceDataRows(), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));

		if (fDebugLevel >= 4)
		{
			if (fDebugLevel >= 5) HLTInfo("Allocating Debug Output Memory");
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100]), pClusterData[iSlice].NumberOfClusters() );
		}

		if (GPUSync("Initialization (3)", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		StandalonePerfTime(firstSlice + iSlice, 1);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows(), HLTCA_GPU_THREAD_COUNT_FINDER, 0, cudaStreams[iSlice & 1]>>>(iSlice);

		if (GPUSync("Neighbours finder", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 2);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 2) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-2, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Neighbours Cleaner", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 3);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 4) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-6, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Start Hits Finder", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 4);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Start Hits Sorter", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 5);

		if (fDebugLevel >= 2)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 3) HLTInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *fSlaveTrackers[firstSlice + iSlice].NTracklets(), iSlice);
			if (*fSlaveTrackers[firstSlice + iSlice].NTracklets() > HLTCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				HLTError("HLTCA_GPU_MAX_TRACKLETS constant insuffisant");
				ResetHelperThreads(1);
				return(1);
			}
		}

		if (fDebugLevel >= 4 && *fSlaveTrackers[firstSlice + iSlice].NTracklets())
		{
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletStartHits(), fGpuTracker[iSlice].TrackletTmpStartHits(), pClusterData[iSlice].NumberOfClusters() * sizeof(AliHLTTPCCAHitId), cudaMemcpyDeviceToHost));
			if (fDebugMask & 8)
			{
				*fOutFile << "Temporary ";
				fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
			}
			uint3* tmpMemory = (uint3*) malloc(sizeof(uint3) * fSlaveTrackers[firstSlice + iSlice].Param().NRows());
			GPUFailedMsg(cudaMemcpy(tmpMemory, fGpuTracker[iSlice].RowStartHitCountOffset(), fSlaveTrackers[firstSlice + iSlice].Param().NRows() * sizeof(uint3), cudaMemcpyDeviceToHost));
			if (fDebugMask & 16)
			{
				*fOutFile << "Start Hits Sort Vector:" << std::endl;
				for (int i = 1;i < fSlaveTrackers[firstSlice + iSlice].Param().NRows() - 5;i++)
				{
					*fOutFile << "Row: " << i << ", Len: " << tmpMemory[i].x << ", Offset: " << tmpMemory[i].y << ", New Offset: " << tmpMemory[i].z << std::endl;
				}
			}
			free(tmpMemory);
#endif

			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fSlaveTrackers[firstSlice + iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 32) fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
		}

		StandalonePerfTime(firstSlice + iSlice, 6);

		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
	}

	for (int i = 0;i < fNHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	StandalonePerfTime(firstSlice, 7);

#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
	int nHardCollisions = 0;

RestartTrackletConstructor:
	if (fDebugLevel >= 3) HLTInfo("Initialising Tracklet Constructor Scheduler");
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		AliHLTTPCCATrackletConstructorInit<<<HLTCA_GPU_MAX_TRACKLETS /* *fSlaveTrackers[firstSlice + iSlice].NTracklets() */ / HLTCA_GPU_THREAD_COUNT + 1, HLTCA_GPU_THREAD_COUNT>>>(iSlice);
		if (GPUSync("Tracklet Initializer", -1, iSlice + firstSlice) RANDOM_ERROR)
		{
			cudaThreadSynchronize();
			cuCtxPopCurrent((CUcontext*) fCudaContext);
			return(1);
		}
		if (fDebugMask & 64) DumpRowBlocks(&fSlaveTrackers[firstSlice], iSlice);
	}
#endif

	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");
	AliHLTTPCCATrackletConstructorGPU<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (GPUSync("Tracklet Constructor", -1, firstSlice) RANDOM_ERROR)
	{
		cudaThreadSynchronize();
		cuCtxPopCurrent((CUcontext*) fCudaContext);
		return(1);
	}

	StandalonePerfTime(firstSlice, 8);

	if (fDebugLevel >= 4)
	{
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			if (fDebugMask & 64) DumpRowBlocks(&fSlaveTrackers[firstSlice], iSlice, false);
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5)
			{
				HLTInfo("Obtained %d tracklets", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			}
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemorySize(), cudaMemcpyDeviceToHost));
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fGpuTracker[iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			if (0 && fSlaveTrackers[firstSlice + iSlice].NTracklets() && fSlaveTrackers[firstSlice + iSlice].Tracklet(0).NHits() < 0)
			{
				cudaThreadSynchronize();
				cuCtxPopCurrent((CUcontext*) fCudaContext);
				printf("INTERNAL ERROR\n");
				return(1);
			}
			if (fDebugMask & 128) fSlaveTrackers[firstSlice + iSlice].DumpTrackletHits(*fOutFile);
		}
	}

	int runSlices = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += runSlices)
	{
		if (runSlices < HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		if (fDebugLevel >= 3) HLTInfo("Running HLT Tracklet selector (Slice %d to %d)", iSlice, iSlice + runSlices);
		AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<selectorBlockCount, HLTCA_GPU_THREAD_COUNT_SELECTOR, 0, cudaStreams[iSlice]>>>(iSlice, CAMath::Min(runSlices, sliceCountLocal - iSlice));
		if (GPUSync("Tracklet Selector", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			cudaThreadSynchronize();
			cuCtxPopCurrent((CUcontext*) fCudaContext);
			return(1);
		}
	}
	StandalonePerfTime(firstSlice, 9);

	char *tmpMemoryGlobalTracking = NULL;
	fSliceOutputReady = 0;
	
	if (Reconstruct_Base_StartGlobal(pOutput, tmpMemoryGlobalTracking)) return(1);

	int tmpSlice = 0, tmpSlice2 = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (fDebugLevel >= 3) HLTInfo("Transfering Tracks from GPU to Host");

		while(tmpSlice < sliceCountLocal && (tmpSlice == iSlice || cudaStreamQuery(cudaStreams[tmpSlice]) == (cudaError_t) CUDA_SUCCESS))
		{
			if (GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice].CommonMemory(), fGpuTracker[tmpSlice].CommonMemory(), fGpuTracker[tmpSlice].CommonMemorySize(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice])) RANDOM_ERROR)
			{
				ResetHelperThreads(1);
				ActivateThreadContext();
				return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
			}
			tmpSlice++;
		}

		while (tmpSlice2 < tmpSlice && (tmpSlice2 == iSlice ? cudaStreamSynchronize(cudaStreams[tmpSlice2]) : cudaStreamQuery(cudaStreams[tmpSlice2])) == (cudaError_t) CUDA_SUCCESS)
		{
			if (*fSlaveTrackers[firstSlice + tmpSlice2].NTracks() > 0)
			{
				GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice2].Tracks(), fGpuTracker[tmpSlice2].Tracks(), sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + tmpSlice2].NTracks(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice2]));
				GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice2].TrackHits(), fGpuTracker[tmpSlice2].TrackHits(), sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + tmpSlice2].NTrackHits(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice2]));
			}
			tmpSlice2++;
		}

		if (GPUFailedMsg(cudaStreamSynchronize(cudaStreams[iSlice])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
		}

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().HitWeights(), fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign() * sizeof(int), cudaMemcpyDeviceToHost));
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
			if (fDebugMask & 256) fSlaveTrackers[firstSlice + iSlice].DumpHitWeights(*fOutFile);
#endif
			if (fDebugMask & 512) fSlaveTrackers[firstSlice + iSlice].DumpTrackHits(*fOutFile);
		}

		if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
			if ((fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError == HLTCA_GPU_ERROR_SCHEDULE_COLLISION || fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError == HLTCA_GPU_ERROR_WRONG_ROW)&& nHardCollisions++ < 10)
			{
				if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError == HLTCA_GPU_ERROR_SCHEDULE_COLLISION)
				{
					HLTWarning("Hard scheduling collision occured, rerunning Tracklet Constructor (Slice %d)", firstSlice + iSlice);
				}
				else
				{
					HLTWarning("Tracklet Constructor returned invalid row (Slice %d)", firstSlice + iSlice);
				}
				if (fDebugLevel >= 4)
				{
					ResetHelperThreads(1);
					return(1);
				}
				for (int i = 0;i < sliceCountLocal;i++)
				{
					cudaThreadSynchronize();
					GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + i].CommonMemory(), fGpuTracker[i].CommonMemory(), fGpuTracker[i].CommonMemorySize(), cudaMemcpyDeviceToHost));
					*fSlaveTrackers[firstSlice + i].NTracks() = 0;
					*fSlaveTrackers[firstSlice + i].NTrackHits() = 0;
					fSlaveTrackers[firstSlice + i].GPUParameters()->fGPUError = HLTCA_GPU_ERROR_NONE;
					GPUFailedMsg(cudaMemcpy(fGpuTracker[i].CommonMemory(), fSlaveTrackers[firstSlice + i].CommonMemory(), fGpuTracker[i].CommonMemorySize(), cudaMemcpyHostToDevice));
					PreInitRowBlocks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(fGpuTracker[i].RowBlockPos(), fGpuTracker[i].RowBlockTracklets(), fGpuTracker[i].Data().HitWeights(), fSlaveTrackers[firstSlice + i].Data().NumberOfHitsPlusAlign());
				}
				goto RestartTrackletConstructor;
			}
#endif
			HLTError("GPU Tracker returned Error Code %d in slice %d", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError, firstSlice + iSlice);
			ResetHelperThreads(1);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(pOutput, iSlice, firstSlice)) return(1);

		if (fDebugLevel >= 4)
		{
			delete[] fSlaveTrackers[firstSlice + iSlice].HitMemory();
			delete[] fSlaveTrackers[firstSlice + iSlice].TrackletMemory();
		}
	}

	if (Reconstruct_Base_Finalize(pOutput, tmpMemoryGlobalTracking, firstSlice)) return(1);

	/*for (int i = firstSlice;i < firstSlice + sliceCountLocal;i++)
	{
		fSlaveTrackers[i].DumpOutput(stdout);
	}*/

	/*static int runnum = 0;
	std::ofstream tmpOut;
	char buffer[1024];
	sprintf(buffer, "GPUtracks%d.out", runnum++);
	tmpOut.open(buffer);
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		fSlaveTrackers[firstSlice + iSlice].DumpTrackHits(tmpOut);
	}
	tmpOut.close();*/

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* stageAtSync = (char*) malloc(100000000);
	GPUFailedMsg(cudaMemcpy(stageAtSync, fGpuTracker[0].StageAtSync(), 100 * 1000 * 1000, cudaMemcpyDeviceToHost));
	cudaFree(fGpuTracker[0].StageAtSync());

	FILE* fp = fopen("profile.txt", "w+");
	FILE* fp2 = fopen("profile.bmp", "w+b");
	int nEmptySync = 0, fEmpty;

	const int bmpheight = 8192;
	BITMAPFILEHEADER bmpFH;
	BITMAPINFOHEADER bmpIH;
	ZeroMemory(&bmpFH, sizeof(bmpFH));
	ZeroMemory(&bmpIH, sizeof(bmpIH));

	bmpFH.bfType = 19778; //"BM"
	bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + (fConstructorBlockCount * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR / 32 * 33 - 1) * bmpheight ;
	bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

	bmpIH.biSize = sizeof(bmpIH);
	bmpIH.biWidth = fConstructorBlockCount * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR / 32 * 33 - 1;
	bmpIH.biHeight = bmpheight;
	bmpIH.biPlanes = 1;
	bmpIH.biBitCount = 32;

	fwrite(&bmpFH, 1, sizeof(bmpFH), fp2);
	fwrite(&bmpIH, 1, sizeof(bmpIH), fp2); 	

	for (int i = 0;i < bmpheight * fConstructorBlockCount * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;i += fConstructorBlockCount * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR)
	{
		fEmpty = 1;
		for (int j = 0;j < fConstructorBlockCount * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;j++)
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

	cuCtxPopCurrent((CUcontext*) fCudaContext);
	return(0);
}

__global__ void ClearPPHitWeights(int sliceCount)
{
	//Clear HitWeights

	for (int k = 0;k < sliceCount;k++)
	{
		AliHLTTPCCATracker &tracker = ((AliHLTTPCCATracker*) gAliHLTTPCCATracker)[k];
		int4* const pHitWeights = (int4*) tracker.Data().HitWeights();
		const int dwCount = tracker.Data().NumberOfHitsPlusAlign();
		const int stride = get_global_size(0);
		int4 i0;
		i0.x = i0.y = i0.z = i0.w = 0;

		for (int i = get_global_id(0);i < dwCount * sizeof(int) / sizeof(int4);i += stride)
		{
			pHitWeights[i] = i0;
		}
	}
}

int AliHLTTPCCAGPUTrackerNVCC::ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function for small events (PP)

	memcpy(fGpuTracker, &fSlaveTrackers[firstSlice], sizeof(AliHLTTPCCATracker) * sliceCountLocal);

	if (fDebugLevel >= 3) HLTInfo("Allocating GPU Tracker memory and initializing constants");

	char* tmpSliceMemHost = (char*) SliceDataMemory(fHostLockedMemory, 0);
	char* tmpSliceMemGpu = (char*) SliceDataMemory(fGPUMemory, 0);

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		StandalonePerfTime(firstSlice + iSlice, 0);

		//Initialize GPU Slave Tracker
		if (fDebugLevel >= 3) HLTInfo("Creating Slice Data");
		fSlaveTrackers[firstSlice + iSlice].SetGPUSliceDataMemory(tmpSliceMemHost, RowMemory(fHostLockedMemory, firstSlice + iSlice));
		fSlaveTrackers[firstSlice + iSlice].ReadEvent(&pClusterData[iSlice]);
		if (fSlaveTrackers[firstSlice + iSlice].Data().MemorySize() > HLTCA_GPU_SLICE_DATA_MEMORY)
		{
			HLTError("Insufficiant Slice Data Memory");
			return(1);
		}

		//Make this a GPU Tracker
		fGpuTracker[iSlice].SetGPUTracker();
		fGpuTracker[iSlice].SetGPUTrackerCommonMemory((char*) CommonMemory(fGPUMemory, iSlice));

		fGpuTracker[iSlice].SetGPUSliceDataMemory(tmpSliceMemGpu, RowMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].SetPointersSliceData(&pClusterData[iSlice], false);

		tmpSliceMemHost += fSlaveTrackers[firstSlice + iSlice].Data().MemorySize();
		tmpSliceMemHost = alignPointer(tmpSliceMemHost, 64 * 1024);
		tmpSliceMemGpu += fSlaveTrackers[firstSlice + iSlice].Data().MemorySize();
		tmpSliceMemGpu = alignPointer(tmpSliceMemGpu, 64 * 1024);

		//Set Pointers to GPU Memory
		char* tmpMem = (char*) GlobalMemory(fGPUMemory, iSlice);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Hits Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerHitsMemory(tmpMem, pClusterData[iSlice].NumberOfClusters());
		tmpMem = alignPointer(tmpMem, 64 * 1024);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Tracklet Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTrackletsMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
		tmpMem = alignPointer(tmpMem, 64 * 1024);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Track Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
		tmpMem = alignPointer(tmpMem, 64 * 1024);

		if (fGpuTracker[iSlice].TrackMemorySize() >= HLTCA_GPU_TRACKS_MEMORY)
		{
			HLTError("Insufficiant Track Memory");
			return(1);
		}

		if (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice) > HLTCA_GPU_GLOBAL_MEMORY)
		{
			HLTError("Insufficiant Global Memory");
			return(1);
		}

		//Initialize Startup Constants
		*fSlaveTrackers[firstSlice + iSlice].NTracklets() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTracks() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTrackHits() = 0;
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError = 0;

		fGpuTracker[iSlice].SetGPUTextureBase(fGpuTracker[0].Data().Memory());

		if (GPUSync("Initialization", -1, iSlice + firstSlice)) return(1);
		StandalonePerfTime(firstSlice + iSlice, 1);
	}

#ifdef HLTCA_GPU_TEXTURE_FETCH
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<ushort2>();
	size_t offset;
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, fGpuTracker[0].Data().Memory(), &channelDescu2, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
	{
		HLTError("Error binding CUDA Texture ushort2 (Offset %d)", (int) offset);
		return(1);
	}
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<unsigned short>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, fGpuTracker[0].Data().Memory(), &channelDescu, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
	{
		HLTError("Error binding CUDA Texture ushort (Offset %d)", (int) offset);
		return(1);
	}
	cudaChannelFormatDesc channelDescs = cudaCreateChannelDesc<signed short>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefs, fGpuTracker[0].Data().Memory(), &channelDescs, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset)
	{
		HLTError("Error binding CUDA Texture short (Offset %d)", (int) offset);
		return(1);
	}
#endif

	//Copy Tracker Object to GPU Memory
	if (fDebugLevel >= 3) HLTInfo("Copying Tracker objects to GPU");
	GPUFailedMsg(cudaMemcpyToSymbol(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker) * sliceCountLocal, 0, cudaMemcpyHostToDevice));

	//Copy Data to GPU Global Memory
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		GPUFailedMsg(cudaMemcpy(fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyHostToDevice));
		//printf("%lld %lld %d %d\n", (size_t) (char*) fGpuTracker[iSlice].Data().Memory(), (size_t) (char*) fSlaveTrackers[firstSlice + iSlice].Data().Memory(), (int) (size_t) fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), (int) (size_t) fSlaveTrackers[firstSlice + iSlice].Data().MemorySize());
	}
	//GPUFailedMsg(cudaMemcpy(SliceDataMemory(fGPUMemory, 0), SliceDataMemory(fHostLockedMemory, 0), tmpSliceMemHost - (char*) SliceDataMemory(fHostLockedMemory, 0), cudaMemcpyHostToDevice));
	//printf("%lld %lld %d\n", (size_t) (char*) SliceDataMemory(fGPUMemory, 0), (size_t) (char*) SliceDataMemory(fHostLockedMemory, 0), (int) (size_t) (tmpSliceMemHost - (char*) SliceDataMemory(fHostLockedMemory, 0)));
	GPUFailedMsg(cudaMemcpy(fGpuTracker[0].CommonMemory(), fSlaveTrackers[firstSlice].CommonMemory(), fSlaveTrackers[firstSlice].CommonMemorySize() * sliceCountLocal, cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(fGpuTracker[0].SliceDataRows(), fSlaveTrackers[firstSlice].SliceDataRows(), (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow) * sliceCountLocal, cudaMemcpyHostToDevice));

	if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder");
	AliHLTTPCCAProcessMultiA<AliHLTTPCCANeighboursFinder> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_FINDER>>>(0, sliceCountLocal, fSlaveTrackers[firstSlice].Param().NRows());
	if (GPUSync("Neighbours finder", -1, firstSlice)) return 1;
	StandalonePerfTime(firstSlice, 2);
	if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner");
	AliHLTTPCCAProcessMultiA<AliHLTTPCCANeighboursCleaner> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(0, sliceCountLocal, fSlaveTrackers[firstSlice].Param().NRows() - 2);
	if (GPUSync("Neighbours Cleaner", -1, firstSlice)) return 1;
	StandalonePerfTime(firstSlice, 3);
	if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder");
	AliHLTTPCCAProcessMultiA<AliHLTTPCCAStartHitsFinder> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(0, sliceCountLocal, fSlaveTrackers[firstSlice].Param().NRows() - 6);
	if (GPUSync("Start Hits Finder", -1, firstSlice)) return 1;
	StandalonePerfTime(firstSlice, 4);

	ClearPPHitWeights <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(sliceCountLocal);
	if (GPUSync("Clear Hit Weights", -1, firstSlice)) return 1;

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
	}

	StandalonePerfTime(firstSlice, 7);

	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");
	AliHLTTPCCATrackletConstructorGPUPP<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>(0, sliceCountLocal);
	if (GPUSync("Tracklet Constructor PP", -1, firstSlice)) return 1;

	StandalonePerfTime(firstSlice, 8);

	AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<selectorBlockCount, HLTCA_GPU_THREAD_COUNT_SELECTOR>>>(0, sliceCountLocal);
	if (GPUSync("Tracklet Selector", -1, firstSlice)) return 1;
	StandalonePerfTime(firstSlice, 9);

	GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice].CommonMemory(), fGpuTracker[0].CommonMemory(), fSlaveTrackers[firstSlice].CommonMemorySize() * sliceCountLocal, cudaMemcpyDeviceToHost));

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (fDebugLevel >= 3) HLTInfo("Transfering Tracks from GPU to Host");

		GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Tracks(), fGpuTracker[iSlice].Tracks(), sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + iSlice].NTracks(), cudaMemcpyDeviceToHost));
		GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackHits(), fGpuTracker[iSlice].TrackHits(), sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + iSlice].NTrackHits(), cudaMemcpyDeviceToHost));

		if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError)
		{
			HLTError("GPU Tracker returned Error Code %d", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());
		fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNLocalTracks = fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNTracks;
		fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNLocalTrackHits = fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNTrackHits;
	}

	if (fGlobalTracking && sliceCountLocal == fgkNSlices)
	{
		char tmpMemory[sizeof(AliHLTTPCCATracklet)
#ifdef EXTERN_ROW_HITS
		+ HLTCA_ROW_COUNT * sizeof(int)
#endif
		+ 16];

		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			if (fSlaveTrackers[iSlice].CommonMemory()->fNTracklets)
			{
				HLTError("Slave tracker tracklets found where none expected, memory not freed!\n");
			}
			fSlaveTrackers[iSlice].SetGPUTrackerTrackletsMemory(&tmpMemory[0], 1, fConstructorBlockCount);
			fSlaveTrackers[iSlice].CommonMemory()->fNTracklets = 1;
		}

		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			int sliceLeft = (iSlice + (fgkNSlices / 2 - 1)) % (fgkNSlices / 2);
			int sliceRight = (iSlice + 1) % (fgkNSlices / 2);
			if (iSlice >= fgkNSlices / 2)
			{
				sliceLeft += fgkNSlices / 2;
				sliceRight += fgkNSlices / 2;
			}
			fSlaveTrackers[iSlice].PerformGlobalTracking(fSlaveTrackers[sliceLeft], fSlaveTrackers[sliceRight], HLTCA_GPU_MAX_TRACKS);		
		}

		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			printf("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, fSlaveTrackers[iSlice].CommonMemory()->fNLocalTracks, fSlaveTrackers[iSlice].CommonMemory()->fNTracks, fSlaveTrackers[iSlice].CommonMemory()->fNLocalTrackHits, fSlaveTrackers[iSlice].CommonMemory()->fNTrackHits);
		}
	}

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		fSlaveTrackers[firstSlice + iSlice].SetOutput(&pOutput[iSlice]);
		fSlaveTrackers[firstSlice + iSlice].WriteOutputPrepare();
		fSlaveTrackers[firstSlice + iSlice].WriteOutput();
	}

	StandalonePerfTime(firstSlice, 10);

	if (fDebugLevel >= 3) HLTInfo("GPU Reconstruction finished");

	return(0);
}

int AliHLTTPCCAGPUTrackerNVCC::ExitGPU_Runtime()
{
	//Uninitialize CUDA
	cuCtxPushCurrent(*((CUcontext*) fCudaContext));

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
			cudaStreamDestroy(((cudaStream_t*) fpCudaStreams)[i]);
		}
		free(fpCudaStreams);
		fGpuTracker = NULL;
		cudaFreeHost(fHostLockedMemory);
	}

	if (GPUFailedMsg(cudaThreadExit()))
	{
		HLTError("Could not uninitialize GPU");
		return(1);
	}

	cuCtxDestroy(*((CUcontext*) fCudaContext));

	cudaDeviceReset();

	HLTInfo("CUDA Uninitialized");
	fCudaInitialized = 0;
	return(0);
}

int AliHLTTPCCAGPUTrackerNVCC::RefitMergedTracks(AliHLTTPCGMMerger* Merger)
{
#ifndef HLTCA_GPU_MERGER
	HLTError("HLTCA_GPU_MERGER compile flag not set");
	return(1);
#else
	if (!fCudaInitialized)
	{
		HLTError("GPU Merger not initialized");
		return(1);
	}
	unsigned long long int a, b, c, d, e;
	AliHLTTPCCATracker::StandaloneQueryFreq(&e);

	char* gpumem = (char*) fGPUMergerMemory;
	float *X, *Y, *Z, *Angle;
	unsigned int *RowType;
	AliHLTTPCGMMergedTrack* tracks;
	float* field;
	AliHLTTPCCAParam* param;

	gpumem = alignPointer(gpumem, 1024 * 1024);

	AssignMemory(X, gpumem, Merger->NClusters());
	AssignMemory(Y, gpumem, Merger->NClusters());
	AssignMemory(Z, gpumem, Merger->NClusters());
	AssignMemory(Angle, gpumem, Merger->NClusters());
	AssignMemory(RowType, gpumem, Merger->NClusters());
	AssignMemory(tracks, gpumem, Merger->NOutputTracks());
	AssignMemory(field, gpumem, 6);
	AssignMemory(param, gpumem, 1);

	if ((size_t) (gpumem - (char*) fGPUMergerMemory) > (size_t) fGPUMergerMaxMemory)
	{
		HLTError("Insufficiant GPU Merger Memory");
	}

	cuCtxPushCurrent(*((CUcontext*) fCudaContext));

	if (fDebugLevel >= 2) HLTInfo("Running GPU Merger (%d/%d)", Merger->NOutputTrackClusters(), Merger->NClusters());
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
	GPUFailedMsg(cudaMemcpy(X, Merger->ClusterX(), Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(Y, Merger->ClusterY(), Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(Z, Merger->ClusterZ(), Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(Angle, Merger->ClusterAngle(), Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(RowType, Merger->ClusterRowType(), Merger->NOutputTrackClusters() * sizeof(unsigned int), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(tracks, Merger->OutputTracks(), Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(field, Merger->PolinomialFieldBz(), 6 * sizeof(float), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(param, fSlaveTrackers[0].pParam(), sizeof(AliHLTTPCCAParam), cudaMemcpyHostToDevice));
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
	RefitTracks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(tracks, Merger->NOutputTracks(), field, X, Y, Z, RowType, Angle, param);
	GPUFailedMsg(cudaThreadSynchronize());
	AliHLTTPCCATracker::StandaloneQueryTime(&c);
	GPUFailedMsg(cudaMemcpy(Merger->ClusterX(), X, Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy(Merger->ClusterY(), Y, Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy(Merger->ClusterZ(), Z, Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy(Merger->ClusterAngle(), Angle, Merger->NOutputTrackClusters() * sizeof(float), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy(Merger->ClusterRowType(), RowType, Merger->NOutputTrackClusters() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy((void*) Merger->OutputTracks(), tracks, Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaThreadSynchronize());
	AliHLTTPCCATracker::StandaloneQueryTime(&d);
	if (fDebugLevel >= 2) HLTInfo("GPU Merger Finished");

	if (fDebugLevel > 0)
	{
		int copysize = 4 * Merger->NOutputTrackClusters() * sizeof(float) + Merger->NOutputTrackClusters() * sizeof(unsigned int) + Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(AliHLTTPCCAParam);
		double speed = (double) copysize * (double) e / (double) (b - a) / 1e9;
		printf("GPU Fit:\tCopy To:\t%lld us (%lf GB/s)\n", (b - a) * 1000000 / e, speed);
		printf("\t\tFit:\t%lld us\n", (c - b) * 1000000 / e);
		speed = (double) copysize * (double) e / (double) (d - c) / 1e9;
		printf("\t\tCopy From:\t%lld us (%lf GB/s)\n", (d - c) * 1000000 / e, speed);
	}

	cuCtxPopCurrent((CUcontext*) fCudaContext);
	return(0);
#endif
}

int AliHLTTPCCAGPUTrackerNVCC::GPUMergerAvailable()
{
	return(1);
}

void AliHLTTPCCAGPUTrackerNVCC::ActivateThreadContext()
{
	cuCtxPushCurrent(*((CUcontext*) fCudaContext));
}
void AliHLTTPCCAGPUTrackerNVCC::ReleaseThreadContext()
{
	cuCtxPopCurrent((CUcontext*) fCudaContext);
}

void AliHLTTPCCAGPUTrackerNVCC::SynchronizeGPU()
{
	cudaThreadSynchronize();
}

AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate()
{
	return new AliHLTTPCCAGPUTrackerNVCC;
}

void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr)
{
	delete ptr;
}

