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

#include "AliHLTTPCCAGPUTrackerOpenCL.h"

#include <opencl.h>

__constant__ float4 gAliHLTTPCCATracker[HLTCA_GPU_TRACKER_CONSTANT_MEM / sizeof( float4 )];

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCATrack.h" 

#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATracker.h"

#include "AliHLTTPCCAProcess.h"

#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCAStartHitsSorter.h"
#include "AliHLTTPCCATrackletConstructor.h"

ClassImp( AliHLTTPCCAGPUTrackerOpenCL )

AliHLTTPCCAGPUTrackerOpenCL::AliHLTTPCCAGPUTrackerOpenCL()
{
	fCudaContext = (void*) new CUcontext;
};

AliHLTTPCCAGPUTrackerOpenCL::~AliHLTTPCCAGPUTrackerOpenCL()
{
	delete (CUcontext*) fCudaContext;
};

int AliHLTTPCCAGPUTrackerOpenCL::InitGPU_Runtime(int sliceCount, int forceDeviceID)
{
	//Find best CUDA device, initialize and allocate memory

	cudaDeviceProp fCudaDeviceProp;

#ifndef CUDA_DEVICE_EMULATION
	int count, bestDevice = -1;
	long long int bestDeviceSpeed = 0, deviceSpeed;
	if (CudaFailedMsg(cudaGetDeviceCount(&count)))
	{
		HLTError("Error getting CUDA Device Count");
		return(1);
	}
	if (fDebugLevel >= 2) HLTInfo("Available CUDA devices:");
#ifdef FERMI
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
		if (CudaFailedMsg(cudaGetDeviceProperties(&fCudaDeviceProp, i))) continue;
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

	if (fGPUMemSize > fCudaDeviceProp.totalGlobalMem || CudaFailedMsg(cudaMalloc(&fGPUMemory, (size_t) fGPUMemSize)))
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
	if (CudaFailedMsg(cudaMallocHost(&fHostLockedMemory, hostMemSize)))
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
		CudaFailedMsg(cudaMemset(fGPUMemory, 143, (size_t) fGPUMemSize));
	}

	fpCudaStreams = malloc(CAMath::Max(3, fSliceCount) * sizeof(cudaStream_t));
	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;
	for (int i = 0;i < CAMath::Max(3, fSliceCount);i++)
	{
		if (CudaFailedMsg(cudaStreamCreate(&cudaStreams[i])))
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

bool AliHLTTPCCAGPUTrackerOpenCL::CudaFailedMsgA(cudaError_t error, const char* file, int line)
{
	//Check for CUDA Error and in the case of an error display the corresponding error string
	if (error == cudaSuccess) return(false);
	HLTWarning("CUDA Error: %d / %s (%s:%d)", error, cudaGetErrorString(error), file, line);
	return(true);
}

int AliHLTTPCCAGPUTrackerOpenCL::CUDASync(char* state, int sliceLocal, int slice)
{
	//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards

	if (fDebugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		HLTError("Cuda Error %s while running kernel (%s) (Slice %d; %d/%d)", cudaGetErrorString(cuErr), state, sliceLocal, slice, fgkNSlices);
		return(1);
	}
	if (CudaFailedMsg(cudaThreadSynchronize()))
	{
		HLTError("CUDA Error while synchronizing (%s) (Slice %d; %d/%d)", state, sliceLocal, slice, fgkNSlices);
		return(1);
	}
	if (fDebugLevel >= 3) HLTInfo("CUDA Sync Done");
	return(0);
}

__global__ void PreInitRowBlocks(int4* const RowBlockPos, int* const RowBlockTracklets, int* const SliceDataHitWeights, int nSliceDataHits)
{
	//Initialize GPU RowBlocks and HitWeights
	int4* const sliceDataHitWeights4 = (int4*) SliceDataHitWeights;
	const int stride = blockDim.x * gridDim.x;
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		sliceDataHitWeights4[i] = i0;
}

int AliHLTTPCCAGPUTrackerOpenCL::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function

	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;

	if (Reconstruct_Base_Init(pOutput, pClusterData, firstSlice, sliceCountLocal)) return(1);

	//Copy Tracker Object to GPU Memory
	if (fDebugLevel >= 3) HLTInfo("Copying Tracker objects to GPU");

	CudaFailedMsg(cudaMemcpyToSymbolAsync(gAliHLTTPCCATracker, fGpuTracker, sizeof(AliHLTTPCCATracker) * sliceCountLocal, 0, cudaMemcpyHostToDevice, cudaStreams[0]));
	if (CUDASync("Initialization (1)", 0, firstSlice) RANDOM_ERROR)
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
		if (CUDASync("Initialization (2)", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		CudaFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
		CudaFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));
		CudaFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].SliceDataRows(), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), cudaMemcpyHostToDevice, cudaStreams[iSlice & 1]));

		if (fDebugLevel >= 4)
		{
			if (fDebugLevel >= 5) HLTInfo("Allocating Debug Output Memory");
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100]), pClusterData[iSlice].NumberOfClusters() );
		}

		if (CUDASync("Initialization (3)", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		StandalonePerfTime(firstSlice + iSlice, 1);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows(), HLTCA_GPU_THREAD_COUNT_FINDER, 0, cudaStreams[iSlice & 1]>>>(iSlice);

		if (CUDASync("Neighbours finder", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 2);

		if (fDebugLevel >= 4)
		{
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 2) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-2, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (CUDASync("Neighbours Cleaner", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 3);

		if (fDebugLevel >= 4)
		{
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 4) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-6, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (CUDASync("Start Hits Finder", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 4);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, sliceCountLocal);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (CUDASync("Start Hits Sorter", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 5);

		if (fDebugLevel >= 2)
		{
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
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
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fSlaveTrackers[firstSlice + iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
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

	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");
	AliHLTTPCCATrackletConstructorGPU<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (CUDASync("Tracklet Constructor", 0, firstSlice) RANDOM_ERROR)
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
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5)
			{
				HLTInfo("Obtained %d tracklets", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			}
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemorySize(), cudaMemcpyDeviceToHost));
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fGpuTracker[iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 128) fSlaveTrackers[firstSlice + iSlice].DumpTrackletHits(*fOutFile);
		}
	}

	int runSlices = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += runSlices)
	{
		if (runSlices < HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		if (fDebugLevel >= 3) HLTInfo("Running HLT Tracklet selector (Slice %d to %d)", iSlice, iSlice + runSlices);
		AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<selectorBlockCount, HLTCA_GPU_THREAD_COUNT_SELECTOR, 0, cudaStreams[iSlice]>>>(iSlice, CAMath::Min(runSlices, sliceCountLocal - iSlice));
		if (CUDASync("Tracklet Selector", iSlice, iSlice + firstSlice) RANDOM_ERROR)
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
			if (CudaFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice].CommonMemory(), fGpuTracker[tmpSlice].CommonMemory(), fGpuTracker[tmpSlice].CommonMemorySize(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice])) RANDOM_ERROR)
			{
				ResetHelperThreads(1);
				ActivateThreadContext();
				return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
			}
			tmpSlice++;
		}

		while (tmpSlice2 < tmpSlice && (tmpSlice2 == iSlice ? cudaStreamSynchronize(cudaStreams[tmpSlice2]) : cudaStreamQuery(cudaStreams[tmpSlice2])) == (cudaError_t) CUDA_SUCCESS)
		{
			CudaFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice2].Tracks(), fGpuTracker[tmpSlice2].Tracks(), sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + tmpSlice2].NTracks(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice2]));
			CudaFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice2].TrackHits(), fGpuTracker[tmpSlice2].TrackHits(), sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + tmpSlice2].NTrackHits(), cudaMemcpyDeviceToHost, cudaStreams[tmpSlice2]));
			tmpSlice2++;
		}

		if (CudaFailedMsg(cudaStreamSynchronize(cudaStreams[iSlice])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
		}

		if (fDebugLevel >= 4)
		{
			CudaFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().HitWeights(), fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign() * sizeof(int), cudaMemcpyDeviceToHost));
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
			if (fDebugMask & 256) fSlaveTrackers[firstSlice + iSlice].DumpHitWeights(*fOutFile);
#endif
			if (fDebugMask & 512) fSlaveTrackers[firstSlice + iSlice].DumpTrackHits(*fOutFile);
		}

		if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			HLTError("GPU Tracker returned Error Code %d in slice %d", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError, firstSlice + iSlice);
			ResetHelperThreads(1);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(pOutput, iSlice, firstSlice)) return(1);
	}

	if (Reconstruct_Base_Finalize(pOutput, tmpMemoryGlobalTracking, firstSlice)) return(1);

	cuCtxPopCurrent((CUcontext*) fCudaContext);
	return(0);
}

int AliHLTTPCCAGPUTrackerOpenCL::ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	HLTFatal("Not implemented in OpenCL");
	return(1);
}

int AliHLTTPCCAGPUTrackerOpenCL::ExitGPU_Runtime()
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

	if (CudaFailedMsg(cudaThreadExit()))
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

int AliHLTTPCCAGPUTrackerOpenCL::RefitMergedTracks(AliHLTTPCGMMerger* Merger)
{
	HLTFatal("Not implemented in OpenCL");
	return(1);
}

void AliHLTTPCCAGPUTrackerOpenCL::ActivateThreadContext()
{
}
void AliHLTTPCCAGPUTrackerOpenCL::ReleaseThreadContext()
{
}

void AliHLTTPCCAGPUTrackerOpenCL::SynchronizeGPU()
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

