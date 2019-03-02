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

#define PASCAL
#include "AliHLTTPCCAGPUTrackerNVCC.h"
#include "AliHLTTPCCAGPUTrackerCommon.h"
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size(dim) (blockDim.x * gridDim.x)
#define get_num_groups(dim) (gridDim.x)
#define get_local_id(dim) (threadIdx.x)
#define get_local_size(dim) (blockDim.x)
#define get_group_id(dim) (blockIdx.x)

#include <cuda.h>
#include <sm_20_atomic_functions.h>

__constant__ float4 gAliHLTTPCCATracker[HLTCA_GPU_TRACKER_CONSTANT_MEM / sizeof( float4 )];
#ifdef HLTCA_GPU_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
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
#include "AliHLTTPCGMPhysicalTrackModel.cxx"
#include "AliHLTTPCGMPropagator.cxx"
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
	double bestDeviceSpeed = -1, deviceSpeed;
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
		int deviceOK = fCudaDeviceProp.major < 9 && (fCudaDeviceProp.major > reqVerMaj || (fCudaDeviceProp.major == reqVerMaj && fCudaDeviceProp.minor >= reqVerMin)) && (size_t) free >= (size_t) (fGPUMemSize + 100 * 1024 + 1024);

		if (fDebugLevel >= 0) HLTInfo("%s%2d: %s (Rev: %d.%d - Mem Avail %lld / %lld)%s", deviceOK ? " " : "[", i, fCudaDeviceProp.name, fCudaDeviceProp.major, fCudaDeviceProp.minor, (long long int) free, (long long int) fCudaDeviceProp.totalGlobalMem, deviceOK ? "" : " ]");
		deviceSpeed = (double) fCudaDeviceProp.multiProcessorCount * (double) fCudaDeviceProp.clockRate * (double) fCudaDeviceProp.warpSize * (double) free * (double) fCudaDeviceProp.major * (double) fCudaDeviceProp.major;
		if (deviceOK)
		{
			if (deviceSpeed > bestDeviceSpeed)
			{
				bestDevice = i;
				bestDeviceSpeed = deviceSpeed;
			}
			else
			{
				if (fDebugLevel >= 0) HLTInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
			}
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
#ifdef HLTCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	fConstructorBlockCount = fCudaDeviceProp.multiProcessorCount;
#else
	fConstructorBlockCount = fCudaDeviceProp.multiProcessorCount * HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
#endif
	fConstructorThreadCount = HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
	fSelectorBlockCount = fCudaDeviceProp.multiProcessorCount * HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;

	if (fCudaDeviceProp.major < 1 || (fCudaDeviceProp.major == 1 && fCudaDeviceProp.minor < 2))
	{
		HLTError( "Unsupported CUDA Device" );
		return(1);
	}

#ifdef HLTCA_GPU_USE_TEXTURES
	if (HLTCA_GPU_SLICE_DATA_MEMORY * sliceCount > (size_t) fCudaDeviceProp.maxTexture1DLinear)
	{
		HLTError("Invalid maximum texture size of device: %lld < %lld\n", (long long int) fCudaDeviceProp.maxTexture1DLinear, (long long int) (HLTCA_GPU_SLICE_DATA_MEMORY * sliceCount));
		return(1);
	}
#endif

	int nStreams = HLTCA_GPU_NUM_STREAMS == 0 ? CAMath::Max(3, fSliceCount) : HLTCA_GPU_NUM_STREAMS;
	if (nStreams < 3)
	{
		HLTError("Invalid number of streams");
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
	if (fDebugLevel >= 1) HLTInfo("GPU Memory used: %lld", fGPUMemSize);
	if (GPUFailedMsg(cudaMallocHost(&fHostLockedMemory, fHostMemSize)))
	{
		cudaFree(fGPUMemory);
		cudaThreadExit();
		HLTError("Error allocating Page Locked Host Memory");
		return(1);
	}
	if (fDebugLevel >= 1) HLTInfo("Host Memory used: %lld", fHostMemSize);

	if (fDebugLevel >= 1)
	{
		GPUFailedMsg(cudaMemset(fGPUMemory, 143, (size_t) fGPUMemSize));
	}
	
	fpCudaStreams = malloc(nStreams * sizeof(cudaStream_t));
	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;
	for (int i = 0;i < nStreams;i++)
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
	HLTInfo("CUDA Initialisation successfull (Device %d: %s, Thread %d, Max slices: %d, %lld bytes used)", fCudaDevice, fCudaDeviceProp.name, fThreadId, fSliceCount, fGPUMemSize);

	return(0);
}

bool AliHLTTPCCAGPUTrackerNVCC::GPUFailedMsgA(cudaError_t error, const char* file, int line)
{
	//Check for CUDA Error and in the case of an error display the corresponding error string
	if (error == cudaSuccess) return(false);
	HLTWarning("CUDA Error: %d / %s (%s:%d)", error, cudaGetErrorString(error), file, line);
	return(true);
}

int AliHLTTPCCAGPUTrackerNVCC::GPUSync(const char* state, int stream, int slice)
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
	if (fDebugLevel >= 3) HLTInfo("GPU Sync Done");
	return(0);
}

__global__ void PreInitRowBlocks(int* const SliceDataHitWeights, int nSliceDataHits)
{
	//Initialize GPU RowBlocks and HitWeights
	int4* const sliceDataHitWeights4 = (int4*) SliceDataHitWeights;
	const int stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	for (int i = get_global_id(0);i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		sliceDataHitWeights4[i] = i0;
}

int AliHLTTPCCAGPUTrackerNVCC::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function

	cudaStream_t* const cudaStreams = (cudaStream_t*) fpCudaStreams;

	if (Reconstruct_Base_Init(pOutput, pClusterData, firstSlice, sliceCountLocal)) return(1);

#ifdef HLTCA_GPU_USE_TEXTURES
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<cahit2>();
	size_t offset;
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, fGpuTracker[0].Data().Memory(), &channelDescu2, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		HLTError("Error binding CUDA Texture cahit2 (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<calink>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, fGpuTracker[0].Data().Memory(), &channelDescu, sliceCountLocal * HLTCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		HLTError("Error binding CUDA Texture calink (Offset %d)", (int) offset);
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
	bool globalSymbolDone = false;
	if (GPUSync("Initialization (1)", 0, firstSlice) RANDOM_ERROR)
	{
		ResetHelperThreads(0);
		return(1);
	}

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(pClusterData, iSlice, firstSlice)) return(1);

		int useStream = HLTCA_GPU_NUM_STREAMS == 0 ? (iSlice & 1) : (iSlice % HLTCA_GPU_NUM_STREAMS);
		//Initialize temporary memory where needed
		if (fDebugLevel >= 3) HLTInfo("Copying Slice Data to GPU and initializing temporary memory");		
		PreInitRowBlocks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[HLTCA_GPU_NUM_STREAMS == 0 ? 2 : useStream]>>>(fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign());
		if (GPUSync("Initialization (2)", 2, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		fSlaveTrackers[firstSlice + iSlice].StartTimer(0);
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), cudaMemcpyHostToDevice, cudaStreams[useStream]));
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyHostToDevice, cudaStreams[useStream]));
		GPUFailedMsg(cudaMemcpyAsync(fGpuTracker[iSlice].SliceDataRows(), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), cudaMemcpyHostToDevice, cudaStreams[useStream]));

		if (fDebugLevel >= 4)
		{
			if (fDebugLevel >= 5) HLTInfo("Allocating Debug Output Memory");
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS);
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100]), pClusterData[iSlice].NumberOfClusters() );
		}
		
		if (HLTCA_GPU_NUM_STREAMS && useStream && globalSymbolDone == false)
		{
			cudaStreamSynchronize(cudaStreams[0]);
			globalSymbolDone = true;
		}

		if (GPUSync("Initialization (3)", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(0);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		fSlaveTrackers[firstSlice + iSlice].StartTimer(1);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows(), HLTCA_GPU_THREAD_COUNT_FINDER, 0, cudaStreams[useStream]>>>(iSlice);

		if (GPUSync("Neighbours finder", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(1);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 2) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, sliceCountLocal);
		fSlaveTrackers[firstSlice + iSlice].StartTimer(2);
		AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-2, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Neighbours Cleaner", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(2);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().Memory(), fGpuTracker[iSlice].Data().Memory(), fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 4) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		fSlaveTrackers[firstSlice + iSlice].StartTimer(3);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-6, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Start Hits Finder", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(3);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, sliceCountLocal);
		fSlaveTrackers[firstSlice + iSlice].StartTimer(4);
		AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Start Hits Sorter", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(4);

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

		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());

#ifdef HLTCA_GPU_CONSTRUCTOR_SINGLE_SLICE
		fSlaveTrackers[firstSlice + iSlice].StartTimer(6);
		AliHLTTPCCATrackletConstructorSingleSlice<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR, 0, cudaStreams[useStream]>>>(firstSlice + iSlice);
		if (GPUSync("Tracklet Constructor", useStream, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(6);
#endif
	}

	for (int i = 0;i < fNHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

#ifdef HLTCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	cudaThreadSynchronize();
#else
	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");
	fSlaveTrackers[firstSlice].StartTimer(6);
	AliHLTTPCCATrackletConstructorGPU<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (GPUSync("Tracklet Constructor", -1, firstSlice) RANDOM_ERROR)
	{
		cudaThreadSynchronize();
		cuCtxPopCurrent((CUcontext*) fCudaContext);
		return(1);
	}
	fSlaveTrackers[firstSlice].StopTimer(6);
#endif //HLTCA_GPU_CONSTRUCTOR_SINGLE_SLICE

	if (fDebugLevel >= 4)
	{
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugLevel >= 5)
			{
				HLTInfo("Obtained %d tracklets", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			}
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemory(), fGpuTracker[iSlice].TrackletMemorySize(), cudaMemcpyDeviceToHost));
			GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].HitMemory(), fGpuTracker[iSlice].HitMemory(), fGpuTracker[iSlice].HitMemorySize(), cudaMemcpyDeviceToHost));
			if (fDebugMask & 128) fSlaveTrackers[firstSlice + iSlice].DumpTrackletHits(*fOutFile);
			delete[] fSlaveTrackers[firstSlice + iSlice].TrackletMemory();
		}
	}

	int runSlices = 0;
	int useStream = 0;
	int streamMap[36];
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += runSlices)
	{
		if (runSlices < HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		runSlices = CAMath::Min(runSlices, sliceCountLocal - iSlice);
		if (fSelectorBlockCount < runSlices) runSlices = fSelectorBlockCount;
		if (HLTCA_GPU_NUM_STREAMS && useStream + 1 == HLTCA_GPU_NUM_STREAMS) runSlices = sliceCountLocal - iSlice;
		if (fSelectorBlockCount < runSlices)
		{
			HLTError("Insufficient number of blocks for tracklet selector");
			cuCtxPopCurrent((CUcontext*) fCudaContext);
			return(1);
		}
		
		if (fDebugLevel >= 3) HLTInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
		fSlaveTrackers[firstSlice + iSlice].StartTimer(7);
		AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<fSelectorBlockCount, HLTCA_GPU_THREAD_COUNT_SELECTOR, 0, cudaStreams[useStream]>>>(iSlice, runSlices);
		if (GPUSync("Tracklet Selector", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			cudaThreadSynchronize();
			cuCtxPopCurrent((CUcontext*) fCudaContext);
			return(1);
		}
		fSlaveTrackers[firstSlice + iSlice].StopTimer(7);
		for (int k = iSlice;k < iSlice + runSlices;k++) streamMap[k] = useStream;
		useStream++;
	}

	char *tmpMemoryGlobalTracking = NULL;
	fSliceOutputReady = 0;
	
	if (Reconstruct_Base_StartGlobal(pOutput, tmpMemoryGlobalTracking)) return(1);
	
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemory(), fGpuTracker[iSlice].CommonMemorySize(), cudaMemcpyDeviceToHost, cudaStreams[streamMap[iSlice]])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
		}
	}

	int tmpSlice = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (fDebugLevel >= 3) HLTInfo("Transfering Tracks from GPU to Host");

		while (tmpSlice < sliceCountLocal && (tmpSlice == iSlice ? cudaStreamSynchronize(cudaStreams[streamMap[tmpSlice]]) : cudaStreamQuery(cudaStreams[streamMap[tmpSlice]])) == (cudaError_t) CUDA_SUCCESS)
		{
			if (*fSlaveTrackers[firstSlice + tmpSlice].NTracks() > 0)
			{
				int useStream = HLTCA_GPU_NUM_STREAMS ? streamMap[tmpSlice] : tmpSlice;
				GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice].Tracks(), fGpuTracker[tmpSlice].Tracks(), sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + tmpSlice].NTracks(), cudaMemcpyDeviceToHost, cudaStreams[useStream]));
				GPUFailedMsg(cudaMemcpyAsync(fSlaveTrackers[firstSlice + tmpSlice].TrackHits(), fGpuTracker[tmpSlice].TrackHits(), sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + tmpSlice].NTrackHits(), cudaMemcpyDeviceToHost, cudaStreams[useStream]));
			}
			tmpSlice++;
		}

		int useStream = HLTCA_GPU_NUM_STREAMS ? streamMap[iSlice] : iSlice;
		if (GPUFailedMsg(cudaStreamSynchronize(cudaStreams[useStream])) RANDOM_ERROR)
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
			const char* errorMsgs[] = HLTCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError];
			HLTError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError, errorMsg, firstSlice + iSlice, fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHits());

			ResetHelperThreads(1);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(pOutput, iSlice, firstSlice)) return(1);
		if (fDebugLevel >= 4)
		{
			delete[] fSlaveTrackers[firstSlice + iSlice].HitMemory();
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
		int nStreams = HLTCA_GPU_NUM_STREAMS == 0 ? CAMath::Max(3, fSliceCount) : HLTCA_GPU_NUM_STREAMS;
		for (int i = 0;i < nStreams;i++)
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

int AliHLTTPCCAGPUTrackerNVCC::RefitMergedTracks(AliHLTTPCGMMerger* Merger, bool resetTimers)
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

	HighResTimer timer;
	static double times[3] = {};
	static int nCount = 0;
	if (resetTimers)
	{
		for (unsigned int k = 0;k < sizeof(times) / sizeof(times[0]);k++) times[k] = 0;
		nCount = 0;
	}
	char* gpumem = (char*) fGPUMergerMemory;
	AliHLTTPCGMMergedTrackHit *clusters;
	AliHLTTPCGMMergedTrack* tracks;
	AliHLTTPCGMPolynomialField* field;
	AliHLTTPCCAParam* param;

	AssignMemory(clusters, gpumem, Merger->NClusters());
	AssignMemory(tracks, gpumem, Merger->NOutputTracks());
	AssignMemory(field, gpumem, 1);
	AssignMemory(param, gpumem, 1);

	if ((size_t) (gpumem - (char*) fGPUMergerMemory) > (size_t) fGPUMergerMaxMemory)
	{
		HLTError("Insufficiant GPU Merger Memory");
	}

	cuCtxPushCurrent(*((CUcontext*) fCudaContext));

	if (fDebugLevel >= 2) HLTInfo("Running GPU Merger (%d/%d)", Merger->NOutputTrackClusters(), Merger->NClusters());
	timer.Start();
	GPUFailedMsg(cudaMemcpy(clusters, Merger->Clusters(), Merger->NOutputTrackClusters() * sizeof(clusters[0]), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(tracks, Merger->OutputTracks(), Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(field, Merger->pField(), sizeof(AliHLTTPCGMPolynomialField), cudaMemcpyHostToDevice));
	GPUFailedMsg(cudaMemcpy(param, &Merger->SliceParam(), sizeof(AliHLTTPCCAParam), cudaMemcpyHostToDevice));
	times[0] += timer.GetCurrentElapsedTime(true);
	RefitTracks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT>>>(tracks, Merger->NOutputTracks(), field, clusters, param);
	GPUFailedMsg(cudaThreadSynchronize());
	times[1] += timer.GetCurrentElapsedTime(true);
	GPUFailedMsg(cudaMemcpy(Merger->Clusters(), clusters, Merger->NOutputTrackClusters() * sizeof(clusters[0]), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaMemcpy((void*) Merger->OutputTracks(), tracks, Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack), cudaMemcpyDeviceToHost));
	GPUFailedMsg(cudaThreadSynchronize());
	times[2] += timer.GetCurrentElapsedTime();
	if (fDebugLevel >= 2) HLTInfo("GPU Merger Finished");
	nCount++;

	if (fDebugLevel > 0)
	{
		int copysize = 4 * Merger->NOutputTrackClusters() * sizeof(float) + Merger->NOutputTrackClusters() * sizeof(unsigned int) + Merger->NOutputTracks() * sizeof(AliHLTTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(AliHLTTPCCAParam);
		double speed = (double) copysize / times[0] * nCount / 1e9;
		printf("GPU Fit:\tCopy To:\t%1.0f us (%lf GB/s)\n", times[0] * 1000000 / nCount, speed);
		printf("\t\tFit:\t%1.0f us\n", times[1] * 1000000 / nCount);
		speed = (double) copysize / times[2] * nCount / 1e9;
		printf("\t\tCopy From:\t%1.0f us (%lf GB/s)\n", times[2] * 1000000 / nCount, speed);
	}
	
	if (!HLTCA_TIMING_SUM)
	{
		for (int i = 0;i < 3;i++) times[i] = 0;
		nCount = 0;
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
