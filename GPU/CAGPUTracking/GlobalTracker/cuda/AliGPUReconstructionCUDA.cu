#define GPUCA_GPUTYPE_PASCAL

#include "AliGPUReconstructionCUDA.h"
#include "AliGPUReconstructionCUDAInternals.h"

#define GPUCA_GPUTYPE_PASCAL
#include "AliGPUTPCGPUTrackerCommon.h"
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size(dim) (blockDim.x * gridDim.x)
#define get_num_groups(dim) (gridDim.x)
#define get_local_id(dim) (threadIdx.x)
#define get_local_size(dim) (blockDim.x)
#define get_group_id(dim) (blockIdx.x)

#include <cuda.h>
#include <sm_20_atomic_functions.h>

#include "AliGPUCADataTypes.h"
#include "AliCAGPULogging.h"
__constant__ uint4 gGPUConstantMemBuffer[(sizeof(AliGPUCAConstantMem) + sizeof(uint4) - 1) / sizeof(uint4)];
__constant__ char& gGPUConstantMemBufferChar = (char&) gGPUConstantMemBuffer;
__constant__ AliGPUCAConstantMem& gGPUConstantMem = (AliGPUCAConstantMem&) gGPUConstantMemBufferChar;

AliGPUCAConstantMem AliGPUCAConstantMemDummy;
#ifdef GPUCA_GPU_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

//Include CXX Files, GPUd() macro will then produce CUDA device code out of the tracker source code
#include "AliGPUTPCTrackParam.cxx"
#include "AliGPUTPCTrack.cxx"

#include "AliGPUTPCHitArea.cxx"
#include "AliGPUTPCGrid.cxx"
#include "AliGPUTPCRow.cxx"
#include "AliGPUCAParam.cxx"
#include "AliGPUTPCTracker.cxx"

#include "AliGPUTPCProcess.h"

#include "AliGPUTPCTrackletSelector.cxx"
#include "AliGPUTPCNeighboursFinder.cxx"
#include "AliGPUTPCNeighboursCleaner.cxx"
#include "AliGPUTPCStartHitsFinder.cxx"
#include "AliGPUTPCStartHitsSorter.cxx"
#include "AliGPUTPCTrackletConstructor.cxx"

#ifdef GPUCA_GPU_MERGER
#include "AliGPUTPCGMMerger.h"
#include "AliGPUTPCGMTrackParam.cxx"
#include "AliGPUTPCGMPhysicalTrackModel.cxx"
#include "AliGPUTPCGMPropagator.cxx"

#include "AliGPUTRDTrack.cxx"
#include "AliGPUTRDTracker.cxx"
#include "AliGPUTRDTrackletWord.cxx"
#ifdef HAVE_O2HEADERS
#include "TRDGeometryBase.cxx"
#endif
#endif

#ifdef HAVE_O2HEADERS
#include "ITStrackingCUDA/TrackerTraitsNV.h"
#ifndef GPUCA_O2_LIB
#include "TrackerTraitsNV.cu"
#include "Context.cu"
#include "Stream.cu"
#include "DeviceStoreNV.cu"
#include "Utils.cu"
#endif
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsNV : public TrackerTraits {}; }}
#endif

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

AliGPUReconstructionCUDA::AliGPUReconstructionCUDA(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mInternals = new AliGPUReconstructionCUDAInternals;
	mProcessingSettings.deviceType = CUDA;
	mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsNV);
}

AliGPUReconstructionCUDA::~AliGPUReconstructionCUDA()
{
	mITSTrackerTraits.reset(nullptr); //Make sure we destroy the ITS tracker before we exit CUDA
	cudaDeviceReset();
	delete mInternals;
}

AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg)
{
	return new AliGPUReconstructionCUDA(cfg);
}

int AliGPUReconstructionCUDA::InitDevice_Runtime()
{
	//Find best CUDA device, initialize and allocate memory

	cudaDeviceProp cudaDeviceProp;

	int count, bestDevice = -1;
	double bestDeviceSpeed = -1, deviceSpeed;
	if (GPUFailedMsg(cudaGetDeviceCount(&count)))
	{
		CAGPUError("Error getting CUDA Device Count");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Available CUDA devices:");
	const int reqVerMaj = 2;
	const int reqVerMin = 0;
	for (int i = 0;i < count;i++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Examining device %d\n", i);
		size_t free, total;
		cuInit(0);
		CUdevice tmpDevice;
		cuDeviceGet(&tmpDevice, i);
		CUcontext tmpContext;
		cuCtxCreate(&tmpContext, 0, tmpDevice);
		if(cuMemGetInfo(&free, &total)) std::cout << "Error\n";
		cuCtxDestroy(tmpContext);
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Obtained current memory usage for device %d\n", i);
		if (GPUFailedMsg(cudaGetDeviceProperties(&cudaDeviceProp, i))) continue;
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Obtained device properties for device %d\n", i);
		int deviceOK = true;
		const char* deviceFailure = "";
		if (cudaDeviceProp.major >= 9) {deviceOK = false; deviceFailure = "Invalid Revision";}
		else if (cudaDeviceProp.major < reqVerMaj || (cudaDeviceProp.major == reqVerMaj && cudaDeviceProp.minor < reqVerMin)) {deviceOK = false; deviceFailure = "Too low device revision";}
		else if (free < mDeviceMemorySize) {deviceOK = false; deviceFailure = "Insufficient GPU memory";}

		deviceSpeed = (double) cudaDeviceProp.multiProcessorCount * (double) cudaDeviceProp.clockRate * (double) cudaDeviceProp.warpSize * (double) free * (double) cudaDeviceProp.major * (double) cudaDeviceProp.major;
		if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem Avail %lld / %lld)%s %s", deviceOK ? " " : "[", i, cudaDeviceProp.name, cudaDeviceProp.major, cudaDeviceProp.minor, (long long int) free, (long long int) cudaDeviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
		if (!deviceOK) continue;
		if (deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
		else
		{
			if (mDeviceProcessingSettings.debugLevel >= 0) CAGPUInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
		}
	}
	if (bestDevice == -1)
	{
		CAGPUWarning("No %sCUDA Device available, aborting CUDA Initialisation", count ? "appropriate " : "");
		CAGPUImportant("Requiring Revision %d.%d, Mem: %lld", reqVerMaj, reqVerMin, (long long int) mDeviceMemorySize);
		return(1);
	}

	if (mDeviceProcessingSettings.deviceNum > -1)
	{
		if (mDeviceProcessingSettings.deviceNum < (signed) count)
		{
			bestDevice = mDeviceProcessingSettings.deviceNum;
		}
		else
		{
			CAGPUWarning("Requested device ID %d non existend, falling back to default device id %d", mDeviceProcessingSettings.deviceNum, bestDevice);
		}
	}
	fDeviceId = bestDevice;

	cudaGetDeviceProperties(&cudaDeviceProp ,fDeviceId );

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		CAGPUInfo("Using CUDA Device %s with Properties:", cudaDeviceProp.name);
		CAGPUInfo("totalGlobalMem = %lld", (unsigned long long int) cudaDeviceProp.totalGlobalMem);
		CAGPUInfo("sharedMemPerBlock = %lld", (unsigned long long int) cudaDeviceProp.sharedMemPerBlock);
		CAGPUInfo("regsPerBlock = %d", cudaDeviceProp.regsPerBlock);
		CAGPUInfo("warpSize = %d", cudaDeviceProp.warpSize);
		CAGPUInfo("memPitch = %lld", (unsigned long long int) cudaDeviceProp.memPitch);
		CAGPUInfo("maxThreadsPerBlock = %d", cudaDeviceProp.maxThreadsPerBlock);
		CAGPUInfo("maxThreadsDim = %d %d %d", cudaDeviceProp.maxThreadsDim[0], cudaDeviceProp.maxThreadsDim[1], cudaDeviceProp.maxThreadsDim[2]);
		CAGPUInfo("maxGridSize = %d %d %d", cudaDeviceProp.maxGridSize[0], cudaDeviceProp.maxGridSize[1], cudaDeviceProp.maxGridSize[2]);
		CAGPUInfo("totalConstMem = %lld", (unsigned long long int) cudaDeviceProp.totalConstMem);
		CAGPUInfo("major = %d", cudaDeviceProp.major);
		CAGPUInfo("minor = %d", cudaDeviceProp.minor);
		CAGPUInfo("clockRate = %d", cudaDeviceProp.clockRate);
		CAGPUInfo("memoryClockRate = %d", cudaDeviceProp.memoryClockRate);
		CAGPUInfo("multiProcessorCount = %d", cudaDeviceProp.multiProcessorCount);
		CAGPUInfo("textureAlignment = %lld", (unsigned long long int) cudaDeviceProp.textureAlignment);
	}
#ifdef GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	fConstructorBlockCount = cudaDeviceProp.multiProcessorCount;
#else
	fConstructorBlockCount = cudaDeviceProp.multiProcessorCount * GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
#endif
	fConstructorThreadCount = GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;
	fSelectorBlockCount = cudaDeviceProp.multiProcessorCount * GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;

	if (cudaDeviceProp.major < 1 || (cudaDeviceProp.major == 1 && cudaDeviceProp.minor < 2))
	{
		CAGPUError( "Unsupported CUDA Device" );
		return(1);
	}

#ifdef GPUCA_GPU_USE_TEXTURES
	if (GPUCA_GPU_SLICE_DATA_MEMORY * NSLICES > (size_t) cudaDeviceProp.maxTexture1DLinear)
	{
		CAGPUError("Invalid maximum texture size of device: %lld < %lld\n", (long long int) cudaDeviceProp.maxTexture1DLinear, (long long int) (GPUCA_GPU_SLICE_DATA_MEMORY * NSLICES));
		return(1);
	}
#endif

	int nStreams = GPUCA_GPU_NUM_STREAMS == 0 ? 3 : GPUCA_GPU_NUM_STREAMS;
	if (nStreams < 3)
	{
		CAGPUError("Invalid number of streams");
		return(1);
	}

	if (cuCtxCreate(&mInternals->CudaContext, CU_CTX_SCHED_AUTO, fDeviceId) != CUDA_SUCCESS)
	{
		CAGPUError("Could not set CUDA Device!");
		return(1);
	}

	if (mDeviceMemorySize > cudaDeviceProp.totalGlobalMem || GPUFailedMsg(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize)))
	{
		CAGPUError("CUDA Memory Allocation Error");
		cudaDeviceReset();
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("GPU Memory used: %lld", (long long int) mDeviceMemorySize);
	if (GPUFailedMsg(cudaMallocHost(&mHostMemoryBase, mHostMemorySize)))
	{
		cudaFree(mDeviceMemoryBase);
		cudaDeviceReset();
		CAGPUError("Error allocating Page Locked Host Memory");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("Host Memory used: %lld", (long long int) mHostMemorySize);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(mHostMemoryBase, 0, mHostMemorySize);
		if (GPUFailedMsg(cudaMemset(mDeviceMemoryBase, 143, mDeviceMemorySize)))
		{
			cudaFree(mDeviceMemoryBase);
			cudaDeviceReset();
			CAGPUError("Error during CUDA memset");
			return(1);
		}
	}

	mInternals->CudaStreams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
	for (int i = 0;i < nStreams;i++)
	{
		if (GPUFailedMsg(cudaStreamCreate(&mInternals->CudaStreams[i])))
		{
			cudaFree(mDeviceMemoryBase);
			cudaFreeHost(mHostMemoryBase);
			cudaDeviceReset();
			CAGPUError("Error creating CUDA Stream");
			return(1);
		}
	}
	
	void* devPtrConstantMem;
	if (GPUFailedMsg(cudaGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer)))
	{
		CAGPUError("Error getting ptr to constant memory");
		ResetHelperThreads(0);
		return 1;
	}
	mDeviceParam = (AliGPUCAParam*) ((char*) devPtrConstantMem + ((char*) &AliGPUCAConstantMemDummy.param - (char*) &AliGPUCAConstantMemDummy));

	cuCtxPopCurrent(&mInternals->CudaContext);
	CAGPUInfo("CUDA Initialisation successfull (Device %d: %s, Thread %d, %lld/%lld bytes used)", fDeviceId, cudaDeviceProp.name, fThreadId, (long long int) mHostMemorySize, (long long int) mDeviceMemorySize);

	return(0);
}

int AliGPUReconstructionCUDA::GPUSync(const char* state, int stream, int slice)
{
	//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards

	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		CAGPUError("Cuda Error %s while running kernel (%s) (Stream %d; %d/%d)", cudaGetErrorString(cuErr), state, stream, slice, NSLICES);
		return(1);
	}
	if (SynchronizeGPU())
	{
		CAGPUError("CUDA Error while synchronizing (%s) (Stream %d; %d/%d)", state, stream, slice, NSLICES);
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Sync Done");
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

int AliGPUReconstructionCUDA::RunTPCTrackingSlices()
{
	//Primary reconstruction function
	if (fGPUStuck)
	{
		CAGPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	if (Reconstruct_Base_Init()) return(1);

#ifdef GPUCA_GPU_USE_TEXTURES
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<cahit2>();
	size_t offset;
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, fGpuTracker[0].Data().Memory(), &channelDescu2, NSLICES * GPUCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		CAGPUError("Error binding CUDA Texture cahit2 (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<calink>();
	if (GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, fGpuTracker[0].Data().Memory(), &channelDescu, NSLICES * GPUCA_GPU_SLICE_DATA_MEMORY)) || offset RANDOM_ERROR)
	{
		CAGPUError("Error binding CUDA Texture calink (Offset %d)", (int) offset);
		ResetHelperThreads(0);
		return(1);
	}
#endif

	//Copy Tracker Object to GPU Memory
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Tracker objects to GPU");
#ifdef GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* tmpMem;
	if (GPUFailedMsg(cudaMalloc(&tmpMem, 100000000)))
	{
		CAGPUError("Error allocating CUDA profile memory");
		ResetHelperThreads(0);
		return(1);
	}
	fGpuTracker[0].fStageAtSync = tmpMem;
	if (GPUFailedMsg(cudaMemset(fGpuTracker[0].StageAtSync(), 0, 100000000)))
	{
		CAGPUError("Error clearing stageatsync");
		ResetHelperThreads(0);
		return 1;
	}
#endif
	if (GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, &mParam, sizeof(AliGPUCAParam), (char*) &AliGPUCAConstantMemDummy.param - (char*) &AliGPUCAConstantMemDummy, cudaMemcpyHostToDevice, mInternals->CudaStreams[0])))
	{
		CAGPUError("Error writing to constant memory");
		ResetHelperThreads(0);
		return 1;
	}
	
	if (GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, fGpuTracker, sizeof(AliGPUTPCTracker) * NSLICES, (char*) AliGPUCAConstantMemDummy.tpcTrackers - (char*) &AliGPUCAConstantMemDummy, cudaMemcpyHostToDevice, mInternals->CudaStreams[0])))
	{
		CAGPUError("Error writing to constant memory");
		ResetHelperThreads(0);
		return 1;
	}
	
	bool globalSymbolDone = false;
	if (GPUSync("Initialization (1)", 0, 0) RANDOM_ERROR)
	{
		ResetHelperThreads(0);
		return(1);
	}

	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(iSlice)) return(1);
		
		int useStream = GPUCA_GPU_NUM_STREAMS == 0 ? (iSlice & 1) : (iSlice % GPUCA_GPU_NUM_STREAMS);
		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		PreInitRowBlocks<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT, 0, mInternals->CudaStreams[GPUCA_GPU_NUM_STREAMS == 0 ? 2 : useStream]>>>(fGpuTracker[iSlice].Data().HitWeights(), mTPCSliceTrackersCPU[iSlice].Data().NumberOfHitsPlusAlign());
		if (GPUSync("Initialization (2)", 2, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		mTPCSliceTrackersCPU[iSlice].StartTimer(0);
		if (TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResInput(), useStream) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResRows(), useStream) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), useStream))
		{
			CAGPUError("Error copying data to GPU");
			ResetHelperThreads(0);
			return 1;
		}

		if (GPUCA_GPU_NUM_STREAMS && useStream && globalSymbolDone == false)
		{
			cudaStreamSynchronize(mInternals->CudaStreams[0]);
			globalSymbolDone = true;
		}

		if (GPUSync("Initialization (3)", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(0);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, NSLICES);
		mTPCSliceTrackersCPU[iSlice].StartTimer(1);
		AliGPUTPCProcess<AliGPUTPCNeighboursFinder> <<<GPUCA_ROW_COUNT, GPUCA_GPU_THREAD_COUNT_FINDER, 0, mInternals->CudaStreams[useStream]>>>(iSlice);

		if (GPUSync("Neighbours finder", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(1);

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			memcpy(mTPCSliceTrackersCPU[iSlice].LinkTmpMemory(), Res(mTPCSliceTrackersCPU[iSlice].Data().MemoryResScratch()).Ptr(), Res(mTPCSliceTrackersCPU[iSlice].Data().MemoryResScratch()).Size());
			if (mDeviceProcessingSettings.debugMask & 2) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, NSLICES);
		mTPCSliceTrackersCPU[iSlice].StartTimer(2);
		AliGPUTPCProcess<AliGPUTPCNeighboursCleaner> <<<GPUCA_ROW_COUNT - 2, GPUCA_GPU_THREAD_COUNT, 0, mInternals->CudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Neighbours Cleaner", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(2);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, NSLICES);
		mTPCSliceTrackersCPU[iSlice].StartTimer(3);
		AliGPUTPCProcess<AliGPUTPCStartHitsFinder> <<<GPUCA_ROW_COUNT - 6, GPUCA_GPU_THREAD_COUNT, 0, mInternals->CudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Start Hits Finder", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(3);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, NSLICES);
		mTPCSliceTrackersCPU[iSlice].StartTimer(4);
		AliGPUTPCProcess<AliGPUTPCStartHitsSorter> <<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT, 0, mInternals->CudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Start Hits Sorter", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(4);

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *mTPCSliceTrackersCPU[iSlice].NTracklets(), iSlice);
			if (*mTPCSliceTrackersCPU[iSlice].NTracklets() > GPUCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				CAGPUError("GPUCA_GPU_MAX_TRACKLETS constant insuffisant");
				ResetHelperThreads(1);
				return(1);
			}
		}

		if (mDeviceProcessingSettings.debugLevel >= 4 && *mTPCSliceTrackersCPU[iSlice].NTracklets())
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 32) mTPCSliceTrackersCPU[iSlice].DumpStartHits(mDebugFile);
		}

#ifdef GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor (Slice %d/%d)", iSlice, NSLICES)
		mTPCSliceTrackersCPU[iSlice].StartTimer(6);
		AliGPUTPCTrackletConstructorSingleSlice<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR, 0, mInternals->CudaStreams[useStream]>>>(iSlice);
		if (GPUSync("Tracklet Constructor", useStream, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(6);
#endif
	}

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

#ifdef GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	SynchronizeGPU();
#else
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor");
	mTPCSliceTrackersCPU[0].StartTimer(6);
	AliGPUTPCTrackletConstructorGPU<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (GPUSync("Tracklet Constructor", -1, 0) RANDOM_ERROR)
	{
		SynchronizeGPU;
		cuCtxPopCurrent(&mInternals->CudaContext);
		return(1);
	}
	mTPCSliceTrackersCPU[0].StopTimer(6);
#endif //GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE

	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			CAGPUInfo("Obtained %d tracklets", *mTPCSliceTrackersCPU[iSlice].NTracklets());
			if (mDeviceProcessingSettings.debugMask & 128) mTPCSliceTrackersCPU[iSlice].DumpTrackletHits(mDebugFile);
		}
	}

	int runSlices = 0;
	int useStream = 0;
	int streamMap[NSLICES];
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice += runSlices)
	{
		if (runSlices < GPUCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		runSlices = CAMath::Min(runSlices, NSLICES - iSlice);
		if (fSelectorBlockCount < runSlices) runSlices = fSelectorBlockCount;
		if (GPUCA_GPU_NUM_STREAMS && useStream + 1 == GPUCA_GPU_NUM_STREAMS) runSlices = NSLICES - iSlice;
		if (fSelectorBlockCount < runSlices)
		{
			CAGPUError("Insufficient number of blocks for tracklet selector");
			cuCtxPopCurrent(&mInternals->CudaContext);
			return(1);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
		mTPCSliceTrackersCPU[iSlice].StartTimer(7);
		AliGPUTPCProcessMulti<AliGPUTPCTrackletSelector><<<fSelectorBlockCount, GPUCA_GPU_THREAD_COUNT_SELECTOR, 0, mInternals->CudaStreams[useStream]>>>(iSlice, runSlices);
		if (GPUSync("Tracklet Selector", iSlice, iSlice) RANDOM_ERROR)
		{
			SynchronizeGPU();
			cuCtxPopCurrent(&mInternals->CudaContext);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(7);
		for (unsigned int k = iSlice;k < iSlice + runSlices;k++) streamMap[k] = useStream;
		useStream++;
	}

	fSliceOutputReady = 0;

	if (Reconstruct_Base_StartGlobal()) return(1);

	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), streamMap[iSlice]) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			return(1);
		}
	}

	unsigned int tmpSlice = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Transfering Tracks from GPU to Host");

		while (tmpSlice < NSLICES && (tmpSlice == iSlice ? cudaStreamSynchronize(mInternals->CudaStreams[streamMap[tmpSlice]]) : cudaStreamQuery(mInternals->CudaStreams[streamMap[tmpSlice]])) == (cudaError_t) CUDA_SUCCESS)
		{
			if (*mTPCSliceTrackersCPU[tmpSlice].NTracks() > 0)
			{
				useStream = GPUCA_GPU_NUM_STREAMS ? streamMap[tmpSlice] : tmpSlice;
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTracks(), useStream);
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTrackHits(), useStream);
			}
			tmpSlice++;
		}

		useStream = GPUCA_GPU_NUM_STREAMS ? streamMap[iSlice] : iSlice;
		if (GPUFailedMsg(cudaStreamSynchronize(mInternals->CudaStreams[useStream])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			return(1);
		}

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) mTPCSliceTrackersCPU[iSlice].DumpHitWeights(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 512) mTPCSliceTrackersCPU[iSlice].DumpTrackHits(mDebugFile);
		}

		if (mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError];
			CAGPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, mTPCSliceTrackersCPU[iSlice].Data().NumberOfHits());

			ResetHelperThreads(1);
			return(1);
		}
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Tracks Transfered: %d / %d", *mTPCSliceTrackersCPU[iSlice].NTracks(), *mTPCSliceTrackersCPU[iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(iSlice)) return(1);
	}
	if (Reconstruct_Base_Finalize()) return(1);

	/*for (unsigned int i = 0;i < NSLICES;i++)
	{
		mTPCSliceTrackersCPU[i].DumpOutput(stdout);
	}*/

	/*static int runnum = 0;
	std::ofstream tmpOut;
	char buffer[1024];
	sprintf(buffer, "GPUtracks%d.out", runnum++);
	tmpOut.open(buffer);
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		mTPCSliceTrackersCPU[iSlice].DumpTrackHits(tmpOut);
	}
	tmpOut.close();*/

#ifdef GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
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
	bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + (fConstructorBlockCount * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR / 32 * 33 - 1) * bmpheight ;
	bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

	bmpIH.biSize = sizeof(bmpIH);
	bmpIH.biWidth = fConstructorBlockCount * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR / 32 * 33 - 1;
	bmpIH.biHeight = bmpheight;
	bmpIH.biPlanes = 1;
	bmpIH.biBitCount = 32;

	fwrite(&bmpFH, 1, sizeof(bmpFH), fp2);
	fwrite(&bmpIH, 1, sizeof(bmpIH), fp2);

	for (int i = 0;i < bmpheight * fConstructorBlockCount * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;i += fConstructorBlockCount * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR)
	{
		fEmpty = 1;
		for (int j = 0;j < fConstructorBlockCount * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;j++)
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
		//if (nEmptySync == GPUCA_GPU_SCHED_ROW_STEP + 2) break;
	}

	fclose(fp);
	fclose(fp2);
	free(stageAtSync);
#endif

	cuCtxPopCurrent(&mInternals->CudaContext);
	return(0);
}

int AliGPUReconstructionCUDA::ExitDevice_Runtime()
{
	//Uninitialize CUDA
	cuCtxPushCurrent(mInternals->CudaContext);

	SynchronizeGPU();
	if (mDeviceMemoryBase)
	{
		cudaFree(mDeviceMemoryBase);
		mDeviceMemoryBase = NULL;
	}
	if (mHostMemoryBase)
	{
		int nStreams = GPUCA_GPU_NUM_STREAMS == 0 ? 3 : GPUCA_GPU_NUM_STREAMS;
		for (int i = 0;i < nStreams;i++)
		{
			cudaStreamDestroy(mInternals->CudaStreams[i]);
		}
		free(mInternals->CudaStreams);
		fGpuTracker = NULL;
		cudaFreeHost(mHostMemoryBase);
		mHostMemoryBase = NULL;
	}

	if (GPUFailedMsg(cudaDeviceReset()))
	{
		CAGPUError("Could not uninitialize GPU");
		return(1);
	}

	cuCtxDestroy(mInternals->CudaContext);

	CAGPUInfo("CUDA Uninitialized");
	return(0);
}

int AliGPUReconstructionCUDA::RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers)
{
#ifndef GPUCA_GPU_MERGER
	CAGPUError("GPUCA_GPU_MERGER compile flag not set");
	return(1);
#else

	HighResTimer timer;
	static double times[3] = {};
	static int nCount = 0;
	if (resetTimers)
	{
		for (unsigned int k = 0;k < sizeof(times) / sizeof(times[0]);k++) times[k] = 0;
		nCount = 0;
	}
	cuCtxPushCurrent(mInternals->CudaContext);

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Running GPU Merger (%d/%d)", Merger->NOutputTrackClusters(), Merger->NClusters());
	timer.Start();

	memcpy((void*) fGpuMerger, (void*) Merger, sizeof(AliGPUTPCGMMerger));

	fGpuMerger->InitGPUProcessor((AliGPUReconstruction*) this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
	ResetRegisteredMemoryPointers(Merger->MemoryResRefit());
	
	GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, fGpuMerger, sizeof(*Merger), (char*) &AliGPUCAConstantMemDummy.tpcMerger - (char*) &AliGPUCAConstantMemDummy, cudaMemcpyHostToDevice));
	TransferMemoryResourceLinkToGPU(Merger->MemoryResRefit());
	times[0] += timer.GetCurrentElapsedTime(true);
	
	RefitTracks<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT>>>(fGpuMerger->OutputTracks(), Merger->NOutputTracks(), fGpuMerger->Clusters());
	if (SynchronizeGPU()) return(1);
	times[1] += timer.GetCurrentElapsedTime(true);
	
	TransferMemoryResourceLinkToHost(Merger->MemoryResRefit());
	if (SynchronizeGPU()) return(1);
	times[2] += timer.GetCurrentElapsedTime();
	
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("GPU Merger Finished");
	nCount++;

	if (mDeviceProcessingSettings.debugLevel > 0)
	{
		int copysize = 4 * Merger->NOutputTrackClusters() * sizeof(float) + Merger->NOutputTrackClusters() * sizeof(unsigned int) + Merger->NOutputTracks() * sizeof(AliGPUTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(AliGPUCAParam);
		double speed = (double) copysize / times[0] * nCount / 1e9;
		printf("GPU Fit:\tCopy To:\t%1.0f us (%lf GB/s)\n", times[0] * 1000000 / nCount, speed);
		printf("\t\tFit:\t%1.0f us\n", times[1] * 1000000 / nCount);
		speed = (double) copysize / times[2] * nCount / 1e9;
		printf("\t\tCopy From:\t%1.0f us (%lf GB/s)\n", times[2] * 1000000 / nCount, speed);
	}

	if (!GPUCA_TIMING_SUM)
	{
		for (int i = 0;i < 3;i++) times[i] = 0;
		nCount = 0;
	}

	cuCtxPopCurrent((CUcontext*) &mInternals->CudaContext);
	return(0);
#endif
}

int AliGPUReconstructionCUDA::TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to GPU: %s\n", res->Name());
	if (stream == -1) return GPUFailedMsg(cudaMemcpy(res->PtrDevice(), res->Ptr(), res->Size(), cudaMemcpyHostToDevice));
	else return GPUFailedMsg(cudaMemcpyAsync(res->PtrDevice(), res->Ptr(), res->Size(), cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
}

int AliGPUReconstructionCUDA::TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to Host: %s\n", res->Name());
	if (stream == -1) return GPUFailedMsg(cudaMemcpy(res->Ptr(), res->PtrDevice(), res->Size(), cudaMemcpyDeviceToHost));
	return GPUFailedMsg(cudaMemcpyAsync(res->Ptr(), res->PtrDevice(), res->Size(), cudaMemcpyDeviceToHost, mInternals->CudaStreams[stream]));
}

int AliGPUReconstructionCUDA::GPUMergerAvailable() const
{
#ifdef GPUCA_GPU_MERGER
	return(1);
#else
	return(0);
#endif
}

void AliGPUReconstructionCUDA::ActivateThreadContext()
{
	cuCtxPushCurrent(mInternals->CudaContext);
}
void AliGPUReconstructionCUDA::ReleaseThreadContext()
{
	cuCtxPopCurrent(&mInternals->CudaContext);
}

int AliGPUReconstructionCUDA::SynchronizeGPU()
{
	GPUFailedMsg(cudaDeviceSynchronize());
	return(0);
}
