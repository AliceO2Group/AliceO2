#include <cuda.h>
#include <sm_20_atomic_functions.h>
#define GPUCA_GPUTYPE_PASCAL

#include "AliGPUReconstructionCUDA.h"
#include "AliGPUReconstructionCUDAInternals.h"
#include "AliGPUReconstructionCommon.h"

__constant__ uint4 gGPUConstantMemBuffer[(sizeof(AliGPUCAConstantMem) + sizeof(uint4) - 1) / sizeof(uint4)];
__constant__ char& gGPUConstantMemBufferChar = (char&) gGPUConstantMemBuffer;
__constant__ AliGPUCAConstantMem& gGPUConstantMem = (AliGPUCAConstantMem&) gGPUConstantMemBufferChar;

#ifdef GPUCA_GPU_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

#ifdef HAVE_O2HEADERS
#include "ITStrackingCUDA/TrackerTraitsNV.h"
#else
namespace o2 { namespace ITS { class TrackerTraitsNV : public TrackerTraits {}; }}
#endif

#define DEVICE_KERNELS_PRE
#include "AliGPUDeviceKernels.h"

template <class TProcess, typename... Args> GPUg() void runKernelCUDA(int iSlice, Args... args)
{
	AliGPUTPCTracker &tracker = gGPUConstantMem.tpcTrackers[iSlice];
	GPUshared() typename TProcess::AliGPUTPCSharedMemory smem;
	TProcess::Thread(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, tracker, args...);
}

template <class TProcess, typename... Args> GPUg() void runKernelCUDAMulti(int firstSlice, int nSliceCount, Args... args)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	AliGPUTPCTracker &tracker = gGPUConstantMem.tpcTrackers[firstSlice + iSlice];
	GPUshared() typename TProcess::AliGPUTPCSharedMemory smem;
	TProcess::Thread(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, tracker, args...);
}

template <class T, typename... Args> int AliGPUReconstructionCUDABackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
	if (x.device == krnlDeviceType::CPU) return AliGPUReconstructionCPU::runKernelBackend<T> (x, y, z, args...);
	if (z.evList) for (int k = 0;k < z.nEvents;k++) GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[x.stream], ((cudaEvent_t*) z.evList)[k], 0));
	if (y.num <= 1)
	{
		runKernelCUDA<T> <<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>>(y.start, args...);
	}
	else
	{
		runKernelCUDAMulti<T> <<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>> (y.start, y.num, args...);
	}
	if (z.ev) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) z.ev, mInternals->CudaStreams[x.stream]));
	return 0;
}

AliGPUReconstructionCUDABackend::AliGPUReconstructionCUDABackend(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mInternals = new AliGPUReconstructionCUDAInternals;
	mProcessingSettings.deviceType = CUDA;
	mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsNV);
}

AliGPUReconstructionCUDABackend::~AliGPUReconstructionCUDABackend()
{
	mITSTrackerTraits.reset(nullptr); //Make sure we destroy the ITS tracker before we exit CUDA
	cudaDeviceReset();
	delete mInternals;
}

AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUCASettingsProcessing& cfg)
{
	return new AliGPUReconstructionCUDA(cfg);
}

int AliGPUReconstructionCUDABackend::InitDevice_Runtime()
{
	//Find best CUDA device, initialize and allocate memory

	cudaDeviceProp cudaDeviceProp;

	int count, bestDevice = -1;
	double bestDeviceSpeed = -1, deviceSpeed;
	if (GPUFailedMsgI(cudaGetDeviceCount(&count)))
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
		if (GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp, i))) continue;
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

	mNStreams = std::max(GPUCA_GPU_NUM_STREAMS, 3);

	if (cuCtxCreate(&mInternals->CudaContext, CU_CTX_SCHED_AUTO, fDeviceId) != CUDA_SUCCESS)
	{
		CAGPUError("Could not set CUDA Device!");
		return(1);
	}

	if (mDeviceMemorySize > cudaDeviceProp.totalGlobalMem || GPUFailedMsgI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize)))
	{
		CAGPUError("CUDA Memory Allocation Error");
		cudaDeviceReset();
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("GPU Memory used: %lld", (long long int) mDeviceMemorySize);
	if (GPUFailedMsgI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize)))
	{
		CAGPUError("Error allocating Page Locked Host Memory");
		cudaDeviceReset();
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("Host Memory used: %lld", (long long int) mHostMemorySize);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(mHostMemoryBase, 0, mHostMemorySize);
		if (GPUFailedMsgI(cudaMemset(mDeviceMemoryBase, 143, mDeviceMemorySize)))
		{
			CAGPUError("Error during CUDA memset");
			cudaDeviceReset();
			return(1);
		}
	}

	for (int i = 0;i < mNStreams;i++)
	{
		if (GPUFailedMsgI(cudaStreamCreate(&mInternals->CudaStreams[i])))
		{
			CAGPUError("Error creating CUDA Stream");
			cudaDeviceReset();
			return(1);
		}
	}
	
	void* devPtrConstantMem;
	if (GPUFailedMsgI(cudaGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer)))
	{
		CAGPUError("Error getting ptr to constant memory");
		cudaDeviceReset();
		return 1;
	}
	mDeviceConstantMem = (AliGPUCAConstantMem*) devPtrConstantMem;
	
	cudaEvent_t *events = (cudaEvent_t*) &mEvents;
	for (unsigned int i = 0;i < sizeof(mEvents) / sizeof(cudaEvent_t);i++)
	{
		if (GPUFailedMsgI(cudaEventCreate(&events[i])))
		{
			CAGPUError("Error creating event");
			cudaDeviceReset();
			return 1;
		}
	}

	ReleaseThreadContext();
	CAGPUInfo("CUDA Initialisation successfull (Device %d: %s, Thread %d, %lld/%lld bytes used)", fDeviceId, cudaDeviceProp.name, fThreadId, (long long int) mHostMemorySize, (long long int) mDeviceMemorySize);

	return(0);
}

int AliGPUReconstructionCUDABackend::RunTPCTrackingSlices()
{
	int retVal = RunTPCTrackingSlices_internal();
	if (retVal) SynchronizeGPU();
	if (retVal >= 2)
	{
		ResetHelperThreads(retVal >= 3);
	}
	ReleaseThreadContext();
	return(retVal != 0);
}

int AliGPUReconstructionCUDABackend::RunTPCTrackingSlices_internal()
{
	//Primary reconstruction function
	if (fGPUStuck)
	{
		CAGPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	if (Reconstruct_Base_Init()) return(1);
	if (PrepareTextures()) return(2);

	//Copy Tracker Object to GPU Memory
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Tracker objects to GPU");
	if (PrepareProfile()) return 2;
	
	WriteToConstantMemory((char*) &mDeviceConstantMem->param - (char*) mDeviceConstantMem, &mParam, sizeof(AliGPUCAParam), mNStreams - 1);
	WriteToConstantMemory((char*) mDeviceConstantMem->tpcTrackers - (char*) mDeviceConstantMem, mWorkersShadow->tpcTrackers, sizeof(AliGPUTPCTracker) * NSLICES, mNStreams - 1, &mEvents.init);
	
	for (int i = 0;i < mNStreams - 1;i++)
	{
		mStreamInit[i] = false;
	}
	mStreamInit[mNStreams - 1] = true;

	if (GPUDebug("Initialization (1)", 0, 0) RANDOM_ERROR)
	{
		return(2);
	}

	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(iSlice)) return(1);
		
		int useStream = (iSlice % mNStreams);
		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		runKernel<AliGPUMemClean16>({fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT, useStream}, krnlRunRangeNone, {}, mWorkersShadow->tpcTrackers[iSlice].Data().HitWeights(), mWorkersShadow->tpcTrackers[iSlice].Data().NumberOfHitsPlusAlign() * sizeof(*mWorkersShadow->tpcTrackers[iSlice].Data().HitWeights()));
		if (GPUDebug("Initialization (2)", 2, iSlice) RANDOM_ERROR)
		{
			return(3);
		}

		//Copy Data to GPU Global Memory
		mWorkers->tpcTrackers[iSlice].StartTimer(0);
		TransferMemoryResourceLinkToGPU(mWorkers->tpcTrackers[iSlice].Data().MemoryResInput(), useStream);
		TransferMemoryResourceLinkToGPU(mWorkers->tpcTrackers[iSlice].Data().MemoryResRows(), useStream);
		TransferMemoryResourceLinkToGPU(mWorkers->tpcTrackers[iSlice].MemoryResCommon(), useStream);

		if (GPUDebug("Initialization (3)", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(0);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, NSLICES);
		mWorkers->tpcTrackers[iSlice].StartTimer(1);
		runKernel<AliGPUTPCNeighboursFinder>({GPUCA_ROW_COUNT, GPUCA_GPU_THREAD_COUNT_FINDER, useStream}, {iSlice}, {nullptr, mStreamInit[useStream] ? nullptr : &mEvents.init});
		if (GPUDebug("Neighbours finder", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(1);
		mStreamInit[useStream] = true;

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mWorkers->tpcTrackers[iSlice].Data(), -1, true);
			memcpy(mWorkers->tpcTrackers[iSlice].LinkTmpMemory(), Res(mWorkers->tpcTrackers[iSlice].Data().MemoryResScratch()).Ptr(), Res(mWorkers->tpcTrackers[iSlice].Data().MemoryResScratch()).Size());
			if (mDeviceProcessingSettings.debugMask & 2) mWorkers->tpcTrackers[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, NSLICES);
		mWorkers->tpcTrackers[iSlice].StartTimer(2);
		runKernel<AliGPUTPCNeighboursCleaner>({GPUCA_ROW_COUNT - 2, GPUCA_GPU_THREAD_COUNT, useStream}, {iSlice});
		if (GPUDebug("Neighbours Cleaner", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(2);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&mWorkers->tpcTrackers[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) mWorkers->tpcTrackers[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, NSLICES);
		mWorkers->tpcTrackers[iSlice].StartTimer(3);
		runKernel<AliGPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, GPUCA_GPU_THREAD_COUNT, useStream}, {iSlice});
		if (GPUDebug("Start Hits Finder", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(3);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, NSLICES);
		mWorkers->tpcTrackers[iSlice].StartTimer(4);
		runKernel<AliGPUTPCStartHitsSorter>({fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT, useStream}, {iSlice});
		if (GPUDebug("Start Hits Sorter", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(4);

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(mWorkers->tpcTrackers[iSlice].MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *mWorkers->tpcTrackers[iSlice].NTracklets(), iSlice);
			if (*mWorkers->tpcTrackers[iSlice].NTracklets() > GPUCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				CAGPUError("GPUCA_GPU_MAX_TRACKLETS constant insuffisant");
				return(3);
			}
		}

		if (mDeviceProcessingSettings.debugLevel >= 4 && *mWorkers->tpcTrackers[iSlice].NTracklets())
		{
			TransferMemoryResourcesToHost(&mWorkers->tpcTrackers[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 32) mWorkers->tpcTrackers[iSlice].DumpStartHits(mDebugFile);
		}

#ifdef GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor (Slice %d/%d)", iSlice, NSLICES)
		mWorkers->tpcTrackers[iSlice].StartTimer(6);
		AliGPUTPCTrackletConstructorSingleSlice<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR, 0, mInternals->CudaStreams[useStream]>>>(iSlice);
		if (GPUDebug("Tracklet Constructor", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(6);
#endif
	}
	ReleaseEvent(&mEvents.init);

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

#ifdef GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	SynchronizeGPU();
#else
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor");
	mWorkers->tpcTrackers[0].StartTimer(6);
	AliGPUTPCTrackletConstructorGPU<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (GPUDebug("Tracklet Constructor", -1, 0) RANDOM_ERROR)
	{
		return(1);
	}
	mWorkers->tpcTrackers[0].StopTimer(6);
#endif //GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
	ReleaseEvent(&mEvents.constructor);

	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			TransferMemoryResourcesToHost(&mWorkers->tpcTrackers[iSlice], -1, true);
			CAGPUInfo("Obtained %d tracklets", *mWorkers->tpcTrackers[iSlice].NTracklets());
			if (mDeviceProcessingSettings.debugMask & 128) mWorkers->tpcTrackers[iSlice].DumpTrackletHits(mDebugFile);
		}
	}

	unsigned int runSlices = 0;
	int useStream = 0;
	int streamMap[NSLICES];
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice += runSlices)
	{
		if (runSlices < GPUCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		runSlices = CAMath::Min(runSlices, NSLICES - iSlice);
		if (fSelectorBlockCount < runSlices) runSlices = fSelectorBlockCount;

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
		mWorkers->tpcTrackers[iSlice].StartTimer(7);
		runKernel<AliGPUTPCTrackletSelector>({fSelectorBlockCount, GPUCA_GPU_THREAD_COUNT_SELECTOR, useStream}, {iSlice, runSlices});
		if (GPUDebug("Tracklet Selector", useStream, iSlice) RANDOM_ERROR)
		{
			return(1);
		}
		mWorkers->tpcTrackers[iSlice].StopTimer(7);
		for (unsigned int k = iSlice;k < iSlice + runSlices;k++)
		{
			TransferMemoryResourceLinkToHost(mWorkers->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents.selector[k]);
			streamMap[k] = useStream;
		}
		useStream++;
		if (useStream >= mNStreams) useStream = 0;
	}

	fSliceOutputReady = 0;

	if (Reconstruct_Base_StartGlobal()) return(1);

	bool transferRunning[NSLICES];
	for (unsigned int i = 0;i < NSLICES;i++) transferRunning[i] = true;
	unsigned int tmpSlice = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Transfering Tracks from GPU to Host");

		if (tmpSlice == iSlice) SynchronizeEvents(&mEvents.selector[iSlice]);
		while (tmpSlice < NSLICES && (tmpSlice == iSlice || IsEventDone(&mEvents.selector[tmpSlice])))
		{
			ReleaseEvent(&mEvents.selector[tmpSlice]);
			if (*mWorkers->tpcTrackers[tmpSlice].NTracks() > 0)
			{
				TransferMemoryResourceLinkToHost(mWorkers->tpcTrackers[tmpSlice].MemoryResTracks(), streamMap[tmpSlice]);
				TransferMemoryResourceLinkToHost(mWorkers->tpcTrackers[tmpSlice].MemoryResTrackHits(), streamMap[tmpSlice], &mEvents.selector[tmpSlice]);
			}
			else
			{
				transferRunning[tmpSlice] = false;
			}
			tmpSlice++;
		}

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mWorkers->tpcTrackers[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) mWorkers->tpcTrackers[iSlice].DumpHitWeights(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 512) mWorkers->tpcTrackers[iSlice].DumpTrackHits(mDebugFile);
		}

		if (mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError];
			CAGPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, mWorkers->tpcTrackers[iSlice].Data().NumberOfHits());
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) if (transferRunning[iSlice2]) ReleaseEvent(&mEvents.selector[iSlice2]);
			return(3);
		}

		if (transferRunning[iSlice]) SynchronizeEvents(&mEvents.selector[iSlice]);
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Tracks Transfered: %d / %d", *mWorkers->tpcTrackers[iSlice].NTracks(), *mWorkers->tpcTrackers[iSlice].NTrackHits());
		if (Reconstruct_Base_FinishSlices(iSlice)) return(1);
	}
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++) if (transferRunning[iSlice]) ReleaseEvent(&mEvents.selector[iSlice]);

	if (Reconstruct_Base_Finalize()) return(1);

	if (DoProfile()) return(1);
	
	if (mDeviceProcessingSettings.debugMask & 1024)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			mWorkers->tpcTrackers[i].DumpOutput(stdout);
		}
	}

	return(0);
}

int AliGPUReconstructionCUDABackend::ExitDevice_Runtime()
{
	//Uninitialize CUDA
	ActivateThreadContext();

	SynchronizeGPU();

	GPUFailedMsgI(cudaFree(mDeviceMemoryBase));
	mDeviceMemoryBase = nullptr;

	for (int i = 0;i < mNStreams;i++)
	{
		GPUFailedMsgI(cudaStreamDestroy(mInternals->CudaStreams[i]));
	}

	GPUFailedMsgI(cudaFreeHost(mHostMemoryBase));
	mHostMemoryBase = nullptr;
	
	cudaEvent_t *events = (cudaEvent_t*) &mEvents;
	for (unsigned int i = 0;i < sizeof(mEvents) & sizeof(cudaEvent_t);i++)
	{
		GPUFailedMsgI(cudaEventDestroy(events[i]));
	}

	if (GPUFailedMsgI(cudaDeviceReset()))
	{
		CAGPUError("Could not uninitialize GPU");
		return(1);
	}

	cuCtxDestroy(mInternals->CudaContext);

	CAGPUInfo("CUDA Uninitialized");
	return(0);
}

int AliGPUReconstructionCUDABackend::DoTRDGPUTracking()
{
#ifndef GPUCA_GPU_MERGER
	CAGPUError("GPUCA_GPU_MERGER compile flag not set");
	return(1);
#else
	ActivateThreadContext();
	SetupGPUProcessor(&mWorkers->trdTracker);
	mWorkersShadow->trdTracker.SetGeometry((AliGPUTRDGeometry*) mProcDevice.fTrdGeometry);

	GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, &mWorkersShadow->trdTracker, sizeof(mWorkersShadow->trdTracker), (char*) &mDeviceConstantMem->trdTracker - (char*) mDeviceConstantMem, cudaMemcpyHostToDevice));

	TransferMemoryResourcesToGPU(&mWorkers->trdTracker);

	DoTrdTrackingGPU<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT_TRD>>>();
	SynchronizeGPU();

	TransferMemoryResourcesToHost(&mWorkers->trdTracker);
	SynchronizeGPU();

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("GPU TRD tracker Finished");

	ReleaseThreadContext();
	return(0);
#endif
}

int AliGPUReconstructionCUDABackend::RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers)
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
	ActivateThreadContext();

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Running GPU Merger (%d/%d)", Merger->NOutputTrackClusters(), Merger->NClusters());
	timer.Start();

	SetupGPUProcessor(Merger);
	mWorkersShadow->tpcMerger.OverrideSliceTracker(mDeviceConstantMem->tpcTrackers);
	
	GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, &mWorkersShadow->tpcMerger, sizeof(mWorkersShadow->tpcMerger), (char*) &mDeviceConstantMem->tpcMerger - (char*) mDeviceConstantMem, cudaMemcpyHostToDevice));
	TransferMemoryResourceLinkToGPU(Merger->MemoryResRefit());
	times[0] += timer.GetCurrentElapsedTime(true);
	
	RefitTracks<<<fConstructorBlockCount, GPUCA_GPU_THREAD_COUNT>>>(mWorkersShadow->tpcMerger.OutputTracks(), mWorkersShadow->tpcMerger.NOutputTracks(), mWorkersShadow->tpcMerger.Clusters());
	SynchronizeGPU();
	times[1] += timer.GetCurrentElapsedTime(true);
	
	TransferMemoryResourceLinkToHost(Merger->MemoryResRefit());
	SynchronizeGPU();
	times[2] += timer.GetCurrentElapsedTime();
	
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("GPU Merger Finished");
	nCount++;

	if (mDeviceProcessingSettings.debugLevel > 0)
	{
		int copysize = 4 * Merger->NOutputTrackClusters() * sizeof(float) + Merger->NOutputTrackClusters() * sizeof(unsigned int) + Merger->NOutputTracks() * sizeof(AliGPUTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(AliGPUCAParam);
		double speed = (double) copysize / times[0] * nCount / 1e9;
		printf("GPU Fit:\tCopy To:\t%'7d us (%6.3f GB/s)\n", (int) (times[0] * 1000000 / nCount), speed);
		printf("\t\tFit:\t\t%'7d us\n", (int) (times[1] * 1000000 / nCount));
		speed = (double) copysize / times[2] * nCount / 1e9;
		printf("\t\tCopy From:\t%'7d us (%6.3f GB/s)\n", (int) (times[2] * 1000000 / nCount), speed);
	}

	if (!GPUCA_TIMING_SUM)
	{
		for (int i = 0;i < 3;i++) times[i] = 0;
		nCount = 0;
	}

	ReleaseThreadContext();
	return(0);
#endif
}

void AliGPUReconstructionCUDABackend::TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
	//if (evList == nullptr) nEvents = 0;
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to GPU: %s\n", res->Name());
	if (stream == -1)
	{
		GPUFailedMsg(cudaMemcpy(res->PtrDevice(), res->Ptr(), res->Size(), cudaMemcpyHostToDevice));
	}
	else
	{
		if (evList == nullptr) nEvents = 0;
		for (int k = 0;k < nEvents;k++) GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[stream], ((cudaEvent_t*) evList)[k], 0));
		GPUFailedMsg(cudaMemcpyAsync(res->PtrDevice(), res->Ptr(), res->Size(), cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
		if (ev) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream]));
	}
}

void AliGPUReconstructionCUDABackend::TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
	//if (evList == nullptr) nEvents = 0;
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to Host: %s\n", res->Name());
	if (stream == -1)
	{
		GPUFailedMsg(cudaMemcpy(res->Ptr(), res->PtrDevice(), res->Size(), cudaMemcpyDeviceToHost));
	}
	else
	{
		if (evList == nullptr) nEvents = 0;
		for (int k = 0;k < nEvents;k++) GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[stream], ((cudaEvent_t*) evList)[k], 0));
		GPUFailedMsg(cudaMemcpyAsync(res->Ptr(), res->PtrDevice(), res->Size(), cudaMemcpyDeviceToHost, mInternals->CudaStreams[stream]));
		if (ev) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream]));
	}
}

void AliGPUReconstructionCUDABackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
	if (stream == -1) GPUFailedMsg(cudaMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice));
	else GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
	if (ev && stream != -1) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream]));
}

void AliGPUReconstructionCUDABackend::ReleaseEvent(deviceEvent* ev) {}

int AliGPUReconstructionCUDABackend::GPUMergerAvailable() const
{
#ifdef GPUCA_GPU_MERGER
	return(1);
#else
	return(0);
#endif
}

void AliGPUReconstructionCUDABackend::ActivateThreadContext()
{
	cuCtxPushCurrent(mInternals->CudaContext);
}
void AliGPUReconstructionCUDABackend::ReleaseThreadContext()
{
	cuCtxPopCurrent(&mInternals->CudaContext);
}

void AliGPUReconstructionCUDABackend::SynchronizeGPU()
{
	GPUFailedMsg(cudaDeviceSynchronize());
}

void AliGPUReconstructionCUDABackend::SynchronizeStream(int stream)
{
	GPUFailedMsg(cudaStreamSynchronize(mInternals->CudaStreams[stream]));
}

void AliGPUReconstructionCUDABackend::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
	for (int i = 0;i < nEvents;i++)
	{
		GPUFailedMsg(cudaEventSynchronize(((cudaEvent_t*) evList)[i]));
	}
}

int AliGPUReconstructionCUDABackend::IsEventDone(deviceEvent* evList, int nEvents)
{
	for (int i = 0;i < nEvents;i++)
	{
		cudaError_t retVal = cudaEventSynchronize(((cudaEvent_t*) evList)[i]);
		if (retVal == cudaErrorNotReady) return 0;
		GPUFailedMsg(retVal);
	}
	return(1);
}

int AliGPUReconstructionCUDABackend::GPUDebug(const char* state, int stream, int slice)
{
	//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards, in case of debugmode
	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		CAGPUError("Cuda Error %s while running kernel (%s) (Stream %d; Slice %d/%d)", cudaGetErrorString(cuErr), state, stream, slice, NSLICES);
		return(1);
	}
	if (GPUFailedMsgI(cudaDeviceSynchronize()))
	{
		CAGPUError("CUDA Error while synchronizing (%s) (Stream %d; Slice %d/%d)", state, stream, slice, NSLICES);
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Sync Done");
	return(0);
}

int AliGPUReconstructionCUDABackend::PrepareTextures()
{
#ifdef GPUCA_GPU_USE_TEXTURES
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<cahit2>();
	size_t offset;
	GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, mWorkersShadow->tpcTrackers[0].Data().Memory(), &channelDescu2, NSLICES * GPUCA_GPU_SLICE_DATA_MEMORY));
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<calink>();
	GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, mWorkersShadow->tpcTrackers[0].Data().Memory(), &channelDescu, NSLICES * GPUCA_GPU_SLICE_DATA_MEMORY));
#endif
	return(0);
}

int AliGPUReconstructionCUDABackend::PrepareProfile()
{
#ifdef GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* tmpMem;
	GPUFailedMsg(cudaMalloc(&tmpMem, 100000000));
	mWorkersShadow->tpcTrackers[0].fStageAtSync = tmpMem;
	GPUFailedMsg(cudaMemset(mWorkersShadow->tpcTrackers[0].StageAtSync(), 0, 100000000));
#endif
	return 0;
}

int AliGPUReconstructionCUDABackend::DoProfile()
{
#ifdef GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* stageAtSync = (char*) malloc(100000000);
	GPUFailedMsg(cudaMemcpy(stageAtSync, mWorkersShadow->tpcTrackers[0].StageAtSync(), 100 * 1000 * 1000, cudaMemcpyDeviceToHost));
	cudaFree(mWorkersShadow->tpcTrackers[0].StageAtSync());

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
	return 0;
}
