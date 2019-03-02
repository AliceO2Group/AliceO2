#include <cuda.h>
#include <sm_20_atomic_functions.h>
#define GPUCA_GPUTYPE_PASCAL

#include "AliGPUReconstructionCUDA.h"
#include "AliGPUReconstructionCUDAInternals.h"
#include "AliGPUReconstructionIncludes.h"

__constant__ uint4 gGPUConstantMemBuffer[(sizeof(AliGPUConstantMem) + sizeof(uint4) - 1) / sizeof(uint4)];
__constant__ char& gGPUConstantMemBufferChar = (char&) gGPUConstantMemBuffer;
__constant__ AliGPUConstantMem& gGPUConstantMem = (AliGPUConstantMem&) gGPUConstantMemBufferChar;

#ifdef GPUCA_USE_TEXTURES
texture<cahit2, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu2;
texture<calink, cudaTextureType1D, cudaReadModeElementType> gAliTexRefu;
#endif

#ifdef HAVE_O2HEADERS
#include "ITStrackingCUDA/TrackerTraitsNV.h"
#else
namespace o2 { namespace ITS { class TrackerTraitsNV : public TrackerTraits {}; }}
#endif

#include "AliGPUReconstructionIncludesDevice.h"

template <class T, int I, typename... Args> GPUg() void runKernelCUDA(int iSlice, Args... args)
{
	GPUshared() typename T::AliGPUTPCSharedMemory smem;
	T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Worker(gGPUConstantMem)[iSlice], args...);
}

template <class T, int I, typename... Args> GPUg() void runKernelCUDAMulti(int firstSlice, int nSliceCount, Args... args)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	GPUshared() typename T::AliGPUTPCSharedMemory smem;
	T::template Thread<I>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Worker(gGPUConstantMem)[firstSlice + iSlice], args...);
}

template <class T, int I, typename... Args> int AliGPUReconstructionCUDABackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
	if (x.device == krnlDeviceType::CPU) return AliGPUReconstructionCPU::runKernelImpl(classArgument<T, I>(), x, y, z, args...);
	if (z.evList) for (int k = 0;k < z.nEvents;k++) GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[x.stream], ((cudaEvent_t*) z.evList)[k], 0));
	if (y.num <= 1)
	{
		runKernelCUDA<T, I> <<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>>(y.start, args...);
	}
	else
	{
		runKernelCUDAMulti<T, I> <<<x.nBlocks, x.nThreads, 0, mInternals->CudaStreams[x.stream]>>> (y.start, y.num, args...);
	}
	if (z.ev) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) z.ev, mInternals->CudaStreams[x.stream]));
	return 0;
}

AliGPUReconstructionCUDABackend::AliGPUReconstructionCUDABackend(const AliGPUSettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mInternals = new AliGPUReconstructionCUDAInternals;
	mProcessingSettings.deviceType = DeviceType::CUDA;
}

AliGPUReconstructionCUDABackend::~AliGPUReconstructionCUDABackend()
{
	mChains.clear(); //Make sure we destroy the ITS tracker before we exit CUDA
	GPUFailedMsgI(cudaDeviceReset());
	delete mInternals;
}

AliGPUReconstruction* AliGPUReconstruction_Create_CUDA(const AliGPUSettingsProcessing& cfg)
{
	return new AliGPUReconstructionCUDA(cfg);
}

void AliGPUReconstructionCUDABackend::GetITSTraits(std::unique_ptr<o2::ITS::TrackerTraits>& trackerTraits, std::unique_ptr<o2::ITS::VertexerTraits>& vertexerTraits)
{
	trackerTraits.reset(new o2::ITS::TrackerTraitsNV);
	vertexerTraits.reset(new o2::ITS::VertexerTraits);
}

int AliGPUReconstructionCUDABackend::InitDevice_Runtime()
{
	//Find best CUDA device, initialize and allocate memory

	cudaDeviceProp cudaDeviceProp;

	int count, bestDevice = -1;
	double bestDeviceSpeed = -1, deviceSpeed;
	if (GPUFailedMsgI(cudaGetDeviceCount(&count)))
	{
		GPUError("Error getting CUDA Device Count");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("Available CUDA devices:");
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
		if (mDeviceProcessingSettings.debugLevel >= 2) GPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem Avail %lld / %lld)%s %s", deviceOK ? " " : "[", i, cudaDeviceProp.name, cudaDeviceProp.major, cudaDeviceProp.minor, (long long int) free, (long long int) cudaDeviceProp.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
		if (!deviceOK) continue;
		if (deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
		else
		{
			if (mDeviceProcessingSettings.debugLevel >= 0) GPUInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
		}
	}
	if (bestDevice == -1)
	{
		GPUWarning("No %sCUDA Device available, aborting CUDA Initialisation", count ? "appropriate " : "");
		GPUImportant("Requiring Revision %d.%d, Mem: %lld", reqVerMaj, reqVerMin, (long long int) mDeviceMemorySize);
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
			GPUWarning("Requested device ID %d non existend, falling back to default device id %d", mDeviceProcessingSettings.deviceNum, bestDevice);
		}
	}
	fDeviceId = bestDevice;

	GPUFailedMsgI(cudaGetDeviceProperties(&cudaDeviceProp ,fDeviceId));

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		GPUInfo("Using CUDA Device %s with Properties:", cudaDeviceProp.name);
		GPUInfo("totalGlobalMem = %lld", (unsigned long long int) cudaDeviceProp.totalGlobalMem);
		GPUInfo("sharedMemPerBlock = %lld", (unsigned long long int) cudaDeviceProp.sharedMemPerBlock);
		GPUInfo("regsPerBlock = %d", cudaDeviceProp.regsPerBlock);
		GPUInfo("warpSize = %d", cudaDeviceProp.warpSize);
		GPUInfo("memPitch = %lld", (unsigned long long int) cudaDeviceProp.memPitch);
		GPUInfo("maxThreadsPerBlock = %d", cudaDeviceProp.maxThreadsPerBlock);
		GPUInfo("maxThreadsDim = %d %d %d", cudaDeviceProp.maxThreadsDim[0], cudaDeviceProp.maxThreadsDim[1], cudaDeviceProp.maxThreadsDim[2]);
		GPUInfo("maxGridSize = %d %d %d", cudaDeviceProp.maxGridSize[0], cudaDeviceProp.maxGridSize[1], cudaDeviceProp.maxGridSize[2]);
		GPUInfo("totalConstMem = %lld", (unsigned long long int) cudaDeviceProp.totalConstMem);
		GPUInfo("major = %d", cudaDeviceProp.major);
		GPUInfo("minor = %d", cudaDeviceProp.minor);
		GPUInfo("clockRate = %d", cudaDeviceProp.clockRate);
		GPUInfo("memoryClockRate = %d", cudaDeviceProp.memoryClockRate);
		GPUInfo("multiProcessorCount = %d", cudaDeviceProp.multiProcessorCount);
		GPUInfo("textureAlignment = %lld", (unsigned long long int) cudaDeviceProp.textureAlignment);
	}
	mCoreCount = cudaDeviceProp.multiProcessorCount;

	if (cudaDeviceProp.major < 1 || (cudaDeviceProp.major == 1 && cudaDeviceProp.minor < 2))
	{
		GPUError( "Unsupported CUDA Device" );
		return(1);
	}

#ifdef GPUCA_USE_TEXTURES
	if (GPUCA_SLICE_DATA_MEMORY * NSLICES > (size_t) cudaDeviceProp.maxTexture1DLinear)
	{
		GPUError("Invalid maximum texture size of device: %lld < %lld\n", (long long int) cudaDeviceProp.maxTexture1DLinear, (long long int) (GPUCA_SLICE_DATA_MEMORY * NSLICES));
		return(1);
	}
#endif

	mNStreams = std::max(mDeviceProcessingSettings.nStreams, 3);

	if (cuCtxCreate(&mInternals->CudaContext, CU_CTX_SCHED_AUTO, fDeviceId) != CUDA_SUCCESS)
	{
		GPUError("Could not set CUDA Device!");
		return(1);
	}

	if (mDeviceMemorySize > cudaDeviceProp.totalGlobalMem || GPUFailedMsgI(cudaMalloc(&mDeviceMemoryBase, mDeviceMemorySize)))
	{
		GPUError("CUDA Memory Allocation Error");
		GPUFailedMsgI(cudaDeviceReset());
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) GPUInfo("GPU Memory used: %lld", (long long int) mDeviceMemorySize);
	if (GPUFailedMsgI(cudaMallocHost(&mHostMemoryBase, mHostMemorySize)))
	{
		GPUError("Error allocating Page Locked Host Memory");
		GPUFailedMsgI(cudaDeviceReset());
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) GPUInfo("Host Memory used: %lld", (long long int) mHostMemorySize);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(mHostMemoryBase, 0xDD, mHostMemorySize);
		if (GPUFailedMsgI(cudaMemset(mDeviceMemoryBase, 0xDD, mDeviceMemorySize)))
		{
			GPUError("Error during CUDA memset");
			GPUFailedMsgI(cudaDeviceReset());
			return(1);
		}
	}

	for (int i = 0;i < mNStreams;i++)
	{
		if (GPUFailedMsgI(cudaStreamCreate(&mInternals->CudaStreams[i])))
		{
			GPUError("Error creating CUDA Stream");
			GPUFailedMsgI(cudaDeviceReset());
			return(1);
		}
	}
	
	void* devPtrConstantMem;
	if (GPUFailedMsgI(cudaGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer)))
	{
		GPUError("Error getting ptr to constant memory");
		GPUFailedMsgI(cudaDeviceReset());
		return 1;
	}
	mDeviceConstantMem = (AliGPUConstantMem*) devPtrConstantMem;
	
	for (unsigned int i = 0;i < mEvents.size();i++)
	{
		cudaEvent_t *events = (cudaEvent_t*) mEvents[i].first;
		for (unsigned int j = 0;j < mEvents[i].second;j++)
		{
			if (GPUFailedMsgI(cudaEventCreate(&events[j])))
			{
				GPUError("Error creating event");
				GPUFailedMsgI(cudaDeviceReset());
				return 1;
			}
		}
	}

	ReleaseThreadContext();
	GPUInfo("CUDA Initialisation successfull (Device %d: %s, Thread %d, %lld/%lld bytes used)", fDeviceId, cudaDeviceProp.name, mThreadId, (long long int) mHostMemorySize, (long long int) mDeviceMemorySize);

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
	
	for (unsigned int i = 0;i < mEvents.size();i++)
	{
		cudaEvent_t *events = (cudaEvent_t*) mEvents[i].first;
		for (unsigned int j = 0;j < mEvents[i].second;j++)
		{
			GPUFailedMsgI(cudaEventDestroy(events[j]));
		}
	}

	if (GPUFailedMsgI(cudaDeviceReset()))
	{
		GPUError("Could not uninitialize GPU");
		return(1);
	}

	cuCtxDestroy(mInternals->CudaContext);

	GPUInfo("CUDA Uninitialized");
	return(0);
}

void AliGPUReconstructionCUDABackend::TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst)
{
	if (!(res->Type() & AliGPUMemoryResource::MEMORY_GPU))
	{
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Skipped transfer of non-GPU memory resource: %s\n", res->Name());
		return;
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf(toGPU ? "Copying to GPU: %s\n" : "Copying to Host: %s\n", res->Name());
	if (stream == -1)
	{
		if (stream == -1) SynchronizeGPU();
		GPUFailedMsg(cudaMemcpy(dst, src, res->Size(), toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
	}
	else
	{
		if (evList == nullptr) nEvents = 0;
		for (int k = 0;k < nEvents;k++) GPUFailedMsg(cudaStreamWaitEvent(mInternals->CudaStreams[stream], ((cudaEvent_t*) evList)[k], 0));
		GPUFailedMsg(cudaMemcpyAsync(dst, src, res->Size(), toGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, mInternals->CudaStreams[stream]));
	}
	if (ev) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream == -1 ? 0 : stream]));
}

void AliGPUReconstructionCUDABackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
	if (stream == -1) GPUFailedMsg(cudaMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice));
	else GPUFailedMsg(cudaMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, cudaMemcpyHostToDevice, mInternals->CudaStreams[stream]));
	if (ev && stream != -1) GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream]));
}

void AliGPUReconstructionCUDABackend::ReleaseEvent(deviceEvent* ev) {}

void AliGPUReconstructionCUDABackend::RecordMarker(deviceEvent* ev, int stream)
{
	GPUFailedMsg(cudaEventRecord(*(cudaEvent_t*) ev, mInternals->CudaStreams[stream]));
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

bool AliGPUReconstructionCUDABackend::IsEventDone(deviceEvent* evList, int nEvents)
{
	for (int i = 0;i < nEvents;i++)
	{
		cudaError_t retVal = cudaEventSynchronize(((cudaEvent_t*) evList)[i]);
		if (retVal == cudaErrorNotReady) return false;
		GPUFailedMsg(retVal);
	}
	return(true);
}

int AliGPUReconstructionCUDABackend::GPUDebug(const char* state, int stream)
{
	//Wait for CUDA-Kernel to finish and check for CUDA errors afterwards, in case of debugmode
	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	cudaError cuErr;
	cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		GPUError("Cuda Error %s while running kernel (%s) (Stream %d)", cudaGetErrorString(cuErr), state, stream);
		return(1);
	}
	if (GPUFailedMsgI(cudaDeviceSynchronize()))
	{
		GPUError("CUDA Error while synchronizing (%s) (Stream %d)", state, stream);
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("GPU Sync Done");
	return(0);
}

int AliGPUReconstructionCUDABackend::PrepareTextures()
{
#ifdef GPUCA_USE_TEXTURES
	cudaChannelFormatDesc channelDescu2 = cudaCreateChannelDesc<cahit2>();
	size_t offset;
	GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu2, mWorkersShadow->tpcTrackers[0].Data().Memory(), &channelDescu2, NSLICES * GPUCA_SLICE_DATA_MEMORY));
	cudaChannelFormatDesc channelDescu = cudaCreateChannelDesc<calink>();
	GPUFailedMsg(cudaBindTexture(&offset, &gAliTexRefu, mWorkersShadow->tpcTrackers[0].Data().Memory(), &channelDescu, NSLICES * GPUCA_SLICE_DATA_MEMORY));
#endif
	return(0);
}

void AliGPUReconstructionCUDABackend::SetThreadCounts()
{
	fThreadCount = GPUCA_THREAD_COUNT;
	fBlockCount = mCoreCount;
	fConstructorBlockCount = fBlockCount * (mDeviceProcessingSettings.trackletConstructorInPipeline ? 1 : GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER);
	fSelectorBlockCount = fBlockCount * GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER;
	fConstructorThreadCount = GPUCA_THREAD_COUNT_CONSTRUCTOR;
	fSelectorThreadCount = GPUCA_THREAD_COUNT_SELECTOR;
	fFinderThreadCount = GPUCA_THREAD_COUNT_FINDER;
	fTRDThreadCount = GPUCA_THREAD_COUNT_TRD;
}
