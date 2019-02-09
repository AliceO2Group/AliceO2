#include "hip/hip_runtime.h"
#define GPUCA_GPUTYPE_HIP

#include "AliGPUReconstructionHIP.h"
#include "AliGPUReconstructionHIPInternals.h"
#include "AliGPUReconstructionCommon.h"

__constant__ uint4 gGPUConstantMemBuffer[(sizeof(AliGPUCAConstantMem) + sizeof(uint4) - 1) / sizeof(uint4)];
__constant__ char& gGPUConstantMemBufferChar = (char&) gGPUConstantMemBuffer;
__constant__ AliGPUCAConstantMem& gGPUConstantMem = (AliGPUCAConstantMem&) gGPUConstantMemBufferChar;

namespace o2 { namespace ITS { class TrackerTraitsHIP : public TrackerTraits {}; }}

#define DEVICE_KERNELS_PRE
#include "AliGPUDeviceKernels.h"

template <class T, int I, typename... Args> GPUg() void runKernelHIP(int iSlice, Args... args)
{
	GPUshared() typename T::AliGPUTPCSharedMemory smem;
	T::template Thread<I>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, T::Worker(gGPUConstantMem)[iSlice], args...);
}

template <class T, int I, typename... Args> GPUg() void runKernelHIPMulti(int firstSlice, int nSliceCount, Args... args)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	GPUshared() typename T::AliGPUTPCSharedMemory smem;
	T::template Thread<I>(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), smem, T::Worker(gGPUConstantMem)[firstSlice + iSlice], args...);
}

template <class T, int I, typename... Args> int AliGPUReconstructionHIPBackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, Args... args)
{
	if (x.device == krnlDeviceType::CPU) return AliGPUReconstructionCPU::runKernelBackend<T, I> (x, y, z, args...);
	if (z.evList) for (int k = 0;k < z.nEvents;k++) GPUFailedMsg(hipStreamWaitEvent(mInternals->HIPStreams[x.stream], ((hipEvent_t*) z.evList)[k], 0));
	if (y.num <= 1)
	{
		hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIP<T, I, Args...>), dim3(x.nBlocks), dim3(x.nThreads), 0, mInternals->HIPStreams[x.stream], y.start, args...);
	}
	else
	{
		hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIPMulti<T, I, Args...>), dim3(x.nBlocks), dim3(x.nThreads), 0, mInternals->HIPStreams[x.stream], y.start, y.num, args...);
	}
	if (z.ev) GPUFailedMsg(hipEventRecord(*(hipEvent_t*) z.ev, mInternals->HIPStreams[x.stream]));
	return 0;
}

AliGPUReconstructionHIPBackend::AliGPUReconstructionHIPBackend(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mInternals = new AliGPUReconstructionHIPInternals;
	mProcessingSettings.deviceType = HIP;
	mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsHIP);
}

AliGPUReconstructionHIPBackend::~AliGPUReconstructionHIPBackend()
{
	mITSTrackerTraits.reset(nullptr); //Make sure we destroy the ITS tracker before we exit HIP
	GPUFailedMsgI(hipDeviceReset());
	delete mInternals;
}

AliGPUReconstruction* AliGPUReconstruction_Create_HIP(const AliGPUCASettingsProcessing& cfg)
{
	return new AliGPUReconstructionHIP(cfg);
}

int AliGPUReconstructionHIPBackend::InitDevice_Runtime()
{
	//Find best HIP device, initialize and allocate memory

	hipDeviceProp_t hipDeviceProp_t;

	int count, bestDevice = -1;
	double bestDeviceSpeed = -1, deviceSpeed;
	if (GPUFailedMsgI(hipGetDeviceCount(&count)))
	{
		CAGPUError("Error getting HIP Device Count");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Available HIP devices:");
	const int reqVerMaj = 2;
	const int reqVerMin = 0;
	for (int i = 0;i < count;i++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Examining device %d\n", i);
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Obtained current memory usage for device %d\n", i);
		if (GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp_t, i))) continue;
		if (mDeviceProcessingSettings.debugLevel >= 4) printf("Obtained device properties for device %d\n", i);
		int deviceOK = true;
		const char* deviceFailure = "";
		if (hipDeviceProp_t.major >= 9) {deviceOK = false; deviceFailure = "Invalid Revision";}
		else if (hipDeviceProp_t.major < reqVerMaj || (hipDeviceProp_t.major == reqVerMaj && hipDeviceProp_t.minor < reqVerMin)) {deviceOK = false; deviceFailure = "Too low device revision";}

		deviceSpeed = (double) hipDeviceProp_t.multiProcessorCount * (double) hipDeviceProp_t.clockRate * (double) hipDeviceProp_t.warpSize * (double) hipDeviceProp_t.major * (double) hipDeviceProp_t.major;
		if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUImportant("Device %s%2d: %s (Rev: %d.%d - Mem %lld)%s %s", deviceOK ? " " : "[", i, hipDeviceProp_t.name, hipDeviceProp_t.major, hipDeviceProp_t.minor, (long long int) hipDeviceProp_t.totalGlobalMem, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
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
		CAGPUWarning("No %sHIP Device available, aborting HIP Initialisation", count ? "appropriate " : "");
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

	GPUFailedMsgI(hipGetDeviceProperties(&hipDeviceProp_t ,fDeviceId));

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		CAGPUInfo("Using HIP Device %s with Properties:", hipDeviceProp_t.name);
		CAGPUInfo("totalGlobalMem = %lld", (unsigned long long int) hipDeviceProp_t.totalGlobalMem);
		CAGPUInfo("sharedMemPerBlock = %lld", (unsigned long long int) hipDeviceProp_t.sharedMemPerBlock);
		CAGPUInfo("regsPerBlock = %d", hipDeviceProp_t.regsPerBlock);
		CAGPUInfo("warpSize = %d", hipDeviceProp_t.warpSize);
		CAGPUInfo("maxThreadsPerBlock = %d", hipDeviceProp_t.maxThreadsPerBlock);
		CAGPUInfo("maxThreadsDim = %d %d %d", hipDeviceProp_t.maxThreadsDim[0], hipDeviceProp_t.maxThreadsDim[1], hipDeviceProp_t.maxThreadsDim[2]);
		CAGPUInfo("maxGridSize = %d %d %d", hipDeviceProp_t.maxGridSize[0], hipDeviceProp_t.maxGridSize[1], hipDeviceProp_t.maxGridSize[2]);
		CAGPUInfo("totalConstMem = %lld", (unsigned long long int) hipDeviceProp_t.totalConstMem);
		CAGPUInfo("major = %d", hipDeviceProp_t.major);
		CAGPUInfo("minor = %d", hipDeviceProp_t.minor);
		CAGPUInfo("clockRate = %d", hipDeviceProp_t.clockRate);
		CAGPUInfo("memoryClockRate = %d", hipDeviceProp_t.memoryClockRate);
		CAGPUInfo("multiProcessorCount = %d", hipDeviceProp_t.multiProcessorCount);
	}

	fThreadCount = GPUCA_GPU_THREAD_COUNT;
	fBlockCount = hipDeviceProp_t.multiProcessorCount;
	fConstructorBlockCount = hipDeviceProp_t.multiProcessorCount * (mDeviceProcessingSettings.trackletConstructorInPipeline ? 1 : GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER);
	fSelectorBlockCount = hipDeviceProp_t.multiProcessorCount * GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;
	fConstructorThreadCount = GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;
	fSelectorThreadCount = GPUCA_GPU_THREAD_COUNT_SELECTOR;
	fFinderThreadCount = GPUCA_GPU_THREAD_COUNT_FINDER;
	fTRDThreadCount = GPUCA_GPU_THREAD_COUNT_TRD;

	if (hipDeviceProp_t.major < 1 || (hipDeviceProp_t.major == 1 && hipDeviceProp_t.minor < 2))
	{
		CAGPUError( "Unsupported HIP Device" );
		return(1);
	}

	mNStreams = std::max(mDeviceProcessingSettings.nStreams, 3);

	if (mDeviceMemorySize > hipDeviceProp_t.totalGlobalMem || GPUFailedMsgI(hipMalloc(&mDeviceMemoryBase, mDeviceMemorySize)))
	{
		CAGPUError("HIP Memory Allocation Error");
		GPUFailedMsgI(hipDeviceReset());
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("GPU Memory used: %lld", (long long int) mDeviceMemorySize);
	if (GPUFailedMsgI(hipHostMalloc(&mHostMemoryBase, mHostMemorySize)))
	{
		CAGPUError("Error allocating Page Locked Host Memory");
		GPUFailedMsgI(hipDeviceReset());
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("Host Memory used: %lld", (long long int) mHostMemorySize);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(mHostMemoryBase, 0, mHostMemorySize);
		if (GPUFailedMsgI(hipMemset(mDeviceMemoryBase, 143, mDeviceMemorySize)))
		{
			CAGPUError("Error during HIP memset");
			GPUFailedMsgI(hipDeviceReset());
			return(1);
		}
	}

	for (int i = 0;i < mNStreams;i++)
	{
		if (GPUFailedMsgI(hipStreamCreate(&mInternals->HIPStreams[i])))
		{
			CAGPUError("Error creating HIP Stream");
			GPUFailedMsgI(hipDeviceReset());
			return(1);
		}
	}
	
	void* devPtrConstantMem;
	if (GPUFailedMsgI(hipGetSymbolAddress(&devPtrConstantMem, gGPUConstantMemBuffer)))
	{
		CAGPUError("Error getting ptr to constant memory");
		GPUFailedMsgI(hipDeviceReset());
		return 1;
	}
	mDeviceConstantMem = (AliGPUCAConstantMem*) devPtrConstantMem;
	
	hipEvent_t *events = (hipEvent_t*) &mEvents;
	for (unsigned int i = 0;i < sizeof(mEvents) / sizeof(hipEvent_t);i++)
	{
		if (GPUFailedMsgI(hipEventCreate(&events[i])))
		{
			CAGPUError("Error creating event");
			GPUFailedMsgI(hipDeviceReset());
			return 1;
		}
	}

	ReleaseThreadContext();
	CAGPUInfo("HIP Initialisation successfull (Device %d: %s, Thread %d, %lld/%lld bytes used)", fDeviceId, hipDeviceProp_t.name, fThreadId, (long long int) mHostMemorySize, (long long int) mDeviceMemorySize);

	return(0);
}

int AliGPUReconstructionHIPBackend::ExitDevice_Runtime()
{
	//Uninitialize HIP
	ActivateThreadContext();

	SynchronizeGPU();

	GPUFailedMsgI(hipFree(mDeviceMemoryBase));
	mDeviceMemoryBase = nullptr;

	for (int i = 0;i < mNStreams;i++)
	{
		GPUFailedMsgI(hipStreamDestroy(mInternals->HIPStreams[i]));
	}

	GPUFailedMsgI(hipHostFree(mHostMemoryBase));
	mHostMemoryBase = nullptr;
	
	hipEvent_t *events = (hipEvent_t*) &mEvents;
	for (unsigned int i = 0;i < sizeof(mEvents) / sizeof(hipEvent_t);i++)
	{
		GPUFailedMsgI(hipEventDestroy(events[i]));
	}

	if (GPUFailedMsgI(hipDeviceReset()))
	{
		CAGPUError("Could not uninitialize GPU");
		return(1);
	}

	CAGPUInfo("HIP Uninitialized");
	return(0);
}

void AliGPUReconstructionHIPBackend::TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf(toGPU ? "Copying to GPU: %s\n" : "Copying to Host: %s\n", res->Name());
	if (stream == -1)
	{
		if (stream == -1) SynchronizeGPU();
		GPUFailedMsg(hipMemcpy(dst, src, res->Size(), toGPU ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost));
	}
	else
	{
		if (evList == nullptr) nEvents = 0;
		for (int k = 0;k < nEvents;k++) GPUFailedMsg(hipStreamWaitEvent(mInternals->HIPStreams[stream], ((hipEvent_t*) evList)[k], 0));
		GPUFailedMsg(hipMemcpyAsync(dst, src, res->Size(), toGPU ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, mInternals->HIPStreams[stream]));
	}
	if (ev) GPUFailedMsg(hipEventRecord(*(hipEvent_t*) ev, mInternals->HIPStreams[stream == -1 ? 0 : stream]));
}

void AliGPUReconstructionHIPBackend::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
	if (stream == -1) GPUFailedMsg(hipMemcpyToSymbol(gGPUConstantMemBuffer, src, size, offset, hipMemcpyHostToDevice));
	else GPUFailedMsg(hipMemcpyToSymbolAsync(gGPUConstantMemBuffer, src, size, offset, hipMemcpyHostToDevice, mInternals->HIPStreams[stream]));
	if (ev && stream != -1) GPUFailedMsg(hipEventRecord(*(hipEvent_t*) ev, mInternals->HIPStreams[stream]));
}

void AliGPUReconstructionHIPBackend::ReleaseEvent(deviceEvent* ev) {}

void AliGPUReconstructionHIPBackend::RecordMarker(deviceEvent* ev, int stream)
{
	GPUFailedMsg(hipEventRecord(*(hipEvent_t*) ev, mInternals->HIPStreams[stream]));
}

void AliGPUReconstructionHIPBackend::ActivateThreadContext(){}
void AliGPUReconstructionHIPBackend::ReleaseThreadContext(){}

void AliGPUReconstructionHIPBackend::SynchronizeGPU()
{
	GPUFailedMsg(hipDeviceSynchronize());
}

void AliGPUReconstructionHIPBackend::SynchronizeStream(int stream)
{
	GPUFailedMsg(hipStreamSynchronize(mInternals->HIPStreams[stream]));
}

void AliGPUReconstructionHIPBackend::SynchronizeEvents(deviceEvent* evList, int nEvents)
{
	for (int i = 0;i < nEvents;i++)
	{
		GPUFailedMsg(hipEventSynchronize(((hipEvent_t*) evList)[i]));
	}
}

int AliGPUReconstructionHIPBackend::IsEventDone(deviceEvent* evList, int nEvents)
{
	for (int i = 0;i < nEvents;i++)
	{
		hipError_t retVal = hipEventSynchronize(((hipEvent_t*) evList)[i]);
		if (retVal == hipErrorNotReady) return 0;
		GPUFailedMsg(retVal);
	}
	return(1);
}

int AliGPUReconstructionHIPBackend::GPUDebug(const char* state, int stream, int slice)
{
	//Wait for HIP-Kernel to finish and check for HIP errors afterwards, in case of debugmode
	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	hipError_t cuErr;
	cuErr = hipGetLastError();
	if (cuErr != hipSuccess)
	{
		CAGPUError("HIP Error %s while running kernel (%s) (Stream %d; Slice %d/%d)", hipGetErrorString(cuErr), state, stream, slice, NSLICES);
		return(1);
	}
	if (GPUFailedMsgI(hipDeviceSynchronize()))
	{
		CAGPUError("HIP Error while synchronizing (%s) (Stream %d; Slice %d/%d)", state, stream, slice, NSLICES);
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Sync Done");
	return(0);
}
