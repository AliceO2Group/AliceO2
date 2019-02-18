#define GPUCA_ALIGPURECONSTRUCTIONCPU_IMPLEMENTATION
#include "AliGPUReconstructionCPU.h"
#include "AliGPUReconstructionIncludes.h"
#include "AliGPUChain.h"

#include "AliGPUTPCClusterData.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCSliceOutTrack.h"
#include "AliGPUTPCSliceOutCluster.h"
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUTPCGMMergedTrackHit.h"
#include "AliGPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliGPUTPCMCInfo.h"
#include "AliGPUTRDTrack.h"
#include "AliGPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "ClusterNativeAccessExt.h"
#include "AliGPUTRDTrackletLabels.h"
#include "AliGPUMemoryResource.h"
#include "AliGPUConstantMem.h"

#include "AliGPUQA.h"
#include "AliGPUDisplay.h"

#define GPUCA_LOGGING_PRINTF
#include "AliGPULogging.h"

#ifndef _WIN32
#include <unistd.h>
#endif

AliGPUReconstruction* AliGPUReconstruction::AliGPUReconstruction_Create_CPU(const AliGPUSettingsProcessing& cfg)
{
	return new AliGPUReconstructionCPU(cfg);
}

template <class T, int I, typename... Args> int AliGPUReconstructionCPUBackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
	if (x.device == krnlDeviceType::Device) throw std::runtime_error("Cannot run device kernel on host");
	unsigned int num = y.num == 0 || y.num == -1 ? 1 : y.num;
	for (unsigned int k = 0;k < num;k++)
	{
		for (unsigned int iB = 0; iB < x.nBlocks; iB++)
		{
			typename T::AliGPUTPCSharedMemory smem;
			T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Worker(*mHostConstantMem)[y.start + k], args...);
		}
	}
	return 0;
}

void AliGPUReconstructionCPU::TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) {}
void AliGPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) {}
int AliGPUReconstructionCPU::GPUDebug(const char* state, int stream) {return 0;}
void AliGPUReconstructionCPU::TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU)
{
	int inc = toGPU ? AliGPUMemoryResource::MEMORY_INPUT : AliGPUMemoryResource::MEMORY_OUTPUT;
	int exc = toGPU ? AliGPUMemoryResource::MEMORY_OUTPUT : AliGPUMemoryResource::MEMORY_INPUT;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		AliGPUMemoryResource& res = mMemoryResources[i];
		if (res.mPtr == nullptr) continue;
		if (proc && res.mProcessor != proc) continue;
		if (!(res.mType & AliGPUMemoryResource::MEMORY_GPU) || (res.mType & AliGPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) continue;
		if (!mDeviceProcessingSettings.keepAllMemory && !(all && !(res.mType & exc)) && !(res.mType & inc)) continue;
		if (toGPU) TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
		else TransferMemoryResourceToHost(&mMemoryResources[i], stream);
	}
}

int AliGPUReconstructionCPU::GetThread()
{
	//Get Thread ID
#ifdef _WIN32
	return((int) (size_t) GetCurrentThread());
#else
	return((int) syscall (SYS_gettid));
#endif
}

int AliGPUReconstructionCPU::InitDevice()
{
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
	{
		mHostMemoryPermanent = mHostMemoryBase = operator new(GPUCA_HOST_MEMORY_SIZE);
		mHostMemorySize = GPUCA_HOST_MEMORY_SIZE;
		ClearAllocatedMemory();
	}
	SetThreadCounts();
	mThreadId = GetThread();
	return 0;
}

int AliGPUReconstructionCPU::ExitDevice()
{
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
	{
		operator delete(mHostMemoryBase);
		mHostMemoryPool = mHostMemoryBase = mHostMemoryPermanent = nullptr;
		mHostMemorySize = 0;
	}
	return 0;
}

void AliGPUReconstructionCPU::SetThreadCounts()
{
	fThreadCount = fBlockCount = fConstructorBlockCount = fSelectorBlockCount = fConstructorThreadCount = fSelectorThreadCount = fFinderThreadCount = fTRDThreadCount = 1;
}

void AliGPUReconstructionCPU::SetThreadCounts(RecoStep step)
{
	if (IsGPU() && mRecoSteps != mRecoStepsGPU)
	{
		if (!(mRecoStepsGPU & step)) AliGPUReconstructionCPU::SetThreadCounts();
		else SetThreadCounts();
	}
}

int AliGPUReconstructionCPU::RunStandalone()
{
	mStatNEvents++;

	if (mThreadId != GetThread())
	{
		if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("Thread changed, migrating context, Previous Thread: %d, New Thread: %d", mThreadId, GetThread());
		mThreadId = GetThread();
	}
	
	for (unsigned int i = 0;i < mChains.size();i++)
	{
		if (mChains[i]->RunStandalone()) return 1;
	}
	
	if (GetDeviceProcessingSettings().debugLevel >= 1)
	{
		if (GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
		{
			printf("Memory Allocation: Host %'lld / %'lld, Device %'lld / %'lld, %d chunks\n",
			(long long int) ((char*) mHostMemoryPool - (char*) mHostMemoryBase), (long long int) mHostMemorySize, (long long int) ((char*) mDeviceMemoryPool - (char*) mDeviceMemoryBase), (long long int) mDeviceMemorySize, (int) mMemoryResources.size());
		}
	}

	return 0;
}

void AliGPUReconstructionCPU::ResetDeviceProcessorTypes()
{
	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		if (mProcessors[i].proc->mDeviceProcessor) mProcessors[i].proc->mDeviceProcessor->InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
	}
}
