#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUReconstructionIncludes.h"

#include "AliGPUTPCTracker.h"
#include "AliGPUTPCSliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(_WIN32)
#include "../utils/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#endif
#include <string.h>

MEM_CLASS_PRE() class AliGPUTPCRow;

#define SemLockName "AliceHLTTPCGPUTrackerInitLockSem"

AliGPUReconstructionDeviceBase::AliGPUReconstructionDeviceBase(const AliGPUSettingsProcessing& cfg) : AliGPUReconstructionCPU(cfg)
{
}

AliGPUReconstructionDeviceBase::~AliGPUReconstructionDeviceBase()
{
	// make d'tor such that vtable is created for this class
	// needed for build with AliRoot
}

void* AliGPUReconstructionDeviceBase::helperWrapper(void* arg)
{
	AliGPUReconstructionDeviceBase::helperParam* par = (AliGPUReconstructionDeviceBase::helperParam*) arg;
	AliGPUReconstructionDeviceBase* cls = par->fCls;

	AliGPUTPCTracker* tmpTracker = new AliGPUTPCTracker;

	if (cls->mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("\tHelper thread %d starting", par->fNum);

	//cpu_set_t mask;
	//CPU_ZERO(&mask);
	//CPU_SET(par->fNum * 2 + 2, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);

	while(pthread_mutex_lock(&((pthread_mutex_t*) par->fMutex)[0]) == 0 && par->fTerminate == false)
	{
		int mustRunSlice19 = 0;
		for (unsigned int i = par->fNum + 1;i < NSLICES;i += cls->mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			//if (cls->mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Running, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
			if (par->fPhase)
			{
				if (cls->param().rec.GlobalTracking)
				{
					int realSlice = i + 1;
					if (realSlice % (NSLICES / 2) < 1) realSlice -= NSLICES / 2;

					if (realSlice % (NSLICES / 2) != 1)
					{
						cls->GlobalTracking(realSlice, par->fNum + 1);
					}

					if (realSlice == 19)
					{
						mustRunSlice19 = 1;
					}
					else
					{
						while (cls->fSliceLeftGlobalReady[realSlice] == 0 || cls->fSliceRightGlobalReady[realSlice] == 0)
						{
							if (par->fReset) goto ResetHelperThread;
						}
						cls->WriteOutput(realSlice, par->fNum + 1);
					}
				}
				else
				{
					while (cls->fSliceOutputReady < (int) i)
					{
						if (par->fReset) goto ResetHelperThread;
					}
					cls->WriteOutput(i, par->fNum + 1);
				}
			}
			else
			{
				if (cls->ReadEvent(i, par->fNum + 1)) par->fError = 1;
				par->fDone = i + 1;
			}
			//if (cls->mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Finished, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
		}
		if (mustRunSlice19)
		{
			while (cls->fSliceLeftGlobalReady[19] == 0 || cls->fSliceRightGlobalReady[19] == 0)
			{
				if (par->fReset) goto ResetHelperThread;
			}
			cls->WriteOutput(19, par->fNum + 1);
		}
ResetHelperThread:
		cls->ResetThisHelperThread(par);
	}
	if (cls->mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("\tHelper thread %d terminating", par->fNum);
	delete tmpTracker;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
	pthread_exit(nullptr);
	return(nullptr);
}

void AliGPUReconstructionDeviceBase::ResetThisHelperThread(AliGPUReconstructionDeviceBase::helperParam* par)
{
	if (par->fReset) GPUImportant("GPU Helper Thread %d reseting", par->fNum);
	par->fReset = false;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
}

void AliGPUReconstructionDeviceBase::ReleaseGlobalLock(void* sem)
{
	//Release the global named semaphore that locks GPU Initialization
#ifdef _WIN32
	HANDLE* h = (HANDLE*) sem;
	ReleaseSemaphore(*h, 1, nullptr);
	CloseHandle(*h);
	delete h;
#else
	sem_t* pSem = (sem_t*) sem;
	sem_post(pSem);
	sem_unlink(SemLockName);
#endif
}

void AliGPUReconstructionDeviceBase::ResetHelperThreads(int helpers)
{
	GPUImportant("Error occurred, GPU tracker helper threads will be reset (Number of threads %d (%d))", mDeviceProcessingSettings.nDeviceHelperThreads, fNSlaveThreads);
	SynchronizeGPU();
	ReleaseThreadContext();
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fReset = true;
		if (helpers || i >= mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}
	GPUImportant("GPU Tracker helper threads have ben reset");
}

int AliGPUReconstructionDeviceBase::StartHelperThreads()
{
	int nThreads = mDeviceProcessingSettings.nDeviceHelperThreads;
	if (nThreads)
	{
		fHelperParams = new helperParam[nThreads];
		if (fHelperParams == nullptr)
		{
			GPUError("Memory allocation error");
			ExitDevice();
			return(1);
		}
		for (int i = 0;i < nThreads;i++)
		{
			fHelperParams[i].fCls = this;
			fHelperParams[i].fTerminate = false;
			fHelperParams[i].fReset = false;
			fHelperParams[i].fNum = i;
			fHelperParams[i].fMutex = malloc(2 * sizeof(pthread_mutex_t));
			if (fHelperParams[i].fMutex == nullptr)
			{
				GPUError("Memory allocation error");
				ExitDevice();
				return(1);
			}
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_init(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j], nullptr))
				{
					GPUError("Error creating pthread mutex");
					ExitDevice();
					return(1);
				}

				pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]);
			}
			fHelperParams[i].fThreadId = (void*) malloc(sizeof(pthread_t));

			if (pthread_create((pthread_t*) fHelperParams[i].fThreadId, nullptr, helperWrapper, &fHelperParams[i]))
			{
				GPUError("Error starting slave thread");
				ExitDevice();
				return(1);
			}
		}
	}
	fNSlaveThreads = nThreads;
	return(0);
}

int AliGPUReconstructionDeviceBase::StopHelperThreads()
{
	if (fNSlaveThreads)
	{
		for (int i = 0;i < fNSlaveThreads;i++)
		{
			fHelperParams[i].fTerminate = true;
			if (pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]))
			{
				GPUError("Error unlocking mutex to terminate slave");
				return(1);
			}
			if (pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]))
			{
				GPUError("Error locking mutex");
				return(1);
			}
			if (pthread_join( *((pthread_t*) fHelperParams[i].fThreadId), nullptr))
			{
				GPUError("Error waiting for thread to terminate");
				return(1);
			}
			free(fHelperParams[i].fThreadId);
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]))
				{
					GPUError("Error unlocking mutex before destroying");
					return(1);
				}
				pthread_mutex_destroy(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]);
			}
			free(fHelperParams[i].fMutex);
		}
		delete[] fHelperParams;
	}
	fNSlaveThreads = 0;
	return(0);
}

void AliGPUReconstructionDeviceBase::WaitForHelperThreads()
{
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}
}

void AliGPUReconstructionDeviceBase::RunHelperThreads(int phase)
{
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fDone = 0;
		fHelperParams[i].fError = 0;
		fHelperParams[i].fPhase = phase;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}
}

int AliGPUReconstructionDeviceBase::InitDevice()
{
	//cpu_set_t mask;
	//CPU_ZERO(&mask);
	//CPU_SET(0, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);

	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		GPUError("Individual memory allocation strategy unsupported for device\n");
		return(1);
	}
	if (mDeviceProcessingSettings.nStreams > GPUCA_GPUCA_MAX_STREAMS)
	{
		GPUError("Too many straems requested %d > %d\n", mDeviceProcessingSettings.nStreams, GPUCA_GPUCA_MAX_STREAMS);
		return(1);
	}
	mThreadId = GetThread();

#ifdef _WIN32
	HANDLE* semLock = nullptr;
	if (mDeviceProcessingSettings.globalInitMutex)
	{
		semLock = new HANDLE;
		*semLock = CreateSemaphore(nullptr, 1, 1, SemLockName);
		if (*semLock == nullptr)
		{
			GPUError("Error creating GPUInit Semaphore");
			return(1);
		}
		WaitForSingleObject(*semLock, INFINITE);
	}
#else
	sem_t* semLock = nullptr;
	if (mDeviceProcessingSettings.globalInitMutex)
	{
		semLock = sem_open(SemLockName, O_CREAT, 0x01B6, 1);
		if (semLock == SEM_FAILED)
		{
			GPUError("Error creating GPUInit Semaphore");
			return(1);
		}
		timespec semtime;
		clock_gettime(CLOCK_REALTIME, &semtime);
		semtime.tv_sec += 10;
		while (sem_timedwait(semLock, &semtime) != 0)
		{
			GPUError("Global Lock for GPU initialisation was not released for 10 seconds, assuming another thread died");
			GPUWarning("Resetting the global lock");
			sem_post(semLock);
		}
	}
#endif

	mDeviceMemorySize = GPUCA_GPUCA_MEMORY_SIZE;
	mHostMemorySize = GPUCA_HOST_MEMORY_SIZE;
	int retVal = InitDevice_Runtime();
	if (retVal)
	{
		GPUImportant("GPU Tracker initialization failed");
		return(1);
	}
	
	if (mDeviceProcessingSettings.globalInitMutex) ReleaseGlobalLock(semLock);
	
	mDeviceMemoryPermanent = mDeviceMemoryBase;
	mHostMemoryPermanent = mHostMemoryBase;
	ClearAllocatedMemory();

	mProcShadow.InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_SLAVE);
	mProcDevice.InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE, &mProcShadow);
	mProcShadow.mMemoryResWorkers = RegisterMemoryAllocation(&mProcShadow, &AliGPUProcessorWorkers::SetPointersDeviceProcessor, AliGPUMemoryResource::MEMORY_PERMANENT, "Workers");
	mProcShadow.mMemoryResFlat = RegisterMemoryAllocation(&mProcShadow, &AliGPUProcessorWorkers::SetPointersFlatObjects, AliGPUMemoryResource::MEMORY_PERMANENT, "Workers");
	AllocateRegisteredMemory(mProcShadow.mMemoryResWorkers);
	memcpy((void*) &mWorkersShadow->trdTracker, (const void*) &workers()->trdTracker, sizeof(workers()->trdTracker));
	AllocateRegisteredMemory(mProcShadow.mMemoryResFlat);
	if (PrepareFlatObjects())
	{
		GPUError("Error preparing flat objects on GPU");
		ExitDevice_Runtime();
		return(1);
	}
	if (mRecoStepsGPU & RecoStep::TPCSliceTracking)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i], &workers()->tpcTrackers[i]);
			RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i].Data(), &workers()->tpcTrackers[i].Data());
		}
	}
	if (mRecoStepsGPU & RecoStep::TPCMerging) RegisterGPUDeviceProcessor(&mWorkersShadow->tpcMerger, &workers()->tpcMerger);
	if (mRecoStepsGPU & RecoStep::TRDTracking) RegisterGPUDeviceProcessor(&mWorkersShadow->trdTracker, &workers()->trdTracker);

	if (StartHelperThreads()) return(1);

	SetThreadCounts();

	GPUInfo("GPU Tracker initialization successfull"); //Verbosity reduced because GPU backend will print GPUImportant message!

	return(retVal);
}

void* AliGPUReconstructionDeviceBase::AliGPUProcessorWorkers::SetPointersDeviceProcessor(void* mem)
{
	//Don't run constructor / destructor here, this will be just local memcopy of Processors in GPU Memory
	computePointerWithAlignment(mem, mWorkersProc, 1);
	return mem;
}

void* AliGPUReconstructionDeviceBase::AliGPUProcessorWorkers::SetPointersFlatObjects(void* mem)
{
	if (mRec->GetTPCTransform())
	{
		computePointerWithAlignment(mem, fTpcTransform, 1);
		computePointerWithAlignment(mem, fTpcTransformBuffer, mRec->GetTPCTransform()->getFlatBufferSize());
	}
	if (mRec->GetTRDGeometry())
	{
		computePointerWithAlignment(mem, fTrdGeometry, 1);
	}
	return mem;
}

int AliGPUReconstructionDeviceBase::PrepareFlatObjects()
{
	if (mTPCFastTransform)
	{
		memcpy((void*) mProcShadow.fTpcTransform, (const void*) mTPCFastTransform.get(), sizeof(*mTPCFastTransform));
		memcpy((void*) mProcShadow.fTpcTransformBuffer, (const void*) mTPCFastTransform->getFlatBufferPtr(), mTPCFastTransform->getFlatBufferSize());
		mProcShadow.fTpcTransform->clearInternalBufferPtr();
		mProcShadow.fTpcTransform->setFutureBufferAddress(mProcDevice.fTpcTransformBuffer);
	}
#ifndef GPUCA_ALIROOT_LIB
	if (mTRDGeometry)
	{
		memcpy((void*) mProcShadow.fTrdGeometry, (const void*) mTRDGeometry.get(), sizeof(*mTRDGeometry));
		mProcShadow.fTrdGeometry->clearInternalBufferPtr();
	}
#endif
	TransferMemoryResourceLinkToGPU(mProcShadow.mMemoryResFlat);
	return 0;
}

int AliGPUReconstructionDeviceBase::ExitDevice()
{
	if (StopHelperThreads()) return(1);

	int retVal = ExitDevice_Runtime();
	mWorkersShadow = mWorkersDevice = nullptr;
	mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
	mHostMemorySize = mDeviceMemorySize = 0;
	
	return retVal;
}

int AliGPUReconstructionDeviceBase::DoTRDGPUTracking()
{
#ifdef GPUCA_BUILD_TRD
	ActivateThreadContext();
	SetupGPUProcessor(&workers()->trdTracker, false);
	mWorkersShadow->trdTracker.SetGeometry((AliGPUTRDGeometry*) mProcDevice.fTrdGeometry);

	WriteToConstantMemory((char*) &mDeviceConstantMem->trdTracker - (char*) mDeviceConstantMem, &mWorkersShadow->trdTracker, sizeof(mWorkersShadow->trdTracker), 0);
	TransferMemoryResourcesToGPU(&workers()->trdTracker);

	runKernel<AliGPUTRDTrackerGPU>({fBlockCount, fTRDThreadCount, 0}, nullptr, krnlRunRangeNone);
	SynchronizeGPU();

	TransferMemoryResourcesToHost(&workers()->trdTracker);
	SynchronizeGPU();

	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("GPU TRD tracker Finished");

	ReleaseThreadContext();
#endif
	return(0);
}

int AliGPUReconstructionDeviceBase::RefitMergedTracks(bool resetTimers)
{
	auto* Merger = &workers()->tpcMerger;
	if (!mRecoStepsGPU.isSet(RecoStep::TPCMerging)) return AliGPUReconstructionCPU::RefitMergedTracks(resetTimers);
	
	HighResTimer timer;
	static double times[3] = {};
	static int nCount = 0;
	if (resetTimers)
	{
		for (unsigned int k = 0;k < sizeof(times) / sizeof(times[0]);k++) times[k] = 0;
		nCount = 0;
	}
	ActivateThreadContext();

	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("Running GPU Merger (%d/%d)", Merger->NOutputTrackClusters(), Merger->NClusters());
	timer.Start();

	SetupGPUProcessor(Merger, false);
	mWorkersShadow->tpcMerger.OverrideSliceTracker(mDeviceConstantMem->tpcTrackers);
	
	WriteToConstantMemory((char*) &mDeviceConstantMem->tpcMerger - (char*) mDeviceConstantMem, &mWorkersShadow->tpcMerger, sizeof(mWorkersShadow->tpcMerger), 0);
	TransferMemoryResourceLinkToGPU(Merger->MemoryResRefit());
	times[0] += timer.GetCurrentElapsedTime(true);
	
	runKernel<AliGPUTPCGMMergerTrackFit>({fBlockCount, fThreadCount, 0}, nullptr, krnlRunRangeNone);
	SynchronizeGPU();
	times[1] += timer.GetCurrentElapsedTime(true);
	
	TransferMemoryResourceLinkToHost(Merger->MemoryResRefit());
	SynchronizeGPU();
	times[2] += timer.GetCurrentElapsedTime();
	
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("GPU Merger Finished");
	nCount++;

	if (mDeviceProcessingSettings.debugLevel > 0)
	{
		int copysize = 4 * Merger->NOutputTrackClusters() * sizeof(float) + Merger->NOutputTrackClusters() * sizeof(unsigned int) + Merger->NOutputTracks() * sizeof(AliGPUTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(AliGPUParam);
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
}

int AliGPUReconstructionDeviceBase::GetMaxThreads()
{
	int retVal = fTRDThreadCount * fBlockCount;
	return std::max(retVal, AliGPUReconstruction::GetMaxThreads());
}
