#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUReconstructionCommon.h"

#include "AliGPUTPCTracker.h"
#include "AliGPUTPCSliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(_WIN32)
#include "../cmodules/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#endif
#include <string.h>

MEM_CLASS_PRE() class AliGPUTPCRow;

#define SemLockName "AliceHLTTPCCAGPUTrackerInitLockSem"

AliGPUReconstructionDeviceBase::AliGPUReconstructionDeviceBase(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionCPU(cfg)
{
}

AliGPUReconstructionDeviceBase::~AliGPUReconstructionDeviceBase()
{
	// make d'tor such that vtable is created for this class
	// needed for build with AliRoot
}

#ifdef GPUCA_ENABLE_GPU_TRACKER

int AliGPUReconstructionDeviceBase::GlobalTracking(int iSlice, int threadId, AliGPUReconstructionDeviceBase::helperParam* hParam)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker running Global Tracking for slice %d on thread %d\n", iSlice, threadId);}

	int sliceLeft = (iSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
	int sliceRight = (iSlice + 1) % (NSLICES / 2);
	if (iSlice >= (int) NSLICES / 2)
	{
		sliceLeft += NSLICES / 2;
		sliceRight += NSLICES / 2;
	}
	while (fSliceOutputReady < iSlice || fSliceOutputReady < sliceLeft || fSliceOutputReady < sliceRight)
	{
		if (hParam != nullptr && hParam->fReset) return(1);
	}

	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceLeft]);
	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceRight]);
	workers()->tpcTrackers[iSlice].PerformGlobalTracking(workers()->tpcTrackers[sliceLeft], workers()->tpcTrackers[sliceRight], GPUCA_GPU_MAX_TRACKS, GPUCA_GPU_MAX_TRACKS);
	pthread_mutex_unlock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceLeft]);
	pthread_mutex_unlock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceRight]);

	fSliceLeftGlobalReady[sliceLeft] = 1;
	fSliceRightGlobalReady[sliceRight] = 1;
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);}
	return(0);
}

void* AliGPUReconstructionDeviceBase::helperWrapper(void* arg)
{
	AliGPUReconstructionDeviceBase::helperParam* par = (AliGPUReconstructionDeviceBase::helperParam*) arg;
	AliGPUReconstructionDeviceBase* cls = par->fCls;

	AliGPUTPCTracker* tmpTracker = new AliGPUTPCTracker;

	if (cls->mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("\tHelper thread %d starting", par->fNum);

	//cpu_set_t mask;
	//CPU_ZERO(&mask);
	//CPU_SET(par->fNum * 2 + 2, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);

	while(pthread_mutex_lock(&((pthread_mutex_t*) par->fMutex)[0]) == 0 && par->fTerminate == false)
	{
		int mustRunSlice19 = 0;
		for (unsigned int i = par->fNum + 1;i < NSLICES;i += cls->mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			//if (cls->mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("\tHelper Thread %d Running, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
			if (par->fPhase)
			{
				if (cls->param().rec.GlobalTracking)
				{
					int realSlice = i + 1;
					if (realSlice % (NSLICES / 2) < 1) realSlice -= NSLICES / 2;

					if (realSlice % (NSLICES / 2) != 1)
					{
						cls->GlobalTracking(realSlice, par->fNum + 1, par);
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
			//if (cls->mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("\tHelper Thread %d Finished, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
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
	if (cls->mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("\tHelper thread %d terminating", par->fNum);
	delete tmpTracker;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
	pthread_exit(nullptr);
	return(nullptr);
}

void AliGPUReconstructionDeviceBase::ResetThisHelperThread(AliGPUReconstructionDeviceBase::helperParam* par)
{
	if (par->fReset) CAGPUImportant("GPU Helper Thread %d reseting", par->fNum);
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

int AliGPUReconstructionDeviceBase::ReadEvent(int iSlice, int threadId)
{
	if (workers()->tpcTrackers[iSlice].ReadEvent()) return(1);
	return(0);
}

void AliGPUReconstructionDeviceBase::WriteOutput(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker running WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
	workers()->tpcTrackers[iSlice].SetOutput(&mSliceOutput[iSlice]);
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_lock((pthread_mutex_t*) fHelperMemMutex);
	workers()->tpcTrackers[iSlice].WriteOutputPrepare();
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_unlock((pthread_mutex_t*) fHelperMemMutex);
	workers()->tpcTrackers[iSlice].WriteOutput();
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker finished WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
}

void AliGPUReconstructionDeviceBase::ResetHelperThreads(int helpers)
{
	CAGPUImportant("Error occurred, GPU tracker helper threads will be reset (Number of threads %d (%d))", mDeviceProcessingSettings.nDeviceHelperThreads, fNSlaveThreads);
	SynchronizeGPU();
	ReleaseThreadContext();
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fReset = true;
		if (helpers || i >= mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}
	CAGPUImportant("GPU Tracker helper threads have ben reset");
}

int AliGPUReconstructionDeviceBase::StartHelperThreads()
{
	int nThreads = mDeviceProcessingSettings.nDeviceHelperThreads;
	if (nThreads)
	{
		fHelperParams = new helperParam[nThreads];
		if (fHelperParams == nullptr)
		{
			CAGPUError("Memory allocation error");
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
				CAGPUError("Memory allocation error");
				ExitDevice();
				return(1);
			}
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_init(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j], nullptr))
				{
					CAGPUError("Error creating pthread mutex");
					ExitDevice();
					return(1);
				}

				pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]);
			}
			fHelperParams[i].fThreadId = (void*) malloc(sizeof(pthread_t));

			if (pthread_create((pthread_t*) fHelperParams[i].fThreadId, nullptr, helperWrapper, &fHelperParams[i]))
			{
				CAGPUError("Error starting slave thread");
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
				CAGPUError("Error unlocking mutex to terminate slave");
				return(1);
			}
			if (pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]))
			{
				CAGPUError("Error locking mutex");
				return(1);
			}
			if (pthread_join( *((pthread_t*) fHelperParams[i].fThreadId), nullptr))
			{
				CAGPUError("Error waiting for thread to terminate");
				return(1);
			}
			free(fHelperParams[i].fThreadId);
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]))
				{
					CAGPUError("Error unlocking mutex before destroying");
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

int AliGPUReconstructionDeviceBase::GetThread()
{
	//Get Thread ID
#ifdef _WIN32
	return((int) (size_t) GetCurrentThread());
#else
	return((int) syscall (SYS_gettid));
#endif
}

void AliGPUReconstructionDeviceBase::TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU)
{
	int inc = toGPU ? AliGPUMemoryResource::MEMORY_INPUT : AliGPUMemoryResource::MEMORY_OUTPUT;
	int exc = toGPU ? AliGPUMemoryResource::MEMORY_OUTPUT : AliGPUMemoryResource::MEMORY_INPUT;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		AliGPUMemoryResource& res = mMemoryResources[i];
		if (proc && res.mProcessor != proc) continue;
		if (!(res.mType & AliGPUMemoryResource::MEMORY_GPU) || (res.mType & AliGPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) continue;
		if (!mDeviceProcessingSettings.keepAllMemory && !(all && !(res.mType & exc)) && !(res.mType & inc)) continue;
		if (toGPU) TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
		else TransferMemoryResourceToHost(&mMemoryResources[i], stream);
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
		CAGPUError("Individual memory allocation strategy unsupported for device\n");
		return(1);
	}
	if (mDeviceProcessingSettings.nStreams > GPUCA_GPU_MAX_STREAMS)
	{
		CAGPUError("Too many straems requested %d > %d\n", mDeviceProcessingSettings.nStreams, GPUCA_GPU_MAX_STREAMS);
		return(1);
	}
	if (!mDeviceProcessingSettings.trackletConstructorInPipeline) mDeviceProcessingSettings.trackletSelectorInPipeline = false;

#ifdef _WIN32
	HANDLE* semLock = nullptr;
	if (mDeviceProcessingSettings.globalInitMutex)
	{
		semLock = new HANDLE;
		*semLock = CreateSemaphore(nullptr, 1, 1, SemLockName);
		if (*semLock == nullptr)
		{
			CAGPUError("Error creating GPUInit Semaphore");
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
			CAGPUError("Error creating GPUInit Semaphore");
			return(1);
		}
		timespec semtime;
		clock_gettime(CLOCK_REALTIME, &semtime);
		semtime.tv_sec += 10;
		while (sem_timedwait(semLock, &semtime) != 0)
		{
			CAGPUError("Global Lock for GPU initialisation was not released for 10 seconds, assuming another thread died");
			CAGPUWarning("Resetting the global lock");
			sem_post(semLock);
		}
	}
#endif

	fThreadId = GetThread();

	mDeviceMemorySize = GPUCA_GPU_MEMORY_SIZE;
	mHostMemorySize = GPUCA_HOST_MEMORY_SIZE;
	int retVal = InitDevice_Runtime();
	if (retVal)
	{
		CAGPUImportant("GPU Tracker initialization failed");
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
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i], &workers()->tpcTrackers[i]);
		RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i].Data(), &workers()->tpcTrackers[i].Data());
	}
	RegisterGPUDeviceProcessor(&mWorkersShadow->tpcMerger, &workers()->tpcMerger);
	if (PrepareFlatObjects())
	{
		CAGPUError("Error preparing flat objects on GPU");
		ExitDevice_Runtime();
		return(1);
	}
	RegisterGPUDeviceProcessor(&mWorkersShadow->trdTracker, &workers()->trdTracker);

	if (StartHelperThreads()) return(1);

	fHelperMemMutex = malloc(sizeof(pthread_mutex_t));
	if (fHelperMemMutex == nullptr)
	{
		CAGPUError("Memory allocation error");
		ExitDevice_Runtime();
		return(1);
	}

	if (pthread_mutex_init((pthread_mutex_t*) fHelperMemMutex, nullptr))
	{
		CAGPUError("Error creating pthread mutex");
		ExitDevice_Runtime();
		free(fHelperMemMutex);
		return(1);
	}

	fSliceGlobalMutexes = malloc(sizeof(pthread_mutex_t) * NSLICES);
	if (fSliceGlobalMutexes == nullptr)
	{
		CAGPUError("Memory allocation error");
		ExitDevice_Runtime();
		return(1);
	}
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		if (pthread_mutex_init(&((pthread_mutex_t*) fSliceGlobalMutexes)[i], nullptr))
		{
			CAGPUError("Error creating pthread mutex");
			ExitDevice_Runtime();
			return(1);
		}
	}

	CAGPUInfo("GPU Tracker initialization successfull"); //Verbosity reduced because GPU backend will print CAGPUImportant message!

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
		mProcShadow.fTpcTransform->clearInternalBufferUniquePtr();
		mProcShadow.fTpcTransform->setFutureBufferAddress(mProcDevice.fTpcTransformBuffer);
	}
#ifndef GPUCA_ALIROOT_LIB
	if (mTRDGeometry)
	{
		memcpy((void*) mProcShadow.fTrdGeometry, (const void*) mTRDGeometry.get(), sizeof(*mTRDGeometry));
		mProcShadow.fTrdGeometry->clearInternalBufferUniquePtr();
	}
#endif
	TransferMemoryResourceLinkToGPU(mProcShadow.mMemoryResFlat);
	return 0;
}

int AliGPUReconstructionDeviceBase::ExitDevice()
{
	if (StopHelperThreads()) return(1);
	pthread_mutex_destroy((pthread_mutex_t*) fHelperMemMutex);
	free(fHelperMemMutex);

	for (unsigned int i = 0;i < NSLICES;i++) pthread_mutex_destroy(&((pthread_mutex_t*) fSliceGlobalMutexes)[i]);
	free(fSliceGlobalMutexes);

	int retVal = ExitDevice_Runtime();
	mWorkersShadow = mWorkersDevice = nullptr;
	mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
	mHostMemorySize = mDeviceMemorySize = 0;

	return retVal;
}

int AliGPUReconstructionDeviceBase::RunTPCTrackingSlices()
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

int AliGPUReconstructionDeviceBase::RunTPCTrackingSlices_internal()
{
	//Primary reconstruction function
	if (fGPUStuck)
	{
		CAGPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	
	if (fThreadId != GetThread())
	{
		CAGPUDebug("CUDA thread changed, migrating context, Previous Thread: %d, New Thread: %d", fThreadId, GetThread());
		fThreadId = GetThread();
	}

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Running GPU Tracker");

	ActivateThreadContext();
	
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Allocating GPU Tracker memory and initializing constants");

	int offset = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		workers()->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
		offset += mIOPtrs.nClusterData[iSlice];
	}
	
	memcpy((void*) mWorkersShadow, (const void*) workers(), sizeof(*workers()));
	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		if (mProcessors[i].proc->mDeviceProcessor) mProcessors[i].proc->mDeviceProcessor->InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
	}

	try
	{
		PrepareEvent();
	}
	catch (const std::bad_alloc& e)
	{
		printf("Memory Allocation Error\n");
		return(2);
	}
	
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUMem = (char*) mDeviceMemoryBase;
		//Initialize Startup Constants
		*workers()->tpcTrackers[iSlice].NTracklets() = 0;
		*workers()->tpcTrackers[iSlice].NTracks() = 0;
		*workers()->tpcTrackers[iSlice].NTrackHits() = 0;
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUFixedBlockCount = NSLICES > fConstructorBlockCount ? (iSlice < fConstructorBlockCount) : fConstructorBlockCount * (iSlice + 1) / NSLICES - fConstructorBlockCount * (iSlice) / NSLICES;
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError = 0;
		workers()->tpcTrackers[iSlice].GPUParameters()->fNextTracklet = ((fConstructorBlockCount + NSLICES - 1 - iSlice) / NSLICES) * fConstructorThreadCount;
		mWorkersShadow->tpcTrackers[iSlice].SetGPUTextureBase(mDeviceMemoryBase);
	}

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fDone = 0;
		fHelperParams[i].fError = 0;
		fHelperParams[i].fPhase = 0;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}
	
	if (PrepareTextures()) return(2);

	//Copy Tracker Object to GPU Memory
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Tracker objects to GPU");
	if (PrepareProfile()) return 2;
	
	WriteToConstantMemory((char*) &mDeviceConstantMem->param - (char*) mDeviceConstantMem, &param(), sizeof(AliGPUCAParam), mNStreams - 1);
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
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Creating Slice Data (Slice %d)", iSlice);
		if (iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) == 0)
		{
			if (ReadEvent(iSlice, 0))
			{
				CAGPUError("Error reading event");
				return(3);
			}
		}
		else
		{
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Waiting for helper thread %d", iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1);
			while(fHelperParams[iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1].fDone < (int) iSlice);
			if (fHelperParams[iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1].fError)
			{
				return(3);
			}
		}

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			if (!mDeviceProcessingSettings.comparableDebutOutput) mDebugFile << std::endl << std::endl << "Reconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
			if (mDeviceProcessingSettings.debugMask & 1) workers()->tpcTrackers[iSlice].DumpSliceData(mDebugFile);
		}
		
		int useStream = (iSlice % mNStreams);
		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		runKernel<AliGPUMemClean16>({fBlockCount, fThreadCount, useStream}, krnlRunRangeNone, {}, mWorkersShadow->tpcTrackers[iSlice].Data().HitWeights(), mWorkersShadow->tpcTrackers[iSlice].Data().NumberOfHitsPlusAlign() * sizeof(*mWorkersShadow->tpcTrackers[iSlice].Data().HitWeights()));
		if (GPUDebug("Initialization (2)", 2, iSlice) RANDOM_ERROR)
		{
			return(3);
		}

		//Copy Data to GPU Global Memory
		workers()->tpcTrackers[iSlice].StartTimer(0);
		TransferMemoryResourceLinkToGPU(workers()->tpcTrackers[iSlice].Data().MemoryResInput(), useStream);
		TransferMemoryResourceLinkToGPU(workers()->tpcTrackers[iSlice].Data().MemoryResRows(), useStream);
		TransferMemoryResourceLinkToGPU(workers()->tpcTrackers[iSlice].MemoryResCommon(), useStream);

		if (GPUDebug("Initialization (3)", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		workers()->tpcTrackers[iSlice].StopTimer(0);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, NSLICES);
		workers()->tpcTrackers[iSlice].StartTimer(1);
		runKernel<AliGPUTPCNeighboursFinder>({GPUCA_ROW_COUNT, fFinderThreadCount, useStream}, {iSlice}, {nullptr, mStreamInit[useStream] ? nullptr : &mEvents.init});
		if (GPUDebug("Neighbours finder", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		workers()->tpcTrackers[iSlice].StopTimer(1);
		mStreamInit[useStream] = true;

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice].Data(), -1, true);
			memcpy(workers()->tpcTrackers[iSlice].LinkTmpMemory(), Res(workers()->tpcTrackers[iSlice].Data().MemoryResScratch()).Ptr(), Res(workers()->tpcTrackers[iSlice].Data().MemoryResScratch()).Size());
			if (mDeviceProcessingSettings.debugMask & 2) workers()->tpcTrackers[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, NSLICES);
		workers()->tpcTrackers[iSlice].StartTimer(2);
		runKernel<AliGPUTPCNeighboursCleaner>({GPUCA_ROW_COUNT - 2, fThreadCount, useStream}, {iSlice});
		if (GPUDebug("Neighbours Cleaner", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		workers()->tpcTrackers[iSlice].StopTimer(2);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) workers()->tpcTrackers[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, NSLICES);
		workers()->tpcTrackers[iSlice].StartTimer(3);
		runKernel<AliGPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, fThreadCount, useStream}, {iSlice});
		if (GPUDebug("Start Hits Finder", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		workers()->tpcTrackers[iSlice].StopTimer(3);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, NSLICES);
		workers()->tpcTrackers[iSlice].StartTimer(4);
		runKernel<AliGPUTPCStartHitsSorter>({fBlockCount, fThreadCount, useStream}, {iSlice});
		if (GPUDebug("Start Hits Sorter", useStream, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		workers()->tpcTrackers[iSlice].StopTimer(4);

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(workers()->tpcTrackers[iSlice].MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *workers()->tpcTrackers[iSlice].NTracklets(), iSlice);
			if (*workers()->tpcTrackers[iSlice].NTracklets() > GPUCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				CAGPUError("GPUCA_GPU_MAX_TRACKLETS constant insuffisant");
				return(3);
			}
		}

		if (mDeviceProcessingSettings.debugLevel >= 4 && *workers()->tpcTrackers[iSlice].NTracklets())
		{
			TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 32) workers()->tpcTrackers[iSlice].DumpStartHits(mDebugFile);
		}

		if (mDeviceProcessingSettings.trackletConstructorInPipeline)
		{
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor (Slice %d/%d)", iSlice, NSLICES)
			workers()->tpcTrackers[iSlice].StartTimer(6);
			runKernel<AliGPUTPCTrackletConstructor>({fConstructorBlockCount, fConstructorThreadCount, useStream}, {iSlice});
			if (GPUDebug("Tracklet Constructor", useStream, iSlice) RANDOM_ERROR)
			{
				return(3);
			}
			workers()->tpcTrackers[iSlice].StopTimer(6);
		}
	}
	ReleaseEvent(&mEvents.init);

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	if (mDeviceProcessingSettings.trackletConstructorInPipeline)
	{
		SynchronizeGPU();
	}
	else
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor");
		for (int i = 0;i < mNStreams;i++)
		{
			RecordMarker(&mEvents.stream[i], i);
		}
		workers()->tpcTrackers[0].StartTimer(6);
		runKernel<AliGPUTPCTrackletConstructor, 1>({fConstructorBlockCount, fConstructorThreadCount, 0}, krnlRunRangeNone, {&mEvents.constructor, mEvents.stream, mNStreams});
		for (int i = 0;i < mNStreams;i++)
		{
			ReleaseEvent(&mEvents.stream[i]);
		}
		if (GPUDebug("Tracklet Constructor", 0, 0) RANDOM_ERROR)
		{
			return(1);
		}
		workers()->tpcTrackers[0].StopTimer(6);
		SynchronizeEvents(&mEvents.constructor);
		ReleaseEvent(&mEvents.constructor);
	}

	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice], -1, true);
			CAGPUInfo("Obtained %d tracklets", *workers()->tpcTrackers[iSlice].NTracklets());
			if (mDeviceProcessingSettings.debugMask & 128) workers()->tpcTrackers[iSlice].DumpTrackletHits(mDebugFile);
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
		workers()->tpcTrackers[iSlice].StartTimer(7);
		runKernel<AliGPUTPCTrackletSelector>({fSelectorBlockCount, fSelectorThreadCount, useStream}, {iSlice, (int) runSlices});
		if (GPUDebug("Tracklet Selector", useStream, iSlice) RANDOM_ERROR)
		{
			return(1);
		}
		workers()->tpcTrackers[iSlice].StopTimer(7);
		for (unsigned int k = iSlice;k < iSlice + runSlices;k++)
		{
			TransferMemoryResourceLinkToHost(workers()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents.selector[k]);
			streamMap[k] = useStream;
		}
		useStream++;
		if (useStream >= mNStreams) useStream = 0;
	}

	fSliceOutputReady = 0;

	if (param().rec.GlobalTracking)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			fSliceLeftGlobalReady[i] = 0;
			fSliceRightGlobalReady[i] = 0;
		}
		memset(fGlobalTrackingDone, 0, NSLICES);
		memset(fWriteOutputDone, 0, NSLICES);

	}
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fPhase = 1;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}

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
			if (*workers()->tpcTrackers[tmpSlice].NTracks() > 0)
			{
				TransferMemoryResourceLinkToHost(workers()->tpcTrackers[tmpSlice].MemoryResTracks(), streamMap[tmpSlice]);
				TransferMemoryResourceLinkToHost(workers()->tpcTrackers[tmpSlice].MemoryResTrackHits(), streamMap[tmpSlice], &mEvents.selector[tmpSlice]);
			}
			else
			{
				transferRunning[tmpSlice] = false;
			}
			tmpSlice++;
		}

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) workers()->tpcTrackers[iSlice].DumpHitWeights(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 512) workers()->tpcTrackers[iSlice].DumpTrackHits(mDebugFile);
		}

		if (workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError];
			CAGPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, workers()->tpcTrackers[iSlice].Data().NumberOfHits());
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) if (transferRunning[iSlice2]) ReleaseEvent(&mEvents.selector[iSlice2]);
			return(3);
		}

		if (transferRunning[iSlice]) SynchronizeEvents(&mEvents.selector[iSlice]);
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Tracks Transfered: %d / %d", *workers()->tpcTrackers[iSlice].NTracks(), *workers()->tpcTrackers[iSlice].NTrackHits());
		
		workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks = workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks;
		workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits = workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1));
		fSliceOutputReady = iSlice;

		if (param().rec.GlobalTracking)
		{
			if (iSlice % (NSLICES / 2) == 2)
			{
				int tmpId = iSlice % (NSLICES / 2) - 1;
				if (iSlice >= NSLICES / 2) tmpId += NSLICES / 2;
				GlobalTracking(tmpId, 0, nullptr);
				fGlobalTrackingDone[tmpId] = 1;
			}
			for (unsigned int tmpSlice3a = 0;tmpSlice3a < iSlice;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
			{
				unsigned int tmpSlice3 = tmpSlice3a + 1;
				if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
				if (tmpSlice3 >= iSlice) break;

				unsigned int sliceLeft = (tmpSlice3 + (NSLICES / 2 - 1)) % (NSLICES / 2);
				unsigned int sliceRight = (tmpSlice3 + 1) % (NSLICES / 2);
				if (tmpSlice3 >= (int) NSLICES / 2)
				{
					sliceLeft += NSLICES / 2;
					sliceRight += NSLICES / 2;
				}

				if (tmpSlice3 % (NSLICES / 2) != 1 && fGlobalTrackingDone[tmpSlice3] == 0 && sliceLeft < iSlice && sliceRight < iSlice)
				{
					GlobalTracking(tmpSlice3, 0, nullptr);
					fGlobalTrackingDone[tmpSlice3] = 1;
				}

				if (fWriteOutputDone[tmpSlice3] == 0 && fSliceLeftGlobalReady[tmpSlice3] && fSliceRightGlobalReady[tmpSlice3])
				{
					WriteOutput(tmpSlice3, 0);
					fWriteOutputDone[tmpSlice3] = 1;
				}
			}
		}
		else
		{
			if (iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) == 0)
			{
				WriteOutput(iSlice, 0);
			}
		}
	}
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++) if (transferRunning[iSlice]) ReleaseEvent(&mEvents.selector[iSlice]);

	if (param().rec.GlobalTracking)
	{
		for (unsigned int tmpSlice3a = 0;tmpSlice3a < NSLICES;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			unsigned int tmpSlice3 = (tmpSlice3a + 1);
			if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
			if (fGlobalTrackingDone[tmpSlice3] == 0) GlobalTracking(tmpSlice3, 0, nullptr);
		}
		for (unsigned int tmpSlice3a = 0;tmpSlice3a < NSLICES;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			unsigned int tmpSlice3 = (tmpSlice3a + 1);
			if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
			if (fWriteOutputDone[tmpSlice3] == 0)
			{
				while (fSliceLeftGlobalReady[tmpSlice3] == 0 || fSliceRightGlobalReady[tmpSlice3] == 0);
				WriteOutput(tmpSlice3, 0);
			}
		}
	}

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	if (param().rec.GlobalTracking)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3)
		{
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				CAGPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits, workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits);
			}
		}
	}

	if (DoProfile()) return(1);
	
	if (mDeviceProcessingSettings.debugMask & 1024)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			workers()->tpcTrackers[i].DumpOutput(stdout);
		}
	}

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Reconstruction finished");

	return(0);
}

const AliGPUTPCTracker* AliGPUReconstructionDeviceBase::CPUTracker(int iSlice) {return &workers()->tpcTrackers[iSlice];}

int AliGPUReconstructionDeviceBase::PrepareTextures()
{
	return 0;
}

int AliGPUReconstructionDeviceBase::DoStuckProtection(int stream, void* event)
{
	return 0;
}

int AliGPUReconstructionDeviceBase::PrepareProfile()
{
	return 0;
}

int AliGPUReconstructionDeviceBase::DoProfile()
{
	return 0;
}

int AliGPUReconstructionDeviceBase::GetMaxThreads()
{
	int retVal = fTRDThreadCount * fBlockCount;
	return std::max(retVal, AliGPUReconstruction::GetMaxThreads());
}

#endif //GPUCA_ENABLE_GPU_TRACKER
