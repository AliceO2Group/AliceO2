#include "AliGPUReconstructionDeviceBase.h"
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "AliGPUReconstructionCommon.h"

#include "AliGPUTPCTracker.h"
#include "AliCAGPULogging.h"
#include "AliGPUTPCSliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(_WIN32)
#include "../cmodules/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#include <errno.h>
#endif

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

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
	mWorkers->tpcTrackers[iSlice].PerformGlobalTracking(mWorkers->tpcTrackers[sliceLeft], mWorkers->tpcTrackers[sliceRight], GPUCA_GPU_MAX_TRACKS, GPUCA_GPU_MAX_TRACKS);
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
				if (cls->mParam.rec.GlobalTracking)
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
#ifdef GPUCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliGPUTPCTracker::StandaloneQueryTime(&a);
#endif
	if (mWorkers->tpcTrackers[iSlice].ReadEvent()) return(1);
#ifdef GPUCA_GPU_TIME_PROFILE
	AliGPUTPCTracker::StandaloneQueryTime(&b);
	CAGPUInfo("Read %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
	return(0);
}

void AliGPUReconstructionDeviceBase::WriteOutput(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker running WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
	mWorkers->tpcTrackers[iSlice].SetOutput(&mSliceOutput[iSlice]);
#ifdef GPUCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliGPUTPCTracker::StandaloneQueryTime(&a);
#endif
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_lock((pthread_mutex_t*) fHelperMemMutex);
	mWorkers->tpcTrackers[iSlice].WriteOutputPrepare();
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_unlock((pthread_mutex_t*) fHelperMemMutex);
	mWorkers->tpcTrackers[iSlice].WriteOutput();
#ifdef GPUCA_GPU_TIME_PROFILE
	AliGPUTPCTracker::StandaloneQueryTime(&b);
	CAGPUInfo("Write %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
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

int AliGPUReconstructionDeviceBase::TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU)
{
	int inc = toGPU ? AliGPUMemoryResource::MEMORY_INPUT : AliGPUMemoryResource::MEMORY_OUTPUT;
	int exc = toGPU ? AliGPUMemoryResource::MEMORY_OUTPUT : AliGPUMemoryResource::MEMORY_INPUT;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		AliGPUMemoryResource& res = mMemoryResources[i];
		if (proc && res.mProcessor != proc) continue;
		if (!(res.mType & AliGPUMemoryResource::MEMORY_GPU) || (res.mType & AliGPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) continue;
		if (!mDeviceProcessingSettings.keepAllMemory && !(all && !(res.mType & exc)) && !(res.mType & inc)) continue;
		if (toGPU ? TransferMemoryResourceToGPU(&mMemoryResources[i], stream) : TransferMemoryResourceToHost(&mMemoryResources[i], stream)) return 1;
	}
	return 0;
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
	memcpy((void*) &mWorkersShadow->trdTracker, (const void*) &mWorkers->trdTracker, sizeof(mWorkers->trdTracker));
	AllocateRegisteredMemory(mProcShadow.mMemoryResFlat);
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i], &mWorkers->tpcTrackers[i]);
		RegisterGPUDeviceProcessor(&mWorkersShadow->tpcTrackers[i].Data(), &mWorkers->tpcTrackers[i].Data());
	}
	RegisterGPUDeviceProcessor(&mWorkersShadow->tpcMerger, &mWorkers->tpcMerger);
	if (PrepareFlatObjects())
	{
		CAGPUError("Error preparing flat objects on GPU");
		ExitDevice_Runtime();
		return(1);
	}
	RegisterGPUDeviceProcessor(&mWorkersShadow->trdTracker, &mWorkers->trdTracker);

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
	if (TransferMemoryResourceLinkToGPU(mProcShadow.mMemoryResFlat)) return 1;
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

int AliGPUReconstructionDeviceBase::Reconstruct_Base_Init()
{
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
		mWorkers->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
		offset += mIOPtrs.nClusterData[iSlice];
	}
	
	memcpy((void*) mWorkersShadow, (const void*) mWorkers.get(), sizeof(*mWorkers));
	for (unsigned int i = 0;i < mProcessors.size();i++)
	{
		if (mProcessors[i].proc->mDeviceProcessor) mProcessors[i].proc->mDeviceProcessor->InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
	}

#ifdef GPUCA_GPU_TIME_PROFILE
	AliGPUTPCTracker::StandaloneQueryFreq(&fProfTimeC);
	AliGPUTPCTracker::StandaloneQueryTime(&fProfTimeD);
#endif

	try
	{
		PrepareEvent();
	}
	catch (const std::bad_alloc& e)
	{
		printf("Memory Allocation Error\n");
		ResetHelperThreads(0);
		return(1);
	}
	
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUMem = (char*) mDeviceMemoryBase;
		//Initialize Startup Constants
		*mWorkers->tpcTrackers[iSlice].NTracklets() = 0;
		*mWorkers->tpcTrackers[iSlice].NTracks() = 0;
		*mWorkers->tpcTrackers[iSlice].NTrackHits() = 0;
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUFixedBlockCount = NSLICES > fConstructorBlockCount ? (iSlice < fConstructorBlockCount) : fConstructorBlockCount * (iSlice + 1) / NSLICES - fConstructorBlockCount * (iSlice) / NSLICES;
		mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError = 0;
		mWorkers->tpcTrackers[iSlice].GPUParameters()->fNextTracklet = ((fConstructorBlockCount + NSLICES - 1 - iSlice) / NSLICES) * fConstructorThreadCount;
		mWorkersShadow->tpcTrackers[iSlice].SetGPUTextureBase(mDeviceMemoryBase);
	}

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fDone = 0;
		fHelperParams[i].fError = 0;
		fHelperParams[i].fPhase = 0;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}

	return(0);
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_SliceInit(unsigned int iSlice)
{
	//Initialize GPU Slave Tracker
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Creating Slice Data (Slice %d)", iSlice);
	if (iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) == 0)
	{
		if (ReadEvent(iSlice, 0))
		{
			CAGPUError("Error reading event");
			ResetHelperThreads(1);
			return(1);
		}
	}
	else
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Waiting for helper thread %d", iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1);
		while(fHelperParams[iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1].fDone < (int) iSlice);
		if (fHelperParams[iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1].fError)
		{
			ResetHelperThreads(1);
			return(1);
		}
	}

	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		if (!mDeviceProcessingSettings.comparableDebutOutput) mDebugFile << std::endl << std::endl << "Reconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
		if (mDeviceProcessingSettings.debugMask & 1) mWorkers->tpcTrackers[iSlice].DumpSliceData(mDebugFile);
	}
	return(0);
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_FinishSlices(unsigned int iSlice)
{
	mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks = mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks;
	mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits = mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1));
	fSliceOutputReady = iSlice;

	if (mParam.rec.GlobalTracking)
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
	return(0);
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_StartGlobal()
{
	if (mParam.rec.GlobalTracking)
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
	return(0);
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_Finalize()
{
	if (mParam.rec.GlobalTracking)
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

	if (mParam.rec.GlobalTracking)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3)
		{
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				CAGPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTrackHits);
			}
		}
	}

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Reconstruction finished");
	return(0);
}

const AliGPUTPCTracker* AliGPUReconstructionDeviceBase::CPUTracker(int iSlice) {return &mWorkers->tpcTrackers[iSlice];}

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
	int retVal = GPUCA_GPU_THREAD_COUNT_TRD * fConstructorBlockCount;
	return std::max(retVal, AliGPUReconstruction::GetMaxThreads());
}

#endif //GPUCA_ENABLE_GPU_TRACKER
