#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUReconstructionIncludes.h"

#include "AliGPUTPCTracker.h"
#include "AliGPUTPCSliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(_WIN32)
#include "../utils/pthread_mutex_win32_wrapper.h"
#else
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

void* AliGPUReconstructionDeviceBase::helperWrapper_static(void* arg)
{
	AliGPUReconstructionHelpers::helperParam* par = (AliGPUReconstructionHelpers::helperParam*) arg;
	AliGPUReconstructionDeviceBase* cls = par->fCls;
	return cls->helperWrapper(par);
}

void* AliGPUReconstructionDeviceBase::helperWrapper(AliGPUReconstructionHelpers::helperParam* par)
{
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("\tHelper thread %d starting", par->fNum);

	//cpu_set_t mask; //TODO add option
	//CPU_ZERO(&mask);
	//CPU_SET(par->fNum * 2 + 2, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);

	par->fMutex[0].lock();
	while(par->fTerminate == false)
	{
		for (int i = par->fNum + 1;i < par->fCount;i += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			//if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Running, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
			if ((par->fFunctionCls->*par->fFunction)(i, par->fNum + 1, par)) par->fError = 1;
			if (par->fReset) break;
			par->fDone = i + 1;
			//if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("\tHelper Thread %d Finished, Slice %d+%d, Phase %d", par->fNum, i, par->fPhase);
		}
		ResetThisHelperThread(par);
		par->fMutex[0].lock();
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("\tHelper thread %d terminating", par->fNum);
	par->fMutex[1].unlock();
	pthread_exit(nullptr);
	return(nullptr);
}

void AliGPUReconstructionDeviceBase::ResetThisHelperThread(AliGPUReconstructionHelpers::helperParam* par)
{
	if (par->fReset) GPUImportant("GPU Helper Thread %d reseting", par->fNum);
	par->fReset = false;
	par->fMutex[1].unlock();
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
		fHelperParams = new AliGPUReconstructionHelpers::helperParam[nThreads];
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
			for (int j = 0;j < 2;j++)
			{
				fHelperParams[i].fMutex[j].lock();
			}

			if (pthread_create(&fHelperParams[i].fThreadId, nullptr, helperWrapper_static, &fHelperParams[i]))
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
			fHelperParams[i].fMutex[0].unlock();
			fHelperParams[i].fMutex[1].lock();
			if (pthread_join(fHelperParams[i].fThreadId, nullptr))
			{
				GPUError("Error waiting for thread to terminate");
				return(1);
			}
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

void AliGPUReconstructionDeviceBase::RunHelperThreads(int (AliGPUReconstructionHelpers::helperDelegateBase::* function)(int i, int t, AliGPUReconstructionHelpers::helperParam* p), AliGPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
{
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fDone = 0;
		fHelperParams[i].fError = 0;
		fHelperParams[i].fFunction = function;
		fHelperParams[i].fFunctionCls = functionCls;
		fHelperParams[i].fCount = count;
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
	if (mDeviceProcessingSettings.nStreams > GPUCA_MAX_STREAMS)
	{
		GPUError("Too many straems requested %d > %d\n", mDeviceProcessingSettings.nStreams, GPUCA_MAX_STREAMS);
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

	mDeviceMemorySize = GPUCA_MEMORY_SIZE;
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
	mProcShadow.mMemoryResWorkers = RegisterMemoryAllocation(&mProcShadow, &AliGPUProcessorWorkers::SetPointersDeviceProcessor, AliGPUMemoryResource::MEMORY_PERMANENT | AliGPUMemoryResource::MEMORY_HOST, "Workers");
	AllocateRegisteredMemory(mProcShadow.mMemoryResWorkers);

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

int AliGPUReconstructionDeviceBase::ExitDevice()
{
	if (StopHelperThreads()) return(1);

	int retVal = ExitDevice_Runtime();
	mWorkersShadow = nullptr;
	mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
	mHostMemorySize = mDeviceMemorySize = 0;
	
	return retVal;
}

int AliGPUReconstructionDeviceBase::GetMaxThreads()
{
	int retVal = fTRDThreadCount * fBlockCount;
	return std::max(retVal, AliGPUReconstruction::GetMaxThreads());
}
