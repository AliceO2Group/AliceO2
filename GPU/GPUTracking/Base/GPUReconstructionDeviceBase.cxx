// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionDeviceBase.cxx
/// \author David Rohr

#include "GPUReconstructionDeviceBase.h"
#include "GPUReconstructionIncludes.h"

#include "GPUTPCTracker.h"
#include "GPUTPCSliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(_WIN32)
#include "../utils/pthread_mutex_win32_wrapper.h"
#else
#include <errno.h>
#include <unistd.h>
#endif
#include <string.h>

MEM_CLASS_PRE() class GPUTPCRow;

#define SemLockName "AliceHLTTPCGPUTrackerInitLockSem"

GPUReconstructionDeviceBase::GPUReconstructionDeviceBase(const GPUSettingsProcessing& cfg) : GPUReconstructionCPU(cfg)
{
}

GPUReconstructionDeviceBase::~GPUReconstructionDeviceBase()
{
	// make d'tor such that vtable is created for this class
	// needed for build with AliRoot
}

void* GPUReconstructionDeviceBase::helperWrapper_static(void* arg)
{
	GPUReconstructionHelpers::helperParam* par = (GPUReconstructionHelpers::helperParam*) arg;
	GPUReconstructionDeviceBase* cls = par->fCls;
	return cls->helperWrapper(par);
}

void* GPUReconstructionDeviceBase::helperWrapper(GPUReconstructionHelpers::helperParam* par)
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

void GPUReconstructionDeviceBase::ResetThisHelperThread(GPUReconstructionHelpers::helperParam* par)
{
	if (par->fReset) GPUImportant("GPU Helper Thread %d reseting", par->fNum);
	par->fReset = false;
	par->fMutex[1].unlock();
}

int GPUReconstructionDeviceBase::GetGlobalLock(void* &pLock)
{
#ifdef _WIN32
	HANDLE* semLock = new HANDLE;
	*semLock = CreateSemaphore(nullptr, 1, 1, SemLockName);
	if (*semLock == nullptr)
	{
		GPUError("Error creating GPUInit Semaphore");
		return(1);
	}
	WaitForSingleObject(*semLock, INFINITE);
#elif !defined(__APPLE__) //GPU not supported on MacOS anyway
	sem_t* semLock = sem_open(SemLockName, O_CREAT, 0x01B6, 1);
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
#else
	void* semLock = nullptr;
#endif
	pLock = semLock;
	return 0;
}

void GPUReconstructionDeviceBase::ReleaseGlobalLock(void* sem)
{
	//Release the global named semaphore that locks GPU Initialization
#ifdef _WIN32
	HANDLE* h = (HANDLE*) sem;
	ReleaseSemaphore(*h, 1, nullptr);
	CloseHandle(*h);
	delete h;
#elif !defined(__APPLE__) //GPU not supported on MacOS anyway
	sem_t* pSem = (sem_t*) sem;
	sem_post(pSem);
	sem_unlink(SemLockName);
#endif
}

void GPUReconstructionDeviceBase::ResetHelperThreads(int helpers)
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

int GPUReconstructionDeviceBase::StartHelperThreads()
{
	int nThreads = mDeviceProcessingSettings.nDeviceHelperThreads;
	if (nThreads)
	{
		fHelperParams = new GPUReconstructionHelpers::helperParam[nThreads];
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

int GPUReconstructionDeviceBase::StopHelperThreads()
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

void GPUReconstructionDeviceBase::WaitForHelperThreads()
{
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}
}

void GPUReconstructionDeviceBase::RunHelperThreads(int (GPUReconstructionHelpers::helperDelegateBase::* function)(int i, int t, GPUReconstructionHelpers::helperParam* p), GPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
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

int GPUReconstructionDeviceBase::InitDevice()
{
	//cpu_set_t mask;
	//CPU_ZERO(&mask);
	//CPU_SET(0, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);

	if (mDeviceProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL)
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

	void* semLock = nullptr;
	if (mDeviceProcessingSettings.globalInitMutex && GetGlobalLock(semLock)) return(1);

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

	mProcShadow.InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_SLAVE);
	mProcShadow.mMemoryResWorkers = RegisterMemoryAllocation(&mProcShadow, &GPUProcessorWorkers::SetPointersDeviceProcessor, GPUMemoryResource::MEMORY_PERMANENT | GPUMemoryResource::MEMORY_HOST, "Workers");
	AllocateRegisteredMemory(mProcShadow.mMemoryResWorkers);

	if (StartHelperThreads()) return(1);

	SetThreadCounts();

	GPUInfo("GPU Tracker initialization successfull"); //Verbosity reduced because GPU backend will print GPUImportant message!

	return(retVal);
}

void* GPUReconstructionDeviceBase::GPUProcessorWorkers::SetPointersDeviceProcessor(void* mem)
{
	//Don't run constructor / destructor here, this will be just local memcopy of Processors in GPU Memory
	computePointerWithAlignment(mem, mWorkersProc, 1);
	return mem;
}

int GPUReconstructionDeviceBase::ExitDevice()
{
	if (StopHelperThreads()) return(1);

	int retVal = ExitDevice_Runtime();
	mWorkersShadow = nullptr;
	mHostMemoryPool = mHostMemoryBase = mDeviceMemoryPool = mDeviceMemoryBase = mHostMemoryPermanent = mDeviceMemoryPermanent = nullptr;
	mHostMemorySize = mDeviceMemorySize = 0;
	
	return retVal;
}

int GPUReconstructionDeviceBase::GetMaxThreads()
{
	int retVal = fTRDThreadCount * fBlockCount;
	return std::max(retVal, GPUReconstruction::GetMaxThreads());
}
