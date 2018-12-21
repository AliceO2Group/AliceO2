#include "AliGPUReconstructionDeviceBase.h"
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAGPUTrackerCommon.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracker.h"
#include "AliCAGPULogging.h"
#include "AliHLTTPCCASliceOutput.h"

#ifdef __CINT__
typedef int cudaError_t
#elif defined(WIN32)
#include "../cmodules/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#include <errno.h>
#endif

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

MEM_CLASS_PRE() class AliHLTTPCCARow;

#define SemLockName "AliceHLTTPCCAGPUTrackerInitLockSem"

AliGPUReconstructionDeviceBase::AliGPUReconstructionDeviceBase(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstruction(cfg)
{
}

#ifdef HLTCA_ENABLE_GPU_TRACKER

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
		if (hParam != NULL && hParam->fReset) return(1);
	}

	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceLeft]);
	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceRight]);
	mTPCSliceTrackersCPU[iSlice].PerformGlobalTracking(mTPCSliceTrackersCPU[sliceLeft], mTPCSliceTrackersCPU[sliceRight], HLTCA_GPU_MAX_TRACKS, HLTCA_GPU_MAX_TRACKS);
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

	AliHLTTPCCATracker* tmpTracker = new AliHLTTPCCATracker;

#ifdef HLTCA_STANDALONE
	if (cls->mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("\tHelper thread %d starting", par->fNum);
#endif

#if defined(HLTCA_STANDALONE) & !defined(WIN32)
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(par->fNum * 2 + 2, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);
#endif

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
#ifdef HLTCA_STANDALONE
	if (cls->mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("\tHelper thread %d terminating", par->fNum);
#endif
	delete tmpTracker;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
	pthread_exit(NULL);
	return(NULL);
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
#ifdef WIN32
	HANDLE* h = (HANDLE*) sem;
	ReleaseSemaphore(*h, 1, NULL);
	CloseHandle(*h);
	delete h;
#else
	sem_t* pSem = (sem_t*) sem;
	sem_post(pSem);
	sem_unlink(SemLockName);
#endif
}

int AliGPUReconstructionDeviceBase::CheckMemorySizes(int sliceCount)
{
	//Check constants for correct memory sizes
	if (sizeof(AliHLTTPCCATracker) * sliceCount > HLTCA_GPU_TRACKER_OBJECT_MEMORY)
	{
		CAGPUError("Insufficiant Tracker Object Memory for %d slices", sliceCount);
		return(1);
	}

	if (NSLICES * AliHLTTPCCATracker::CommonMemorySize() > HLTCA_GPU_COMMON_MEMORY)
	{
		CAGPUError("Insufficiant Common Memory");
		return(1);
	}

	if (NSLICES * (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow) > HLTCA_GPU_ROWS_MEMORY)
	{
		CAGPUError("Insufficiant Row Memory");
		return(1);
	}

	if (mDeviceProcessingSettings.debugLevel >= 3)
	{
		CAGPUInfo("Memory usage: Tracker Object %lld / %lld, Common Memory %lld / %lld, Row Memory %lld / %lld",
			(long long int) sizeof(AliHLTTPCCATracker) * sliceCount, (long long int) HLTCA_GPU_TRACKER_OBJECT_MEMORY,
			(long long int) (NSLICES * AliHLTTPCCATracker::CommonMemorySize()), (long long int) HLTCA_GPU_COMMON_MEMORY,
			(long long int) (NSLICES * (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow)), (long long int) HLTCA_GPU_ROWS_MEMORY);
	}
	return(0);
}

int AliGPUReconstructionDeviceBase::ReadEvent(int iSlice, int threadId)
{
	mTPCSliceTrackersCPU[iSlice].SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, iSlice), RowMemory(fHostLockedMemory, iSlice));
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
#endif
	if (mTPCSliceTrackersCPU[iSlice].ReadEvent(&mClusterData[iSlice])) return(1);
#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
	CAGPUInfo("Read %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
	return(0);
}

void AliGPUReconstructionDeviceBase::WriteOutput(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUDebug("GPU Tracker running WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
	mTPCSliceTrackersCPU[iSlice].SetOutput(&mSliceOutput[iSlice]);
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
#endif
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_lock((pthread_mutex_t*) fHelperMemMutex);
	mTPCSliceTrackersCPU[iSlice].WriteOutputPrepare();
	if (mDeviceProcessingSettings.nDeviceHelperThreads) pthread_mutex_unlock((pthread_mutex_t*) fHelperMemMutex);
	mTPCSliceTrackersCPU[iSlice].WriteOutput();
#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
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
		if (fHelperParams == NULL)
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
			if (fHelperParams[i].fMutex == NULL)
			{
				CAGPUError("Memory allocation error");
				ExitDevice();
				return(1);
			}
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_init(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j], NULL))
				{
					CAGPUError("Error creating pthread mutex");
					ExitDevice();
					return(1);
				}

				pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]);
			}
			fHelperParams[i].fThreadId = (void*) malloc(sizeof(pthread_t));

			if (pthread_create((pthread_t*) fHelperParams[i].fThreadId, NULL, helperWrapper, &fHelperParams[i]))
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
			if (pthread_join( *((pthread_t*) fHelperParams[i].fThreadId), NULL))
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
#ifdef WIN32
	return((int) (size_t) GetCurrentThread());
#else
	return((int) syscall (SYS_gettid));
#endif
}

int AliGPUReconstructionDeviceBase::InitDevice()
{
#if defined(HLTCA_STANDALONE) & !defined(WIN32)
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);
#endif

	if (CheckMemorySizes(NSLICES)) return(1);

#ifdef WIN32
	HANDLE* semLock;
	if (mDeviceProcessingSettings.globalInitMutex)
	{
		semLock = new HANDLE;
		*semLock = CreateSemaphore(NULL, 1, 1, SemLockName);
		if (*semLock == NULL)
		{
			CAGPUError("Error creating GPUInit Semaphore");
			return(1);
		}
		WaitForSingleObject(*semLock, INFINITE);
	}
#else
	sem_t* semLock;
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

#ifdef HLTCA_GPU_MERGER
	fGPUMergerMaxMemory = HLTCA_GPU_MERGER_MEMORY;
#endif
	const size_t trackerGPUMem  = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + NSLICES * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_GLOBAL_MEMORY);
	const size_t trackerHostMem = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + NSLICES * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + HLTCA_GPU_TRACKER_OBJECT_MEMORY;
	fGPUMemSize = trackerGPUMem + fGPUMergerMaxMemory + HLTCA_GPU_MEMALIGN;
	fHostMemSize = trackerHostMem + fGPUMergerMaxMemory + HLTCA_GPU_MEMALIGN;
	int retVal = InitDevice_Runtime();
	
	if (mDeviceProcessingSettings.globalInitMutex) ReleaseGlobalLock(semLock);
	
	fGPUMergerMemory = getPointerWithAlignmentNoUpdate<char>(fGPUMemory, trackerGPUMem, HLTCA_GPU_MEMALIGN);
	fGPUMergerHostMemory = getPointerWithAlignmentNoUpdate<char>(fHostLockedMemory, trackerHostMem, HLTCA_GPU_MEMALIGN);
	if (retVal)
	{
		CAGPUImportant("GPU Tracker initialization failed");
		return(1);
	}

	//Don't run constructor / destructor here, this will be just local memcopy of Tracker in GPU Memory
	fGpuTracker = (AliHLTTPCCATracker*) TrackerMemory(fHostLockedMemory, 0);

	for (unsigned int i = 0;i < NSLICES;i++)
	{
		mTPCSliceTrackersCPU[i].SetGPUTracker();
		mTPCSliceTrackersCPU[i].SetGPUTrackerCommonMemory((char*) CommonMemory(fHostLockedMemory, i));
		mTPCSliceTrackersCPU[i].SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, i), RowMemory(fHostLockedMemory, i));
		mTPCSliceTrackersCPU[i].InitializeRows(&mParam);
	}

	if (StartHelperThreads()) return(1);

	fHelperMemMutex = malloc(sizeof(pthread_mutex_t));
	if (fHelperMemMutex == NULL)
	{
		CAGPUError("Memory allocation error");
		ExitDevice_Runtime();
		return(1);
	}

	if (pthread_mutex_init((pthread_mutex_t*) fHelperMemMutex, NULL))
	{
		CAGPUError("Error creating pthread mutex");
		ExitDevice_Runtime();
		free(fHelperMemMutex);
		return(1);
	}

	fSliceGlobalMutexes = malloc(sizeof(pthread_mutex_t) * NSLICES);
	if (fSliceGlobalMutexes == NULL)
	{
		CAGPUError("Memory allocation error");
		ExitDevice_Runtime();
		return(1);
	}
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		if (pthread_mutex_init(&((pthread_mutex_t*) fSliceGlobalMutexes)[i], NULL))
		{
			CAGPUError("Error creating pthread mutex");
			ExitDevice_Runtime();
			return(1);
		}
	}

	CAGPUInfo("GPU Tracker initialization successfull"); //Verbosity reduced because GPU backend will print CAGPUImportant message!

	return(retVal);
}

int AliGPUReconstructionDeviceBase::ExitDevice()
{
	if (StopHelperThreads()) return(1);
	pthread_mutex_destroy((pthread_mutex_t*) fHelperMemMutex);
	free(fHelperMemMutex);

	for (unsigned int i = 0;i < NSLICES;i++) pthread_mutex_destroy(&((pthread_mutex_t*) fSliceGlobalMutexes)[i]);
	free(fSliceGlobalMutexes);

	return(ExitDevice_Runtime());
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_FinishSlices(unsigned int iSlice)
{
	mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTracks = mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks;
	mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTrackHits = mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTrackHits;
	if (mParam.rec.GlobalTracking) mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracklets = 1;

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1));
	fSliceOutputReady = iSlice;

	if (mParam.rec.GlobalTracking)
	{
		if (iSlice % (NSLICES / 2) == 2)
		{
			int tmpId = iSlice % (NSLICES / 2) - 1;
			if (iSlice >= NSLICES / 2) tmpId += NSLICES / 2;
			GlobalTracking(tmpId, 0, NULL);
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
				GlobalTracking(tmpSlice3, 0, NULL);
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

int AliGPUReconstructionDeviceBase::Reconstruct_Base_Finalize(char*& tmpMemoryGlobalTracking)
{
	if (mParam.rec.GlobalTracking)
	{
		for (unsigned int tmpSlice3a = 0;tmpSlice3a < NSLICES;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
		{
			unsigned int tmpSlice3 = (tmpSlice3a + 1);
			if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
			if (fGlobalTrackingDone[tmpSlice3] == 0) GlobalTracking(tmpSlice3, 0, NULL);
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
		free(tmpMemoryGlobalTracking);
		if (mDeviceProcessingSettings.debugLevel >= 3)
		{
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				CAGPUDebug("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTracks, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTrackHits, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTrackHits);
			}
		}
	}

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Reconstruction finished");
	return(0);
}

int AliGPUReconstructionDeviceBase::Reconstruct_Base_StartGlobal(char*& tmpMemoryGlobalTracking)
{
	if (mParam.rec.GlobalTracking)
	{
		int tmpmemSize = sizeof(AliHLTTPCCATracklet)
#ifdef EXTERN_ROW_HITS
		+ HLTCA_ROW_COUNT * sizeof(int)
#endif
		+ 16;
		tmpMemoryGlobalTracking = (char*) malloc(tmpmemSize * NSLICES);
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			fSliceLeftGlobalReady[i] = 0;
			fSliceRightGlobalReady[i] = 0;
		}
		memset(fGlobalTrackingDone, 0, NSLICES);
		memset(fWriteOutputDone, 0, NSLICES);

		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			mTPCSliceTrackersCPU[iSlice].SetGPUTrackerTrackletsMemory(tmpMemoryGlobalTracking + (tmpmemSize * iSlice), 1);
		}
	}
	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		fHelperParams[i].fPhase = 1;
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
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
		mDebugFile << std::endl << std::endl << "Reconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
#endif
		if (mDeviceProcessingSettings.debugMask & 1) mTPCSliceTrackersCPU[iSlice].DumpSliceData(mDebugFile);
	}

	if (mTPCSliceTrackersCPU[iSlice].Data().MemorySize() > HLTCA_GPU_SLICE_DATA_MEMORY RANDOM_ERROR)
	{
		CAGPUError("Insufficiant Slice Data Memory: Slice %d, Needed %lld, Available %lld", iSlice, (long long int) mTPCSliceTrackersCPU[iSlice].Data().MemorySize(), (long long int) HLTCA_GPU_SLICE_DATA_MEMORY);
		ResetHelperThreads(1);
		return(1);
	}

	if (mDeviceProcessingSettings.debugLevel >= 3)
	{
		CAGPUInfo("GPU Slice Data Memory Used: %lld/%lld", (long long int) mTPCSliceTrackersCPU[iSlice].Data().MemorySize(), (long long int) HLTCA_GPU_SLICE_DATA_MEMORY);
	}
	return(0);
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
	memcpy((void*) fGpuTracker, (void*) mTPCSliceTrackersCPU, sizeof(AliHLTTPCCATracker) * NSLICES);

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Allocating GPU Tracker memory and initializing constants");

#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryFreq(&fProfTimeC);
	AliHLTTPCCATracker::StandaloneQueryTime(&fProfTimeD);
#endif

	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		mClusterData[iSlice].SetClusterData(iSlice, mIOPtrs.nClusterData[iSlice], mIOPtrs.clusterData[iSlice]);

		//Make this a GPU Tracker
		fGpuTracker[iSlice].SetGPUTracker();
		fGpuTracker[iSlice].SetGPUTrackerCommonMemory((char*) CommonMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].SetGPUSliceDataMemory(SliceDataMemory(fGPUMemory, iSlice), RowMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].SetPointersSliceData(&mClusterData[iSlice], false);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUMem = (char*) fGPUMemory;

		//Set Pointers to GPU Memory
		char* tmpMem = (char*) GlobalMemory(fGPUMemory, iSlice);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Initialising GPU Hits Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerHitsMemory(tmpMem, mClusterData[iSlice].NumberOfClusters());
		tmpMem += getAlignment(tmpMem, HLTCA_GPU_MEMALIGN);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Initialising GPU Tracklet Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTrackletsMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS);
		tmpMem += getAlignment(tmpMem, HLTCA_GPU_MEMALIGN);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Initialising GPU Track Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKS, mClusterData[iSlice].NumberOfClusters());
		tmpMem += getAlignment(tmpMem, HLTCA_GPU_MEMALIGN);

		if (fGpuTracker[iSlice].TrackMemorySize() >= HLTCA_GPU_TRACKS_MEMORY RANDOM_ERROR)
		{
			CAGPUError("Insufficiant Track Memory");
			ResetHelperThreads(0);
			return(1);
		}

		if ((size_t) (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice)) > HLTCA_GPU_GLOBAL_MEMORY RANDOM_ERROR)
		{
			CAGPUError("Insufficiant Global Memory (%lld < %lld)", (long long int) (size_t) (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice)), (long long int) HLTCA_GPU_GLOBAL_MEMORY);
			ResetHelperThreads(0);
			return(1);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3)
		{
			CAGPUInfo("GPU Global Memory Used: %lld/%lld, Page Locked Tracks Memory used: %lld / %lld", (long long int) (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice)), (long long int) HLTCA_GPU_GLOBAL_MEMORY, (long long int) fGpuTracker[iSlice].TrackMemorySize(), (long long int) HLTCA_GPU_TRACKS_MEMORY);
		}

		//Initialize Startup Constants
		*mTPCSliceTrackersCPU[iSlice].NTracklets() = 0;
		*mTPCSliceTrackersCPU[iSlice].NTracks() = 0;
		*mTPCSliceTrackersCPU[iSlice].NTrackHits() = 0;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount = (int) NSLICES > fConstructorBlockCount ? ((int) iSlice < fConstructorBlockCount) : fConstructorBlockCount * (iSlice + 1) / NSLICES - fConstructorBlockCount * (iSlice) / NSLICES;
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Blocks for Slice %d: %d", iSlice, fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError = 0;
		mTPCSliceTrackersCPU[iSlice].GPUParameters()->fNextTracklet = ((fConstructorBlockCount + NSLICES - 1 - iSlice) / NSLICES) * fConstructorThreadCount;
		fGpuTracker[iSlice].SetGPUTextureBase(fGpuTracker[0].Data().Memory());
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

const AliHLTTPCCATracker* AliGPUReconstructionDeviceBase::CPUTracker(int iSlice) {return &mTPCSliceTrackersCPU[iSlice];}

#endif //HLTCA_ENABLE_GPU_TRACKER
