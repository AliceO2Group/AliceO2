// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//					David Rohr <drohr@kip.uni-heidelberg.de>				*
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "AliHLTTPCCAGPUTrackerBase.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAGPUTrackerCommon.h"

ClassImp( AliHLTTPCCAGPUTrackerBase )

int AliHLTTPCCAGPUTrackerBase::GlobalTracking(int iSlice, int threadId, AliHLTTPCCAGPUTrackerBase::helperParam* hParam)
{
	if (fDebugLevel >= 3) {HLTDebug("GPU Tracker running Global Tracking for slice %d on thread %d\n", iSlice, threadId);}

	int sliceLeft = (iSlice + (fgkNSlices / 2 - 1)) % (fgkNSlices / 2);
	int sliceRight = (iSlice + 1) % (fgkNSlices / 2);
	if (iSlice >= fgkNSlices / 2)
	{
		sliceLeft += fgkNSlices / 2;
		sliceRight += fgkNSlices / 2;
	}
	while (fSliceOutputReady < iSlice || fSliceOutputReady < sliceLeft || fSliceOutputReady < sliceRight)
	{
		if (hParam != NULL && hParam->fReset) return(1);
	}

	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceLeft]);
	pthread_mutex_lock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceRight]);
	fSlaveTrackers[iSlice].PerformGlobalTracking(fSlaveTrackers[sliceLeft], fSlaveTrackers[sliceRight], HLTCA_GPU_MAX_TRACKS);
	pthread_mutex_unlock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceLeft]);
	pthread_mutex_unlock(&((pthread_mutex_t*) fSliceGlobalMutexes)[sliceRight]);

	fSliceLeftGlobalReady[sliceLeft] = 1;
	fSliceRightGlobalReady[sliceRight] = 1;
	if (fDebugLevel >= 3) {HLTDebug("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);}
	return(0);
}

void* AliHLTTPCCAGPUTrackerBase::helperWrapper(void* arg)
{
	AliHLTTPCCAGPUTrackerBase::helperParam* par = (AliHLTTPCCAGPUTrackerBase::helperParam*) arg;
	AliHLTTPCCAGPUTrackerBase* cls = par->fCls;

	AliHLTTPCCATracker* tmpTracker = new AliHLTTPCCATracker;

#ifdef HLTCA_STANDALONE
	if (cls->fDebugLevel >= 2) HLTInfo("\tHelper thread %d starting", par->fNum);
#endif

#if defined(HLTCA_STANDALONE) & !defined(_WIN32)
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(par->fNum * 2 + 2, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);
#endif

	while(pthread_mutex_lock(&((pthread_mutex_t*) par->fMutex)[0]) == 0 && par->fTerminate == false)
	{
		if (par->CPUTracker)
		{
			for (int i = 0;i < cls->fNSlicesPerCPUTracker;i++)
			{
				int myISlice = cls->fSliceCount - cls->fNCPUTrackers * cls->fNSlicesPerCPUTracker + (par->fNum - cls->fNHelperThreads) * cls->fNSlicesPerCPUTracker + i;
#ifdef HLTCA_STANDALONE
				if (cls->fDebugLevel >= 3) HLTInfo("\tHelper Thread %d Doing full CPU tracking, Slice %d", par->fNum, myISlice);
#endif
				if (myISlice >= 0)
				{
					tmpTracker->Initialize(cls->fSlaveTrackers[par->fFirstSlice + myISlice].Param());
					tmpTracker->ReadEvent(&par->pClusterData[myISlice]);
					tmpTracker->DoTracking();
					tmpTracker->SetOutput(&par->pOutput[myISlice]);
					pthread_mutex_lock((pthread_mutex_t*) cls->fHelperMemMutex);
					tmpTracker->WriteOutputPrepare();
					pthread_mutex_unlock((pthread_mutex_t*) cls->fHelperMemMutex);
					tmpTracker->WriteOutput();

					/*cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetGPUSliceDataMemory((char*) new uint4[HLTCA_GPU_SLICE_DATA_MEMORY/sizeof(uint4)], (char*) new uint4[HLTCA_GPU_ROWS_MEMORY/sizeof(uint4)]);
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].ReadEvent(&par->pClusterData[myISlice]);
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetPointersTracklets(HLTCA_GPU_MAX_TRACKLETS);
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetPointersHits(par->pClusterData[myISlice].NumberOfClusters());
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetPointersTracks(HLTCA_GPU_MAX_TRACKS, par->pClusterData[myISlice].NumberOfClusters());
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ cls->fSlaveTrackers[par->fFirstSlice + myISlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS, cls->fConstructorBlockCount);
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ cls->fSlaveTrackers[par->fFirstSlice + myISlice].HitMemorySize()/sizeof( uint4 ) + 100]), par->pClusterData[myISlice].NumberOfClusters());
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].SetGPUTrackerTracksMemory(reinterpret_cast<char*> ( new uint4 [ cls->fSlaveTrackers[par->fFirstSlice + myISlice].TrackMemorySize()/sizeof( uint4 ) + 100]), HLTCA_GPU_MAX_TRACKS, par->pClusterData[myISlice].NumberOfClusters());
					cls->fSlaveTrackers[par->fFirstSlice + myISlice].DoTracking();
					cls->WriteOutput(par->pOutput, par->fFirstSlice, myISlice, par->fNum + 1);
					delete[] cls->fSlaveTrackers[par->fFirstSlice + myISlice].HitMemory();
					delete[] cls->fSlaveTrackers[par->fFirstSlice + myISlice].TrackletMemory();
					delete[] cls->fSlaveTrackers[par->fFirstSlice + myISlice].TrackMemory();*/
				}
#ifdef HLTCA_STANDALONE
				if (cls->fDebugLevel >= 3) HLTInfo("\tHelper Thread %d Finished, Slice %d", par->fNum, myISlice);
#endif
			}
		}
		else
		{
			int mustRunSlice19 = 0;
			for (int i = par->fNum + 1;i < par->fSliceCount;i += cls->fNHelperThreads + 1)
			{
				//if (cls->fDebugLevel >= 3) HLTInfo("\tHelper Thread %d Running, Slice %d+%d, Phase %d", par->fNum, par->fFirstSlice, i, par->fPhase);
				if (par->fPhase)
				{
					if (cls->fUseGlobalTracking)
					{
						int realSlice = i + 1;
						if (realSlice % (fgkNSlices / 2) < 1) realSlice -= fgkNSlices / 2;

						if (realSlice % (fgkNSlices / 2) != 1)
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
							cls->WriteOutput(par->pOutput, par->fFirstSlice, realSlice, par->fNum + 1);
						}
					}
					else
					{
						while (cls->fSliceOutputReady < i)
						{
							if (par->fReset) goto ResetHelperThread;
						}
						cls->WriteOutput(par->pOutput, par->fFirstSlice, i, par->fNum + 1);
					}
				}
				else
				{
					cls->ReadEvent(par->pClusterData, par->fFirstSlice, i, par->fNum + 1);
					par->fDone = i + 1;
				}
				//if (cls->fDebugLevel >= 3) HLTInfo("\tHelper Thread %d Finished, Slice %d+%d, Phase %d", par->fNum, par->fFirstSlice, i, par->fPhase);
			}
			if (mustRunSlice19)
			{
				while (cls->fSliceLeftGlobalReady[19] == 0 || cls->fSliceRightGlobalReady[19] == 0)
				{
					if (par->fReset) goto ResetHelperThread;
				}
				cls->WriteOutput(par->pOutput, par->fFirstSlice, 19, par->fNum + 1);
			}
		}
ResetHelperThread:
		cls->ResetThisHelperThread(par);
	}
#ifdef HLTCA_STANDALONE
	if (cls->fDebugLevel >= 2) HLTInfo("\tHelper thread %d terminating", par->fNum);
#endif
	delete tmpTracker;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
	pthread_exit(NULL);
	return(NULL);
}

void AliHLTTPCCAGPUTrackerBase::ResetThisHelperThread(AliHLTTPCCAGPUTrackerBase::helperParam* par)
{
	if (par->fReset) HLTImportant("GPU Helper Thread %d reseting", par->fNum);
	par->fReset = false;
	pthread_mutex_unlock(&((pthread_mutex_t*) par->fMutex)[1]);
}

#define SemLockName "AliceHLTTPCCAGPUTrackerInitLockSem"

AliHLTTPCCAGPUTrackerBase::AliHLTTPCCAGPUTrackerBase() :
fGpuTracker(NULL),
fGPUMemory(NULL),
fHostLockedMemory(NULL),
fGPUMergerMemory(NULL),
fGPUMergerHostMemory(NULL),
fGPUMergerMaxMemory(0),
fDebugLevel(0),
fDebugMask(0xFFFFFFFF),
fOutFile(NULL),
fGPUMemSize(0),
fSliceCount(HLTCA_GPU_DEFAULT_MAX_SLICE_COUNT),
fCudaDevice(0),
fOutputControl(NULL),
fThreadId(0),
fCudaInitialized(0),
fPPMode(0),
fSelfheal(0),
fConstructorBlockCount(30),
selectorBlockCount(30),
fNHelperThreads(HLTCA_GPU_DEFAULT_HELPER_THREADS),
fHelperParams(NULL),
fHelperMemMutex(NULL),
fSliceOutputReady(0),
fSliceGlobalMutexes(NULL),
fNCPUTrackers(0),
fNSlicesPerCPUTracker(0),
fGlobalTracking(0),
fUseGlobalTracking(0),
fNSlaveThreads(0)
{}

AliHLTTPCCAGPUTrackerBase::~AliHLTTPCCAGPUTrackerBase()
{
}

void AliHLTTPCCAGPUTrackerBase::ReleaseGlobalLock(void* sem)
{
	//Release the global named semaphore that locks GPU Initialization
#ifdef R__WIN32
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

int AliHLTTPCCAGPUTrackerBase::CheckMemorySizes(int sliceCount)
{
	//Check constants for correct memory sizes
	if (sizeof(AliHLTTPCCATracker) * sliceCount > HLTCA_GPU_TRACKER_OBJECT_MEMORY)
	{
		HLTError("Insufficiant Tracker Object Memory for %d slices", sliceCount);
		return(1);
	}

	if (fgkNSlices * AliHLTTPCCATracker::CommonMemorySize() > HLTCA_GPU_COMMON_MEMORY)
	{
		HLTError("Insufficiant Common Memory");
		return(1);
	}

	if (fgkNSlices * (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow) > HLTCA_GPU_ROWS_MEMORY)
	{
		HLTError("Insufficiant Row Memory");
		return(1);
	}

	if (fDebugLevel >= 3)
	{
		HLTInfo("Memory usage: Tracker Object %d / %d, Common Memory %d / %d, Row Memory %d / %d", (int) sizeof(AliHLTTPCCATracker) * sliceCount, HLTCA_GPU_TRACKER_OBJECT_MEMORY, (int) (fgkNSlices * AliHLTTPCCATracker::CommonMemorySize()), HLTCA_GPU_COMMON_MEMORY, (int) (fgkNSlices * (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow)), HLTCA_GPU_ROWS_MEMORY);
	}
	return(0);
}

void AliHLTTPCCAGPUTrackerBase::SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile)
{
	//Set Debug Level and Debug output File if applicable
	fDebugLevel = dwLevel;
	if (NewOutFile) fOutFile = NewOutFile;
}

int AliHLTTPCCAGPUTrackerBase::SetGPUTrackerOption(char* OptionName, int OptionValue)
{
	//Set a specific GPU Tracker Option
	if (strcmp(OptionName, "PPMode") == 0)
	{
		fPPMode = OptionValue;
	}
	else if (strcmp(OptionName, "DebugMask") == 0)
	{
		fDebugMask = OptionValue;
	}
	else if (strcmp(OptionName, "HelperThreads") == 0)
	{
		fNHelperThreads = OptionValue;
	}
	else if (strcmp(OptionName, "CPUTrackers") == 0)
	{
		fNCPUTrackers = OptionValue;
	}
	else if (strcmp(OptionName, "SlicesPerCPUTracker") == 0)
	{
		fNSlicesPerCPUTracker = OptionValue;
	}
	else if (strcmp(OptionName, "GlobalTracking") == 0)
	{
		fGlobalTracking = OptionValue;
	}
	else
	{
		HLTError("Unknown Option: %s", OptionName);
		return(1);
	}

	if (fNHelperThreads + fNCPUTrackers > fNSlaveThreads && fCudaInitialized)
	{
		HLTInfo("Insufficient Slave Threads available (%d), creating additional Slave Threads (%d+%d)\n", fNSlaveThreads, fNHelperThreads, fNCPUTrackers);
		StopHelperThreads();
		StartHelperThreads();
	}

	return(0);
}

#ifdef HLTCA_STANDALONE
void AliHLTTPCCAGPUTrackerBase::StandalonePerfTime(int iSlice, int i)
{
	//Run Performance Query for timer i of slice iSlice
	if (fDebugLevel >= 1)
	{
		AliHLTTPCCATracker::StandaloneQueryTime( fSlaveTrackers[iSlice].PerfTimer(i));
	}
}
#else
void AliHLTTPCCAGPUTrackerBase::StandalonePerfTime(int /*iSlice*/, int /*i*/) {}
#endif

int AliHLTTPCCAGPUTrackerBase::SelfHealReconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	if (!fSelfheal)
	{
		ReleaseThreadContext();
		return(1);
	}
	static bool selfHealing = false;
	if (selfHealing)
	{
		HLTError("Selfhealing failed, giving up");
		ReleaseThreadContext();
		return(1);
	}
	else
	{
		HLTError("Unsolvable CUDA error occured, trying to reinitialize GPU");
	}			
	selfHealing = true;
	ExitGPU();
	if (InitGPU(fSliceCount, fCudaDevice))
	{
		HLTError("Could not reinitialize CUDA device, disabling GPU tracker");
		ExitGPU();
		return(1);
	}
	HLTInfo("GPU tracker successfully reinitialized, restarting tracking");
	int retVal = Reconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal);
	selfHealing = false;
	return(retVal);
}

void AliHLTTPCCAGPUTrackerBase::ReadEvent(AliHLTTPCCAClusterData* pClusterData, int firstSlice, int iSlice, int threadId)
{
	fSlaveTrackers[firstSlice + iSlice].SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, iSlice), RowMemory(fHostLockedMemory, firstSlice + iSlice));
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
#endif
	fSlaveTrackers[firstSlice + iSlice].ReadEvent(&pClusterData[iSlice]);
#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
	HLTInfo("Read %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
}

void AliHLTTPCCAGPUTrackerBase::WriteOutput(AliHLTTPCCASliceOutput** pOutput, int firstSlice, int iSlice, int threadId)
{
	if (fDebugLevel >= 3) {HLTDebug("GPU Tracker running WriteOutput for slice %d on thread %d\n", firstSlice + iSlice, threadId);}
	fSlaveTrackers[firstSlice + iSlice].SetOutput(&pOutput[iSlice]);
#ifdef HLTCA_GPU_TIME_PROFILE
	unsigned long long int a, b;
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
#endif
	if (fNHelperThreads) pthread_mutex_lock((pthread_mutex_t*) fHelperMemMutex);
	fSlaveTrackers[firstSlice + iSlice].WriteOutputPrepare();
	if (fNHelperThreads) pthread_mutex_unlock((pthread_mutex_t*) fHelperMemMutex);
	fSlaveTrackers[firstSlice + iSlice].WriteOutput();
#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
	HLTInfo("Write %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
	if (fDebugLevel >= 3) {HLTDebug("GPU Tracker finished WriteOutput for slice %d on thread %d\n", firstSlice + iSlice, threadId);}
}

int AliHLTTPCCAGPUTrackerBase::InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param)
{
	//Initialize Slice Tracker Parameter for a slave tracker
	fSlaveTrackers[iSlice].Initialize(param);
	if (fSlaveTrackers[iSlice].Param().NRows() != HLTCA_ROW_COUNT)
	{
		HLTError("Error, Slice Tracker %d Row Count of %d exceeds Constant of %d", iSlice, fSlaveTrackers[iSlice].Param().NRows(), HLTCA_ROW_COUNT);
		return(1);
	}
	return(0);
}

void AliHLTTPCCAGPUTrackerBase::ResetHelperThreads(int helpers)
{
	HLTImportant("Error occurred, GPU tracker helper threads will be reset (Number of threads %d/%d)", fNHelperThreads, fNCPUTrackers);
	SynchronizeGPU();
	ReleaseThreadContext();
	for (int i = 0;i < fNHelperThreads + fNCPUTrackers;i++)
	{
		fHelperParams[i].fReset = true;
		if (helpers || i >= fNHelperThreads) pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}
	HLTImportant("GPU Tracker helper threads have ben reset");
}

int AliHLTTPCCAGPUTrackerBase::StartHelperThreads()
{
	int nThreads = fNHelperThreads + fNCPUTrackers;
	if (nThreads)
	{
		fHelperParams = new helperParam[nThreads];
		if (fHelperParams == NULL)
		{
			HLTError("Memory allocation error");
			ExitGPU();
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
				HLTError("Memory allocation error");
				ExitGPU();
				return(1);
			}
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_init(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j], NULL))
				{
					HLTError("Error creating pthread mutex");
					ExitGPU();
					return(1);
				}

				pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]);
			}
			fHelperParams[i].fThreadId = (void*) malloc(sizeof(pthread_t));

			if (pthread_create((pthread_t*) fHelperParams[i].fThreadId, NULL, helperWrapper, &fHelperParams[i]))
			{
				HLTError("Error starting slave thread");
				ExitGPU();
				return(1);
			}
		}
	}
	fNSlaveThreads = nThreads;
	return(0);
}

int AliHLTTPCCAGPUTrackerBase::StopHelperThreads()
{
	if (fNSlaveThreads)
	{
		for (int i = 0;i < fNSlaveThreads;i++)
		{
			fHelperParams[i].fTerminate = true;
			if (pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]))
			{
				HLTError("Error unlocking mutex to terminate slave");
				return(1);
			}
			if (pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]))
			{
				HLTError("Error locking mutex");
				return(1);
			}
			if (pthread_join( *((pthread_t*) fHelperParams[i].fThreadId), NULL))
			{
				HLTError("Error waiting for thread to terminate");
				return(1);
			}
			free(fHelperParams[i].fThreadId);
			for (int j = 0;j < 2;j++)
			{
				if (pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[j]))
				{
					HLTError("Error unlocking mutex before destroying");
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

void AliHLTTPCCAGPUTrackerBase::SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val)
{
	//Set Output Control Pointers
	fOutputControl = val;
	for (int i = 0;i < fgkNSlices;i++)
	{
		fSlaveTrackers[i].SetOutputControl(val);
	}
}

int AliHLTTPCCAGPUTrackerBase::GetThread()
{
	//Get Thread ID
#ifdef R__WIN32
	return((int) (size_t) GetCurrentThread());
#else
	return((int) syscall (SYS_gettid));
#endif
}

unsigned long long int* AliHLTTPCCAGPUTrackerBase::PerfTimer(int iSlice, unsigned int i)
{
	//Returns pointer to PerfTimer i of slice iSlice
	return(fSlaveTrackers ? fSlaveTrackers[iSlice].PerfTimer(i) : NULL);
}

const AliHLTTPCCASliceOutput::outputControlStruct* AliHLTTPCCAGPUTrackerBase::OutputControl() const
{
	//Return Pointer to Output Control Structure
	return fOutputControl;
}

int AliHLTTPCCAGPUTrackerBase::GetSliceCount() const
{
	//Return max slice count processable
	return(fSliceCount);
}

char* AliHLTTPCCAGPUTrackerBase::MergerBaseMemory()
{
	return(alignPointer((char*) fGPUMergerHostMemory, 1024 * 1024));
}

int AliHLTTPCCAGPUTrackerBase::IsInitialized()
{
	return(fCudaInitialized);
}

int AliHLTTPCCAGPUTrackerBase::InitGPU(int sliceCount, int forceDeviceID)
{
#if defined(HLTCA_STANDALONE) & !defined(_WIN32)
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0, &mask);
	//sched_setaffinity(0, sizeof(mask), &mask);
#endif

	if (sliceCount == -1) sliceCount = fSliceCount;

	if (CheckMemorySizes(sliceCount)) return(1);

#ifdef R__WIN32
	HANDLE* semLock = new HANDLE;
	*semLock = CreateSemaphore(NULL, 1, 1, SemLockName);
	if (*semLock == NULL)
	{
		HLTError("Error creating GPUInit Semaphore");
		return(1);
	}
	WaitForSingleObject(*semLock, INFINITE);
#else
	sem_t* semLock = sem_open(SemLockName, O_CREAT, 0x01B6, 1);
	if (semLock == SEM_FAILED)
	{
		HLTError("Error creating GPUInit Semaphore");
		return(1);
	}
	timespec semtime;
	clock_gettime(CLOCK_REALTIME, &semtime);
	semtime.tv_sec += 10;
	while (sem_timedwait(semLock, &semtime) != 0)
	{
		HLTError("Global Lock for GPU initialisation was not released for 10 seconds, assuming another thread died");
		HLTWarning("Resetting the global lock");
		sem_post(semLock);
	}
#endif

	fThreadId = GetThread();

	fGPUMemSize = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + sliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_GLOBAL_MEMORY);

#ifdef HLTCA_GPU_MERGER
	fGPUMergerMaxMemory = 2000000 * 5 * sizeof(float);
	fGPUMemSize += fGPUMergerMaxMemory;
#endif

	int retVal = InitGPU_Runtime(sliceCount, forceDeviceID);
	ReleaseGlobalLock(semLock);

	if (retVal)
	{
		HLTImportant("GPU Tracker initialization failed");
		return(1);
	}

	fSliceCount = sliceCount;
	//Don't run constructor / destructor here, this will be just local memcopy of Tracker in GPU Memory
	fGpuTracker = (AliHLTTPCCATracker*) TrackerMemory(fHostLockedMemory, 0);

	for (int i = 0;i < fgkNSlices;i++)
	{
		fSlaveTrackers[i].SetGPUTracker();
		fSlaveTrackers[i].SetGPUTrackerCommonMemory((char*) CommonMemory(fHostLockedMemory, i));
		fSlaveTrackers[i].SetGPUSliceDataMemory(SliceDataMemory(fHostLockedMemory, i), RowMemory(fHostLockedMemory, i));
	}

	if (StartHelperThreads()) return(1);

	fHelperMemMutex = malloc(sizeof(pthread_mutex_t));
	if (fHelperMemMutex == NULL)
	{
		HLTError("Memory allocation error");
		ExitGPU_Runtime();
		return(1);
	}

	if (pthread_mutex_init((pthread_mutex_t*) fHelperMemMutex, NULL))
	{
		HLTError("Error creating pthread mutex");
		ExitGPU_Runtime();
		free(fHelperMemMutex);
		return(1);
	}

	fSliceGlobalMutexes = malloc(sizeof(pthread_mutex_t) * fgkNSlices);
	if (fSliceGlobalMutexes == NULL)
	{
		HLTError("Memory allocation error");
		ExitGPU_Runtime();
		return(1);
	}
	for (int i = 0;i < fgkNSlices;i++)
	{
		if (pthread_mutex_init(&((pthread_mutex_t*) fSliceGlobalMutexes)[i], NULL))
		{
			HLTError("Error creating pthread mutex");
			ExitGPU_Runtime();
			return(1);
		}
	}

	fCudaInitialized = 1;
	HLTInfo("GPU Tracker initialization successfull"); //Verbosity reduced because GPU backend will print HLTImportant message!

#if defined(HLTCA_STANDALONE) & !defined(CUDA_DEVICE_EMULATION)
	if (fDebugLevel < 2 && 0)
	{
		//Do one initial run for Benchmark reasons
		const int useDebugLevel = fDebugLevel;
		fDebugLevel = 0;
		AliHLTTPCCAClusterData* tmpCluster = new AliHLTTPCCAClusterData[sliceCount];

		std::ifstream fin;

		AliHLTTPCCAParam tmpParam;
		AliHLTTPCCASliceOutput::outputControlStruct tmpOutputControl;

		fin.open("events/settings.dump");
		int tmpCount;
		fin >> tmpCount;
		for (int i = 0;i < sliceCount;i++)
		{
			fSlaveTrackers[i].SetOutputControl(&tmpOutputControl);
			tmpParam.ReadSettings(fin);
			InitializeSliceParam(i, tmpParam);
		}
		fin.close();

		fin.open("eventspbpbc/event.0.dump", std::ifstream::binary);
		for (int i = 0;i < sliceCount;i++)
		{
			tmpCluster[i].StartReading(i, 0);
			tmpCluster[i].ReadEvent(fin);
		}
		fin.close();

		AliHLTTPCCASliceOutput **tmpOutput = new AliHLTTPCCASliceOutput*[sliceCount];
		memset(tmpOutput, 0, sliceCount * sizeof(AliHLTTPCCASliceOutput*));

		Reconstruct(tmpOutput, tmpCluster, 0, sliceCount);
		for (int i = 0;i < sliceCount;i++)
		{
			free(tmpOutput[i]);
			tmpOutput[i] = NULL;
			fSlaveTrackers[i].SetOutputControl(NULL);
		}
		delete[] tmpOutput;
		delete[] tmpCluster;
		fDebugLevel = useDebugLevel;
	}
#endif

	return(retVal);
}

int AliHLTTPCCAGPUTrackerBase::ExitGPU()
{
	if (StopHelperThreads()) return(1);
	pthread_mutex_destroy((pthread_mutex_t*) fHelperMemMutex);
	free(fHelperMemMutex);

	for (int i = 0;i < fgkNSlices;i++) pthread_mutex_destroy(&((pthread_mutex_t*) fSliceGlobalMutexes)[i]);
	free(fSliceGlobalMutexes);

	return(ExitGPU_Runtime());
}

int AliHLTTPCCAGPUTrackerBase::Reconstruct_Base_FinishSlices(AliHLTTPCCASliceOutput** pOutput, int& iSlice, int& firstSlice)
{
	fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNLocalTracks = fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNTracks;
	fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNLocalTrackHits = fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNTrackHits;
	if (fUseGlobalTracking) fSlaveTrackers[firstSlice + iSlice].CommonMemory()->fNTracklets = 1;

	if (fDebugLevel >= 3) HLTInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (fNHelperThreads + 1));
	fSliceOutputReady = iSlice;

	if (fUseGlobalTracking)
	{
		if (iSlice % (fgkNSlices / 2) == 2)
		{
			int tmpId = iSlice % (fgkNSlices / 2) - 1;
			if (iSlice >= fgkNSlices / 2) tmpId += fgkNSlices / 2;
			GlobalTracking(tmpId, 0, NULL);
			fGlobalTrackingDone[tmpId] = 1;
		}
		for (int tmpSlice3a = 0;tmpSlice3a < iSlice;tmpSlice3a += fNHelperThreads + 1)
		{
			int tmpSlice3 = tmpSlice3a + 1;
			if (tmpSlice3 % (fgkNSlices / 2) < 1) tmpSlice3 -= (fgkNSlices / 2);
			if (tmpSlice3 >= iSlice) break;

			int sliceLeft = (tmpSlice3 + (fgkNSlices / 2 - 1)) % (fgkNSlices / 2);
			int sliceRight = (tmpSlice3 + 1) % (fgkNSlices / 2);
			if (tmpSlice3 >= fgkNSlices / 2)
			{
				sliceLeft += fgkNSlices / 2;
				sliceRight += fgkNSlices / 2;
			}

			if (tmpSlice3 % (fgkNSlices / 2) != 1 && fGlobalTrackingDone[tmpSlice3] == 0 && sliceLeft < iSlice && sliceRight < iSlice)
			{
				GlobalTracking(tmpSlice3, 0, NULL);
				fGlobalTrackingDone[tmpSlice3] = 1;
			}

			if (fWriteOutputDone[tmpSlice3] == 0 && fSliceLeftGlobalReady[tmpSlice3] && fSliceRightGlobalReady[tmpSlice3])
			{
				WriteOutput(pOutput, firstSlice, tmpSlice3, 0);
				fWriteOutputDone[tmpSlice3] = 1;
			}
		}
	}
	else
	{
		if (iSlice % (fNHelperThreads + 1) == 0)
		{
			WriteOutput(pOutput, firstSlice, iSlice, 0);
		}
	}
	return(0);
}

int AliHLTTPCCAGPUTrackerBase::Reconstruct_Base_Finalize(AliHLTTPCCASliceOutput** pOutput, char*& tmpMemoryGlobalTracking, int& firstSlice)
{
	if (fUseGlobalTracking)
	{
		for (int tmpSlice3a = 0;tmpSlice3a < fgkNSlices;tmpSlice3a += fNHelperThreads + 1)
		{
			int tmpSlice3 = (tmpSlice3a + 1);
			if (tmpSlice3 % (fgkNSlices / 2) < 1) tmpSlice3 -= (fgkNSlices / 2);
			if (fGlobalTrackingDone[tmpSlice3] == 0) GlobalTracking(tmpSlice3, 0, NULL);
		}
		for (int tmpSlice3a = 0;tmpSlice3a < fgkNSlices;tmpSlice3a += fNHelperThreads + 1)
		{
			int tmpSlice3 = (tmpSlice3a + 1);
			if (tmpSlice3 % (fgkNSlices / 2) < 1) tmpSlice3 -= (fgkNSlices / 2);
			if (fWriteOutputDone[tmpSlice3] == 0)
			{
				while (fSliceLeftGlobalReady[tmpSlice3] == 0 || fSliceRightGlobalReady[tmpSlice3] == 0);
				WriteOutput(pOutput, firstSlice, tmpSlice3, 0);
			}
		}
	}

	for (int i = 0;i < fNHelperThreads + fNCPUTrackers;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	if (fUseGlobalTracking)
	{
		free(tmpMemoryGlobalTracking);
		if (fDebugLevel >= 3)
		{
			for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
			{
				HLTDebug("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, fSlaveTrackers[iSlice].CommonMemory()->fNLocalTracks, fSlaveTrackers[iSlice].CommonMemory()->fNTracks, fSlaveTrackers[iSlice].CommonMemory()->fNLocalTrackHits, fSlaveTrackers[iSlice].CommonMemory()->fNTrackHits);
			}
		}
	}

	StandalonePerfTime(firstSlice, 10);

	if (fDebugLevel >= 3) HLTInfo("GPU Reconstruction finished");
	return(0);
}

int AliHLTTPCCAGPUTrackerBase::Reconstruct_Base_StartGlobal(AliHLTTPCCASliceOutput** pOutput, char*& tmpMemoryGlobalTracking)
{
	if (fUseGlobalTracking)
	{
		int tmpmemSize = sizeof(AliHLTTPCCATracklet)
#ifdef EXTERN_ROW_HITS
		+ HLTCA_ROW_COUNT * sizeof(int)
#endif
		+ 16;
		tmpMemoryGlobalTracking = (char*) malloc(tmpmemSize * fgkNSlices);
		for (int i = 0;i < fgkNSlices;i++)
		{
			fSliceLeftGlobalReady[i] = 0;
			fSliceRightGlobalReady[i] = 0;
		}
		memset(fGlobalTrackingDone, 0, fgkNSlices);
		memset(fWriteOutputDone, 0, fgkNSlices);

		for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
		{
			fSlaveTrackers[iSlice].SetGPUTrackerTrackletsMemory(tmpMemoryGlobalTracking + (tmpmemSize * iSlice), 1, fConstructorBlockCount);
		}
	}
	for (int i = 0;i < fNHelperThreads;i++)
	{
		fHelperParams[i].fPhase = 1;
		fHelperParams[i].pOutput = pOutput;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}
	return(0);
}

int AliHLTTPCCAGPUTrackerBase::Reconstruct_Base_SliceInit(AliHLTTPCCAClusterData* pClusterData, int& iSlice, int& firstSlice)
{
	StandalonePerfTime(firstSlice + iSlice, 0);

	//Initialize GPU Slave Tracker
	if (fDebugLevel >= 3) HLTInfo("Creating Slice Data (Slice %d)", iSlice);
	if (iSlice % (fNHelperThreads + 1) == 0)
	{
		ReadEvent(pClusterData, firstSlice, iSlice, 0);
	}
	else
	{
		if (fDebugLevel >= 3) HLTInfo("Waiting for helper thread %d", iSlice % (fNHelperThreads + 1) - 1);
		while(fHelperParams[iSlice % (fNHelperThreads + 1) - 1].fDone < iSlice);
	}

	if (fDebugLevel >= 4)
	{
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
		*fOutFile << std::endl << std::endl << "Reconstruction: " << iSlice << "/" << sliceCountLocal << " Total Slice: " << fSlaveTrackers[firstSlice + iSlice].Param().ISlice() << " / " << fgkNSlices << std::endl;
#endif
		if (fDebugMask & 1) fSlaveTrackers[firstSlice + iSlice].DumpSliceData(*fOutFile);
	}

	if (fSlaveTrackers[firstSlice + iSlice].Data().MemorySize() > HLTCA_GPU_SLICE_DATA_MEMORY RANDOM_ERROR)
	{
		HLTError("Insufficiant Slice Data Memory: Slice %d, Needed %d, Available %d", firstSlice + iSlice, fSlaveTrackers[firstSlice + iSlice].Data().MemorySize(), HLTCA_GPU_SLICE_DATA_MEMORY);
		ResetHelperThreads(1);
		return(1);
	}

	if (fDebugLevel >= 3)
	{
		HLTInfo("GPU Slice Data Memory Used: %d/%d", (int) fSlaveTrackers[firstSlice + iSlice].Data().MemorySize(), HLTCA_GPU_SLICE_DATA_MEMORY);
	}
	return(0);
}

int AliHLTTPCCAGPUTrackerBase::Reconstruct_Base_Init(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int& firstSlice, int& sliceCountLocal)
{
	if (sliceCountLocal == -1) sliceCountLocal = fSliceCount;

	if (!fCudaInitialized)
	{
		HLTError("GPUTracker not initialized");
		return(1);
	}
	if (sliceCountLocal > fSliceCount)
	{
		HLTError("GPU Tracker was initialized to run with %d slices but was called to process %d slices", fSliceCount, sliceCountLocal);
		return(1);
	}
	if (fThreadId != GetThread())
	{
		HLTDebug("CUDA thread changed, migrating context, Previous Thread: %d, New Thread: %d", fThreadId, GetThread());
		fThreadId = GetThread();
	}

	if (fDebugLevel >= 2) HLTInfo("Running GPU Tracker (Slices %d to %d)", fSlaveTrackers[firstSlice].Param().ISlice(), fSlaveTrackers[firstSlice].Param().ISlice() + sliceCountLocal);

	if (sliceCountLocal * sizeof(AliHLTTPCCATracker) > HLTCA_GPU_TRACKER_CONSTANT_MEM)
	{
		HLTError("Insuffissant constant memory (Required %d, Available %d, Tracker %d, Param %d, SliceData %d)", sliceCountLocal * (int) sizeof(AliHLTTPCCATracker), (int) HLTCA_GPU_TRACKER_CONSTANT_MEM, (int) sizeof(AliHLTTPCCATracker), (int) sizeof(AliHLTTPCCAParam), (int) sizeof(AliHLTTPCCASliceData));
		return(1);
	}
	
	ActivateThreadContext();
	if (fPPMode)
	{
		int retVal = ReconstructPP(pOutput, pClusterData, firstSlice, sliceCountLocal);
		ReleaseThreadContext();
		return(retVal);
	}

	for (int i = fNHelperThreads;i < fNCPUTrackers + fNHelperThreads;i++)
	{
		fHelperParams[i].CPUTracker = 1;
		fHelperParams[i].pClusterData = pClusterData;
		fHelperParams[i].pOutput = pOutput;
		fHelperParams[i].fSliceCount = sliceCountLocal;
		fHelperParams[i].fFirstSlice = firstSlice;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}
	sliceCountLocal -= fNCPUTrackers * fNSlicesPerCPUTracker;
	if (sliceCountLocal < 0) sliceCountLocal = 0;

	fUseGlobalTracking = fGlobalTracking && sliceCountLocal == fgkNSlices;

	memcpy(fGpuTracker, &fSlaveTrackers[firstSlice], sizeof(AliHLTTPCCATracker) * sliceCountLocal);

	if (fDebugLevel >= 3) HLTInfo("Allocating GPU Tracker memory and initializing constants");

#ifdef HLTCA_GPU_TIME_PROFILE
	AliHLTTPCCATracker::StandaloneQueryFreq(&fProfTimeC);
	AliHLTTPCCATracker::StandaloneQueryTime(&fProfTimeD);
#endif

	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		//Make this a GPU Tracker
		fGpuTracker[iSlice].SetGPUTracker();
		fGpuTracker[iSlice].SetGPUTrackerCommonMemory((char*) CommonMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].SetGPUSliceDataMemory(SliceDataMemory(fGPUMemory, iSlice), RowMemory(fGPUMemory, iSlice));
		fGpuTracker[iSlice].SetPointersSliceData(&pClusterData[iSlice], false);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUMem = (char*) fGPUMemory;

		//Set Pointers to GPU Memory
		char* tmpMem = (char*) GlobalMemory(fGPUMemory, iSlice);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Hits Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerHitsMemory(tmpMem, pClusterData[iSlice].NumberOfClusters());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Tracklet Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTrackletsMemory(tmpMem, HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fDebugLevel >= 3) HLTInfo("Initialising GPU Track Memory");
		tmpMem = fGpuTracker[iSlice].SetGPUTrackerTracksMemory(tmpMem, HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
		tmpMem = alignPointer(tmpMem, 1024 * 1024);

		if (fGpuTracker[iSlice].TrackMemorySize() >= HLTCA_GPU_TRACKS_MEMORY RANDOM_ERROR)
		{
			HLTError("Insufficiant Track Memory");
			ResetHelperThreads(0);
			return(1);
		}

		if (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice) > HLTCA_GPU_GLOBAL_MEMORY RANDOM_ERROR)
		{
			HLTError("Insufficiant Global Memory");
			ResetHelperThreads(0);
			return(1);
		}

		if (fDebugLevel >= 3)
		{
			HLTInfo("GPU Global Memory Used: %d/%d, Page Locked Tracks Memory used: %d / %d", (int) (tmpMem - (char*) GlobalMemory(fGPUMemory, iSlice)), HLTCA_GPU_GLOBAL_MEMORY, (int) fGpuTracker[iSlice].TrackMemorySize(), HLTCA_GPU_TRACKS_MEMORY);
		}

		//Initialize Startup Constants
		*fSlaveTrackers[firstSlice + iSlice].NTracklets() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTracks() = 0;
		*fSlaveTrackers[firstSlice + iSlice].NTrackHits() = 0;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount = sliceCountLocal > fConstructorBlockCount ? (iSlice < fConstructorBlockCount) : fConstructorBlockCount * (iSlice + 1) / sliceCountLocal - fConstructorBlockCount * (iSlice) / sliceCountLocal;
		if (fDebugLevel >= 3) HLTInfo("Blocks for Slice %d: %d", iSlice, fGpuTracker[iSlice].GPUParametersConst()->fGPUFixedBlockCount);
		fGpuTracker[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
		fGpuTracker[iSlice].GPUParametersConst()->fGPUnSlices = sliceCountLocal;
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError = 0;
		fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fNextTracklet = (fConstructorBlockCount / sliceCountLocal + (fConstructorBlockCount % sliceCountLocal > iSlice)) * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
		fGpuTracker[iSlice].SetGPUTextureBase(fGpuTracker[0].Data().Memory());
	}

	for (int i = 0;i < fNHelperThreads;i++)
	{
		fHelperParams[i].CPUTracker = 0;
		fHelperParams[i].fDone = 0;
		fHelperParams[i].fPhase = 0;
		fHelperParams[i].pClusterData = pClusterData;
		fHelperParams[i].fSliceCount = sliceCountLocal;
		fHelperParams[i].fFirstSlice = firstSlice;
		pthread_mutex_unlock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[0]);
	}

	return(0);
}
