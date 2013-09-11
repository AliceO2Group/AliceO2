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

#include "AliHLTTPCCAGPUTrackerBase.h"
#include "AliHLTTPCCAClusterData.h"

ClassImp( AliHLTTPCCAGPUTrackerBase )

int AliHLTTPCCAGPUTrackerBase::GlobalTracking(int iSlice, int threadId, AliHLTTPCCAGPUTrackerBase::helperParam* hParam)
{
	if (fDebugLevel >= 3) printf("GPU Tracker running Global Tracking for slice %d on thread %d\n", iSlice, threadId);

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
	if (fDebugLevel >= 3) printf("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);
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
fpCudaStreams(NULL),
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

template <class T> inline T* AliHLTTPCCAGPUTrackerBase::alignPointer(T* ptr, int alignment)
{
	//Macro to align Pointers.
	//Will align to start at 1 MB segments, this should be consistent with every alignment in the tracker
	//(As long as every single data structure is <= 1 MB)

	size_t adr = (size_t) ptr;
	if (adr % alignment)
	{
		adr += alignment - (adr % alignment);
	}
	return((T*) adr);
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
	printf("Read %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
}

void AliHLTTPCCAGPUTrackerBase::WriteOutput(AliHLTTPCCASliceOutput** pOutput, int firstSlice, int iSlice, int threadId)
{
	if (fDebugLevel >= 3) printf("GPU Tracker running WriteOutput for slice %d on thread %d\n", firstSlice + iSlice, threadId);
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
	printf("Write %d %f %f\n", threadId, ((double) b - (double) a) / (double) fProfTimeC, ((double) a - (double) fProfTimeD) / (double) fProfTimeC);
#endif
	if (fDebugLevel >= 3) printf("GPU Tracker finished WriteOutput for slice %d on thread %d\n", firstSlice + iSlice, threadId);
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
	HLTImportant("GPU Tracker initialization successfull");

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