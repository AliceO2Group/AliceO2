// @(#) $Id: AliHLTTPCCATracker.cxx 34611 2009-09-04 00:22:05Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
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

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAClusterData.h"

#ifdef R__WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <dlfcn.h>
#endif

#ifdef HLTCA_STANDALONE
#include <omp.h>
#endif

int AliHLTTPCCATrackerFramework::InitGPU(int sliceCount, int forceDeviceID)
{
	//Initialize GPU Tracker and determine if GPU available
	int retVal;
	if (!fGPULibAvailable)
	{
		printf("GPU Library not loaded\n");
		return(1);
	}
	if (fGPUTrackerAvailable && (retVal = ExitGPU())) return(retVal);
	retVal = fGPUTracker->InitGPU(sliceCount, forceDeviceID);
	fUseGPUTracker = fGPUTrackerAvailable = retVal == 0;
	fGPUSliceCount = sliceCount;
	return(retVal);
}

int AliHLTTPCCATrackerFramework::ExitGPU()
{
	//Uninitialize GPU Tracker
	if (!fGPUTrackerAvailable) return(0);
	fUseGPUTracker = false;
	fGPUTrackerAvailable = false;
	return(fGPUTracker->ExitGPU());
}

void AliHLTTPCCATrackerFramework::SetGPUDebugLevel(int Level, std::ostream *OutFile, std::ostream *GPUOutFile)
{
	//Set Debug Level for GPU Tracker and also for CPU Tracker for comparison reasons
	fGPUTracker->SetDebugLevel(Level, GPUOutFile);
	fGPUDebugLevel = Level;
	for (int i = 0;i < fgkNSlices;i++)
	{
		fCPUTrackers[i].SetGPUDebugLevel(Level, OutFile);
	}
}

int AliHLTTPCCATrackerFramework::SetGPUTracker(bool enable)
{
	//Enable / disable GPU Tracker
	if (enable && !fGPUTrackerAvailable)
	{
		fUseGPUTracker = false;
		return(1);
	}
	fUseGPUTracker = enable;
	return(0);
}

GPUhd() void AliHLTTPCCATrackerFramework::SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val)
{
	//Set Output Control Pointers
	fOutputControl = val;
	fGPUTracker->SetOutputControl(val);
	for (int i = 0;i < fgkNSlices;i++)
	{
		fCPUTrackers[i].SetOutputControl(val);
	}
}

int AliHLTTPCCATrackerFramework::ProcessSlices(int firstSlice, int sliceCount, AliHLTTPCCAClusterData* pClusterData, AliHLTTPCCASliceOutput** pOutput)
{
	//Process sliceCount slices starting from firstslice, in is pClusterData array, out pOutput array
	if (fUseGPUTracker)
	{
		if (fGPUTracker->Reconstruct(pOutput, pClusterData, firstSlice, CAMath::Min(sliceCount, fgkNSlices - firstSlice))) return(1);
	}
	else
	{
#ifdef HLTCA_STANDALONE
		if (fOutputControl->fOutputPtr && omp_get_max_threads() > 1)
		{
			printf("fOutputPtr must not be used with OpenMP\n");
			return(1);
		}

#pragma omp parallel for
#endif
		for (int iSlice = 0;iSlice < CAMath::Min(sliceCount, fgkNSlices - firstSlice);iSlice++)
		{
			fCPUTrackers[firstSlice + iSlice].ReadEvent(&pClusterData[iSlice]);
			fCPUTrackers[firstSlice + iSlice].SetOutput(&pOutput[iSlice]);
			fCPUTrackers[firstSlice + iSlice].Reconstruct();
			fCPUTrackers[firstSlice + iSlice].SetupCommonMemory();
		}
	}
	
	if (fGPUDebugLevel >= 6 && fUseGPUTracker)
	{
	    fUseGPUTracker = 0;
	    ProcessSlices(firstSlice, sliceCount, pClusterData, pOutput);
	    fUseGPUTracker = 1;
	}

	//printf("Slice Tracks Output: %d\n", pOutput[0].NTracks());
	return(0);
}

unsigned long long int* AliHLTTPCCATrackerFramework::PerfTimer(int GPU, int iSlice, int iTimer)
{
	//Performance information for slice trackers
	return(GPU ? fGPUTracker->PerfTimer(iSlice, iTimer) : fCPUTrackers[iSlice].PerfTimer(iTimer));
}

int AliHLTTPCCATrackerFramework::InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param)
{
	//Initialize Tracker Parameters for a slice
	if (fGPUTrackerAvailable && fGPUTracker->InitializeSliceParam(iSlice, param)) return(1);
	fCPUTrackers[iSlice].Initialize(param);
	return(0);
}

AliHLTTPCCATrackerFramework::AliHLTTPCCATrackerFramework(int allowGPU) :	fGPULibAvailable(false), fGPUTrackerAvailable(false), fUseGPUTracker(false), fGPUDebugLevel(0), fGPUSliceCount(0), fGPUTracker(NULL), fGPULib(NULL), fOutputControl( NULL ), fCPUSliceCount(fgkNSlices)
{
	//Constructor
#ifdef R__WIN32
	HMODULE hGPULib = LoadLibraryEx("cagpu.dll", NULL, NULL);
#else
	void* hGPULib = dlopen("cagpu.so", RTLD_NOW);
#endif
	if (hGPULib == NULL)
	{
		printf("Error Opening cagpu library\n");
		fGPUTracker = new AliHLTTPCCAGPUTracker;
#ifndef R__WIN32
		printf("%d\n", dlerror());
#endif
	}
	else
	{
#ifdef R__WIN32
		FARPROC createFunc = GetProcAddress(hGPULib, "AliHLTTPCCAGPUTrackerNVCCCreate");
#else
		void* createFunc = (void*) dlsym(hGPULib, "AliHLTTPCCAGPUTrackerNVCCCreate");
#endif
		if (createFunc == NULL)
		{
			printf("Error Creating GPU Tracker\n");
#ifdef R__WIN32
			FreeLibrary(hGPULib);
#else
			dlclose(hGPULib);
#endif
			fGPUTracker = new AliHLTTPCCAGPUTracker;
		}
		else
		{
			AliHLTTPCCAGPUTracker* (*tmp)() = (AliHLTTPCCAGPUTracker* (*)()) createFunc;
			fGPUTracker = tmp();
			fGPULibAvailable = true;
			fGPULib = (void*) (size_t) hGPULib;
		}
	}

	if (allowGPU && fGPULibAvailable)
	{
		fUseGPUTracker = (fGPUTrackerAvailable= (fGPUTracker->InitGPU() == 0));
		fGPUSliceCount = fGPUTrackerAvailable ? fGPUTracker->GetSliceCount() : 0;
	}
}

AliHLTTPCCATrackerFramework::~AliHLTTPCCATrackerFramework()
{
#ifdef R__WIN32
	HMODULE hGPULib = (HMODULE) (size_t) fGPULib;
#else
	void* hGPULib = fGPULib;
#endif
	if (fGPULib)
	{
		if (fGPUTracker)
		{
#ifdef R__WIN32
			FARPROC destroyFunc = GetProcAddress(hGPULib, "AliHLTTPCCAGPUTrackerNVCCDestroy");
#else
			void* destroyFunc = (void*) dlsym(hGPULib, "AliHLTTPCCAGPUTrackerNVCCDestroy");
#endif
			if (destroyFunc == NULL) printf("Error Freeing GPU Tracker\n");
			else
			{
				void (*tmp)(AliHLTTPCCAGPUTracker*) =  (void (*)(AliHLTTPCCAGPUTracker*)) destroyFunc;
				tmp(fGPUTracker);
			}
		}

#ifdef R__WIN32
		FreeLibrary(hGPULib);
#else
		dlclose(hGPULib);
#endif
	}
	else if (fGPUTracker)
	{
		delete fGPUTracker;
	}
	fGPULib = NULL;
	fGPUTracker = NULL;
}
