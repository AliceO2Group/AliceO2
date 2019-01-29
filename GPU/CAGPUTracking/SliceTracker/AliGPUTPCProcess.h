//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAPROCESS_H
#define ALIHLTTPCCAPROCESS_H

/**
 * Definitions needed for AliGPUTPCTracker
 *
 */

#include "AliGPUTPCDef.h"
#include "AliGPUTPCTrackParam.h"

#ifndef __OPENCL__
#include "AliGPUCADataTypes.h"
#endif

MEM_CLASS_PRE()
class AliGPUTPCTracker;

#if defined(__CUDACC__)



#elif defined(__OPENCL__) //__OPENCL__

#else //CPU

template <class TProcess>
void AliGPUTPCProcess(int nBlocks, int nThreads, AliGPUTPCTracker &tracker)
{
	for (int iB = 0; iB < nBlocks; iB++)
	{
		typename TProcess::AliGPUTPCSharedMemory smem;
		for (int iS = 0; iS <= TProcess::NThreadSyncPoints(); iS++)
			for (int iT = 0; iT < nThreads; iT++)
			{
				TProcess::Thread(nBlocks, nThreads, iB, iT, iS, smem, tracker);
			}
	}
}


#endif

#endif //ALIHLTTPCCAPROCESS_H
