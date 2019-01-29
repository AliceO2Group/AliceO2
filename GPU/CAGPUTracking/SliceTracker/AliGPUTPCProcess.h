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

template <class TProcess>
GPUg() void AliGPUTPCProcess(int iSlice)
{
	AliGPUTPCTracker &tracker = gGPUConstantMem.tpcTrackers[iSlice];
	GPUshared() typename TProcess::AliGPUTPCSharedMemory smem;

	for (int iSync = 0; iSync <= TProcess::NThreadSyncPoints(); iSync++)
	{
		GPUsync();
		TProcess::Thread(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, tracker);
	}
}

template <class TProcess>
GPUg() void AliGPUTPCProcessMulti(int firstSlice, int nSliceCount)
{
	const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
	const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
	const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
	const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
	AliGPUTPCTracker &tracker = gGPUConstantMem.tpcTrackers[firstSlice + iSlice];
	GPUshared() typename TProcess::AliGPUTPCSharedMemory smem;

	for (int iSync = 0; iSync <= TProcess::NThreadSyncPoints(); iSync++)
	{
		GPUsync();
		TProcess::Thread(sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), iSync, smem, tracker);
	}
}

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
