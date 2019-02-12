//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLETSELECTOR_H
#define ALIHLTTPCCATRACKLETSELECTOR_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"
#include "AliGPUTPCHitId.h"
#include "AliGPUConstantMem.h"
MEM_CLASS_PRE()
class AliGPUTPCTracker;

/**
 * @class AliGPUTPCTrackletSelector
 *
 */
class AliGPUTPCTrackletSelector
{
public:
	MEM_CLASS_PRE()
	class AliGPUTPCSharedMemory
	{
		friend class AliGPUTPCTrackletSelector;

	protected:
		int fItr0;          // index of the first track in the block
		int fNThreadsTotal; // total n threads
		int fNTracklets;    // n of tracklets
#if GPUCA_GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
		AliGPUTPCHitId fHits[GPUCA_GPUCA_THREAD_COUNT_SELECTOR][GPUCA_GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE];
#endif //GPUCA_GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
	};

	typedef GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) workerType;
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(AliGPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &smem, workerType& tracker);
};

#endif //ALIHLTTPCCATRACKLETSELECTOR_H
