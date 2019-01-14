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
#if GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
		AliGPUTPCHitId fHits[GPUCA_GPU_THREAD_COUNT_SELECTOR][GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE];
#endif //GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
	};

	GPUd() static int NThreadSyncPoints() { return 1; }

	GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
	                          MEM_LOCAL(GPUsharedref() AliGPUTPCSharedMemory) & smem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) & tracker);
};

#endif //ALIHLTTPCCATRACKLETSELECTOR_H
