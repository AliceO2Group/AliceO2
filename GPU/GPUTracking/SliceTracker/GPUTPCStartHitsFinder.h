//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef GPUTPCSTARTHITSFINDER_H
#define GPUTPCSTARTHITSFINDER_H

#include "GPUTPCDef.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCStartHitsFinder
 *
 */
class GPUTPCStartHitsFinder
{
  public:
	MEM_CLASS_PRE()
	class GPUTPCSharedMemory
	{
		friend class GPUTPCStartHitsFinder;

	  public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCSharedMemory()
		    : fIRow(0), fNHits(0), fNRowStartHits(0)
		{
		}

		GPUTPCSharedMemory(const GPUTPCSharedMemory & /*dummy*/)
		    : fIRow(0), fNHits(0), fNRowStartHits(0)
		{
		}
		GPUTPCSharedMemory &operator=(const GPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fIRow;          // row index
		int fNHits;         // n hits in the row
		GPUAtomic(int) fNRowStartHits; //start hits found in the row
	};

	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
    GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //GPUTPCSTARTHITSFINDER_H
