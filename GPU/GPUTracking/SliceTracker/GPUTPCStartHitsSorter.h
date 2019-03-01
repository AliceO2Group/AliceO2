//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef GPUTPCSTARTHITSSORTER_H
#define GPUTPCSTARTHITSSORTER_H

#include "GPUTPCDef.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCStartHitsSorter
 *
 */
class GPUTPCStartHitsSorter
{
public:
	MEM_CLASS_PRE()
	class GPUTPCSharedMemory
	{
		friend class GPUTPCStartHitsSorter;

	  public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCSharedMemory()
		    : fStartRow(0), fNRows(0), fStartOffset(0)
		{
		}

		GPUTPCSharedMemory(const GPUTPCSharedMemory & /*dummy*/)
		    : fStartRow(0), fNRows(0), fStartOffset(0) {}
		GPUTPCSharedMemory &operator=(const GPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fStartRow;    // start row index
		int fNRows;       // number of rows to process
		int fStartOffset; //start offset for hits sorted by this block
	};

	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //GPUTPCSTARTHITSSORTER_H
