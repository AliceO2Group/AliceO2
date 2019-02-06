//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASTARTHITSFINDER_H
#define ALIHLTTPCCASTARTHITSFINDER_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCHitId.h"
#include "AliGPUCADataTypes.h"
MEM_CLASS_PRE()
class AliGPUTPCTracker;

/**
 * @class AliGPUTPCStartHitsFinder
 *
 */
class AliGPUTPCStartHitsFinder
{
  public:
	MEM_CLASS_PRE()
	class AliGPUTPCSharedMemory
	{
		friend class AliGPUTPCStartHitsFinder;

	  public:
#if !defined(GPUCA_GPUCODE)
		AliGPUTPCSharedMemory()
		    : fIRow(0), fNHits(0), fNRowStartHits(0)
		{
		}

		AliGPUTPCSharedMemory(const AliGPUTPCSharedMemory & /*dummy*/)
		    : fIRow(0), fNHits(0), fNRowStartHits(0)
		{
		}
		AliGPUTPCSharedMemory &operator=(const AliGPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fIRow;          // row index
		int fNHits;         // n hits in the row
		int fNRowStartHits; //start hits found in the row
	};

	typedef GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) workerType;
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(AliGPUCAConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //ALIHLTTPCCASTARTHITSFINDER_H
