//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASTARTHITSSORTER_H
#define ALIHLTTPCCASTARTHITSSORTER_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCHitId.h"
#include "AliGPUCADataTypes.h"
MEM_CLASS_PRE()
class AliGPUTPCTracker;

/**
 * @class AliGPUTPCStartHitsSorter
 *
 */
class AliGPUTPCStartHitsSorter
{
public:
	MEM_CLASS_PRE()
	class AliGPUTPCSharedMemory
	{
		friend class AliGPUTPCStartHitsSorter;

	  public:
#if !defined(GPUCA_GPUCODE)
		AliGPUTPCSharedMemory()
		    : fStartRow(0), fNRows(0), fStartOffset(0)
		{
		}

		AliGPUTPCSharedMemory(const AliGPUTPCSharedMemory & /*dummy*/)
		    : fStartRow(0), fNRows(0), fStartOffset(0) {}
		AliGPUTPCSharedMemory &operator=(const AliGPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fStartRow;    // start row index
		int fNRows;       // number of rows to process
		int fStartOffset; //start offset for hits sorted by this block
	};

	typedef GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) workerType;
	MEM_TEMPLATE() GPUd() static workerType* Worker(MEM_TYPE(AliGPUCAConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //ALIHLTTPCCASTARTHITSSORTER_H
