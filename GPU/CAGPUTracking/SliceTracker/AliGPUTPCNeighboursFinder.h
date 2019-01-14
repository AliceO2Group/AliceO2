//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCANEIGHBOURSFINDER_H
#define ALIHLTTPCCANEIGHBOURSFINDER_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"
#include "AliGPUTPCRow.h"
MEM_CLASS_PRE()
class AliGPUTPCTracker;

/**
 * @class AliGPUTPCNeighboursFinder
 *
 */
class AliGPUTPCNeighboursFinder
{
  public:
	MEM_CLASS_PRE()
	class AliGPUTPCSharedMemory
	{
		friend class AliGPUTPCNeighboursFinder;

	  public:
#if !defined(GPUCA_GPUCODE)
		AliGPUTPCSharedMemory()
		    : fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fRow(), fRowUp(), fRowDown()
		{
		}

		AliGPUTPCSharedMemory(const AliGPUTPCSharedMemory & /*dummy*/)
		    : fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fRow(), fRowUp(), fRowDown() {}
		AliGPUTPCSharedMemory &operator=(const AliGPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fNHits;   // n hits
		int fUpNHits; // n hits in the next row
		int fDnNHits; // n hits in the prev row
		float fUpDx;  // x distance to the next row
		float fDnDx;  // x distance to the previous row
		float fUpTx;  // normalized x distance to the next row
		float fDnTx;  // normalized x distance to the previous row
		int fIRow;    // row number
		int fIRowUp;  // next row number
		int fIRowDn;  // previous row number
#if ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
#if defined(GPUCA_GPUCODE)
		float2 fA[GPUCA_GPU_THREAD_COUNT_FINDER][ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
		calink fB[GPUCA_GPU_THREAD_COUNT_FINDER][ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
#else
		float2 fA[ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
		calink fB[ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
#endif
#endif //ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
		MEM_LG(AliGPUTPCRow)
		fRow, fRowUp, fRowDown;
	};

	GPUd() static int NThreadSyncPoints() { return 2; }

	GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
	                          MEM_LOCAL(GPUsharedref() AliGPUTPCSharedMemory) & smem, MEM_CONSTANT(GPUconstant() AliGPUTPCTracker) & tracker);
};

#endif //ALIHLTTPCCANEIGHBOURSFINDER_H
