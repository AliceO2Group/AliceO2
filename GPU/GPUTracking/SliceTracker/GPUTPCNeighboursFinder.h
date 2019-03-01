// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNeighboursFinder.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCNEIGHBOURSFINDER_H
#define GPUTPCNEIGHBOURSFINDER_H

#include "GPUTPCDef.h"
#include "GPUTPCGPUConfig.h"
#include "GPUTPCRow.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursFinder
 *
 */
class GPUTPCNeighboursFinder
{
public:
	MEM_CLASS_PRE()
	class GPUTPCSharedMemory
	{
		friend class GPUTPCNeighboursFinder;

	  public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCSharedMemory()
		    : fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fRow(), fRowUp(), fRowDown()
		{
		}

		GPUTPCSharedMemory(const GPUTPCSharedMemory & /*dummy*/)
		    : fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fRow(), fRowUp(), fRowDown() {}
		GPUTPCSharedMemory &operator=(const GPUTPCSharedMemory & /*dummy*/) { return *this; }
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
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
#if defined(GPUCA_GPUCODE)
		float2 fA[GPUCA_THREAD_COUNT_FINDER][GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
		calink fB[GPUCA_THREAD_COUNT_FINDER][GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
#else
		float2 fA[GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
		calink fB[GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
#endif
#endif //GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
		MEM_LG(GPUTPCRow)
		fRow, fRowUp, fRowDown;
	};

	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //GPUTPCNEIGHBOURSFINDER_H
