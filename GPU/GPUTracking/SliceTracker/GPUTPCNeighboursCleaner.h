// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNeighboursCleaner.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCNEIGHBOURSCLEANER_H
#define GPUTPCNEIGHBOURSCLEANER_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCNeighboursCleaner
 *
 */
class GPUTPCNeighboursCleaner
{
public:
	MEM_CLASS_PRE()
	class GPUTPCSharedMemory
	{
		friend class GPUTPCNeighboursCleaner;

	  public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCSharedMemory()
		    : fIRow(0), fIRowUp(0), fIRowDn(0), fNHits(0)
		{
		}
		GPUTPCSharedMemory(const GPUTPCSharedMemory & /*dummy*/)
		    : fIRow(0), fIRowUp(0), fIRowDn(0), fNHits(0) {}
		GPUTPCSharedMemory &operator=(const GPUTPCSharedMemory & /*dummy*/) { return *this; }
#endif //!GPUCA_GPUCODE

	  protected:
		int fIRow;   // current row index
		int fIRowUp; // current row index
		int fIRowDn; // current row index
		int fNHits;  // number of hits
	};
    
	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType &tracker);
};

#endif //GPUTPCNEIGHBOURSCLEANER_H
