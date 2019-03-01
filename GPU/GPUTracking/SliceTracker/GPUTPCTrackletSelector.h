// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTrackletSelector.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLETSELECTOR_H
#define GPUTPCTRACKLETSELECTOR_H

#include "GPUTPCDef.h"
#include "GPUTPCGPUConfig.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
MEM_CLASS_PRE()
class GPUTPCTracker;

/**
 * @class GPUTPCTrackletSelector
 *
 */
class GPUTPCTrackletSelector
{
public:
	MEM_CLASS_PRE()
	class GPUTPCSharedMemory
	{
		friend class GPUTPCTrackletSelector;

	protected:
		int fItr0;          // index of the first track in the block
		int fNThreadsTotal; // total n threads
		int fNTracklets;    // n of tracklets
#if GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
		GPUTPCHitId fHits[GPUCA_THREAD_COUNT_SELECTOR][GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE];
#endif //GPUCA_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
	};

	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType& tracker);
};

#endif //GPUTPCTRACKLETSELECTOR_H
