// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCStartHitsFinder.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCStartHitsFinder.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"

template <> GPUd() void GPUTPCStartHitsFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &s, workerType &tracker)
{
	// find start hits for tracklets

	if (iThread == 0)
	{
		s.fIRow = iBlock + 1;
		s.fNRowStartHits = 0;
		if (s.fIRow <= GPUCA_ROW_COUNT - 4)
		{
			s.fNHits = tracker.Row(s.fIRow).NHits();
		}
		else
			s.fNHits = -1;
	}
    GPUbarrier();
	GPUglobalref() const MEM_GLOBAL(GPUTPCRow) &row = tracker.Row(s.fIRow);
	GPUglobalref() const MEM_GLOBAL(GPUTPCRow) &rowUp = tracker.Row(s.fIRow + 2);
	for (int ih = iThread; ih < s.fNHits; ih += nThreads)
	{
		if (tracker.HitLinkDownData(row, ih) == CALINK_INVAL && tracker.HitLinkUpData(row, ih) != CALINK_INVAL && tracker.HitLinkUpData(rowUp, tracker.HitLinkUpData(row, ih)) != CALINK_INVAL)
		{
#ifdef GPUCA_SORT_STARTHITS
			GPUglobalref() GPUTPCHitId *const startHits = tracker.TrackletTmpStartHits() + s.fIRow * GPUCA_MAX_ROWSTARTHITS;
			int nextRowStartHits = CAMath::AtomicAddShared(&s.fNRowStartHits, 1);
			if (nextRowStartHits >= GPUCA_MAX_ROWSTARTHITS)
#else
			GPUglobalref() GPUTPCHitId *const startHits = tracker.TrackletStartHits();
			int nextRowStartHits = CAMath::AtomicAdd(tracker.NTracklets(), 1);
			if (nextRowStartHits >= GPUCA_MAX_TRACKLETS)
#endif
			{
				tracker.GPUParameters()->fGPUError = GPUCA_ERROR_TRACKLET_OVERFLOW;
				CAMath::AtomicExch(tracker.NTracklets(), 0);
				break;
			}
			startHits[nextRowStartHits].Set(s.fIRow, ih);
		}
	}
    GPUbarrier();
    
#ifdef GPUCA_SORT_STARTHITS
	if (iThread == 0)
	{
		int nOffset = CAMath::AtomicAdd(tracker.NTracklets(), s.fNRowStartHits);
#ifdef GPUCA_GPUCODE
		tracker.RowStartHitCountOffset()[s.fIRow] = s.fNRowStartHits;
		if (nOffset + s.fNRowStartHits >= GPUCA_MAX_TRACKLETS)
		{
			tracker.GPUParameters()->fGPUError = GPUCA_ERROR_TRACKLET_OVERFLOW;
			CAMath::AtomicExch(tracker.NTracklets(), 0);
		}
#endif
	}
#endif
}
