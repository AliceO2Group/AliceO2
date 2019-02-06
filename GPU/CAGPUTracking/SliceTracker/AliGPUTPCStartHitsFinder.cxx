// @(#) $Id: AliGPUTPCStartHitsFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliGPUTPCStartHitsFinder.h"
#include "AliGPUTPCTracker.h"
#include "AliTPCCommonMath.h"

template <> GPUd() void AliGPUTPCStartHitsFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &s, workerType &tracker)
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
	GPUglobalref() const MEM_GLOBAL(AliGPUTPCRow) &row = tracker.Row(s.fIRow);
	GPUglobalref() const MEM_GLOBAL(AliGPUTPCRow) &rowUp = tracker.Row(s.fIRow + 2);
	for (int ih = iThread; ih < s.fNHits; ih += nThreads)
	{
		if (tracker.HitLinkDownData(row, ih) == CALINK_INVAL && tracker.HitLinkUpData(row, ih) != CALINK_INVAL && tracker.HitLinkUpData(rowUp, tracker.HitLinkUpData(row, ih)) != CALINK_INVAL)
		{
#ifdef GPUCA_GPU_SORT_STARTHITS
			GPUglobalref() AliGPUTPCHitId *const startHits = tracker.TrackletTmpStartHits() + s.fIRow * GPUCA_GPU_MAX_ROWSTARTHITS;
			int nextRowStartHits = CAMath::AtomicAddShared(&s.fNRowStartHits, 1);
			if (nextRowStartHits >= GPUCA_GPU_MAX_TRACKLETS)
			{
				tracker.GPUParameters()->fGPUError = GPUCA_GPU_ERROR_TRACKLET_OVERFLOW;
				CAMath::AtomicExch(tracker.NTracklets(), 0);
			}
#else
			GPUglobalref() AliGPUTPCHitId *const startHits = tracker.TrackletStartHits();
			int nextRowStartHits = CAMath::AtomicAdd(tracker.NTracklets(), 1);
#endif
			startHits[nextRowStartHits].Set(s.fIRow, ih);
		}
	}
    GPUbarrier();
    
#ifdef GPUCA_GPU_SORT_STARTHITS
	if (iThread == 0)
	{
		int nOffset = CAMath::AtomicAdd(tracker.NTracklets(), s.fNRowStartHits);
#ifdef GPUCA_GPUCODE
		tracker.RowStartHitCountOffset()[s.fIRow] = s.fNRowStartHits;
		if (nOffset + s.fNRowStartHits >= GPUCA_GPU_MAX_TRACKLETS)
		{
			tracker.GPUParameters()->fGPUError = GPUCA_GPU_ERROR_TRACKLET_OVERFLOW;
			CAMath::AtomicExch(tracker.NTracklets(), 0);
		}
#endif
	}
#endif
}
