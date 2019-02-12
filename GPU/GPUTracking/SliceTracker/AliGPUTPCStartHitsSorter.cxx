// @(#) $Id: AliGPUTPCStartHitsFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: David Rohr <drohr@kip.uni-heidelberg.de>				*
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

#include "AliGPUTPCStartHitsSorter.h"
#include "AliGPUTPCTracker.h"

template <> GPUd() void AliGPUTPCStartHitsSorter::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &s, workerType &tracker)
{
	//Sorts the Start Hits by Row Index
	if ( iThread == 0 ) {
		const int tmpNRows = GPUCA_ROW_COUNT - 6;
		const int nRows = iBlock == (nBlocks - 1) ? (tmpNRows - (tmpNRows / nBlocks) * (nBlocks - 1)) : (tmpNRows / nBlocks);
		const int nStartRow = (tmpNRows / nBlocks) * iBlock + 1;
		int startOffset2 = 0;

		for (int ir = 1;ir < GPUCA_ROW_COUNT - 5;ir++)
		{
			if (ir < nStartRow) startOffset2 += tracker.RowStartHitCountOffset()[ir];
		}
		s.fStartOffset = startOffset2;
		s.fNRows = nRows;
		s.fStartRow = nStartRow;
	}
    GPUbarrier();
    
	int startOffset = s.fStartOffset;
	for (int ir = 0;ir < s.fNRows;ir++)
	{
		GPUglobalref() AliGPUTPCHitId *const startHits = tracker.TrackletStartHits();
		GPUglobalref() AliGPUTPCHitId *const tmpStartHits = tracker.TrackletTmpStartHits() + (s.fStartRow + ir) * GPUCA_GPUCA_MAX_ROWSTARTHITS;
		const int tmpLen = tracker.RowStartHitCountOffset()[ir + s.fStartRow];			//Length of hits in row stored by StartHitsFinder

		for (int j = iThread;j < tmpLen;j += nThreads)
		{
			startHits[startOffset + j] = tmpStartHits[j];
		}
		startOffset += tmpLen;
	}
}
