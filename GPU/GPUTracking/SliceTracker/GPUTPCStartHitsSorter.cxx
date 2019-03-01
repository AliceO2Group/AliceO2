// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCStartHitsSorter.cxx
/// \author David Rohr

#include "GPUTPCStartHitsSorter.h"
#include "GPUTPCTracker.h"

template <> GPUd() void GPUTPCStartHitsSorter::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &s, workerType &tracker)
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
		GPUglobalref() GPUTPCHitId *const startHits = tracker.TrackletStartHits();
		GPUglobalref() GPUTPCHitId *const tmpStartHits = tracker.TrackletTmpStartHits() + (s.fStartRow + ir) * GPUCA_MAX_ROWSTARTHITS;
		const int tmpLen = tracker.RowStartHitCountOffset()[ir + s.fStartRow];			//Length of hits in row stored by StartHitsFinder

		for (int j = iThread;j < tmpLen;j += nThreads)
		{
			startHits[startOffset + j] = tmpStartHits[j];
		}
		startOffset += tmpLen;
	}
}
