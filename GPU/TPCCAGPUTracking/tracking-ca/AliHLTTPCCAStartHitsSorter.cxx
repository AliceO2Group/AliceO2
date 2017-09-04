// @(#) $Id: AliHLTTPCCAStartHitsFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCAStartHitsSorter.h"
#include "AliHLTTPCCATracker.h"

GPUd() void AliHLTTPCCAStartHitsSorter::Thread
( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
 GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker )
{
	//Sorts the Start Hits by Row Index and create RowBlock Data
	if ( iSync == 0 ) {
		if ( iThread == 0 ) {
			const int gpuFixedBlockCount = tracker.GPUParametersConst()->fGPUFixedBlockCount;
			const int tmpNRows = tracker.Param().NRows() - 6;
			const int nRows = iBlock == (nBlocks - 1) ? (tmpNRows - (tmpNRows / nBlocks) * (nBlocks - 1)) : (tmpNRows / nBlocks);
			const int nStartRow = (tmpNRows / nBlocks) * iBlock + 1;
			int startOffset2 = 0;

			for (int ir = 1;ir < tracker.Param().NRows() - 5;ir++)
			{
				if (ir < nStartRow) startOffset2 += tracker.RowStartHitCountOffset()[ir].x;
			}
			s.fStartOffset = startOffset2;
			s.fNRows = nRows;
			s.fStartRow = nStartRow;
		}
	} else if ( iSync == 1 ) {
		int startOffset = s.fStartOffset;
		for (int ir = 0;ir < s.fNRows;ir++)
		{
			GPUglobalref() AliHLTTPCCAHitId *const startHits = tracker.TrackletStartHits();
			GPUglobalref() AliHLTTPCCAHitId *const tmpStartHits = tracker.TrackletTmpStartHits();
			const int tmpLen = tracker.RowStartHitCountOffset()[ir + s.fStartRow].x;			//Length of hits in row stored by StartHitsFinder
			const int tmpOffset = tracker.RowStartHitCountOffset()[ir + s.fStartRow].y;			//Offset of first hit in row of unsorted array by StartHitsFinder
			if (iThread == 0)
				tracker.RowStartHitCountOffset()[ir + s.fStartRow].z = startOffset;				//Store New Offset Value of sorted array

			for (int j = iThread;j < tmpLen;j += nThreads)
			{
				startHits[startOffset + j] = tmpStartHits[tmpOffset + j];
			}
			startOffset += tmpLen;
		}
	}
}
