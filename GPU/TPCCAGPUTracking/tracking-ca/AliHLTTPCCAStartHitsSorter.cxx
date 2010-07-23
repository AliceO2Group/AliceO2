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
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
	//Sorts the Start Hits by Row Index and create RowBlock Data
  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
		const int gpuFixedBlockCount = tracker.GPUParametersConst()->fGPUFixedBlockCount;
	  const int tmpNRows = tracker.Param().NRows() - 6;
	  int nRows = iBlock == 29 ? (tmpNRows - (tmpNRows / 30) * 29) : (tmpNRows / 30);
	  int nStartRow = (tmpNRows / 30) * iBlock + 1;
      int startOffset = 0;
	  int startOffset2 = 0;
	  int previousBlockEndTracklet = 0;
	  int nCurrentBlock = 0;

      for (int ir = 1;ir < tracker.Param().NRows() - 5;ir++)
	  {
	    if (ir < nStartRow)
			startOffset2 += tracker.RowStartHitCountOffset()[ir].x;

		if (iBlock == nBlocks - 1 && nCurrentBlock < gpuFixedBlockCount)
		{
			startOffset += tracker.RowStartHitCountOffset()[ir].x;
			for (int i = previousBlockEndTracklet + HLTCA_GPU_THREAD_COUNT;i <= startOffset;i += HLTCA_GPU_THREAD_COUNT)
			{
				if (previousBlockEndTracklet / HLTCA_GPU_THREAD_COUNT != i / HLTCA_GPU_THREAD_COUNT)
				{
					tracker.BlockStartingTracklet()[nCurrentBlock].x = previousBlockEndTracklet;
					tracker.BlockStartingTracklet()[nCurrentBlock++].y = HLTCA_GPU_THREAD_COUNT;
					previousBlockEndTracklet += HLTCA_GPU_THREAD_COUNT;
					if (nCurrentBlock == gpuFixedBlockCount)
					{
						break;
					}
				}
			}
			if ((ir + 1) % HLTCA_GPU_SCHED_ROW_STEP == 0 && nCurrentBlock < gpuFixedBlockCount)
			{
				if (previousBlockEndTracklet != startOffset)
				{
					tracker.BlockStartingTracklet()[nCurrentBlock].x = previousBlockEndTracklet;
					tracker.BlockStartingTracklet()[nCurrentBlock++].y = startOffset - previousBlockEndTracklet;
					previousBlockEndTracklet = startOffset;
				}
			}
			if (nCurrentBlock == gpuFixedBlockCount)
			{
				tracker.GPUParameters()->fScheduleFirstDynamicTracklet = previousBlockEndTracklet;
			}
		}
	  }
	  if (iBlock == nBlocks - 1)
	  {
	    if (nCurrentBlock < gpuFixedBlockCount)
		{
			tracker.BlockStartingTracklet()[nCurrentBlock].x = previousBlockEndTracklet;
			tracker.BlockStartingTracklet()[nCurrentBlock++].y = startOffset - previousBlockEndTracklet;
			tracker.GPUParameters()->fScheduleFirstDynamicTracklet = startOffset;
		}
		for (int i = nCurrentBlock;i < HLTCA_GPU_BLOCK_COUNT;i++)
		{
			tracker.BlockStartingTracklet()[i].x = 0;
			tracker.BlockStartingTracklet()[i].y = 0;			
		}
	  }
	  s.fStartOffset = startOffset2;
	  s.fNRows = nRows;
      s.fStartRow = nStartRow;
    }
  } else if ( iSync == 1 ) {
	int startOffset = s.fStartOffset;
    for (int ir = 0;ir < s.fNRows;ir++)
	{
		AliHLTTPCCAHitId *const startHits = tracker.TrackletStartHits();
		AliHLTTPCCAHitId *const tmpStartHits = tracker.TrackletTmpStartHits();
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

