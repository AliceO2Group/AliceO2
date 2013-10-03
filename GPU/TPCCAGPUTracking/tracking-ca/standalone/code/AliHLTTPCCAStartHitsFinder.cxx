// @(#) $Id: AliHLTTPCCAStartHitsFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAMath.h"

GPUdi() void AliHLTTPCCAStartHitsFinder::Thread
( int /*nBlocks*/, int nThreads, int iBlock, int iThread, int iSync,
  GPUsharedref() AliHLTTPCCASharedMemory MEM_LOCAL &s, GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &tracker )
{
  // find start hits for tracklets

  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock + 1;
      s.fNRowStartHits = 0;
      if ( s.fIRow <= s.fNRows - 4 ) {
        s.fNHits = tracker.Row( s.fIRow ).NHits();
        if ( s.fNHits >= ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS ) s.fNHits = ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS - 1;
      } else s.fNHits = -1;
    }
  } else if ( iSync == 1 ) {
#ifdef HLTCA_GPUCODE
    GPUsharedref() volatile int *xxx = &(s.fIRow);
    GPUglobalref() const AliHLTTPCCARow MEM_GLOBAL &row = tracker.Row( *xxx );
	GPUglobalref() const AliHLTTPCCARow MEM_GLOBAL &rowUp = tracker.Row( (*xxx) + 2 );
#else
    const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
	const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRow + 2 );
#endif
    for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
      if (tracker.HitLinkDownData(row, ih) < 0 && tracker.HitLinkUpData(row, ih) >= 0 && tracker.HitLinkUpData(rowUp, tracker.HitLinkUpData(row, ih)) >= 0) {
        int oldNRowStartHits = CAMath::AtomicAddShared( &s.fNRowStartHits, 1 );
#ifdef HLTCA_GPUCODE
        s.fRowStartHits[oldNRowStartHits].Set( *xxx, ih );
#else
        s.fRowStartHits[oldNRowStartHits].Set( s.fIRow, ih );
#endif
      }
    }
  } else if ( iSync == 2 ) {
    if ( iThread == 0 ) {
	  int nOffset = CAMath::AtomicAdd( tracker.NTracklets(), s.fNRowStartHits );
#ifdef HLTCA_GPUCODE
	  if (nOffset + s.fNRowStartHits >= HLTCA_GPU_MAX_TRACKLETS)
	  {
		tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_TRACKLET_OVERFLOW;
		CAMath::AtomicExch( tracker.NTracklets(), 0 );
		nOffset = 0;
	  }
#endif
      s.fNOldStartHits = nOffset;
#ifdef HLTCA_GPU_SORT_STARTHITS
      GPUsharedref() volatile int *yyy = &(s.fIRow);
	  tracker.RowStartHitCountOffset()[*yyy].x = s.fNRowStartHits;
	  tracker.RowStartHitCountOffset()[*yyy].y = nOffset;
#endif
    }
  } else if ( iSync == 3 ) {
#ifdef HLTCA_GPU_SORT_STARTHITS
	GPUglobalref() AliHLTTPCCAHitId *const startHits = tracker.TrackletTmpStartHits();
#else
    GPUglobalref() AliHLTTPCCAHitId *const startHits = tracker.TrackletStartHits();
#endif
    for ( int ish = iThread; ish < s.fNRowStartHits; ish += nThreads ) {
      startHits[s.fNOldStartHits+ish] = s.fRowStartHits[ish];
    }
  }
}

