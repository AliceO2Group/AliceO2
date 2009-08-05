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

GPUd() void AliHLTTPCCAStartHitsFinder::Thread
( int /*nBlocks*/, int nThreads, int iBlock, int iThread, int iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  // find start hits for tracklets

  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
      if ( iBlock == 0 ) {
        CAMath::AtomicExch( tracker.NTracklets(), 0 );
      }
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock + 1;
      s.fNRowStartHits = 0;
      if ( s.fIRow <= s.fNRows - 4 ) {
        s.fNHits = tracker.Row( s.fIRow ).NHits();
        if ( s.fNHits >= ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS ) s.fNHits = ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS - 1;
      } else s.fNHits = -1;
    }
  } else if ( iSync == 1 ) {
    const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
    for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
      if ( ( tracker.HitLinkDownData( row, ih ) < 0 ) && ( tracker.HitLinkUpData( row, ih ) >= 0 ) ) {
        int oldNRowStartHits = CAMath::AtomicAdd( &s.fNRowStartHits, 1 );
        s.fRowStartHits[oldNRowStartHits].Set( s.fIRow, ih );
      }
    }
  } else if ( iSync == 2 ) {
    if ( iThread == 0 ) {
	  int nOffset = CAMath::AtomicAdd( tracker.NTracklets(), s.fNRowStartHits );
      s.fNOldStartHits = nOffset;
#ifdef HLTCA_GPU_SORT_STARTHITS
	  tracker.RowStartHitCountOffset()[s.fIRow].x = s.fNRowStartHits;
	  tracker.RowStartHitCountOffset()[s.fIRow].y = nOffset;
#endif
    }
  } else if ( iSync == 3 ) {
#ifdef HLTCA_GPU_SORT_STARTHITS
	AliHLTTPCCAHitId *const startHits = tracker.TrackletTmpStartHits();
#else
    AliHLTTPCCAHitId *const startHits = tracker.TrackletStartHits();
#endif
    for ( int ish = iThread; ish < s.fNRowStartHits; ish += nThreads ) {
      startHits[s.fNOldStartHits+ish] = s.fRowStartHits[ish];
    }
  }
}

