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
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker )
{
  // find start hits for tracklets

  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock + 1;
      s.fNRowStartHits = 0;
      if ( s.fIRow <= s.fNRows - 4 ) {
        s.fNHits = tracker.Row( s.fIRow ).NHits();
      } else s.fNHits = -1;
    }
  } else if ( iSync == 1 ) {
    GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( s.fIRow );
    GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowUp = tracker.Row( s.fIRow + 2 );
    for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
      if (tracker.HitLinkDownData(row, ih) == CALINK_INVAL && tracker.HitLinkUpData(row, ih) != CALINK_INVAL && tracker.HitLinkUpData(rowUp, tracker.HitLinkUpData(row, ih)) != CALINK_INVAL) {
#ifdef HLTCA_GPU_SORT_STARTHITS
        GPUglobalref() AliHLTTPCCAHitId *const startHits = tracker.TrackletTmpStartHits() + s.fIRow * HLTCA_GPU_MAX_ROWSTARTHITS;
        int nextRowStartHits = CAMath::AtomicAddShared( &s.fNRowStartHits, 1 );
        if (nextRowStartHits >= HLTCA_GPU_MAX_TRACKLETS)
        {
          tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_TRACKLET_OVERFLOW;
          CAMath::AtomicExch( tracker.NTracklets(), 0 );
        }
#else
        GPUglobalref() AliHLTTPCCAHitId *const startHits = tracker.TrackletStartHits();
        int nextRowStartHits = CAMath::AtomicAdd( tracker.NTracklets(), 1 );
#endif
        startHits[nextRowStartHits].Set( s.fIRow, ih );
      }
    }
  } else if ( iSync == 2 ) {
#ifdef HLTCA_GPU_SORT_STARTHITS
    if ( iThread == 0 ) {
      int nOffset = CAMath::AtomicAdd( tracker.NTracklets(), s.fNRowStartHits );
#ifdef HLTCA_GPUCODE
      tracker.RowStartHitCountOffset()[s.fIRow] = s.fNRowStartHits;
      if (nOffset + s.fNRowStartHits >= HLTCA_GPU_MAX_TRACKLETS)
      {
        tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_TRACKLET_OVERFLOW;
        CAMath::AtomicExch( tracker.NTracklets(), 0 );
      }
#endif
    }
#endif
  }  
}
