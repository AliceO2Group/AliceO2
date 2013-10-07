// @(#) $Id: AliHLTTPCCANeighboursCleaner.cxx 27042 2008-07-02 12:06:02Z richterm $
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


#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCATracker.h"

GPUdi() void AliHLTTPCCANeighboursCleaner::Thread
( int /*nBlocks*/, int nThreads, int iBlock, int iThread, int iSync,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker )
{
  // *
  // * kill link to the neighbour if the neighbour is not pointed to the cluster
  // *

  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock + 2;
      if ( s.fIRow <= s.fNRows - 3 ) {
        s.fIRowUp = s.fIRow + 2;
        s.fIRowDn = s.fIRow - 2;
        s.fNHits = tracker.Row( s.fIRow ).NHits();
      }
    }
  } else if ( iSync == 1 ) {
    if ( s.fIRow <= s.fNRows - 3 ) {
#ifdef HLTCA_GPUCODE
      int Up = s.fIRowUp;
      int Dn = s.fIRowDn;
      GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( s.fIRow );
      GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowUp = tracker.Row( Up );
      GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowDn = tracker.Row( Dn );
#else
      const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
      const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRowUp );
      const AliHLTTPCCARow &rowDn = tracker.Row( s.fIRowDn );
#endif

      // - look at up link, if it's valid but the down link in the row above doesn't link to us remove
      //   the link
      // - look at down link, if it's valid but the up link in the row below doesn't link to us remove
      //   the link
      for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
        int up = tracker.HitLinkUpData( row, ih );
        if ( up >= 0 ) {
          short upDn = tracker.HitLinkDownData( rowUp, up );
          if ( ( upDn != ih ) ) tracker.SetHitLinkUpData( row, ih, -1 );
        }
        int dn = tracker.HitLinkDownData( row, ih );
        if ( dn >= 0 ) {
          short dnUp = tracker.HitLinkUpData( rowDn, dn );
          if ( dnUp != ih ) tracker.SetHitLinkDownData( row, ih, -1 );
        }
      }
    }
  }
}

