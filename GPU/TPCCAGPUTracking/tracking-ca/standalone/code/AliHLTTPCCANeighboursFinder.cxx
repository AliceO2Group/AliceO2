// @(#) $Id: AliHLTTPCCANeighboursFinder1.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCATracker.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#endif

GPUd() void AliHLTTPCCANeighboursFinder::Thread
( int /*nBlocks*/, int nThreads, int iBlock, int iThread, int iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  //* find neighbours

  if ( iSync == 0 ) {
    if ( iThread == 0 ) {
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock;
      if ( s.fIRow < s.fNRows ) {
        const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
        s.fNHits = row.NHits();

        if ( ( s.fIRow >= 2 ) && ( s.fIRow <= s.fNRows - 3 ) ) {
          s.fIRowUp = s.fIRow + 2;
          s.fIRowDn = s.fIRow - 2;

          // references to the rows above and below
          const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRowUp );
          const AliHLTTPCCARow &rowDn = tracker.Row( s.fIRowDn );

          // the axis perpendicular to the rows
          const float xDn = rowDn.X();
          const float x   = row.X();
          const float xUp = rowUp.X();

          // number of hits in rows above and below
          s.fUpNHits = tracker.Row( s.fIRowUp ).NHits();
          s.fDnNHits = tracker.Row( s.fIRowDn ).NHits();

          // distance of the rows (absolute and relative)
          s.fUpDx = xUp - x;
          s.fDnDx = xDn - x;
          s.fUpTx = xUp / x;
          s.fDnTx = xDn / x;
          // UpTx/DnTx is used to move the HitArea such that central events are preferred (i.e. vertices
          // coming from y = 0, z = 0).

          s.fGridUp = tracker.Row( s.fIRowUp ).Grid();
          s.fGridDn = tracker.Row( s.fIRowDn ).Grid();
        }
      }
    }
  } else if ( iSync == 1 ) {
    if ( s.fIRow < s.fNRows ) {
      if ( ( s.fIRow == 0 ) || ( s.fIRow == s.fNRows - 1 ) || ( s.fIRow == 1 ) || ( s.fIRow == s.fNRows - 2 ) ) {
        const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
        for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
          tracker.SetHitLinkUpData( row, ih, -1 );
          tracker.SetHitLinkDownData( row, ih, -1 );
        }
      } else {
        const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRowUp );
        const AliHLTTPCCARow &rowDn = tracker.Row( s.fIRowDn );

        for ( unsigned int ih = iThread; ih < s.fGridUp.N() + s.fGridUp.Ny() + 2; ih += nThreads ) {
          s.fGridContentUp[ih] = tracker.FirstHitInBin( rowUp, ih );
        }
        for ( unsigned int ih = iThread; ih < s.fGridDn.N() + s.fGridDn.Ny() + 2; ih += nThreads ) {
          s.fGridContentDn[ih] = tracker.FirstHitInBin( rowDn, ih );
        }
      }
    }
  } else if ( iSync == 2 ) {
    if ( ( s.fIRow <= 1 ) || ( s.fIRow >= s.fNRows - 2 ) ) return;

    //const float kAreaSize = 3;
    float chi2Cut = 3.*3.*4 * ( s.fUpDx * s.fUpDx + s.fDnDx * s.fDnDx );
    const float kAreaSize = 3;
    //float chi2Cut = 3.*3.*(s.fUpDx*s.fUpDx + s.fDnDx*s.fDnDx ); //SG
    const int kMaxN = 20;

    const AliHLTTPCCARow &row = tracker.Row( s.fIRow );
    const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRowUp );
    const AliHLTTPCCARow &rowDn = tracker.Row( s.fIRowDn );
    const float y0 = row.Grid().YMin();
    const float z0 = row.Grid().ZMin();
    const float stepY = row.HstepY();
    const float stepZ = row.HstepZ();

    for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {

      unsigned short *neighUp = s.fB[iThread];
      float2 *yzUp = s.fA[iThread];
      //unsigned short neighUp[5];
      //float2 yzUp[5];

      int linkUp = -1;
      int linkDn = -1;

      if ( s.fDnNHits > 0 && s.fUpNHits > 0 ) {

        int nNeighUp = 0;

        // coordinates of the hit in the current row
        const float y = y0 + tracker.HitDataY( row, ih ) * stepY;
        const float z = z0 + tracker.HitDataZ( row, ih ) * stepZ;

        AliHLTTPCCAHitArea areaDn, areaUp;
        // TODO: for NVIDIA GPUs it should use the GridContentUp/-Dn that got copied into shared mem
        areaUp.Init( rowUp, tracker.Data(), y*s.fUpTx, z*s.fUpTx, kAreaSize, kAreaSize );
        areaDn.Init( rowDn, tracker.Data(), y*s.fDnTx, z*s.fDnTx, kAreaSize, kAreaSize );

        do {
          AliHLTTPCCAHit h;
          int i = areaUp.GetNext( tracker, rowUp, tracker.Data(), &h );
          if ( i < 0 ) break;
          neighUp[nNeighUp] = ( unsigned short ) i;
          yzUp[nNeighUp] = CAMath::MakeFloat2( s.fDnDx * ( h.Y() - y ), s.fDnDx * ( h.Z() - z ) );
          if ( ++nNeighUp >= kMaxN ) break;
        } while ( 1 );

        int nNeighDn = 0;

        if ( nNeighUp > 0 ) {

          int bestDn = -1, bestUp = -1;
          float bestD = 1.e10;

          do {
            AliHLTTPCCAHit h;
            int i = areaDn.GetNext( tracker, rowDn, tracker.Data(), &h );
            if ( i < 0 ) break;

            nNeighDn++;
            float2 yzdn = CAMath::MakeFloat2( s.fUpDx * ( h.Y() - y ), s.fUpDx * ( h.Z() - z ) );

            for ( int iUp = 0; iUp < nNeighUp; iUp++ ) {
              float2 yzup = yzUp[iUp];
              float dy = yzdn.x - yzup.x;
              float dz = yzdn.y - yzup.y;
              float d = dy * dy + dz * dz;
              if ( d < bestD ) {
                bestD = d;
                bestDn = i;
                bestUp = iUp;
              }
            }
          } while ( 1 );

          if ( bestD <= chi2Cut ) {
            linkUp = neighUp[bestUp];
            linkDn = bestDn;
          }
        }
#ifdef DRAW
        std::cout << "n NeighUp = " << nNeighUp << ", n NeighDn = " << nNeighDn << std::endl;
#endif

      }

      tracker.SetHitLinkUpData( row, ih, linkUp );
      tracker.SetHitLinkDownData( row, ih, linkDn );
#ifdef DRAW
      std::cout << "Links for row " << s.fIRow << ", hit " << ih << ": " << linkUp << " " << linkDn << std::endl;
      if ( s.fIRow == 22 && ih == 5 ) {
        AliHLTTPCCADisplay::Instance().DrawSliceLink( s.fIRow, ih, -1, -1, 1 );
        AliHLTTPCCADisplay::Instance().DrawSliceHit( s.fIRow, ih, kBlue, 1. );
        AliHLTTPCCADisplay::Instance().Ask();
      }
#endif
    }
  }
}

