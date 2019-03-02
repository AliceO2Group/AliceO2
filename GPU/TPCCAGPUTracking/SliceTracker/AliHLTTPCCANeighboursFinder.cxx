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
#include "AliHLTTPCCAHit.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#endif //DRAW

GPUdi() void AliHLTTPCCANeighboursFinder::Thread
( int /*nBlocks*/, int nThreads, int iBlock, int iThread, int iSync,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker )
{
  //* find neighbours

  if ( iSync == 0 ) {
#ifdef HLTCA_GPUCODE
	for (int i = iThread;i < sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += nThreads)
	{
		reinterpret_cast<GPUsharedref() int*>(&s.fRow)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock])[i];
		if (iBlock >= 2 && iBlock <= tracker.Param().NRows() - 3)
		{
			reinterpret_cast<GPUsharedref() int*>(&s.fRowUp)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock + 2])[i];
			reinterpret_cast<GPUsharedref() int*>(&s.fRowDown)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock - 2])[i];
		}
	}
	GPUsync();
#endif
    if ( iThread == 0 ) {
      s.fNRows = tracker.Param().NRows();
      s.fIRow = iBlock;
      if ( s.fIRow < s.fNRows ) {
#ifdef HLTCA_GPUCODE
		GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &row = s.fRow;
#else
		GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( s.fIRow );
#endif
        s.fNHits = row.NHits();

        if ( ( s.fIRow >= 2 ) && ( s.fIRow <= s.fNRows - 3 ) ) {
          s.fIRowUp = s.fIRow + 2;
          s.fIRowDn = s.fIRow - 2;

          // references to the rows above and below

#ifdef HLTCA_GPUCODE
          GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &rowUp = s.fRowUp;
          GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &rowDn = s.fRowDown;
#else
          GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowUp = tracker.Row( s.fIRowUp );
          GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowDn = tracker.Row( s.fIRowDn );
#endif
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

          //s.fGridUp = tracker.Row( s.fIRowUp ).Grid();
          //s.fGridDn = tracker.Row( s.fIRowDn ).Grid();
        }
      }
    }
  } else if ( iSync == 1 ) {
    if ( s.fIRow < s.fNRows ) {
      if ( ( s.fIRow <= 1 ) || ( s.fIRow >= s.fNRows - 2 ) ) {
#ifdef HLTCA_GPUCODE
		GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &row = s.fRow;
#else
		GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( s.fIRow );
#endif
        for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {
          tracker.SetHitLinkUpData( row, ih, CALINK_INVAL );
          tracker.SetHitLinkDownData( row, ih, CALINK_INVAL );
        }
      } else {
/*#ifdef HLTCA_GPUCODE
          const AliHLTTPCCARow &rowUp = s.fRowUp;
          const AliHLTTPCCARow &rowDn = s.fRowDown;
#else
          const AliHLTTPCCARow &rowUp = tracker.Row( s.fIRowUp );
          const AliHLTTPCCARow &rowDn = tracker.Row( s.fIRowDn );
#endif

        for ( unsigned int ih = iThread; ih < s.fGridUp.N() + s.fGridUp.Ny() + 2; ih += nThreads ) {
          s.fGridContentUp[ih] = tracker.FirstHitInBin( rowUp, ih );
        }
        for ( unsigned int ih = iThread; ih < s.fGridDn.N() + s.fGridDn.Ny() + 2; ih += nThreads ) {
          s.fGridContentDn[ih] = tracker.FirstHitInBin( rowDn, ih );
        }*/
      }
    }
  } else if ( iSync == 2 ) {

#ifdef HLTCA_GPUCODE
    if ( ( iBlock <= 1 ) || ( iBlock >= s.fNRows - 2 ) ) return;
#else
    if ( ( s.fIRow <= 1 ) || ( s.fIRow >= s.fNRows - 2 ) ) return;
#endif

    float chi2Cut = 3.*3.*4 * ( s.fUpDx * s.fUpDx + s.fDnDx * s.fDnDx );
    //float chi2Cut = 3.*3.*(s.fUpDx*s.fUpDx + s.fDnDx*s.fDnDx ); //SG
#ifdef HLTCA_GPUCODE
		  GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &row = s.fRow;
          GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &rowUp = s.fRowUp;
          GPUsharedref() const MEM_LOCAL(AliHLTTPCCARow) &rowDn = s.fRowDown;
#else
		  GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( s.fIRow );
		  GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowUp = tracker.Row( s.fIRowUp );
          GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow) &rowDn = tracker.Row( s.fIRowDn );
#endif
    const float y0 = row.Grid().YMin();
    const float z0 = row.Grid().ZMin();
    const float stepY = row.HstepY();
    const float stepZ = row.HstepZ();

	for ( int ih = iThread; ih < s.fNHits; ih += nThreads ) {

      int linkUp = -1;
      int linkDn = -1;

      if ( s.fDnNHits > 0 && s.fUpNHits > 0 ) {

       

        // coordinates of the hit in the current row
#if defined(HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS)
		cahit2 tmpval = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(cahit2) + row.HitNumberOffset() + ih);
        const float y = y0 + tmpval.x * stepY;
        const float z = z0 + tmpval.y * stepZ;
#else
        const float y = y0 + tracker.HitDataY( row, ih ) * stepY;
        const float z = z0 + tracker.HitDataZ( row, ih ) * stepZ;
#endif //HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS

#if ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
#if defined(HLTCA_GPUCODE)
      GPUsharedref() calink *neighUp = s.fB[iThread];
      GPUsharedref() float2 *yzUp = s.fA[iThread];
#else
      GPUsharedref() calink *neighUp = s.fB;
      GPUsharedref() float2 *yzUp = s.fA;
#endif
#if defined(HLTCA_GPUCODE) & HLTCA_GPU_MAXN > ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP
	  calink neighUp2[HLTCA_GPU_MAXN - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP];
	  float2 yzUp2[HLTCA_GPU_MAXN - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP];
#endif
#else
      calink neighUp[HLTCA_GPU_MAXN];
      float2 yzUp[HLTCA_GPU_MAXN];
#endif //ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0

		int nNeighUp = 0;
        AliHLTTPCCAHitArea areaDn, areaUp;

        const float kAngularMultiplier = tracker.Param().GetSearchWindowDZDR();
        const float kAreaSize = tracker.Param().NeighboursSearchArea();
        areaUp.Init( rowUp, tracker.Data(), y*s.fUpTx, kAngularMultiplier != 0. ? z : (z*s.fUpTx), kAreaSize, kAngularMultiplier != 0 ? (s.fUpDx * kAngularMultiplier) : kAreaSize);
        areaDn.Init( rowDn, tracker.Data(), y*s.fDnTx, kAngularMultiplier != 0. ? z : (z*s.fDnTx), kAreaSize, kAngularMultiplier != 0 ? (-s.fDnDx * kAngularMultiplier) : kAreaSize);

        do {
          AliHLTTPCCAHit h;
          int i = areaUp.GetNext( tracker, rowUp, tracker.Data(), &h );
          if ( i < 0 ) break;
#if defined(HLTCA_GPUCODE) & HLTCA_GPU_MAXN > ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP & ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
		  if (nNeighUp >= ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP)
		  {
			neighUp2[nNeighUp - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP] = ( calink ) i;
			yzUp2[nNeighUp - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP] = CAMath::MakeFloat2( s.fDnDx * ( h.Y() - y ), s.fDnDx * ( h.Z() - z ) );
		  }
		  else
#endif
		  {
			neighUp[nNeighUp] = ( calink ) i;
			yzUp[nNeighUp] = CAMath::MakeFloat2( s.fDnDx * ( h.Y() - y ), s.fDnDx * ( h.Z() - z ) );
		  }
          if ( ++nNeighUp >= HLTCA_GPU_MAXN )
          {
              //printf("Neighbors buffer ran full...\n");
              break;
          }
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
#if defined(HLTCA_GPUCODE) & HLTCA_GPU_MAXN > ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP & ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
			  float2 yzup = iUp >= ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP ? yzUp2[iUp - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP] : yzUp[iUp];
#else
              float2 yzup = yzUp[iUp];
#endif

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
#if defined(HLTCA_GPUCODE) & HLTCA_GPU_MAXN > ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP & ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
			linkUp = bestUp >= ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP ? neighUp2[bestUp - ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP] : neighUp[bestUp];
#else
            linkUp = neighUp[bestUp];
#endif
            linkDn = bestDn;
          }
        }
#ifdef DRAW
        std::cout << "n NeighUp = " << nNeighUp << ", n NeighDn = " << nNeighDn << std::endl;
#endif //DRAW
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
#endif //DRAW
    }
  }
}
