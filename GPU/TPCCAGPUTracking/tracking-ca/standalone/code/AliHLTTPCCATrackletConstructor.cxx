// @(#) $Id: AliHLTTPCCATrackletConstructor.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATracklet.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "MemoryAssignmentHelpers.h"

//#include "AliHLTTPCCAPerformance.h"
//#include "TH1D.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#endif


GPUd() void AliHLTTPCCATrackletConstructor::Step0
( int nBlocks, int /*nThreads*/, int iBlock, int iThread,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &/*tParam*/ )
{
  // reconstruction of tracklets, step 0

  r.fIsMemThread = ( iThread < TRACKLET_CONSTRUCTOR_NMEMTHREDS );
  if ( iThread == 0 ) {
    int nTracks = *tracker.NTracklets();
    int nTrPerBlock = nTracks / nBlocks + 1;
    s.fNRows = tracker.Param().NRows();
    s.fItr0 = nTrPerBlock * iBlock;
    s.fItr1 = s.fItr0 + nTrPerBlock;
    if ( s.fItr1 > nTracks ) s.fItr1 = nTracks;
    s.fMinStartRow = 158;
    s.fMaxEndRow = 0;
  }
  if ( iThread < 32 ) {
    s.fMinStartRow32[iThread] = 158;
  }
}


GPUd() void AliHLTTPCCATrackletConstructor::Step1
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int iThread,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam )
{
  // reconstruction of tracklets, step 1

  r.fItr = s.fItr0 + ( iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS );
  r.fGo = ( !r.fIsMemThread ) && ( r.fItr < s.fItr1 );
  r.fSave = r.fGo;
  r.fNHits = 0;

  if ( !r.fGo ) return;

  r.fStage = 0;

  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];

  unsigned int kThread = iThread % 32;//& 00000020;
  if ( SAVE() ) for ( int i = 0; i < 160; i++ ) tracklet.SetRowHit( i, -1 );

  AliHLTTPCCAHitId id = tracker.TrackletStartHits()[r.fItr];
  r.fStartRow = id.RowIndex();
  r.fEndRow = r.fStartRow;
  r.fFirstRow = r.fStartRow;
  r.fLastRow = r.fFirstRow;
  r.fCurrIH =  id.HitIndex();

  CAMath::AtomicMin( &s.fMinStartRow32[kThread], r.fStartRow );
  tParam.SetSinPhi( 0 );
  tParam.SetDzDs( 0 );
  tParam.SetQPt( 0 );
  tParam.SetSignCosPhi( 1 );
  tParam.SetChi2( 0 );
  tParam.SetNDF( -3 );
  tParam.SetCov( 0, 1 );
  tParam.SetCov( 1, 0 );
  tParam.SetCov( 2, 1 );
  tParam.SetCov( 3, 0 );
  tParam.SetCov( 4, 0 );
  tParam.SetCov( 5, 1 );
  tParam.SetCov( 6, 0 );
  tParam.SetCov( 7, 0 );
  tParam.SetCov( 8, 0 );
  tParam.SetCov( 9, 1 );
  tParam.SetCov( 10, 0 );
  tParam.SetCov( 11, 0 );
  tParam.SetCov( 12, 0 );
  tParam.SetCov( 13, 0 );
  tParam.SetCov( 14, 10. );

}

GPUd() void AliHLTTPCCATrackletConstructor::Step2
( int /*nBlocks*/, int nThreads, int /*iBlock*/, int iThread,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &/*r*/, AliHLTTPCCATracker &/*tracker*/, AliHLTTPCCATrackParam &/*tParam*/ )
{
  // reconstruction of tracklets, step 2

  if ( iThread == 0 ) {
    //CAMath::AtomicMinGPU(&s.fMinRow, s.fMinRow32[iThread]);
    int minStartRow = 158;
    int n = ( nThreads > 32 ) ? 32 : nThreads;
    for ( int i = 0; i < n; i++ ) {
      if ( s.fMinStartRow32[i] < minStartRow ) minStartRow = s.fMinStartRow32[i];
    }
    s.fMinStartRow = minStartRow;
  }
}

GPUd() void AliHLTTPCCATrackletConstructor::ReadData
( int iThread, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, int iRow )
{
  // reconstruction of tracklets, read data step

  if ( r.fIsMemThread ) {
    const AliHLTTPCCARow &row = tracker.Row( iRow );
    bool jr = !r.fCurrentData;

    // copy hits, grid content and links

    // FIXME: inefficient copy
    const int numberOfHitsAligned = NextMultipleOf<8>(row.NHits());
    /*ushort2 *sMem1 = reinterpret_cast<ushort2 *>( s.fData[jr] );
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1[i].x = tracker.HitDataY( row, i );
      sMem1[i].y = tracker.HitDataZ( row, i );
    }*/

    ushort_v *sMem1 = reinterpret_cast<ushort_v *>( s.fData[jr] );
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1[i] = tracker.HitDataY( row, i );
    }

    ushort_v *sMem1a = reinterpret_cast<ushort_v *>( s.fData[jr] ) + numberOfHitsAligned;
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1a[i] = tracker.HitDataZ( row, i );
    }

    short *sMem2 = reinterpret_cast<short *>( s.fData[jr] ) + 2 * numberOfHitsAligned;
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem2[i] = tracker.HitLinkUpData( row, i );
    }
	
    unsigned short *sMem3 = reinterpret_cast<unsigned short *>( s.fData[jr] ) + 3 * numberOfHitsAligned;
    const int n = row.FullSize(); // + grid content size
    for ( int i = iThread; i < n; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem3[i] = tracker.FirstHitInBin( row, i );
    }

	/*for (int k = 0;k < 4;k++)		//Copy FirstHitInBint (k = 1, HitLinkUpData (k = 0) to shared memory
	{
		int* const sMem2 = reinterpret_cast<int *> (reinterpret_cast<short *>( s.fData[jr] ) + k * numberOfHitsAligned);
		const int* sourceMem;
		int copyCount;
		switch (k)
		{
		case 0:
			sourceMem = (int*) tracker.HitDataY(row);
			copyCount = numberOfHitsAligned * sizeof(short) / sizeof(*sMem2);
			break;
		case 1:
			sourceMem = (int*) tracker.HitDataZ(row);
			copyCount = numberOfHitsAligned * sizeof(short) / sizeof(*sMem2);			
			break;
		case 2:
			sourceMem = (int*) tracker.HitLinkUpData(row);
			copyCount = numberOfHitsAligned * sizeof(short) / sizeof(*sMem2);
			break;
		case 3:
			sourceMem = (int*) tracker.FirstHitInBin(row);
			copyCount = NextMultipleOf<8>(row.FullSize());
			break;
		}
		for (int i = iThread;i < copyCount;i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
		{
			sMem2[i] = sourceMem[i];
		}
	}*/

  }
}


GPUd() void AliHLTTPCCATrackletConstructor::StoreTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  AliHLTTPCCASharedMemory &/*s*/, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam )
{
  // reconstruction of tracklets, tracklet store step

  if ( !r.fSave ) return;

  //AliHLTTPCCAPerformance::Instance().HNHitsPerTrackCand()->Fill(r.fNHits);

  do {
    {
	//std::cout<<"tracklet to store: "<<r.fItr<<", nhits = "<<r.fNHits<<std::endl;
    }

    if ( r.fNHits < 5 ) {
      r.fNHits = 0;
      break;
    }

    if ( 0 ) {
      if ( 1. / .5 < CAMath::Abs( tParam.QPt() ) ) { //SG!!!
        r.fNHits = 0;
        break;
      }
    }

    {
      bool ok = 1;
      const float *c = tParam.Cov();
      for ( int i = 0; i < 15; i++ ) ok = ok && CAMath::Finite( c[i] );
      for ( int i = 0; i < 5; i++ ) ok = ok && CAMath::Finite( tParam.Par()[i] );
      ok = ok && ( tParam.X() > 50 );

      if ( c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0 ) ok = 0;

      if ( !ok ) {
        r.fNHits = 0;
        break;
      }
    }
  } while ( 0 );

  if ( !SAVE() ) return;

  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];

  tracklet.SetNHits( r.fNHits );

  if ( r.fNHits > 0 ) {
#ifdef DRAW
    if ( 0 ) {
      std::cout << "store tracklet " << r.fItr << ", nhits = " << r.fNHits << std::endl;
      if ( AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kBlue, 1. ) ) {
        AliHLTTPCCADisplay::Instance().Ask();
      }
    }
#endif
    if ( CAMath::Abs( tParam.Par()[4] ) < 1.e-4 ) tParam.SetPar( 4, 1.e-4 );
    tracklet.SetFirstRow( r.fFirstRow );
    tracklet.SetLastRow( r.fLastRow );
    tracklet.SetParam( tParam );
    int w = ( r.fNHits << 16 ) + r.fItr;
    for ( int iRow = 0; iRow < 160; iRow++ ) {
      int ih = tracklet.RowHit( iRow );
      if ( ih >= 0 ) {
        tracker.MaximizeHitWeight( tracker.Row( iRow ), ih, w );
      }
    }
  }
}

GPUd() void AliHLTTPCCATrackletConstructor::UpdateTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam, int iRow )
{
  // reconstruction of tracklets, tracklets update step

  //std::cout<<"Update tracklet: "<<r.fItr<<" "<<r.fGo<<" "<<r.fStage<<" "<<iRow<<std::endl;
  bool drawSearch = 0;//r.fItr==2;
  bool drawFit = 0;//r.fItr==2;
  bool drawFitted = drawFit ;//|| 1;//r.fItr==16;

  if ( !r.fGo ) return;

  const int kMaxRowGap = 4;

  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];

  const AliHLTTPCCARow &row = tracker.Row( iRow );

  float y0 = row.Grid().YMin();
  float stepY = row.HstepY();
  float z0 = row.Grid().ZMin();
  float stepZ = row.HstepZ();
  float stepYi = row.HstepYi();
  float stepZi = row.HstepZi();

  if ( r.fStage == 0 ) { // fitting part
    do {

      if ( iRow < r.fStartRow || r.fCurrIH < 0  ) break;

      if ( ( iRow - r.fStartRow ) % 2 != 0 ) break; // SG!!! - jump over the row

      uint4 *tmpint4 = s.fData[r.fCurrentData];
      ushort2 hh;// = reinterpret_cast<ushort2*>( tmpint4 )[r.fCurrIH];
	  hh.x = reinterpret_cast<ushort_v*>( tmpint4 )[r.fCurrIH];
	  hh.y = reinterpret_cast<ushort_v*>( tmpint4 )[NextMultipleOf<8>(row.NHits()) + r.fCurrIH];

      int oldIH = r.fCurrIH;
      r.fCurrIH = reinterpret_cast<short*>( tmpint4 )[2 * NextMultipleOf<8>(row.NHits()) + r.fCurrIH]; // read from linkup data

      float x = row.X();
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
#ifdef DRAW
      if ( drawFit ) std::cout << " fit tracklet: new hit " << oldIH << ", xyz=" << x << " " << y << " " << z << std::endl;
#endif

      if ( iRow == r.fStartRow ) {
        tParam.SetX( x );
        tParam.SetY( y );
        tParam.SetZ( z );
        r.fLastY = y;
        r.fLastZ = z;
        #ifdef DRAW
        if ( drawFit ) std::cout << " fit tracklet " << r.fItr << ", row " << iRow << " first row" << std::endl;
        #endif
      } else {

        float err2Y, err2Z;
        float dx = x - tParam.X();
        float dy = y - r.fLastY;//tParam.Y();
        float dz = z - r.fLastZ;//tParam.Z();
        r.fLastY = y;
        r.fLastZ = z;

        float ri = 1. / CAMath::Sqrt( dx * dx + dy * dy );
        if ( iRow == r.fStartRow + 2 ) { //SG!!! important - thanks to Matthias
          tParam.SetSinPhi( dy*ri );
          tParam.SetSignCosPhi( dx );
          tParam.SetDzDs( dz*ri );
          //std::cout << "Init. errors... " << r.fItr << std::endl;
          tracker.GetErrors2( iRow, tParam, err2Y, err2Z );
          //std::cout << "Init. errors = " << err2Y << " " << err2Z << std::endl;
          tParam.SetCov( 0, err2Y );
          tParam.SetCov( 2, err2Z );
        }
        if ( drawFit ) {
          #ifdef DRAW
          std::cout << " fit tracklet " << r.fItr << ", row " << iRow << " transporting.." << std::endl;
          std::cout << " params before transport=" << std::endl;
          tParam.Print();
          #endif
        }
        float sinPhi, cosPhi;
        if ( r.fNHits >= 10 && CAMath::Abs( tParam.SinPhi() ) < .99 ) {
          sinPhi = tParam.SinPhi();
          cosPhi = CAMath::Sqrt( 1 - sinPhi * sinPhi );
        } else {
          sinPhi = dy * ri;
          cosPhi = dx * ri;
        }
        #ifdef DRAW
        if ( drawFit ) std::cout << "sinPhi0 = " << sinPhi << ", cosPhi0 = " << cosPhi << std::endl;
        #endif
        if ( !tParam.TransportToX( x, sinPhi, cosPhi, tracker.Param().ConstBz(), -1 ) ) {
          #ifdef DRAW
          if ( drawFit ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not transport!!" << std::endl;
		  #endif
          if ( SAVE() ) tracklet.SetRowHit( iRow, -1 );
          break;
        }
        //std::cout<<"mark1 "<<r.fItr<<std::endl;
        //tParam.Print();
        tracker.GetErrors2( iRow, tParam.GetZ(), sinPhi, cosPhi, tParam.GetDzDs(), err2Y, err2Z );
        //std::cout<<"mark2"<<std::endl;

        if ( drawFit ) {
          #ifdef DRAW
          std::cout << " params after transport=" << std::endl;
          tParam.Print();
          std::cout << "fit tracklet before filter: " << r.fItr << ", row " << iRow << " errs=" << err2Y << " " << err2Z << std::endl;
          AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kBlue, 2., 1 );
          AliHLTTPCCADisplay::Instance().Ask();
		  #endif
        }
        if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
          #ifdef DRAW
          if ( drawFit ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not filter!!" << std::endl;
          #endif
          if ( SAVE() ) tracklet.SetRowHit( iRow, -1 );
          break;
        }
      }
      if ( SAVE() ) tracklet.SetRowHit( iRow, oldIH );
      if ( drawFit ) {
        #ifdef DRAW
        std::cout << "fit tracklet after filter " << r.fItr << ", row " << iRow << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kGreen, 2. );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif
      }
      r.fNHits++;
      r.fLastRow = iRow;
      r.fEndRow = iRow;
      break;
    } while ( 0 );

    if ( r.fCurrIH < 0 ) {
      #ifdef DRAW
      if ( drawFitted ) std::cout << "fitted tracklet " << r.fItr << ", nhits=" << r.fNHits << std::endl;
      #endif
      r.fStage = 1;
      //AliHLTTPCCAPerformance::Instance().HNHitsPerSeed()->Fill(r.fNHits);
      if ( r.fNHits < 3 ) { r.fNHits = 0; r.fGo = 0;}//SG!!!
      if ( CAMath::Abs( tParam.SinPhi() ) > .999 ) {
        #ifdef DRAW
        if ( drawFitted ) std::cout << " fitted tracklet  error: sinPhi=" << tParam.SinPhi() << std::endl;
        #endif
        r.fNHits = 0; r.fGo = 0;
      } else {
        //tParam.SetCosPhi( CAMath::Sqrt(1-tParam.SinPhi()*tParam.SinPhi()) );
      }
      if ( drawFitted ) {
        #ifdef DRAW
        std::cout << "fitted tracklet " << r.fItr << " miss=" << r.fNMissed << " go=" << r.fGo << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kBlue );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif
      }
      if ( r.fGo ) {
        CAMath::AtomicMax( &s.fMaxEndRow, r.fEndRow - 1 );
      }
    }
  } else { // forward/backward searching part
    do {
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "search tracklet " << r.fItr << " row " << iRow << " miss=" << r.fNMissed << " go=" << r.fGo << " stage=" << r.fStage << std::endl;
        #endif
      }

      if ( r.fStage == 2 && ( ( iRow >= r.fEndRow ) ||
                              ( iRow >= r.fStartRow && ( iRow - r.fStartRow ) % 2 == 0 )
                            ) ) break;
      if ( r.fNMissed > kMaxRowGap  ) {
        break;
      }

      r.fNMissed++;

      float x = row.X();
      float err2Y, err2Z;
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "tracklet " << r.fItr << " before transport to row " << iRow << " : " << std::endl;
        tParam.Print();
        #endif
      }
      if ( !tParam.TransportToX( x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz(), .99 ) ) {
        #ifdef DRAW
        if ( drawSearch ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not transport!!" << std::endl;
        #endif
        break;
      }
      if ( row.NHits() < 1 ) {
        // skip empty row
        break;
      }
      if ( drawSearch ) {
		#ifdef DRAW
        std::cout << "tracklet " << r.fItr << " after transport to row " << iRow << " : " << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kBlue, 2., 1 );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif
      }
      uint4 *tmpint4 = s.fData[r.fCurrentData];

      //ushort2 *hits = reinterpret_cast<ushort2*>( tmpint4 );
	  ushort_v *hitsx = reinterpret_cast<ushort_v*>( tmpint4 );
	  ushort_v *hitsy = reinterpret_cast<ushort_v*>( tmpint4 ) + NextMultipleOf<8>(row.NHits());

      float fY = tParam.GetY();
      float fZ = tParam.GetZ();
      int best = -1;

      { // search for the closest hit
        const int fIndYmin = row.Grid().GetBinBounded( fY - 1.f, fZ - 1.f );
        assert( fIndYmin >= 0 );

        int ds;
        int fY0 = ( int ) ( ( fY - y0 ) * stepYi );
        int fZ0 = ( int ) ( ( fZ - z0 ) * stepZi );
        int ds0 = ( ( ( int )1 ) << 30 );
        ds = ds0;

        unsigned int fHitYfst = 1, fHitYlst = 0, fHitYfst1 = 1, fHitYlst1 = 0;

        if ( drawSearch ) {
#ifdef DRAW
          std::cout << " tracklet " << r.fItr << ", row " << iRow << ": grid N=" << row.Grid().N() << std::endl;
          std::cout << " tracklet " << r.fItr << ", row " << iRow << ": minbin=" << fIndYmin << std::endl;
#endif
        }
        {
          int nY = row.Grid().Ny();

          unsigned short *sGridP = ( reinterpret_cast<unsigned short*>( tmpint4 ) ) + 3 * NextMultipleOf<8>(row.NHits());
          fHitYfst = sGridP[fIndYmin];
          fHitYlst = sGridP[fIndYmin+2];
          fHitYfst1 = sGridP[fIndYmin+nY];
          fHitYlst1 = sGridP[fIndYmin+nY+2];
          assert( (signed) fHitYfst <= row.NHits() );
          assert( (signed) fHitYlst <= row.NHits() );
          assert( (signed) fHitYfst1 <= row.NHits() );
          assert( (signed) fHitYlst1 <= row.NHits() );
          if ( drawSearch ) {
#ifdef DRAW
            std::cout << " Grid, row " << iRow << ": nHits=" << row.NHits() << ", grid n=" << row.Grid().N() << ", c[n]=" << sGridP[row.Grid().N()] << std::endl;
            std::cout << "hit steps = " << stepY << " " << stepZ << std::endl;
            std::cout << " Grid bins:" << std::endl;
            for ( unsigned int i = 0; i < row.Grid().N(); i++ ) {
              std::cout << " bin " << i << ": ";
              for ( int j = sGridP[i]; j < sGridP[i+1]; j++ ) {
                ushort2 hh = hits[j];
                float y = y0 + hh.x * stepY;
                float z = z0 + hh.y * stepZ;
                std::cout << "[" << j << "|" << y << "," << z << "] ";
              }
              std::cout << std::endl;
            }
#endif
          }
          if ( sGridP[row.Grid().N()] != row.NHits() ) {
#ifdef DRAW
            std::cout << " grid, row " << iRow << ": nHits=" << row.NHits() << ", grid n=" << row.Grid().N() << ", c[n]=" << sGridP[row.Grid().N()] << std::endl;
            //exit(0);
#endif
          }
        }
        if ( drawSearch ) {
          #ifdef DRAW
          std::cout << " tracklet " << r.fItr << ", row " << iRow << ", yz= " << fY << "," << fZ << ": search hits=" << fHitYfst << " " << fHitYlst << " / " << fHitYfst1 << " " << fHitYlst1 << std::endl;
          std::cout << " hit search :" << std::endl;
          #endif
        }
        for ( unsigned int fIh = fHitYfst; fIh < fHitYlst; fIh++ ) {
          assert( (signed) fIh < row.NHits() );
          ushort2 hh;// = hits[fIh];
		  hh.x = hitsx[fIh];
		  hh.y = hitsy[fIh];
          int ddy = ( int )( hh.x ) - fY0;
          int ddz = ( int )( hh.y ) - fZ0;
          int dds = CAMath::Abs( ddy ) + CAMath::Abs( ddz );
          if ( drawSearch ) {
            #ifdef DRAW
            std::cout << fIh << ": hityz= " << hh.x << " " << hh.y << "(" << hh.x*stepY << " " << hh.y*stepZ << "), trackyz=" << fY0 << " " << fZ0 << "(" << fY0*stepY << " " << fZ0*stepZ << "), dy,dz,ds= " << ddy << " " << ddz << " " << dds << "(" << ddy*stepY << " " << ddz*stepZ << std::endl;
            #endif
          }
          if ( dds < ds ) {
            ds = dds;
            best = fIh;
          }
        }

        for ( unsigned int fIh = fHitYfst1; fIh < fHitYlst1; fIh++ ) {
          ushort2 hh;// = hits[fIh];
		  hh.x = hitsx[fIh];
		  hh.y = hitsy[fIh];
          int ddy = ( int )( hh.x ) - fY0;
          int ddz = ( int )( hh.y ) - fZ0;
          int dds = CAMath::Abs( ddy ) + CAMath::Abs( ddz );
          if ( drawSearch ) {
            #ifdef DRAW
            std::cout << fIh << ": hityz= " << hh.x << " " << hh.y << "(" << hh.x*stepY << " " << hh.y*stepZ << "), trackyz=" << fY0 << " " << fZ0 << "(" << fY0*stepY << " " << fZ0*stepZ << "), dy,dz,ds= " << ddy << " " << ddz << " " << dds << "(" << ddy*stepY << " " << ddz*stepZ << std::endl;
            #endif
          }
          if ( dds < ds ) {
            ds = dds;
            best = fIh;
          }
        }
      }// end of search for the closest hit

      if ( best < 0 ) break;
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "hit search " << r.fItr << ", row " << iRow << " hit " << best << " found" << std::endl;
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best, kRed, 1. );
        AliHLTTPCCADisplay::Instance().Ask();
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best, kWhite, 1 );
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best );
		#endif
      }

      ushort2 hh;// = hits[best];
	  hh.x = hitsx[best];
	  hh.y = hitsy[best];

      //std::cout<<"mark 3, "<<r.fItr<<std::endl;
      //tParam.Print();
      tracker.GetErrors2( iRow, *( ( AliHLTTPCCATrackParam* )&tParam ), err2Y, err2Z );
      //std::cout<<"mark 4"<<std::endl;

      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
      float dy = y - fY;
      float dz = z - fZ;

      const float kFactor = tracker.Param().HitPickUpFactor() * tracker.Param().HitPickUpFactor() * 3.5 * 3.5;
      float sy2 = kFactor * ( tParam.GetErr2Y() +  err2Y );
      float sz2 = kFactor * ( tParam.GetErr2Z() +  err2Z );
      if ( sy2 > 2. ) sy2 = 2.;
      if ( sz2 > 2. ) sz2 = 2.;

      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "dy,sy= " << dy << " " << CAMath::Sqrt( sy2 ) << ", dz,sz= " << dz << " " << CAMath::Sqrt( sz2 ) << std::endl;
        std::cout << "dy,dz= " << dy << " " << dz << ", sy,sz= " << CAMath::Sqrt( sy2 ) << " " << CAMath::Sqrt( sz2 ) << ", sy,sz= " << CAMath::Sqrt( kFactor*( tParam.GetErr2Y() +  err2Y ) ) << " " << CAMath::Sqrt( kFactor*( tParam.GetErr2Z() +  err2Z ) ) << std::endl;
        #endif
      }
      if ( CAMath::FMulRZ( dy, dy ) > sy2 || CAMath::FMulRZ( dz, dz ) > sz2  ) {
        if ( drawSearch ) {
          #ifdef DRAW
          std::cout << "found hit is out of the chi2 window\n " << std::endl;
          #endif
        }
        break;
      }
#ifdef DRAW
      //if( SAVE() ) hitstore[ iRow ] = best;
      //std::cout<<"hit search before filter: "<<r.fItr<<", row "<<iRow<<std::endl;
      //AliHLTTPCCADisplay::Instance().DrawTracklet(tParam, hitstore, kBlue);
      //AliHLTTPCCADisplay::Instance().Ask();
#endif
      if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
        if ( drawSearch ) {
          #ifdef DRAW
          std::cout << "tracklet " << r.fItr << " at row " << iRow << " : can not filter!!!! " << std::endl;
          #endif
        }
        break;
      }
      if ( SAVE() ) tracklet.SetRowHit( iRow, best );
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "tracklet " << r.fItr << " after filter at row " << iRow << " : " << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kRed );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif
      }
      r.fNHits++;
      r.fNMissed = 0;
      if ( r.fStage == 1 ) r.fLastRow = iRow;
      else r.fFirstRow = iRow;
    } while ( 0 );
  }
}



GPUd() void AliHLTTPCCATrackletConstructor::Thread
( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam )
{

  // reconstruction of tracklets
  if ( iSync == 0 ) {
    Step0( nBlocks, nThreads, iBlock, iThread, s, r, tracker, tParam );
  } else if ( iSync == 1 ) {
    Step1( nBlocks, nThreads, iBlock, iThread, s, r, tracker, tParam );
  } else if ( iSync == 2 ) {
    Step2( nBlocks, nThreads, iBlock, iThread, s, r, tracker, tParam );
  }

  else if ( iSync == 3 )

  {
    r.fCurrentData = 1;
    ReadData( iThread, s, r, tracker, s.fMinStartRow );
    r.fCurrentData = 0;
    r.fNMissed = 0;
  } else if ( iSync == 3 + 159 + 1 ) {
    r.fCurrentData = 1;
    int nextRow = s.fMaxEndRow;
    if ( nextRow < 0 ) nextRow = 0;
    ReadData( iThread, s, r, tracker, nextRow );
    r.fCurrentData = 0;
    r.fNMissed = 0;
    r.fStage = 2;
    if ( r.fGo ) {
      const AliHLTTPCCARow &row = tracker.Row( r.fEndRow );
      float x = row.X();
      if ( !tParam.TransportToX( x, tracker.Param().ConstBz(), .999 ) ) r.fGo = 0;
    }
  }

  else if ( iSync <= 3 + 159 + 1 + 159 ) {
    int iRow, nextRow;
    if (  iSync <= 3 + 159 ) {
      iRow = iSync - 4;
      if ( iRow < s.fMinStartRow ) return;
      nextRow = iRow + 1;
      if ( nextRow > 158 ) nextRow = 158;
    } else {
      iRow = 158 - ( iSync - 4 - 159 - 1 );
      if ( iRow > s.fMaxEndRow ) return;
      nextRow = iRow - 1;
      if ( nextRow < 0 ) nextRow = 0;
    }

    if ( r.fIsMemThread ) {
      ReadData( iThread, s, r, tracker, nextRow );
    } else {
      /*UpdateTracklet( nBlocks, nThreads, iBlock, iThread,
                      s, r, tracker, tParam, iRow );*/
    }
    r.fCurrentData = !r.fCurrentData;
  }

  else if ( iSync == 4 + 159*2 + 1 + 1 ) { //
    StoreTracklet( nBlocks, nThreads, iBlock, iThread,
                   s, r, tracker, tParam );
  }
}

