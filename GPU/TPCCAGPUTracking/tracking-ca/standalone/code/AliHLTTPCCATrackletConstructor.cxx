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

#define kMaxRowGap 4


GPUd() void AliHLTTPCCATrackletConstructor::Step0
( int nBlocks, int /*nThreads*/, int iBlock, int iThread,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &/*tParam*/ )
{
  // reconstruction of tracklets, step 0

  r.fIsMemThread = ( iThread < TRACKLET_CONSTRUCTOR_NMEMTHREDS );
  if ( iThread == 0 ) {
    int nTracks = *tracker.NTracklets();
#ifdef HLTCA_GPUCODE
    int nTrPerBlock = blockDim.x;
#else
	int nTrPerBlock = nTracks / nBlocks + 1;
#endif
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

GPUd() void AliHLTTPCCATrackletConstructor::InitTracklet( AliHLTTPCCATrackParam &tParam )
{
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

GPUd() void AliHLTTPCCATrackletConstructor::Step1
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int iThread,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam )
{
  // reconstruction of tracklets, step 1

  r.fItr = s.fItr0 + ( iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS );
  r.fGo = ( !r.fIsMemThread ) && ( r.fItr < s.fItr1 );
#ifdef HLTCA_GPU_SORT_STARTHITS_2
  if (r.fGo)
	r.fGo &= tracker.TrackletStartHit(r.fItr).RowIndex() != 0;
#endif
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

  InitTracklet(tParam);
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
#ifdef HLTCA_GPU_PREFETCHDATA
  if ( r.fIsMemThread ) {
    const AliHLTTPCCARow &row = tracker.Row( iRow );
    bool jr = !r.fCurrentData;

    // copy hits, grid content and links

    // FIXME: inefficient copy
    const int numberOfHitsAligned = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits());

/*	
#ifdef HLTCA_GPU_REORDERHITDATA
    ushort2 *sMem1 = reinterpret_cast<ushort2 *>( s.fData[jr] );
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1[i].x = tracker.HitDataY( row, i );
      sMem1[i].y = tracker.HitDataZ( row, i );
    }
#else
    ushort_v *sMem1 = reinterpret_cast<ushort_v *>( s.fData[jr] );
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1[i] = tracker.HitDataY( row, i );
    }

    ushort_v *sMem1a = reinterpret_cast<ushort_v *>( s.fData[jr] ) + numberOfHitsAligned;
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem1a[i] = tracker.HitDataZ( row, i );
    }
#endif

    short *sMem2 = reinterpret_cast<short *>( s.fData[jr] ) + 2 * numberOfHitsAligned;
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem2[i] = tracker.HitLinkUpData( row, i );
    }
	
    unsigned short *sMem3 = reinterpret_cast<unsigned short *>( s.fData[jr] ) + 3 * numberOfHitsAligned;
    const int n = row.FullSize(); // + grid content size
    for ( int i = iThread; i < n; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem3[i] = tracker.FirstHitInBin( row, i );
    }*/

	for (int k = 0;k < 4;k++)		//Copy HitData (k = 0, 1), FirstHitInBint (k = 3), HitLinkUpData (k = 2) to shared memory
	{
		HLTCA_GPU_ROWCOPY* sharedMem;
		const HLTCA_GPU_ROWCOPY* sourceMem;
		int copyCount;
		switch (k)
		{
		case 0:
			sourceMem = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.HitDataY(row) );
			sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[jr] ) + k * numberOfHitsAligned);
			copyCount = numberOfHitsAligned * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);
			break;
		case 1:
			sourceMem = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.HitDataZ(row) );
			sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[jr] ) + k * numberOfHitsAligned);
			copyCount = numberOfHitsAligned * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);
			break;
		case 2:
			sourceMem = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.HitLinkUpData(row) );
			sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[jr] ) + k * numberOfHitsAligned);
			copyCount = numberOfHitsAligned * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);
			break;
		case 3:
			sourceMem = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.FirstHitInBin(row) );
			sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[jr] ) + k * numberOfHitsAligned);
			copyCount = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.FullSize()) * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);
			break;
		}
		for (int i = iThread;i < copyCount;i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
		{
			sharedMem[i] = sourceMem[i];
		}
	}

	const AliHLTTPCCASliceData &tmpData = tracker.Data();
	const AliHLTTPCCARow &tmpRow = tmpData.Row( iRow );
	const AliHLTTPCCARow* pTmpRow = &tmpRow;
	const HLTCA_GPU_ROWCOPY* const sourceMem = (const HLTCA_GPU_ROWCOPY *) pTmpRow;
	HLTCA_GPU_ROWCOPY* const sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> ( &s.fRow[jr] );
	const int copyCount = NextMultipleOf<sizeof(HLTCA_GPU_ROWCOPY)>(sizeof(AliHLTTPCCARow)) / sizeof(HLTCA_GPU_ROWCOPY);
	for (int i = iThread;i < copyCount;i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		sharedMem[i] = sourceMem[i];
	}

  }
#endif
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
	tracklet.SetFirstRow( CAMath::Min(r.fFirstRow, r.fStartRow) );
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

  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];

#ifdef HLTCA_GPU_PREFETCHDATA
  const AliHLTTPCCARow &row = s.fRow[r.fCurrentData];
#else
  const AliHLTTPCCARow &row = tracker.Row( iRow );
#endif

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

#ifdef HLTCA_GPU_PREFETCHDATA
      uint4 *tmpint4 = s.fData[r.fCurrentData];
#endif
	  ushort2 hh;
#ifdef HLTCA_GPU_REORDERHITDATA
      hh = reinterpret_cast<ushort2*>( tmpint4 )[r.fCurrIH];
#else
#ifdef HLTCA_GPU_PREFETCHDATA
	  hh.x = reinterpret_cast<ushort_v*>( tmpint4 )[r.fCurrIH];
	  hh.y = reinterpret_cast<ushort_v*>( tmpint4 )[NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits()) + r.fCurrIH];
#else
	  hh.x = tracker.HitDataY(tracker.Row(iRow))[r.fCurrIH];
	  hh.y = tracker.HitDataZ(tracker.Row(iRow))[r.fCurrIH];
#endif
#endif

      int oldIH = r.fCurrIH;
#ifdef HLTCA_GPU_PREFETCHDATA
      r.fCurrIH = reinterpret_cast<short*>( tmpint4 )[2 * NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits()) + r.fCurrIH]; // read from linkup data
#else
	  r.fCurrIH = tracker.HitLinkUpData(tracker.Row(iRow))[r.fCurrIH]; // read from linkup data
#endif

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
#ifdef HLTCA_GPU_PREFETCHDATA
      uint4 *tmpint4 = s.fData[r.fCurrentData];
#endif

#ifdef HLTCA_GPU_REORDERHITDATA
      const ushort2 *hits = reinterpret_cast<ushort2*>( tmpint4 );
#else
#ifdef HLTCA_GPU_PREFETCHDATA
	  const ushort_v *hitsx = reinterpret_cast<ushort_v*>( tmpint4 );
	  const ushort_v *hitsy = reinterpret_cast<ushort_v*>( tmpint4 ) + NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits());
#else
	  const ushort_v *hitsx = tracker.HitDataY(tracker.Row(iRow));
	  const ushort_v *hitsy = tracker.HitDataZ(tracker.Row(iRow));
#endif
#endif

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

#ifdef HLTCA_GPU_PREFETCHDATA
		  const unsigned short *sGridP = ( reinterpret_cast<unsigned short*>( tmpint4 ) ) + 3 * NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits());
#else
		  const unsigned short *sGridP = tracker.FirstHitInBin(tracker.Row(iRow));
#endif
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
          ushort2 hh;
#ifdef HLTCA_GPU_REORDERHITDATA
		  hh = hits[fIh];
#else
		  hh.x = hitsx[fIh];
		  hh.y = hitsy[fIh];
#endif
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
          ushort2 hh;
#ifdef HLTCA_GPU_REORDERHITDATA
		  hh = hits[fIh];
#else
		  hh.x = hitsx[fIh];
		  hh.y = hitsy[fIh];
#endif
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

      ushort2 hh;
#ifdef HLTCA_GPU_REORDERHITDATA
	  hh = hits[best];
#else
	  hh.x = hitsx[best];
	  hh.y = hitsy[best];
#endif

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

#ifdef HLTCA_GPUCODE
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
		if (r.fGo)
		{
			if ( r.fStage == 0  &&
				!( iRow < r.fStartRow || r.fCurrIH < 0  ) &&
				!( ( iRow - r.fStartRow ) % 2 != 0 )
			)
			tracker.fStageAtSync[iSync * blockDim.x * gridDim.x + iBlock * blockDim.x + (iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS)] = 1;
			if ( r.fStage != 0)
			{
				if (r.fStage == 1 && r.fNMissed <= 4  )
					tracker.fStageAtSync[iSync * blockDim.x * gridDim.x + iBlock * blockDim.x + (iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS)] = 2;
				if (r.fStage == 2 && iRow < r.fEndRow && iRow >= r.fStartRow && ( iRow - r.fStartRow ) % 2 != 0)
					tracker.fStageAtSync[iSync * blockDim.x * gridDim.x + iBlock * blockDim.x + (iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS)] = 3;
				if (r.fStage == 2 && iRow < r.fStartRow && r.fNMissed <= 4)
					tracker.fStageAtSync[iSync * blockDim.x * gridDim.x + iBlock * blockDim.x + (iThread - TRACKLET_CONSTRUCTOR_NMEMTHREDS)] = 4;
			}
		}
#endif
#endif
		UpdateTracklet( nBlocks, nThreads, iBlock, iThread,
                      s, r, tracker, tParam, iRow );
    }
    r.fCurrentData = !r.fCurrentData;
  }

  else if ( iSync == 4 + 159*2 + 1 + 1 ) { //
    StoreTracklet( nBlocks, nThreads, iBlock, iThread,
                   s, r, tracker, tParam );
  }
}

#ifdef HLTCA_GPUCODE
GPUd() void AliHLTTPCCATrackletConstructor::CopyTrackletTempData( AliHLTTPCCAThreadMemory &rMemSrc, AliHLTTPCCAThreadMemory &rMemDst, AliHLTTPCCATrackParam &tParamSrc, AliHLTTPCCATrackParam &tParamDst)
{
	rMemDst.fStartRow = rMemSrc.fStartRow;
	rMemDst.fEndRow = rMemSrc.fEndRow;
	rMemDst.fFirstRow = rMemSrc.fFirstRow;
	rMemDst.fLastRow = rMemSrc.fLastRow;
	rMemDst.fCurrIH =  rMemSrc.fCurrIH;
	rMemDst.fGo = rMemSrc.fGo;
	rMemDst.fStage = rMemSrc.fStage;
	rMemDst.fNHits = rMemSrc.fNHits;
	rMemDst.fNMissed = rMemSrc.fNMissed;
	rMemDst.fLastY = rMemSrc.fLastY;
	rMemDst.fLastZ = rMemSrc.fLastZ;

	tParamDst.SetSinPhi( tParamSrc.GetSinPhi() );
	tParamDst.SetDzDs( tParamSrc.GetDzDs() );
	tParamDst.SetQPt( tParamSrc.GetQPt() );
	tParamDst.SetSignCosPhi( tParamSrc.GetSignCosPhi() );
	tParamDst.SetChi2( tParamSrc.GetChi2() );
	tParamDst.SetNDF( tParamSrc.GetNDF() );
	tParamDst.SetCov( 0, tParamSrc.GetCov(0) );
	tParamDst.SetCov( 1, tParamSrc.GetCov(1) );
	tParamDst.SetCov( 2, tParamSrc.GetCov(2) );
	tParamDst.SetCov( 3, tParamSrc.GetCov(3) );
	tParamDst.SetCov( 4, tParamSrc.GetCov(4) );
	tParamDst.SetCov( 5, tParamSrc.GetCov(5) );
	tParamDst.SetCov( 6, tParamSrc.GetCov(6) );
	tParamDst.SetCov( 7, tParamSrc.GetCov(7) );
	tParamDst.SetCov( 8, tParamSrc.GetCov(8) );
	tParamDst.SetCov( 9, tParamSrc.GetCov(9) );
	tParamDst.SetCov( 10, tParamSrc.GetCov(10) );
	tParamDst.SetCov( 11, tParamSrc.GetCov(11) );
	tParamDst.SetCov( 12, tParamSrc.GetCov(12) );
	tParamDst.SetCov( 13, tParamSrc.GetCov(13) );
	tParamDst.SetCov( 14, tParamSrc.GetCov(14) );
	tParamDst.SetX( tParamSrc.GetX() );
	tParamDst.SetY( tParamSrc.GetY() );
	tParamDst.SetZ( tParamSrc.GetZ() );
}

GPUd() int AliHLTTPCCATrackletConstructor::FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem, int Reverse, int RowBlock)
{
	__syncthreads();
	int NextTracketlFirstRun = sMem.fNextTrackletFirstRun;
	if (threadIdx.x  == 0)
	{
		if (sMem.fNextTrackletFirstRun)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_START
			const uint2 &nTracklet = tracker.BlockStartingTracklet()[blockIdx.x];
			sMem.fNextTrackletCount = nTracklet.y;
			if (sMem.fNextTrackletCount == 0)
			{
				sMem.fNextTrackletFirstRun = 0;
			}
			else
			{
				if (tracker.TrackletStartHits()[nTracklet.x].RowIndex() / HLTCA_GPU_SCHED_ROW_STEP != RowBlock)
				{
					sMem.fNextTrackletCount = 0;
				}
				else
				{
					sMem.fNextTrackletFirst = nTracklet.x;
					sMem.fNextTrackletNoDummy = 1;
				}
			}
#endif
		}
		else
		{
			const int nFetchTracks = CAMath::Max(CAMath::Min((*tracker.RowBlockPos(Reverse, RowBlock)).x - (*tracker.RowBlockPos(Reverse, RowBlock)).y, HLTCA_GPU_THREAD_COUNT - TRACKLET_CONSTRUCTOR_NMEMTHREDS), 0);
			sMem.fNextTrackletCount = nFetchTracks;
			const int nUseTrack = nFetchTracks ? CAMath::AtomicAdd(&(*tracker.RowBlockPos(Reverse, RowBlock)).y, nFetchTracks) : 0;
			sMem.fNextTrackletFirst = nUseTrack;

			const int nFillTracks = CAMath::Min(nFetchTracks, nUseTrack + nFetchTracks - (*((volatile int2*) (tracker.RowBlockPos(Reverse, RowBlock)))).x);
			if (nFillTracks > 0)
			{
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
				sMem.fNextTrackletCount = nFetchTracks - nFillTracks;
#else
				const int nStartFillTrack = CAMath::AtomicAdd(&(*tracker.RowBlockPos(Reverse, RowBlock)).x, nFillTracks);
				if (nFillTracks + nStartFillTrack >= HLTCA_GPU_MAX_TRACKLETS)
				{
					tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW;
				}
				for (int i = 0;i < nFillTracks;i++)
				{
					tracker.RowBlockTracklets(Reverse, RowBlock)[(nStartFillTrack + i) % HLTCA_GPU_MAX_TRACKLETS] = -3;	//Dummy filling track
				}
#endif
			}
			sMem.fNextTrackletNoDummy = 0;
		}
	}
	__syncthreads();
	if (sMem.fNextTrackletCount == 0)
	{
		return(-2);		//No more track in this RowBlock
	}
	else if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS >= sMem.fNextTrackletCount)
	{
		return(-1);		//No track in this RowBlock for this thread
	}
	else if (NextTracketlFirstRun)
	{
		if (threadIdx.x == 0) sMem.fNextTrackletFirstRun = 0;
		return(sMem.fNextTrackletFirst + threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS);
	}
	else
	{
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
		return(tracker.RowBlockTracklets(Reverse, RowBlock)[(sMem.fNextTrackletFirst + threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) % HLTCA_GPU_MAX_TRACKLETS]);
#else
		const int nTrackPos = sMem.fNextTrackletFirst + threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS;
		volatile int* const ptrTracklet = &tracker.RowBlockTracklets(Reverse, RowBlock)[nTrackPos % HLTCA_GPU_MAX_TRACKLETS];
		int nTracklet;
		int nTryCount = 0;
		while ((nTracklet = *ptrTracklet) == -1)
		{
			for (int i = 0;i < 10000;i++)
				sMem.fNextTrackletStupidDummy++;
			nTryCount++;
			if (nTryCount > 20)
			{
				CAMath::AtomicAdd(&tracker.GPUParameters()->fGPUSchedCollisions, 1);
				return(-1);
			}
		};
		return(nTracklet);
#endif
	}
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorNew()
{
	AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );

#ifdef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
	tracker.BlockStartingTracklet()[0].x = HLTCA_GPU_EMULATION_SINGLE_TRACKLET;
	tracker.BlockStartingTracklet()[0].y = 1;
	for (int i = 1;i < HLTCA_GPU_BLOCK_COUNT;i++)
	{
		tracker.BlockStartingTracklet()[i].x = tracker.BlockStartingTracklet()[i].y = 0;
	}
#endif

	AliHLTTPCCATrackParam tParam;
	AliHLTTPCCAThreadMemory rMem;
	GPUshared() AliHLTTPCCASharedMemory sMem;

#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (threadIdx.x == 0)
	{
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
		sMem.fNextTrackletFirstRun = tracker.GPUParameters()->fStaticStartingTracklets;
#else
		sMem.fNextTrackletFirstRun = 1;
#endif
	}
	__syncthreads();
#endif

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

#endif

#pragma unroll
	for (int iReverse = 0;iReverse < 2;iReverse++)
	{
		for (int iRowBlock = 0;iRowBlock < tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1;iRowBlock++)
		{
			int i;
			while ((i = FetchTracklet(tracker, sMem, iReverse, iRowBlock)) != -2)
			{
#ifdef HLTCA_GPU_RESCHED
				short2 StoreToRowBlock;
				int StorePosition;
				if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
				{
					const int nReverse = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
					const int nRowBlock = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
					sMem.fTrackletStoreCount[nReverse][nRowBlock] = 0;
				}
#endif
				__syncthreads();
				if (i >= 0)
				{
#ifdef HLTCA_GPU_EMULATION_DEBUG_TRACKLET
					if (i == HLTCA_GPU_EMULATION_DEBUG_TRACKLET)
					{
						tracker.GPUParameter()->fGPUSchedCollisions += 1;
					}
#endif
					AliHLTTPCCAThreadMemory &rMemGlobal = tracker.GPUTrackletTemp()[i].fThreadMem;
					AliHLTTPCCATrackParam &tParamGlobal = tracker.GPUTrackletTemp()[i].fParam;
					CopyTrackletTempData( rMemGlobal, rMem, tParamGlobal, tParam );
					rMem.fItr = i;
					rMem.fIsMemThread = 0;
					rMem.fSave = 1;

#ifdef HLTCA_GPU_RESCHED
					StoreToRowBlock.x = iRowBlock + 1;
					StoreToRowBlock.y = iReverse;
					if (iReverse)
					{
						for (int j = tracker.Param().NRows() - 1 - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP;j >= CAMath::Max(0, tracker.Param().NRows() - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP);j--)
						{
							UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
							if (rMem.fNMissed > kMaxRowGap) rMem.fGo = 0;
							if (!rMem.fGo) break;							
						}
						
						if (!rMem.fGo || iRowBlock == tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP)
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}
					else
					{
						for (int j = CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP);j < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, tracker.Param().NRows());j++)
						{
							UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
							if (!rMem.fGo) break;
						}
						if ( (rMem.fNMissed > kMaxRowGap || iRowBlock == tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP) && rMem.fGo )
						{
							const AliHLTTPCCARow &row = tracker.Row( rMem.fEndRow );
							float x = row.X();
							if ( !tParam.TransportToX( x, tracker.Param().ConstBz(), .999 ) )
							{
								rMem.fGo = 0;
							}
							else
							{
								StoreToRowBlock.x = (tracker.Param().NRows() - rMem.fEndRow) / HLTCA_GPU_SCHED_ROW_STEP;
								StoreToRowBlock.y = 1;
								rMem.fNMissed = 0;
								rMem.fStage = 2;
							}
						}

						if (!rMem.fGo)
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}

					if (rMem.fGo && (iRowBlock != tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						CopyTrackletTempData( rMem, rMemGlobal, tParam, tParamGlobal );
						StorePosition = CAMath::AtomicAdd(&sMem.fTrackletStoreCount[StoreToRowBlock.y][StoreToRowBlock.x], 1);
					}
#else
					for (int j = rMem.fStartRow;j < tracker.Param().NRows();j++)
					{
						UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
						if (!rMem.fGo) break;
					}

					rMem.fNMissed = 0;
					rMem.fStage = 2;
					if ( rMem.fGo )
					{
						const AliHLTTPCCARow &row = tracker.Row( rMem.fEndRow );
						float x = row.X();
						if ( !tParam.TransportToX( x, tracker.Param().ConstBz(), .999 ) ) rMem.fGo = 0;
					}

					for (int j = rMem.fEndRow;j >= 0;j--)
					{
						UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
						if (!rMem.fGo) break;
					}

					StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
#endif
				}
#ifdef HLTCA_GPU_RESCHED
				__syncthreads();
				if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
				{
					const int nReverse = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
					const int nRowBlock = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
					if (sMem.fTrackletStoreCount[nReverse][nRowBlock])
					{
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = CAMath::AtomicAdd(&tracker.RowBlockPos(nReverse, nRowBlock)->z, sMem.fTrackletStoreCount[nReverse][nRowBlock]);
#else
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = CAMath::AtomicAdd(&tracker.RowBlockPos(nReverse, nRowBlock)->x, sMem.fTrackletStoreCount[nReverse][nRowBlock]);
#endif
					}
				}
				__syncthreads();
				if (i >= 0 && rMem.fGo && (iRowBlock != tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
				{
					tracker.RowBlockTracklets(StoreToRowBlock.y, StoreToRowBlock.x)[sMem.fTrackletStoreCount[StoreToRowBlock.y][StoreToRowBlock.x] + StorePosition] = i;
				}
				__syncthreads();
#endif
			}
		}
	}
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

#endif
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(int iTracklet, AliHLTTPCCATracker &tracker)
{
	AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];
	typename AliHLTTPCCATrackletConstructor::AliHLTTPCCAThreadMemory &rMem = tracker.GPUTrackletTemp()[iTracklet].fThreadMem;

	rMem.fStartRow = id.RowIndex();
  	rMem.fEndRow = id.RowIndex();
  	rMem.fFirstRow = id.RowIndex();
  	rMem.fLastRow = id.RowIndex();
  	rMem.fCurrIH =  id.HitIndex();

	rMem.fItr = iTracklet;
	rMem.fIsMemThread = 0;
	rMem.fGo = 1;
	rMem.fSave = 1;
	rMem.fStage = 0;
	rMem.fNHits = 0;
	rMem.fNMissed = 0;

	AliHLTTPCCATrackletConstructor::InitTracklet(tracker.GPUTrackletTemp()[iTracklet].fParam);
	const int FirstDynamicTracklet = tracker.GPUParameters()->fScheduleFirstDynamicTracklet;

	AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[iTracklet];
	for ( int j = 0; j < 160; j++ ) tracklet.SetRowHit( j, -1 );

#ifndef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (iTracklet >= FirstDynamicTracklet)
#endif
	{
		const int firstTrackletInRowBlock = CAMath::Max(FirstDynamicTracklet, tracker.RowStartHitCountOffset()[CAMath::Max(id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP * HLTCA_GPU_SCHED_ROW_STEP, 1)].z);
		if (iTracklet == firstTrackletInRowBlock)
		{
			const int firstRowInNextBlock = (id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_SCHED_ROW_STEP;
			int trackletsInRowBlock;
			if (firstRowInNextBlock >= tracker.Param().NRows() - 3)
				trackletsInRowBlock = *tracker.NTracklets() - firstTrackletInRowBlock;
			else
				trackletsInRowBlock = CAMath::Max(FirstDynamicTracklet, tracker.RowStartHitCountOffset()[firstRowInNextBlock].z) - firstTrackletInRowBlock;

			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->x = trackletsInRowBlock;
#ifdef HLTCA_GPU_SCHED_HOST_SYNC
			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->z = trackletsInRowBlock;
#endif
		}
		tracker.RowBlockTracklets(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)[iTracklet - firstTrackletInRowBlock] = iTracklet;
	}
#endif
}

GPUg() void AliHLTTPCCATrackletConstructorInit()
{
	AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= *tracker.NTracklets()) return;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(i, tracker);
}

GPUg() void AliHLTTPCCATrackletConstructorNew()
{
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorNew();
}

#ifdef HLTCA_GPU_SCHED_HOST_SYNC
GPUg() void AliHLTTPCCATrackletConstructorUpdateRowBlockPos()
{
	AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0)
	{
		tracker.GPUParameters()->fStaticStartingTracklets = 0;
	}
	if (i < (tracker.Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1) * 2)
	{
		tracker.RowBlockPos()[i].y = tracker.RowBlockPos()[i].x;
		tracker.RowBlockPos()[i].x = tracker.RowBlockPos()[i].z;
	}
}
#endif
#endif
