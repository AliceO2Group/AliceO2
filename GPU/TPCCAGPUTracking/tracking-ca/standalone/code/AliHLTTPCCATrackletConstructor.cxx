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
#endif //DRAW

#define kMaxRowGap 4

GPUd() void AliHLTTPCCATrackletConstructor::InitTracklet( AliHLTTPCCATrackParam &tParam )
{
  //Initialize Tracklet Parameters using default values
  tParam.InitParam();
}

GPUd() void AliHLTTPCCATrackletConstructor::ReadData
#ifndef HLTCA_GPU_PREFETCHDATA
( int /*iThread*/, AliHLTTPCCASharedMemory& /*s*/, AliHLTTPCCAThreadMemory& /*r*/, AliHLTTPCCATracker& /*tracker*/, int /*iRow*/ )
{
	//Prefetch Data to shared memory
#else
( int iThread, AliHLTTPCCASharedMemory& s, AliHLTTPCCAThreadMemory& r, AliHLTTPCCATracker& tracker, int iRow )
{
  // reconstruction of tracklets, read data step
    const AliHLTTPCCARow &row = tracker.Row( iRow );
    //bool jr = !r.fCurrentData;

    // copy hits, grid content and links

    // FIXME: inefficient copy
    //const int numberOfHitsAligned = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits());

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
#endif //HLTCA_GPU_REORDERHITDATA

    short *sMem2 = reinterpret_cast<short *>( s.fData[jr] ) + 2 * numberOfHitsAligned;
    for ( int i = iThread; i < numberOfHitsAligned; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem2[i] = tracker.HitLinkUpData( row, i );
    }
	
    unsigned short *sMem3 = reinterpret_cast<unsigned short *>( s.fData[jr] ) + 3 * numberOfHitsAligned;
    const int n = row.FullSize(); // + grid content size
    for ( int i = iThread; i < n; i += TRACKLET_CONSTRUCTOR_NMEMTHREDS ) {
      sMem3[i] = tracker.FirstHitInBin( row, i );
    }*/

	/*for (int k = 0;k < 2;k++)
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
		case 1:
			sourceMem = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.FirstHitInBin(row) );
			sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[jr] ) + k * numberOfHitsAligned);
			copyCount = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.FullSize()) * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);
			break;
		}
		for (int i = iThread;i < copyCount;i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
		{
			sharedMem[i] = sourceMem[i];
		}
	}*/

	for (unsigned int i = iThread;i < row.FullSize() * sizeof(ushort_v) / sizeof(HLTCA_GPU_ROWCOPY);i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		reinterpret_cast<HLTCA_GPU_ROWCOPY *> (reinterpret_cast<ushort_v *>( s.fData[!r.fCurrentData] ))[i] = reinterpret_cast<const HLTCA_GPU_ROWCOPY *>( tracker.FirstHitInBin(row) )[i];
	}

	const HLTCA_GPU_ROWCOPY* const sourceMem = (const HLTCA_GPU_ROWCOPY *) &row;
	HLTCA_GPU_ROWCOPY* const sharedMem = reinterpret_cast<HLTCA_GPU_ROWCOPY *> ( &s.fRow[!r.fCurrentData] );
	for (unsigned int i = iThread;i < sizeof(AliHLTTPCCARow) / sizeof(HLTCA_GPU_ROWCOPY);i += TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		sharedMem[i] = sourceMem[i];
	}
#endif //!HLTCA_GPU_PREFETCHDATA
}


GPUd() void AliHLTTPCCATrackletConstructor::StoreTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  AliHLTTPCCASharedMemory
#if defined(HLTCA_GPUCODE) | defined(EXTERN_ROW_HITS)
  &s
#else
  &/*s*/
#endif  //!HLTCA_GPUCODE
  , AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam )
{
  // reconstruction of tracklets, tracklet store step

  //AliHLTTPCCAPerformance::Instance().HNHitsPerTrackCand()->Fill(r.fNHits);

  do {
    {
	//std::cout<<"tracklet to store: "<<r.fItr<<", nhits = "<<r.fNHits<<std::endl;
    }

    if ( r.fNHits < TRACKLET_SELECTOR_MIN_HITS ) {
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
#endif //DRAW
    if ( CAMath::Abs( tParam.Par()[4] ) < 1.e-4 ) tParam.SetPar( 4, 1.e-4 );
	if (r.fStartRow < r.fFirstRow) r.fFirstRow = r.fStartRow;
	tracklet.SetFirstRow( r.fFirstRow );
    tracklet.SetLastRow( r.fLastRow );
#ifdef HLTCA_GPUCODE
    tracklet.SetParam( tParam.fParam );
#else
    tracklet.SetParam( tParam.GetParam() );
#endif //HLTCA_GPUCODE
    int w = ( r.fNHits << 16 ) + r.fItr;
    for ( int iRow = r.fFirstRow; iRow <= r.fLastRow; iRow++ ) {
#ifdef EXTERN_ROW_HITS
      int ih = tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr];
#else
	  int ih = tracklet.RowHit( iRow );
#endif //EXTERN_ROW_HITS
      if ( ih >= 0 ) {
#if defined(HLTCA_GPUCODE) & !defined(HLTCA_GPU_PREFETCHDATA) & !defined(HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY)
    	    tracker.MaximizeHitWeight( s.fRows[ iRow ], ih, w );
#else
	    tracker.MaximizeHitWeight( tracker.Row( iRow ), ih, w );
#endif //HLTCA_GPUCODE & !HLTCA_GPU_PREFETCHDATA & !HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY
      }
    }
  }

}

GPUd() void AliHLTTPCCATrackletConstructor::UpdateTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  AliHLTTPCCASharedMemory 
#if defined(HLTCA_GPUCODE) | defined(EXTERN_ROW_HITS)
  &s
#else
  &/*s*/
#endif //HLTCA_GPUCODE
  , AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam, int iRow )
{
  // reconstruction of tracklets, tracklets update step

  //std::cout<<"Update tracklet: "<<r.fItr<<" "<<r.fGo<<" "<<r.fStage<<" "<<iRow<<std::endl;
  bool drawSearch = 0;//r.fItr==2;
  bool drawFit = 0;//r.fItr==2;
  bool drawFitted = drawFit ;//|| 1;//r.fItr==16;

  if ( !r.fGo ) return;

#ifndef EXTERN_ROW_HITS
  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS

#ifdef HLTCA_GPU_PREFETCHDATA
  const AliHLTTPCCARow &row = s.fRow[r.fCurrentData];
#elif defined(HLTCA_GPUCODE)
  const AliHLTTPCCARow &row = s.fRows[iRow];
#else
  const AliHLTTPCCARow &row = tracker.Row( iRow );
#endif //HLTCA_GPU_PREFETCHDATA

  float y0 = row.Grid().YMin();
  float stepY = row.HstepY();
  float z0 = row.Grid().ZMin();
  float stepZ = row.HstepZ();
  float stepYi = row.HstepYi();
  float stepZi = row.HstepZi();

  if ( r.fStage == 0 ) { // fitting part
    do {

      if ( iRow < r.fStartRow || r.fCurrIH < 0  ) break;

      if ( ( iRow - r.fStartRow ) % 2 != 0 )
	  {
#ifndef EXTERN_ROW_HITS
		  tracklet.SetRowHit(iRow, -1);
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //EXTERN_ROW_HITS
		  break; // SG!!! - jump over the row
	  }

//#ifdef HLTCA_GPU_PREFETCHDATA
//      uint4 *tmpint4 = s.fData[r.fCurrentData];
//#endif
	  ushort2 hh;
//#ifdef HLTCA_GPU_REORDERHITDATA
//      hh = reinterpret_cast<ushort2*>( tmpint4 )[r.fCurrIH];
//#else
//#ifdef HLTCA_GPU_PREFETCHDATA
//	  hh.x = reinterpret_cast<ushort_v*>( tmpint4 )[r.fCurrIH];
//	  hh.y = reinterpret_cast<ushort_v*>( tmpint4 )[NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits()) + r.fCurrIH];
//#else
#if defined(HLTCA_GPU_TEXTURE_FETCH)
	  hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + r.fCurrIH);
#else
	  hh = tracker.HitData(row)[r.fCurrIH];
#endif //HLTCA_GPU_TEXTURE_FETCH
//#endif
//#endif

      int oldIH = r.fCurrIH;
//#ifdef HLTCA_GPU_PREFETCHDATA
//      r.fCurrIH = reinterpret_cast<short*>( tmpint4 )[2 * NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits()) + r.fCurrIH]; // read from linkup data
//#else
#if defined(HLTCA_GPU_TEXTURE_FETCH)
	  r.fCurrIH = tex1Dfetch(gAliTexRefs, ((char*) tracker.Data().HitLinkUpData(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + r.fCurrIH);
#else
	  r.fCurrIH = tracker.HitLinkUpData(row)[r.fCurrIH]; // read from linkup data
#endif //HLTCA_GPU_TEXTURE_FETCH
//#endif

      float x = row.X();
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
#ifdef DRAW
      if ( drawFit ) std::cout << " fit tracklet: new hit " << oldIH << ", xyz=" << x << " " << y << " " << z << std::endl;
#endif //DRAW

      if ( iRow == r.fStartRow ) {
        tParam.SetX( x );
        tParam.SetY( y );
        tParam.SetZ( z );
        r.fLastY = y;
        r.fLastZ = z;
        #ifdef DRAW
        if ( drawFit ) std::cout << " fit tracklet " << r.fItr << ", row " << iRow << " first row" << std::endl;
        #endif //DRAW
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
          #endif //DRAW
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
        #endif //DRAW
        if ( !tParam.TransportToX( x, sinPhi, cosPhi, tracker.Param().ConstBz(), -1 ) ) {
          #ifdef DRAW
          if ( drawFit ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not transport!!" << std::endl;
		  #endif //DRAW
#ifndef EXTERN_ROW_HITS
          tracklet.SetRowHit( iRow, -1 );
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //EXTERN_ROW_HITS
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
		  #endif //DRAW
        }
        if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
          #ifdef DRAW
          if ( drawFit ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not filter!!" << std::endl;
          #endif //DRAW
#ifndef EXTERN_ROW_HITS
          tracklet.SetRowHit( iRow, -1 );
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //EXTERN_ROW_HITS
          break;
        }
      }
#ifndef EXTERN_ROW_HITS
      tracklet.SetRowHit( iRow, oldIH );
#else
	  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = oldIH;
#endif //!EXTERN_ROW_HITS
      if ( drawFit ) {
        #ifdef DRAW
        std::cout << "fit tracklet after filter " << r.fItr << ", row " << iRow << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kGreen, 2. );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif //DRAW
      }
      r.fNHits++;
      r.fLastRow = iRow;
      r.fEndRow = iRow;
      break;
    } while ( 0 );

    if ( r.fCurrIH < 0 ) {
      #ifdef DRAW
      if ( drawFitted ) std::cout << "fitted tracklet " << r.fItr << ", nhits=" << r.fNHits << std::endl;
      #endif //DRAW
      r.fStage = 1;
      //AliHLTTPCCAPerformance::Instance().HNHitsPerSeed()->Fill(r.fNHits);
      if ( r.fNHits < 3 ) { r.fNHits = 0; r.fGo = 0;}//SG!!!
      if ( CAMath::Abs( tParam.SinPhi() ) > .999 ) {
        #ifdef DRAW
        if ( drawFitted ) std::cout << " fitted tracklet  error: sinPhi=" << tParam.SinPhi() << std::endl;
        #endif //DRAW
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
		#endif //DRAW
      }
    }
  } else { // forward/backward searching part
    do {
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "search tracklet " << r.fItr << " row " << iRow << " miss=" << r.fNMissed << " go=" << r.fGo << " stage=" << r.fStage << std::endl;
        #endif //DRAW
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
        #endif //DRAW
      }
      if ( !tParam.TransportToX( x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz(), .99 ) ) {
        #ifdef DRAW
        if ( drawSearch ) std::cout << " tracklet " << r.fItr << ", row " << iRow << ": can not transport!!" << std::endl;
        #endif //DRAW
#ifndef EXTERN_ROW_HITS
		tracklet.SetRowHit(iRow, -1);
#else
		tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //!EXTERN_ROW_HITS
        break;
      }
      if ( row.NHits() < 1 ) {
        // skip empty row
#ifndef EXTERN_ROW_HITS
		  tracklet.SetRowHit(iRow, -1);
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //!EXTERN_ROW_HITS
        break;
      }
      if ( drawSearch ) {
		#ifdef DRAW
        std::cout << "tracklet " << r.fItr << " after transport to row " << iRow << " : " << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kBlue, 2., 1 );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif //DRAW
      }
#ifdef HLTCA_GPU_PREFETCHDATA
      uint4 *tmpint4 = s.fData[r.fCurrentData];
#endif //HLTCA_GPU_PREFETCHDATA

//#ifdef HLTCA_GPU_REORDERHITDATA
//      const ushort2 *hits = reinterpret_cast<ushort2*>( tmpint4 );
//#else
//#ifdef HLTCA_GPU_PREFETCHDATA
//	  const ushort_v *hitsx = reinterpret_cast<ushort_v*>( tmpint4 );
//	  const ushort_v *hitsy = reinterpret_cast<ushort_v*>( tmpint4 ) + NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.NHits());
//#else
#ifndef HLTCA_GPU_TEXTURE_FETCH
	  const ushort2 *hits = tracker.HitData(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH
//#endif
//#endif

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
#endif //DRAW
        }
        {
          int nY = row.Grid().Ny();

//#ifdef HLTCA_GPU_PREFETCHDATA
//		  const unsigned short *sGridP = ( reinterpret_cast<unsigned short*>( tmpint4 ) );
//#else
#ifndef HLTCA_GPU_TEXTURE_FETCH
		  const unsigned short *sGridP = tracker.FirstHitInBin(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH
//#endif

#ifdef HLTCA_GPU_TEXTURE_FETCH
		  fHitYfst = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + fIndYmin);
		  fHitYlst = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + fIndYmin+2);
		  fHitYfst1 = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + fIndYmin+nY);
		  fHitYlst1 = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + fIndYmin+nY+2);
#else
          fHitYfst = sGridP[fIndYmin];
          fHitYlst = sGridP[fIndYmin+2];
          fHitYfst1 = sGridP[fIndYmin+nY];
          fHitYlst1 = sGridP[fIndYmin+nY+2];
#endif //HLTCA_GPU_TEXTURE_FETCH
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
#endif //DRAW
          }
#ifdef DRAW
          if ( sGridP[row.Grid().N()] != row.NHits() ) {
            std::cout << " grid, row " << iRow << ": nHits=" << row.NHits() << ", grid n=" << row.Grid().N() << ", c[n]=" << sGridP[row.Grid().N()] << std::endl;
            //exit(0);
          }
#endif //DRAW
        }
#ifdef DRAW
        if ( drawSearch ) {
          std::cout << " tracklet " << r.fItr << ", row " << iRow << ", yz= " << fY << "," << fZ << ": search hits=" << fHitYfst << " " << fHitYlst << " / " << fHitYfst1 << " " << fHitYlst1 << std::endl;
          std::cout << " hit search :" << std::endl;
        }
#endif //DRAW
        for ( unsigned int fIh = fHitYfst; fIh < fHitYlst; fIh++ ) {
          assert( (signed) fIh < row.NHits() );
          ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
		 hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + fIh);
#else
		  hh = hits[fIh];
#endif //HLTCA_GPU_TEXTURE_FETCH
          int ddy = ( int )( hh.x ) - fY0;
          int ddz = ( int )( hh.y ) - fZ0;
          int dds = CAMath::Abs( ddy ) + CAMath::Abs( ddz );
          if ( drawSearch ) {
            #ifdef DRAW
            std::cout << fIh << ": hityz= " << hh.x << " " << hh.y << "(" << hh.x*stepY << " " << hh.y*stepZ << "), trackyz=" << fY0 << " " << fZ0 << "(" << fY0*stepY << " " << fZ0*stepZ << "), dy,dz,ds= " << ddy << " " << ddz << " " << dds << "(" << ddy*stepY << " " << ddz*stepZ << std::endl;
            #endif //DRAW
          }
          if ( dds < ds ) {
            ds = dds;
            best = fIh;
          }
        }

		for ( unsigned int fIh = fHitYfst1; fIh < fHitYlst1; fIh++ ) {
          ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
		  hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + fIh);
#else
		  hh = hits[fIh];
#endif //HLTCA_GPU_TEXTURE_FETCH
          int ddy = ( int )( hh.x ) - fY0;
          int ddz = ( int )( hh.y ) - fZ0;
          int dds = CAMath::Abs( ddy ) + CAMath::Abs( ddz );
          if ( drawSearch ) {
            #ifdef DRAW
            std::cout << fIh << ": hityz= " << hh.x << " " << hh.y << "(" << hh.x*stepY << " " << hh.y*stepZ << "), trackyz=" << fY0 << " " << fZ0 << "(" << fY0*stepY << " " << fZ0*stepZ << "), dy,dz,ds= " << ddy << " " << ddz << " " << dds << "(" << ddy*stepY << " " << ddz*stepZ << std::endl;
            #endif //DRAW
          }
          if ( dds < ds ) {
            ds = dds;
            best = fIh;
          }
        }
      }// end of search for the closest hit

      if ( best < 0 )
	  {
#ifndef EXTERN_ROW_HITS
		  tracklet.SetRowHit(iRow, -1);
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //!EXTERN_ROW_HITS
		  break;
	  }
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "hit search " << r.fItr << ", row " << iRow << " hit " << best << " found" << std::endl;
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best, kRed, 1. );
        AliHLTTPCCADisplay::Instance().Ask();
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best, kWhite, 1 );
        AliHLTTPCCADisplay::Instance().DrawSliceHit( iRow, best );
		#endif //DRAW
      }

      ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
		 hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + best);
#else
		  hh = hits[best];
#endif //HLTCA_GPU_TEXTURE_FETCH

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
        #endif //DRAW
      }
      if ( CAMath::FMulRZ( dy, dy ) > sy2 || CAMath::FMulRZ( dz, dz ) > sz2  ) {
        if ( drawSearch ) {
          #ifdef DRAW
          std::cout << "found hit is out of the chi2 window\n " << std::endl;
          #endif //DRAW
        }
#ifndef EXTERN_ROW_HITS
		tracklet.SetRowHit(iRow, -1);
#else
		tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //!EXTERN_ROW_HITS
        break;
      }
#ifdef DRAW
      //if( SAVE() ) hitstore[ iRow ] = best;
      //std::cout<<"hit search before filter: "<<r.fItr<<", row "<<iRow<<std::endl;
      //AliHLTTPCCADisplay::Instance().DrawTracklet(tParam, hitstore, kBlue);
      //AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW
      if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
        if ( drawSearch ) {
          #ifdef DRAW
          std::cout << "tracklet " << r.fItr << " at row " << iRow << " : can not filter!!!! " << std::endl;
          #endif //DRAW
        }
        break;
      }
#ifndef EXTERN_ROW_HITS
	  tracklet.SetRowHit( iRow, best );
#else
	  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = best;
#endif //!EXTERN_ROW_HITS
      if ( drawSearch ) {
        #ifdef DRAW
        std::cout << "tracklet " << r.fItr << " after filter at row " << iRow << " : " << std::endl;
        tParam.Print();
        AliHLTTPCCADisplay::Instance().DrawTracklet( tParam, hitstore, kRed );
        AliHLTTPCCADisplay::Instance().Ask();
		#endif //DRAW
      }
      r.fNHits++;
      r.fNMissed = 0;
      if ( r.fStage == 1 ) r.fLastRow = iRow;
      else r.fFirstRow = iRow;
    } while ( 0 );
  }
}

#ifdef HLTCA_GPUCODE
GPUd() void AliHLTTPCCATrackletConstructor::CopyTrackletTempData( AliHLTTPCCAThreadMemory &rMemSrc, AliHLTTPCCAThreadMemory &rMemDst, AliHLTTPCCATrackParam &tParamSrc, AliHLTTPCCATrackParam &tParamDst)
{
	//Copy Temporary Tracklet data from registers to global mem and vice versa
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

GPUd() int AliHLTTPCCATrackletConstructor::FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem, int Reverse, int RowBlock, int &mustInit)
{
	//Fetch a new trackled to be processed by this thread
	__syncthreads();
	int nextTracketlFirstRun = sMem.fNextTrackletFirstRun;
	if (threadIdx.x  == TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		sMem.fNTracklets = *tracker.NTracklets();
		if (sMem.fNextTrackletFirstRun)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_START
			const int iSlice = tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (HLTCA_GPU_BLOCK_COUNT % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % HLTCA_GPU_BLOCK_COUNT != 0)) / HLTCA_GPU_BLOCK_COUNT;
			const int nSliceBlockOffset = HLTCA_GPU_BLOCK_COUNT * iSlice / tracker.GPUParametersConst()->fGPUnSlices;
			const uint2 &nTracklet = tracker.BlockStartingTracklet()[blockIdx.x - nSliceBlockOffset];

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
#endif //HLTCA_GPU_SCHED_FIXED_START
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
				const int nStartFillTrack = CAMath::AtomicAdd(&(*tracker.RowBlockPos(Reverse, RowBlock)).x, nFillTracks);
				if (nFillTracks + nStartFillTrack >= HLTCA_GPU_MAX_TRACKLETS)
				{
					tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW;
				}
				for (int i = 0;i < nFillTracks;i++)
				{
					tracker.RowBlockTracklets(Reverse, RowBlock)[(nStartFillTrack + i) % HLTCA_GPU_MAX_TRACKLETS] = -3;	//Dummy filling track
				}
			}
			sMem.fNextTrackletNoDummy = 0;
		}
	}
	__syncthreads();
	mustInit = 0;
	if (sMem.fNextTrackletCount == 0)
	{
		return(-2);		//No more track in this RowBlock
	}
#if HLTCA_GPU_TRACKLET_CONSTRUCTOR_NMEMTHREDS > 0
	else if (threadIdx.x < TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		return(-1);
	}
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_NMEMTHREDS > 0
	else if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS >= sMem.fNextTrackletCount)
	{
		return(-1);		//No track in this RowBlock for this thread
	}
	else if (nextTracketlFirstRun)
	{
		if (threadIdx.x == TRACKLET_CONSTRUCTOR_NMEMTHREDS) sMem.fNextTrackletFirstRun = 0;
		mustInit = 1;
		return(sMem.fNextTrackletFirst + threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS);
	}
	else
	{
		const int nTrackPos = sMem.fNextTrackletFirst + threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS;
		mustInit = (nTrackPos < tracker.RowBlockPos(Reverse, RowBlock)->w);
		volatile int* const ptrTracklet = &tracker.RowBlockTracklets(Reverse, RowBlock)[nTrackPos % HLTCA_GPU_MAX_TRACKLETS];
		int nTracklet;
		int nTryCount = 0;
		while ((nTracklet = *ptrTracklet) == -1)
		{
			for (int i = 0;i < 20000;i++)
				sMem.fNextTrackletStupidDummy++;
			nTryCount++;
			if (nTryCount > 30)
			{
				tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_SCHEDULE_COLLISION;
				return(-1);
			}
		};
		return(nTracklet);
	}
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(AliHLTTPCCATracker *pTracker)
{
	//Main Tracklet construction function that calls the scheduled (FetchTracklet) and then Processes the tracklet (mainly UpdataTracklet) and at the end stores the tracklet.
	//Can also dispatch a tracklet to be rescheduled
#ifdef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
	pTracker[0].BlockStartingTracklet()[0].x = HLTCA_GPU_EMULATION_SINGLE_TRACKLET;
	pTracker[0].BlockStartingTracklet()[0].y = 1;
	for (int i = 1;i < HLTCA_GPU_BLOCK_COUNT;i++)
	{
		pTracker[0].BlockStartingTracklet()[i].x = pTracker[0].BlockStartingTracklet()[i].y = 0;
	}
#endif //HLTCA_GPU_EMULATION_SINGLE_TRACKLET

	GPUshared() AliHLTTPCCASharedMemory sMem;

#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (threadIdx.x == TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		sMem.fNextTrackletFirstRun = 1;
	}
	__syncthreads();
#endif //HLTCA_GPU_SCHED_FIXED_START

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (threadIdx.x == TRACKLET_CONSTRUCTOR_NMEMTHREDS)
	{
		sMem.fMaxSync = 0;
	}
	int threadSync = 0;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

	for (int iReverse = 0;iReverse < 2;iReverse++)
	{
		for (int iRowBlock = 0;iRowBlock < HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1;iRowBlock++)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_SLICE
			int iSlice = pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + (HLTCA_GPU_BLOCK_COUNT % pTracker[0].GPUParametersConst()->fGPUnSlices != 0 && pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % HLTCA_GPU_BLOCK_COUNT != 0)) / HLTCA_GPU_BLOCK_COUNT;
#else
			for (int iSlice = 0;iSlice < pTracker[0].GPUParametersConst()->fGPUnSlices;iSlice++)
#endif //HLTCA_GPU_SCHED_FIXED_SLICE
			{
				AliHLTTPCCATracker &tracker = pTracker[iSlice];
				if (sMem.fNextTrackletFirstRun && iSlice != tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (HLTCA_GPU_BLOCK_COUNT % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % HLTCA_GPU_BLOCK_COUNT != 0)) / HLTCA_GPU_BLOCK_COUNT)
				{
					continue;
				}
				/*if (!sMem.fNextTrackletFirstRun && tracker.RowBlockPos(1, HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP)->x <= tracker.RowBlockPos(1, HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP)->y)
				{
					continue;
				}*/
				int sharedRowsInitialized = 0;

				int iTracklet;
				int mustInit;
				while ((iTracklet = FetchTracklet(tracker, sMem, iReverse, iRowBlock, mustInit)) != -2)
				{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					CAMath::AtomicMax(&sMem.fMaxSync, threadSync);
					__syncthreads();
					threadSync = CAMath::Min(sMem.fMaxSync, 100000000 / blockDim.x / gridDim.x);
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
#ifndef HLTCA_GPU_PREFETCHDATA
					if (!sharedRowsInitialized)
					{
#ifdef HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY
						if (iReverse)
						{
							for (int i = CAMath::Max(0, HLTCA_ROW_COUNT - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP) * sizeof(AliHLTTPCCARow) / sizeof(int) + threadIdx.x;i < (HLTCA_ROW_COUNT - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP) * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
							{
								reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
							}
						}
						else
						{
							for (int i = CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP) * sizeof(AliHLTTPCCARow) / sizeof(int) + threadIdx.x;i < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, HLTCA_ROW_COUNT) * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
							{
								reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
							}
						}
#else
						for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
						{
							reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
						}
#endif //HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY
						sharedRowsInitialized = 1;
					}
#endif //!HLTCA_GPU_PREFETCHDATA
#ifdef HLTCA_GPU_RESCHED
					short2 storeToRowBlock;
					int storePosition = 0;
					if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = 0;
					}
#endif //HLTCA_GPU_RESCHED
					__syncthreads();
					AliHLTTPCCATrackParam tParam;
					AliHLTTPCCAThreadMemory rMem;

					rMem.fCurrentData = 0;

#ifdef HLTCA_GPU_EMULATION_DEBUG_TRACKLET
					if (iTracklet == HLTCA_GPU_EMULATION_DEBUG_TRACKLET)
					{
						tracker.GPUParameters()->fGPUError = 1;
					}
#endif //HLTCA_GPU_EMULATION_DEBUG_TRACKLET
					AliHLTTPCCAThreadMemory &rMemGlobal = tracker.GPUTrackletTemp()[iTracklet].fThreadMem;
					AliHLTTPCCATrackParam &tParamGlobal = tracker.GPUTrackletTemp()[iTracklet].fParam;
					if (mustInit)
					{
						AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];

						rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
						rMem.fCurrIH = id.HitIndex();
						rMem.fStage = 0;
						rMem.fNHits = 0;
						rMem.fNMissed = 0;

						AliHLTTPCCATrackletConstructor::InitTracklet(tParam);
					}
					else if (iTracklet >= 0)
					{
						CopyTrackletTempData( rMemGlobal, rMem, tParamGlobal, tParam );
					}
#ifdef HLTCA_GPU_PREFETCHDATA
					else if (threadIdx.x < TRACKLET_CONSTRUCTOR_NMEMTHREDS)
					{
						ReadData(threadIdx.x, sMem, rMem, tracker, iReverse ? (HLTCA_ROW_COUNT - 1 - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP) : (CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP)));
					}
#endif //HLTCA_GPU_PREFETCHDATA
					rMem.fItr = iTracklet;
					rMem.fGo = (iTracklet >= 0);

#ifdef HLTCA_GPU_RESCHED
					storeToRowBlock.x = iRowBlock + 1;
					storeToRowBlock.y = iReverse;
#ifdef HLTCA_GPU_PREFETCHDATA
					rMem.fCurrentData ^= 1;
					__syncthreads();
#endif //HLTCA_GPU_PREFETCHDATA
					if (iReverse)
					{
						for (int j = HLTCA_ROW_COUNT - 1 - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP;j >= CAMath::Max(0, HLTCA_ROW_COUNT - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP);j--)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && !(j >= rMem.fEndRow || ( j >= rMem.fStartRow && j - rMem.fStartRow % 2 == 0)))
								pTracker[0].StageAtSync()[threadSync++ * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
#ifdef HLTCA_GPU_PREFETCHDATA
							if (threadIdx.x < TRACKLET_CONSTRUCTOR_NMEMTHREDS && j > CAMath::Max(0, HLTCA_ROW_COUNT - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP))
							{
								ReadData(threadIdx.x, sMem, rMem, tracker, j - 1);
							}
							else
#endif //HLTCA_GPU_PREFETCHDATA
							if (iTracklet >= 0)
							{
								UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
								if (rMem.fNMissed > kMaxRowGap)
								{
									rMem.fGo = 0;
#ifndef HLTCA_GPU_PREFETCHDATA
									break;
#endif //!HLTCA_GPU_PREFETCHDATA
								}
							}
#ifdef HLTCA_GPU_PREFETCHDATA
							__syncthreads();
							rMem.fCurrentData ^= 1;
#endif //HLTCA_GPU_PREFETCHDATA
						}
							
						if (iTracklet >= 0 && (!rMem.fGo || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}
					else
					{
						for (int j = CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP);j < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, HLTCA_ROW_COUNT);j++)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && j >= rMem.fStartRow && (rMem.fStage > 0 || rMem.fCurrIH >= 0 || (j - rMem.fStartRow) % 2 == 0 ))
								pTracker[0].StageAtSync()[threadSync++ * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
#ifdef HLTCA_GPU_PREFETCHDATA
							if (threadIdx.x < TRACKLET_CONSTRUCTOR_NMEMTHREDS && j < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, HLTCA_ROW_COUNT) - 1)
							{
								ReadData(threadIdx.x, sMem, rMem, tracker, j + 1);
							}
							else
#endif	//HLTCA_GPU_PREFETCHDATA
							if (iTracklet >= 0)
							{
								UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
#ifndef HLTCA_GPU_PREFETCHDATA
								//if (rMem.fNMissed > kMaxRowGap || rMem.fGo == 0) break;	//DR!!! CUDA Crashes with this enabled
#endif //!HLTCA_GPU_PREFETCHDATA
							}
#ifdef HLTCA_GPU_PREFETCHDATA
							__syncthreads();
							rMem.fCurrentData ^= 1;
#endif //HLTCA_GPU_PREFETCHDATA
						}
						if (rMem.fGo && (rMem.fNMissed > kMaxRowGap || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
#if defined(HLTCA_GPU_PREFETCHDATA) | !defined(HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY)
							if ( !tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999 ) )
#else
							if ( !tParam.TransportToX( sMem.fRows[ rMem.fEndRow ].X(), tracker.Param().ConstBz(), .999 ) )
#endif //HLTCA_GPU_PREFETCHDATA | !HLTCA_GPU_PREFETCH_ROWBLOCK_ONLY
							{
								rMem.fGo = 0;
							}
							else
							{
								storeToRowBlock.x = (HLTCA_ROW_COUNT - rMem.fEndRow) / HLTCA_GPU_SCHED_ROW_STEP;
								storeToRowBlock.y = 1;
								rMem.fNMissed = 0;
								rMem.fStage = 2;
							}
						}

						if (iTracklet >= 0 && !rMem.fGo)
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}

					if (rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						CopyTrackletTempData( rMem, rMemGlobal, tParam, tParamGlobal );
						storePosition = CAMath::AtomicAdd(&sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x], 1);
					}
#else
					if (threadIdx.x % HLTCA_GPU_WARP_SIZE == 0)
					{
						sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 160;
						sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 0;
					}
					__syncthreads();
					if (iTracklet >= 0)
					{
						CAMath::AtomicMin(&sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
					}
					__syncthreads();
					if (iTracklet >= 0)
					{
						for (int j = sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
						{
							UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
							if (!rMem.fGo) break;
						}

						rMem.fNMissed = 0;
						rMem.fStage = 2;
						if ( rMem.fGo )
						{
							if ( !tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999 ) )  rMem.fGo = 0;
						}
						CAMath::AtomicMax(&sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
					}

					__syncthreads();
					if (iTracklet >= 0)
					{
						for (int j = rMem.fEndRow;j >= 0;j--)
						{
							if (!rMem.fGo) break;
							if (j <= rMem.fEndRow)
							{
								UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
							}
						}

						StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
					}
#endif //HLTCA_GPU_RESCHED

#ifdef HLTCA_GPU_RESCHED
					__syncthreads();
					if (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = (threadIdx.x - TRACKLET_CONSTRUCTOR_NMEMTHREDS) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						if (sMem.fTrackletStoreCount[nReverse][nRowBlock])
						{
							sMem.fTrackletStoreCount[nReverse][nRowBlock] = CAMath::AtomicAdd(&tracker.RowBlockPos(nReverse, nRowBlock)->x, sMem.fTrackletStoreCount[nReverse][nRowBlock]);
						}
					}
					__syncthreads();
					if (iTracklet >= 0 && rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						tracker.RowBlockTracklets(storeToRowBlock.y, storeToRowBlock.x)[sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x] + storePosition] = iTracklet;
					}
					__syncthreads();
#endif //HLTCA_GPU_RESCHED
				}
			}
		}
	}
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(int iTracklet, AliHLTTPCCATracker &tracker)
{
	//Initialize Row Blocks

#ifndef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];
#ifdef HLTCA_GPU_SCHED_FIXED_START
	const int firstDynamicTracklet = tracker.GPUParameters()->fScheduleFirstDynamicTracklet;
	if (iTracklet >= firstDynamicTracklet)
#endif //HLTCA_GPU_SCHED_FIXED_START
	{
#ifdef HLTCA_GPU_SCHED_FIXED_START
		const int firstTrackletInRowBlock = CAMath::Max(firstDynamicTracklet, tracker.RowStartHitCountOffset()[CAMath::Max(id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP * HLTCA_GPU_SCHED_ROW_STEP, 1)].z);
#else
		const int firstTrackletInRowBlock = tracker.RowStartHitCountOffset()[CAMath::Max(id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP * HLTCA_GPU_SCHED_ROW_STEP, 1)].z;
#endif //HLTCA_GPU_SCHED_FIXED_START

		if (iTracklet == firstTrackletInRowBlock)
		{
			const int firstRowInNextBlock = (id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_SCHED_ROW_STEP;
			int trackletsInRowBlock;
			if (firstRowInNextBlock >= HLTCA_ROW_COUNT - 3)
				trackletsInRowBlock = *tracker.NTracklets() - firstTrackletInRowBlock;
			else
#ifdef HLTCA_GPU_SCHED_FIXED_START
				trackletsInRowBlock = CAMath::Max(firstDynamicTracklet, tracker.RowStartHitCountOffset()[firstRowInNextBlock].z) - firstTrackletInRowBlock;
#else
				trackletsInRowBlock = tracker.RowStartHitCountOffset()[firstRowInNextBlock].z - firstTrackletInRowBlock;
#endif //HLTCA_GPU_SCHED_FIXED_START

			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->x = trackletsInRowBlock;
			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->w = trackletsInRowBlock;
		}
		tracker.RowBlockTracklets(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)[iTracklet - firstTrackletInRowBlock] = iTracklet;
	}
#endif //!HLTCA_GPU_EMULATION_SINGLE_TRACKLET
}

GPUg() void AliHLTTPCCATrackletConstructorInit(int iSlice)
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit
	AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[iSlice];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= *tracker.NTracklets()) return;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(i, tracker);
}

GPUg() void AliHLTTPCCATrackletConstructorGPU()
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	AliHLTTPCCATracker *pTracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker);
}

GPUg() void AliHLTTPCCATrackletConstructorGPUPP(int firstSlice, int sliceCount)
{
	if (blockIdx.x >= sliceCount) return;
	AliHLTTPCCATracker *pTracker = &( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + blockIdx.x];
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(pTracker);
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(AliHLTTPCCATracker *tracker)
{
	GPUshared() AliHLTTPCCASharedMemory sMem;
	sMem.fNTracklets = *tracker->NTracklets();

	for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
	{
		reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker->SliceDataRows())[i];
	}

	for (int iTracklet = threadIdx.x;iTracklet < (*tracker->NTracklets() / HLTCA_GPU_THREAD_COUNT + 1) * HLTCA_GPU_THREAD_COUNT;iTracklet += blockDim.x)
	{
		AliHLTTPCCATrackParam tParam;
		AliHLTTPCCAThreadMemory rMem;
		
		if (iTracklet < *tracker->NTracklets())
		{
			AliHLTTPCCAHitId id = tracker->TrackletTmpStartHits()[iTracklet];

			rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
			rMem.fCurrIH = id.HitIndex();
			rMem.fStage = 0;
			rMem.fNHits = 0;
			rMem.fNMissed = 0;

			AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

			rMem.fItr = iTracklet;
			rMem.fGo = 1;
		}

		if (threadIdx.x % HLTCA_GPU_WARP_SIZE == 0)
		{
			sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 160;
			sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 0;
		}
		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			CAMath::AtomicMin(&sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
		}
		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
			{
				UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam, j);
				if (!rMem.fGo) break;
			}

			rMem.fNMissed = 0;
			rMem.fStage = 2;
			if ( rMem.fGo )
			{
				if ( !tParam.TransportToX( tracker->Row( rMem.fEndRow ).X(), tracker->Param().ConstBz(), .999 ) )  rMem.fGo = 0;
			}
			CAMath::AtomicMax(&sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
		}

		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = rMem.fEndRow;j >= 0;j--)
			{
				if (!rMem.fGo) break;
				if (j <= rMem.fEndRow)
				{
					UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam, j);
				}
			}
			StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam );
		}
	}
}

#else //HLTCA_GPUCODE

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorCPU(AliHLTTPCCATracker &tracker)
{
	//Tracklet constructor simple CPU Function that does not neew a scheduler
	GPUshared() AliHLTTPCCASharedMemory sMem;
	sMem.fNTracklets = *tracker.NTracklets();
	for (int iTracklet = 0;iTracklet < *tracker.NTracklets();iTracklet++)
	{
		AliHLTTPCCATrackParam tParam;
		AliHLTTPCCAThreadMemory rMem;
		
		AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];

		rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
		rMem.fCurrIH = id.HitIndex();
		rMem.fStage = 0;
		rMem.fNHits = 0;
		rMem.fNMissed = 0;

		AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

		rMem.fItr = iTracklet;
		rMem.fGo = 1;

		for (int j = rMem.fStartRow;j < tracker.Param().NRows();j++)
		{
			UpdateTracklet(1, 1, 0, iTracklet, sMem, rMem, tracker, tParam, j);
			if (!rMem.fGo) break;
		}

		rMem.fNMissed = 0;
		rMem.fStage = 2;
		if ( rMem.fGo )
		{
			if ( !tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999 ) ) rMem.fGo = 0;
		}

		for (int j = rMem.fEndRow;j >= 0;j--)
		{
			if (!rMem.fGo) break;
			UpdateTracklet( 1, 1, 0, iTracklet, sMem, rMem, tracker, tParam, j);
		}

		StoreTracklet( 1, 1, 0, iTracklet, sMem, rMem, tracker, tParam );
	}
}
#endif //HLTCA_GPUCODE
