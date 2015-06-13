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

#define kMaxRowGap 4

MEM_CLASS_PRE2() GPUdi() void AliHLTTPCCATrackletConstructor::InitTracklet( MEM_LG2(AliHLTTPCCATrackParam) &tParam )
{
  //Initialize Tracklet Parameters using default values
  tParam.InitParam();
}

MEM_CLASS_PRE2() GPUdi() bool AliHLTTPCCATrackletConstructor::CheckCov(MEM_LG2(AliHLTTPCCATrackParam) &tParam)
{
      bool ok = 1;
      const float *c = tParam.Cov();
      for ( int i = 0; i < 15; i++ ) ok = ok && CAMath::Finite( c[i] );
      for ( int i = 0; i < 5; i++ ) ok = ok && CAMath::Finite( tParam.Par()[i] );
      ok = ok && ( tParam.X() > 50 );
	  if ( c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0 ) ok = 0;
	  return(ok);
}


MEM_CLASS_PRE23() GPUdi() void AliHLTTPCCATrackletConstructor::StoreTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory)
#if defined(HLTCA_GPUCODE) | defined(EXTERN_ROW_HITS)
  &s
#else
  &/*s*/
#endif  //!HLTCA_GPUCODE
  , AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_LG2(AliHLTTPCCATracker) &tracker, MEM_LG3(AliHLTTPCCATrackParam) &tParam )
{
  // reconstruction of tracklets, tracklet store step

  do {
    if ( r.fNHits < TRACKLET_SELECTOR_MIN_HITS(tParam.QPt()) ) {
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
	  bool ok = CheckCov(tParam);

      if ( !ok ) {
        r.fNHits = 0;
        break;
      }
    }
  } while ( 0 );

  if ( !SAVE() ) return;

/*#ifndef HLTCA_GPUCODE
  printf("Tracklet %d: Hits %3d NDF %3d Chi %8.4f Sign %f Cov: %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\n", r.fItr, r.fNHits, tParam.GetNDF(), tParam.GetChi2(), tParam.GetSignCosPhi(),
	  tParam.Cov()[0], tParam.Cov()[1], tParam.Cov()[2], tParam.Cov()[3], tParam.Cov()[4], tParam.Cov()[5], tParam.Cov()[6], tParam.Cov()[7], tParam.Cov()[8], tParam.Cov()[9], 
	  tParam.Cov()[10], tParam.Cov()[11], tParam.Cov()[12], tParam.Cov()[13], tParam.Cov()[14]);
#endif*/

  GPUglobalref() MEM_GLOBAL(AliHLTTPCCATracklet) &tracklet = tracker.Tracklets()[r.fItr];

  tracklet.SetNHits( r.fNHits );

  if ( r.fNHits > 0 ) {
    if ( CAMath::Abs( tParam.Par()[4] ) < 1.e-4 ) tParam.SetPar( 4, 1.e-4 );
	if (r.fStartRow < r.fFirstRow) r.fFirstRow = r.fStartRow;
	tracklet.SetFirstRow( r.fFirstRow );
    tracklet.SetLastRow( r.fLastRow );
#ifdef HLTCA_GPUCODE
    tracklet.SetParam( tParam.fParam );
#else
    tracklet.SetParam( tParam.GetParam() );
#endif //HLTCA_GPUCODE
    int w = tracker.CalculateHitWeight(r.fNHits, tParam.GetChi2(), r.fItr);
    tracklet.SetHitWeight(w);
    for ( int iRow = r.fFirstRow; iRow <= r.fLastRow; iRow++ ) {
#ifdef EXTERN_ROW_HITS
      int ih = tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr];
#else
	  int ih = tracklet.RowHit( iRow );
#endif //EXTERN_ROW_HITS
      if ( ih >= 0 ) {
#if defined(HLTCA_GPUCODE)
   	    tracker.MaximizeHitWeight( s.fRows[ iRow ], ih, w );
#else
	    tracker.MaximizeHitWeight( tracker.Row( iRow ), ih, w );
#endif //HLTCA_GPUCODE
      }
    }
  }

}

MEM_CLASS_PRE2() GPUdi() void AliHLTTPCCATrackletConstructor::UpdateTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory)
#if defined(HLTCA_GPUCODE) | defined(EXTERN_ROW_HITS)
  &s
#else
  &/*s*/
#endif //HLTCA_GPUCODE
  , AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, MEM_LG2(AliHLTTPCCATrackParam) &tParam, int iRow )
{
  // reconstruction of tracklets, tracklets update step

  if ( !r.fGo ) return;

#ifndef EXTERN_ROW_HITS
  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS

#if defined(HLTCA_GPUCODE)
  const GPUsharedref() MEM_LOCAL(AliHLTTPCCARow) &row = s.fRows[iRow];
#else
  const GPUglobalref() MEM_GLOBAL(AliHLTTPCCARow) &row = tracker.Row( iRow );
#endif //HLTCA_GPUCODE

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


	  ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
	  hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + r.fCurrIH);
#else
	  hh = tracker.HitData(row)[r.fCurrIH];
#endif //HLTCA_GPU_TEXTURE_FETCH

      int oldIH = r.fCurrIH;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
	  r.fCurrIH = tex1Dfetch(gAliTexRefs, ((char*) tracker.Data().HitLinkUpData(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + r.fCurrIH);
#else
	  r.fCurrIH = tracker.HitLinkUpData(row)[r.fCurrIH]; // read from linkup data
#endif //HLTCA_GPU_TEXTURE_FETCH

      float x = row.X();
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;

      if ( iRow == r.fStartRow ) {
        tParam.SetX( x );
        tParam.SetY( y );
        tParam.SetZ( z );
        r.fLastY = y;
        r.fLastZ = z;
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
        float sinPhi, cosPhi;
        if ( r.fNHits >= 10 && CAMath::Abs( tParam.SinPhi() ) < .99 ) {
          sinPhi = tParam.SinPhi();
          cosPhi = CAMath::Sqrt( 1 - sinPhi * sinPhi );
        } else {
          sinPhi = dy * ri;
          cosPhi = dx * ri;
        }
        if ( !tParam.TransportToX( x, sinPhi, cosPhi, tracker.Param().ConstBz(), -1 ) ) {
#ifndef EXTERN_ROW_HITS
          tracklet.SetRowHit( iRow, -1 );
#else
		  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //EXTERN_ROW_HITS
          break;
        }
        tracker.GetErrors2( iRow, tParam.GetZ(), sinPhi, cosPhi, tParam.GetDzDs(), err2Y, err2Z );

        if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
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
      r.fNHits++;
      r.fLastRow = iRow;
      r.fEndRow = iRow;
      break;
    } while ( 0 );

    if ( r.fCurrIH < 0 ) {
      r.fStage = 1;
      if ( CAMath::Abs( tParam.SinPhi() ) > .999 ) {
        r.fNHits = 0; r.fGo = 0;
      } else {
        //tParam.SetCosPhi( CAMath::Sqrt(1-tParam.SinPhi()*tParam.SinPhi()) );
      }
    }
  } else { // forward/backward searching part
    do {
      if ( r.fStage == 2 && ( ( iRow >= r.fEndRow ) ||
                              ( iRow >= r.fStartRow && ( iRow - r.fStartRow ) % 2 == 0 )
                            ) ) break;
      if ( r.fNMissed > kMaxRowGap  ) {
        break;
      }

      r.fNMissed++;

      float x = row.X();
      float err2Y, err2Z;
      if ( !tParam.TransportToX( x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz(), .99 ) ) {
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

#ifndef HLTCA_GPU_TEXTURE_FETCH
	  GPUglobalref() const ushort2 *hits = tracker.HitData(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH

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

        {
          int nY = row.Grid().Ny();

#ifndef HLTCA_GPU_TEXTURE_FETCH
		  GPUglobalref()  const unsigned short *sGridP = tracker.FirstHitInBin(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH

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
        }

		for ( unsigned int fIh = fHitYfst; fIh < fHitYlst; fIh++ ) {
          assert( (signed) fIh < row.NHits() );
          ushort2 hh;
		  if (r.fStage <= 2 || tracker.HitWeight(row, fIh) >= 0)
		  {

#if defined(HLTCA_GPU_TEXTURE_FETCH)
			hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + fIh);
#else
			hh = hits[fIh];
#endif //HLTCA_GPU_TEXTURE_FETCH
			int ddy = ( int )( hh.x ) - fY0;
			int ddz = ( int )( hh.y ) - fZ0;
			int dds = CAMath::Abs( ddy ) + CAMath::Abs( ddz );
			if ( dds < ds ) {
				ds = dds;
				best = fIh;
			}
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

      ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
		 hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + best);
#else
		  hh = hits[best];
#endif //HLTCA_GPU_TEXTURE_FETCH

      tracker.GetErrors2( iRow, *( ( MEM_LG2(AliHLTTPCCATrackParam)* )&tParam ), err2Y, err2Z );

      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
      float dy = y - fY;
      float dz = z - fZ;

      const float kFactor = tracker.Param().HitPickUpFactor() * tracker.Param().HitPickUpFactor() * 3.5 * 3.5;
      float sy2 = kFactor * ( tParam.GetErr2Y() +  err2Y );
      float sz2 = kFactor * ( tParam.GetErr2Z() +  err2Z );
      if ( sy2 > 2. ) sy2 = 2.;
      if ( sz2 > 2. ) sz2 = 2.;

      if ( CAMath::FMulRZ( dy, dy ) > sy2 || CAMath::FMulRZ( dz, dz ) > sz2  ) {
#ifndef EXTERN_ROW_HITS
		tracklet.SetRowHit(iRow, -1);
#else
		tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = -1;
#endif //!EXTERN_ROW_HITS
        break;
      }
#ifdef GLOBAL_TRACKING_EXTRAPOLATE_ONLY
      if ( r.fStage <= 2)
#endif
		  if (!tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
			break;
		  }
#ifndef EXTERN_ROW_HITS
	  tracklet.SetRowHit( iRow, best );
#else
	  tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = best;
#endif //!EXTERN_ROW_HITS
      r.fNHits++;
      r.fNMissed = 0;
      if ( r.fStage == 1 ) r.fLastRow = iRow;
      else r.fFirstRow = iRow;
    } while ( 0 );
  }
}

#ifdef HLTCA_GPUCODE

#include "AliHLTTPCCATrackletConstructorGPU.h"

#else //HLTCA_GPUCODE

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorCPU(AliHLTTPCCATracker &tracker)
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

GPUdi() int AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGlobalTracking(AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam& tParam, int row, int increment)
{
	AliHLTTPCCAThreadMemory rMem;	
	GPUshared() AliHLTTPCCASharedMemory sMem;
	sMem.fNTracklets = *tracker.NTracklets();
	rMem.fItr = 0;
	rMem.fStage = 3;
	rMem.fNHits = rMem.fNMissed = 0;
	rMem.fGo = 1;
	while (rMem.fGo && row >= 0 && row < tracker.Param().NRows())
	{
		UpdateTracklet(1, 1, 0, 0, sMem, rMem, tracker, tParam, row);
		row += increment;
	}
	if (!CheckCov(tParam)) rMem.fNHits = 0;
	return(rMem.fNHits);
}

#endif //HLTCA_GPUCODE
