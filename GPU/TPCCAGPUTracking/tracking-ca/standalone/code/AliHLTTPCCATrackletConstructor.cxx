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

GPUdi() void AliHLTTPCCATrackletConstructor::InitTracklet( AliHLTTPCCATrackParam &tParam )
{
  //Initialize Tracklet Parameters using default values
  tParam.InitParam();
}


GPUdi() void AliHLTTPCCATrackletConstructor::StoreTracklet
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

  do {
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

GPUdi() void AliHLTTPCCATrackletConstructor::UpdateTracklet
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

  if ( !r.fGo ) return;

#ifndef EXTERN_ROW_HITS
  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS

#if defined(HLTCA_GPUCODE)
  const AliHLTTPCCARow &row = s.fRows[iRow];
#else
  const AliHLTTPCCARow &row = tracker.Row( iRow );
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
	  const ushort2 *hits = tracker.HitData(row);
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
		  const unsigned short *sGridP = tracker.FirstHitInBin(row);
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

      tracker.GetErrors2( iRow, *( ( AliHLTTPCCATrackParam* )&tParam ), err2Y, err2Z );

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
      if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
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
GPUdi() void AliHLTTPCCATrackletConstructor::CopyTrackletTempData( AliHLTTPCCAThreadMemory &rMemSrc, AliHLTTPCCAThreadMemory &rMemDst, AliHLTTPCCATrackParam &tParamSrc, AliHLTTPCCATrackParam &tParamDst)
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

#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem, int Reverse, int RowBlock, int &mustInit)
{
	//Fetch a new trackled to be processed by this thread
	__syncthreads();
	int nextTrackletFirstRun = sMem.fNextTrackletFirstRun;
	if (threadIdx.x == 0)
	{
		sMem.fNTracklets = *tracker.NTracklets();
		if (sMem.fNextTrackletFirstRun)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_START
			const int iSlice = tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x;
			const int nSliceBlockOffset = gridDim.x * iSlice / tracker.GPUParametersConst()->fGPUnSlices;
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
				}
			}
#endif //HLTCA_GPU_SCHED_FIXED_START
		}
		else
		{
			const int4 oldPos = *tracker.RowBlockPos(Reverse, RowBlock);
			const int nFetchTracks = CAMath::Max(CAMath::Min(oldPos.x - oldPos.y, HLTCA_GPU_THREAD_COUNT), 0);
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
					tracker.RowBlockTracklets(Reverse, RowBlock)[(nStartFillTrack + i) % HLTCA_GPU_MAX_TRACKLETS] = -(blockIdx.x * 1000000 + nFetchTracks * 10000 + oldPos.x * 100 + oldPos.y);	//Dummy filling track
				}
			}
		}
	}
	__syncthreads();
	mustInit = 0;
	if (sMem.fNextTrackletCount == 0)
	{
		return(-2);		//No more track in this RowBlock
	}
	else if (threadIdx.x >= sMem.fNextTrackletCount)
	{
		return(-1);		//No track in this RowBlock for this thread
	}
	else if (nextTrackletFirstRun)
	{
		if (threadIdx.x == 0) sMem.fNextTrackletFirstRun = 0;
		mustInit = 1;
		return(sMem.fNextTrackletFirst + threadIdx.x);
	}
	else
	{
		const int nTrackPos = sMem.fNextTrackletFirst + threadIdx.x;
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

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(AliHLTTPCCATracker *pTracker)
{
	//Main Tracklet construction function that calls the scheduled (FetchTracklet) and then Processes the tracklet (mainly UpdataTracklet) and at the end stores the tracklet.
	//Can also dispatch a tracklet to be rescheduled
#ifdef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
	pTracker[0].BlockStartingTracklet()[0].x = HLTCA_GPU_EMULATION_SINGLE_TRACKLET;
	pTracker[0].BlockStartingTracklet()[0].y = 1;
	for (int i = 1;i < gridDim.x;i++)
	{
		pTracker[0].BlockStartingTracklet()[i].x = pTracker[0].BlockStartingTracklet()[i].y = 0;
	}
#endif //HLTCA_GPU_EMULATION_SINGLE_TRACKLET

	GPUshared() AliHLTTPCCASharedMemory sMem;

#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (threadIdx.x == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}
	__syncthreads();
#endif //HLTCA_GPU_SCHED_FIXED_START

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (threadIdx.x == 0)
	{
		sMem.fMaxSync = 0;
	}
	int threadSync = 0;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

	for (int iReverse = 0;iReverse < 2;iReverse++)
	{
		for (volatile int iRowBlock = 0;iRowBlock < HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1;iRowBlock++)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_SLICE
			int iSlice = pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % pTracker[0].GPUParametersConst()->fGPUnSlices != 0 && pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x;
#else
			for (int iSlice = 0;iSlice < pTracker[0].GPUParametersConst()->fGPUnSlices;iSlice++)
#endif //HLTCA_GPU_SCHED_FIXED_SLICE
			{
				AliHLTTPCCATracker &tracker = pTracker[iSlice];
				if (blockIdx.x != 7 && sMem.fNextTrackletFirstRun && iSlice != (tracker.GPUParametersConst()->fGPUnSlices > gridDim.x ? blockIdx.x : (tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x)))
				{
					continue;
				}

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
					if (!sharedRowsInitialized)
					{
						for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
						{
							reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
						}
						sharedRowsInitialized = 1;
					}
#ifdef HLTCA_GPU_RESCHED
					short2 storeToRowBlock;
					int storePosition = 0;
					if (threadIdx.x < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = threadIdx.x / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = threadIdx.x % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = 0;
					}
#else
					mustInit = 1;
#endif //HLTCA_GPU_RESCHED
					__syncthreads();
					AliHLTTPCCATrackParam tParam;
					AliHLTTPCCAThreadMemory rMem;

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
					rMem.fItr = iTracklet;
					rMem.fGo = (iTracklet >= 0);

#ifdef HLTCA_GPU_RESCHED
					storeToRowBlock.x = iRowBlock + 1;
					storeToRowBlock.y = iReverse;
					if (iReverse)
					{
						for (int j = HLTCA_ROW_COUNT - 1 - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP;j >= CAMath::Max(0, HLTCA_ROW_COUNT - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP);j--)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && !(j >= rMem.fEndRow || ( j >= rMem.fStartRow && j - rMem.fStartRow % 2 == 0)))
								pTracker[0].StageAtSync()[threadSync++ * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (iTracklet >= 0)
							{
								UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
								if (rMem.fNMissed > kMaxRowGap && j <= rMem.fStartRow)
								{
									rMem.fGo = 0;
									break;
								}
							}
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
							if (iTracklet >= 0)
							{
								UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
								//if (rMem.fNMissed > kMaxRowGap || rMem.fGo == 0) break;	//DR!!! CUDA Crashes with this enabled
							}
						}
						if (rMem.fGo && (rMem.fNMissed > kMaxRowGap || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
							if ( !tParam.TransportToX( sMem.fRows[ rMem.fEndRow ].X(), tracker.Param().ConstBz(), .999 ) )
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

					__syncthreads();
					if (threadIdx.x < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = threadIdx.x / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = threadIdx.x % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
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
							UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
						}

						StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
					}
#endif //HLTCA_GPU_RESCHED
				}
			}
		}
	}
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(int iTracklet, AliHLTTPCCATracker &tracker)
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

#else //HLTCA_GPU_ALTERNATIVE_SCHEDULER

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(AliHLTTPCCATracker *pTracker)
{
	const int nativeslice = blockIdx.x % pTracker[0].GPUParametersConst()->fGPUnSlices;
	GPUshared() AliHLTTPCCASharedMemory sMem;

	for (int iSlice = 0;iSlice < pTracker[0].GPUParametersConst()->fGPUnSlices;iSlice++)
	{
		if (iSlice) __syncthreads();
		AliHLTTPCCATracker &tracker = pTracker[(nativeslice + iSlice) % pTracker[0].GPUParametersConst()->fGPUnSlices];
		if (threadIdx.x == 0)
		{
			sMem.fNTracklets = *tracker.NTracklets();
		}

		for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
		{
			reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
		}

		int iTracklet;

		/*if (iSlice == 0)
		{
			iTracklet = (blockIdx.x - nativeslice) / tracker.GPUParametersConst()->fGPUnSlices * HLTCA_GPU_THREAD_COUNT;
		}
		else
		{
			if (threadIdx.x == 0)
			{
				if (tracker.GPUParameters()->fNextTracklet < *tracker.NTracklets())
				{
					sMem.fNextTrackletFirst = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, HLTCA_GPU_THREAD_COUNT);
				}
				else
				{
					sMem.fNextTrackletFirst = -1;
				}
			}
		}*/

		__syncthreads();
		for (iTracklet = blockIdx.x * blockDim.x + threadIdx.x;iTracklet < *tracker.NTracklets();iTracklet += blockDim.x * gridDim.x)
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
}

#endif //HLTCA_GPU_ALTERNATIVE_SCHEDULER

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
				UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam, j);
			}
			StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam );
		}
	}
}

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
#endif //HLTCA_GPUCODE
