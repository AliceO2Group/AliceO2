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
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCAHit.h"

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

#ifdef EXTERN_ROW_HITS
#define GETRowHit(iRow) tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr]
#define SETRowHit(iRow, val) tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = val
#else
#define GETRowHit(iRow) tracklet.RowHit(iRow)
#define SETRowHit(iRow, val) tracklet.SetRowHit(iRow, val)
#endif

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
      int ih = GETRowHit(iRow);
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

  if ( r.fStage == 0 ) { // fitting part
    do {

      if ( iRow < r.fStartRow || r.fCurrIH < 0  ) break;
      if ( ( iRow - r.fStartRow ) % 2 != 0 )
	  {
          SETRowHit(iRow, -1);
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
          SETRowHit(iRow, -1);
          break;
        }
        tracker.GetErrors2( iRow, tracker.Param().GetContinuousTracking() ? 125. : tParam.GetZ(), sinPhi, cosPhi, tParam.GetDzDs(), err2Y, err2Z );

        if ( !tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
          SETRowHit(iRow, -1);
          break;
        }
      }
      SETRowHit(iRow, oldIH);
      r.fNHits++;
      r.fLastRow = iRow;
      r.fEndRow = iRow;
      break;
    } while ( 0 );
    
    /*QQQQprintf("Extrapolate Row %d X %f Y %f Z %f SinPhi %f DzDs %f QPt %f", iRow, tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt());
    for (int i = 0;i < 15;i++) printf(" C%d=%6.2f", i, tParam.GetCov(i));
    printf("\n");*/

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
        SETRowHit(iRow, -1);
        break;
      }
      if ( row.NHits() < 1 ) {
        // skip empty row
          SETRowHit(iRow, -1);
        break;
      }

#ifndef HLTCA_GPU_TEXTURE_FETCH
      GPUglobalref() const ushort2 *hits = tracker.HitData(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH
      float fY = tParam.GetY();
      float fZ = tParam.GetZ();
      int best = -1;

      { // search for the closest hit
        tracker.GetErrors2( iRow, *( ( MEM_LG2(AliHLTTPCCATrackParam)* )&tParam ), err2Y, err2Z );
        const float kFactor = tracker.Param().HitPickUpFactor() * tracker.Param().HitPickUpFactor() * 3.5 * 3.5;
        float sy2 = kFactor * ( tParam.GetErr2Y() +  err2Y );
        float sz2 = kFactor * ( tParam.GetErr2Z() +  err2Z );
        if ( sy2 > 2. ) sy2 = 2.;
        if ( sz2 > 2. ) sz2 = 2.;
                                
        int bin, ny, nz;
        row.Grid().GetBinArea(fY, fZ, 1.5, 1.5, bin, ny, nz);
        float ds = 1e6;
#ifndef HLTCA_GPU_TEXTURE_FETCH
        GPUglobalref()  const unsigned short *sGridP = tracker.FirstHitInBin(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH
        for (int k = 0;k <= nz;k++)
        {
          int nBinsY = row.Grid().Ny();
          int mybin = bin + k * nBinsY;
#ifdef HLTCA_GPU_TEXTURE_FETCH
          unsigned int hitFst = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + mybin);
          unsigned int hitLst = tex1Dfetch(gAliTexRefu, ((char*) tracker.Data().FirstHitInBin(row) - tracker.Data().GPUTextureBase()) / sizeof(unsigned short) + mybin + ny + 1);
#else
          unsigned int hitFst = sGridP[mybin];
          unsigned int hitLst = sGridP[mybin + ny + 1];
#endif //HLTCA_GPU_TEXTURE_FETCH                      
          for ( unsigned int ih = hitFst; ih < hitLst; ih++ ) {
            assert( (signed) ih < row.NHits() );
            ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
            hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + ih);
#else
            hh = hits[ih];
#endif //HLTCA_GPU_TEXTURE_FETCH
            float y = y0 + hh.x * stepY;
            float z = z0 + hh.y * stepZ;
            float dy = y - fY;
            float dz = z - fZ;
            if (dy * dy < sy2 && dz * dz < sz2) {
              float dds = HLTCA_Y_FACTOR * fabs(dy) + fabs(dz);
              if ( dds < ds ) {
                ds = dds;
                best = ih;
              }
            }
          }
        }
      }// end of search for the closest hit

      if ( best < 0 )
      {
        SETRowHit(iRow, -1);
        break;
      }

      ushort2 hh;
#if defined(HLTCA_GPU_TEXTURE_FETCH)
      hh = tex1Dfetch(gAliTexRefu2, ((char*) tracker.Data().HitData() - tracker.Data().GPUTextureBase()) / sizeof(ushort2) + row.HitNumberOffset() + best);
#else
      hh = hits[best];
#endif //HLTCA_GPU_TEXTURE_FETCH
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;

      if (!tParam.Filter( y, z, err2Y, err2Z, .99 ) ) {
        break;
      }
      SETRowHit(iRow, best);
      r.fNHits++;
      r.fNMissed = 0;
      if ( r.fStage == 1 ) r.fLastRow = iRow;
      else r.fFirstRow = iRow;
    } while ( 0 );
  }
}

GPUdi() void AliHLTTPCCATrackletConstructor::DoTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker)& tracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem, AliHLTTPCCAThreadMemory& rMem)
{
	int iRow = 0, iRowEnd = tracker.Param().NRows();;
	MEM_PLAIN(AliHLTTPCCATrackParam) tParam;
	if (rMem.fGo)
	{
		AliHLTTPCCAHitId id = tracker.TrackletStartHits()[rMem.fItr];

		rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
		rMem.fCurrIH = id.HitIndex();
		rMem.fNMissed = 0;
		iRow = rMem.fStartRow;
		AliHLTTPCCATrackletConstructor::InitTracklet(tParam);
	}
	rMem.fStage = 0;
	rMem.fNHits = 0;

	for (int k = 0;k < 2;k++)
	{
		for (;iRow != iRowEnd;iRow += rMem.fStage == 2 ? -1 : 1)
		{
			UpdateTracklet(0, 0, 0, 0, sMem, rMem, tracker, tParam, iRow);
		}

		if (rMem.fStage == 2)
		{
			StoreTracklet( 0, 0, 0, 0, sMem, rMem, tracker, tParam );
		}
		else
		{
			rMem.fNMissed = 0;
			rMem.fStage = 2;
			if (rMem.fGo) if (!tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999)) rMem.fGo = 0;
			iRow = rMem.fEndRow;
			iRowEnd = -1;
		}
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
		AliHLTTPCCAThreadMemory rMem;
		rMem.fItr = iTracklet;
		rMem.fGo = 1;

		DoTracklet(tracker, sMem, rMem);
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
