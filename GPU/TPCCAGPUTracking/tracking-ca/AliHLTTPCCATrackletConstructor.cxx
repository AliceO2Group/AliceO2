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

#ifdef HLTCA_GPUCODE
  #define MAKESharedRef(vartype, varname, varglobal, varshared) const GPUsharedref() MEM_LOCAL(vartype) &varname = varshared;
#else
  #define MAKESharedRef(vartype, varname, varglobal, varshared) const GPUglobalref() MEM_GLOBAL(vartype) &varname = varglobal;
#endif

#ifdef HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
  #define TEXTUREFetchCons(type, texture, address, entry) tex1Dfetch(texture, ((char*) address - tracker.Data().GPUTextureBase()) / sizeof(type) + entry);
#else
  #define TEXTUREFetchCons(type, texture, address, entry) address[entry];
#endif

MEM_CLASS_PRE23() GPUdi() void AliHLTTPCCATrackletConstructor::StoreTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_LG2(AliHLTTPCCATracker) &tracker, MEM_LG3(AliHLTTPCCATrackParam) &tParam )
{
  // reconstruction of tracklets, tracklet store step
  if ( r.fNHits < TRACKLET_SELECTOR_MIN_HITS(tParam.QPt()) ||
    !CheckCov(tParam) ||
    AliHLTTPCCAMath::Abs(tParam.GetQPt()) > tracker.Param().MaxTrackQPt() )
  {
    r.fNHits = 0;
  }

/*printf("Tracklet %d: Hits %3d NDF %3d Chi %8.4f Sign %f Cov: %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\n", r.fItr, r.fNHits, tParam.GetNDF(), tParam.GetChi2(), tParam.GetSignCosPhi(),
	  tParam.Cov()[0], tParam.Cov()[1], tParam.Cov()[2], tParam.Cov()[3], tParam.Cov()[4], tParam.Cov()[5], tParam.Cov()[6], tParam.Cov()[7], tParam.Cov()[8], tParam.Cov()[9], 
	  tParam.Cov()[10], tParam.Cov()[11], tParam.Cov()[12], tParam.Cov()[13], tParam.Cov()[14]);*/

  GPUglobalref() MEM_GLOBAL(AliHLTTPCCATracklet) &tracklet = tracker.Tracklets()[r.fItr];

  tracklet.SetNHits( r.fNHits );

  if ( r.fNHits > 0 ) {
    tracklet.SetFirstRow( r.fFirstRow );
    tracklet.SetLastRow( r.fLastRow );
    tracklet.SetParam( tParam.GetParam() );
    int w = tracker.CalculateHitWeight(r.fNHits, tParam.GetChi2(), r.fItr);
    tracklet.SetHitWeight(w);
    for ( int iRow = r.fFirstRow; iRow <= r.fLastRow; iRow++ ) {
      calink ih = GETRowHit(iRow);
      if ( ih != CALINK_INVAL ) {
        MAKESharedRef(AliHLTTPCCARow, row, tracker.Row(iRow), s.fRows[iRow]);
        tracker.MaximizeHitWeight( row, ih, w );
      }
    }
  }

}

MEM_CLASS_PRE2() GPUdi() void AliHLTTPCCATrackletConstructor::UpdateTracklet
( int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
  GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &s, AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, MEM_LG2(AliHLTTPCCATrackParam) &tParam, int iRow )
{
  // reconstruction of tracklets, tracklets update step
#ifndef EXTERN_ROW_HITS
  AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS

  MAKESharedRef(AliHLTTPCCARow, row, tracker.Row(iRow), s.fRows[iRow]);

  float y0 = row.Grid().YMin();
  float stepY = row.HstepY();
  float z0 = row.Grid().ZMin() - tParam.ZOffset();
  float stepZ = row.HstepZ();

  if ( r.fStage == 0 ) { // fitting part
    do {

      if ( iRow < r.fStartRow || r.fCurrIH == CALINK_INVAL  ) break;
      if ( ( iRow - r.fStartRow ) & 1 )
      {
          SETRowHit(iRow, CALINK_INVAL);
          break; // SG!!! - jump over the row
      }

      cahit2 hh = TEXTUREFetchCons(cahit22, gAliTexRefu2, tracker.HitData(row), r.fCurrIH);

      int oldIH = r.fCurrIH;
      r.fCurrIH = TEXTUREFetchCons(calink, gAliTexRefs, tracker.HitLinkUpData(row), r.fCurrIH);

      float x = row.X();
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
	  
      if ( iRow == r.fStartRow ) {
        tParam.SetX( x );
        tParam.SetY( y );
        r.fLastY = y;
        if (tracker.Param().GetContinuousTracking()) {
          tParam.SetZ( 0.f );
          r.fLastZ = 0.f;
          tParam.SetZOffset( z );
        } else {
          tParam.SetZ( z );
          r.fLastZ = z;
          tParam.SetZOffset( 0.f );
        }
      } else {

        float err2Y, err2Z;
        float dx = x - tParam.X();
        float dy, dz;
        if (r.fNHits >= 10)
        {
            dy = y - tParam.Y();
            dz = z - tParam.Z();
        }
        else
        {
            dy = y - r.fLastY;
            dz = z - r.fLastZ;
        }
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
        if (r.fNHits >= 10 && CAMath::Abs( tParam.SinPhi() ) < HLTCA_MAX_SIN_PHI_LOW ) {
          sinPhi = tParam.SinPhi();
          cosPhi = CAMath::Sqrt( 1 - sinPhi * sinPhi );
        } else {
          sinPhi = dy * ri;
          cosPhi = dx * ri;
        }
        if ( !tParam.TransportToX( x, sinPhi, cosPhi, tracker.Param().ConstBz(), CALINK_INVAL ) ) {
          SETRowHit(iRow, CALINK_INVAL);
          break;
        }
        tracker.GetErrors2( iRow, tracker.Param().GetContinuousTracking() ? 125. : tParam.GetZ(), sinPhi, cosPhi, tParam.GetDzDs(), err2Y, err2Z );

        if (r.fNHits >= 10)
        {
          const float kFactor = tracker.Param().HitPickUpFactor() * tracker.Param().HitPickUpFactor() * 3.5 * 3.5;
          float sy2 = kFactor * ( tParam.GetErr2Y() +  err2Y );
          float sz2 = kFactor * ( tParam.GetErr2Z() +  err2Z );
          if ( sy2 > 2. ) sy2 = 2.;
          if ( sz2 > 2. ) sz2 = 2.;
          dy = y - tParam.Y();
          dz = z - tParam.Z();
          if (dy * dy > sy2 || dz * dz > sz2)
          {
            if (++r.fNMissed >= TRACKLET_CONSTRUCTOR_MAX_ROW_GAP_SEED)
            {
              r.fCurrIH = CALINK_INVAL;
            }
            SETRowHit(iRow, CALINK_INVAL);
            break;
          }
        }

        if ( !tParam.Filter( y, z, err2Y, err2Z, HLTCA_MAX_SIN_PHI_LOW ) ) {
          SETRowHit(iRow, CALINK_INVAL);
          break;
        }
      }
      SETRowHit(iRow, oldIH);
      r.fNHitsEndRow = ++r.fNHits;
      r.fLastRow = iRow;
      r.fEndRow = iRow;
      r.fNMissed = 0;
      break;
    } while ( 0 );
    
    /*QQQQprintf("Extrapolate Row %d X %f Y %f Z %f SinPhi %f DzDs %f QPt %f", iRow, tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt());
    for (int i = 0;i < 15;i++) printf(" C%d=%6.2f", i, tParam.GetCov(i));
    printf("\n");*/

    if ( r.fCurrIH == CALINK_INVAL ) {
      r.fStage = 1;
	  r.fLastY = tParam.Y(); //Store last spatial position here to start inward following from here
	  r.fLastZ = tParam.Z();
      if ( CAMath::Abs( tParam.SinPhi() ) > HLTCA_MAX_SIN_PHI ) {
        r.fGo = 0;
      }
    }
  } else { // forward/backward searching part
    do {
      if ( r.fStage == 2 && iRow > r.fEndRow ) break;
      if ( r.fNMissed > TRACKLET_CONSTRUCTOR_MAX_ROW_GAP )
      {
          r.fGo = 0;
          break;
      }

      r.fNMissed++;

      float x = row.X();
      float err2Y, err2Z;
      if ( !tParam.TransportToX( x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz(), HLTCA_MAX_SIN_PHI_LOW ) ) {
        r.fGo = 0;
        SETRowHit(iRow, CALINK_INVAL);
        break;
      }
      if ( row.NHits() < 1 ) {
        SETRowHit(iRow, CALINK_INVAL);
        break;
      }

#ifndef HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
      GPUglobalref() const cahit2 *hits = tracker.HitData(row);
      GPUglobalref() const calink *firsthit = tracker.FirstHitInBin(row);
#endif //!HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
      float fY = tParam.GetY();
      float fZ = tParam.GetZ();
      calink best = CALINK_INVAL;

      { // search for the closest hit
        tracker.GetErrors2( iRow, *( ( MEM_LG2(AliHLTTPCCATrackParam)* )&tParam ), err2Y, err2Z );
        const float kFactor = tracker.Param().HitPickUpFactor() * tracker.Param().HitPickUpFactor() * 3.5 * 3.5;
        float sy2 = kFactor * ( tParam.GetErr2Y() +  err2Y );
        float sz2 = kFactor * ( tParam.GetErr2Z() +  err2Z );
        if ( sy2 > 2. ) sy2 = 2.;
        if ( sz2 > 2. ) sz2 = 2.;
                                
        int bin, ny, nz;
        row.Grid().GetBinArea(fY, fZ + tParam.ZOffset(), 1.5, 1.5, bin, ny, nz);
        float ds = 1e6;

        for (int k = 0;k <= nz;k++)
        {
          int nBinsY = row.Grid().Ny();
          int mybin = bin + k * nBinsY;
          unsigned int hitFst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin);
          unsigned int hitLst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin + ny + 1);
          for ( unsigned int ih = hitFst; ih < hitLst; ih++ ) {
            assert( (signed) ih < row.NHits() );
            cahit2 hh = TEXTUREFetchCons(cahit2, gAliTexRefu2, hits, ih);
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

      if ( best == CALINK_INVAL )
      {
        SETRowHit(iRow, CALINK_INVAL);
        break;
      }

      cahit2 hh = TEXTUREFetchCons(cahit2, gAliTexRefu2, hits, best);
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;
      
      calink oldHit = (r.fStage == 2 && iRow >= r.fStartRow) ? GETRowHit(iRow) : CALINK_INVAL;
      if (oldHit != best && !tParam.Filter( y, z, err2Y, err2Z, HLTCA_MAX_SIN_PHI_LOW, oldHit != CALINK_INVAL))
      {
          if (oldHit != CALINK_INVAL) SETRowHit(iRow, CALINK_INVAL);
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

GPUdi() void AliHLTTPCCATrackletConstructor::DoTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker)& tracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& s, AliHLTTPCCAThreadMemory& r)
{
	int iRow = 0, iRowEnd = tracker.Param().NRows();;
	MEM_PLAIN(AliHLTTPCCATrackParam) tParam;
#ifndef EXTERN_ROW_HITS
	AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS
	if (r.fGo)
	{
		AliHLTTPCCAHitId id = tracker.TrackletStartHits()[r.fItr];

		r.fStartRow = r.fEndRow = r.fFirstRow = r.fLastRow = id.RowIndex();
		r.fCurrIH = id.HitIndex();
		r.fNMissed = 0;
		iRow = r.fStartRow;
		AliHLTTPCCATrackletConstructor::InitTracklet(tParam);
	}
	r.fStage = 0;
	r.fNHits = 0;
	//if (tracker.Param().ISlice() != 35 && tracker.Param().ISlice() != 34 || r.fItr == CALINK_INVAL) {StoreTracklet( 0, 0, 0, 0, s, r, tracker, tParam );return;}

	for (int k = 0;k < 2;k++)
	{
		for (;iRow != iRowEnd;iRow += r.fStage == 2 ? -1 : 1)
		{
			if (!r.fGo) break;
			UpdateTracklet(0, 0, 0, 0, s, r, tracker, tParam, iRow);
		}
		if (!r.fGo && r.fStage == 2)
		{
			for (;iRow >= r.fStartRow;iRow--)
			{
				SETRowHit(iRow, CALINK_INVAL);
			}
		}
		if (r.fStage == 2)
		{
			StoreTracklet( 0, 0, 0, 0, s, r, tracker, tParam );
		}
		else
		{
			r.fNMissed = 0;
			if ((r.fGo = (tParam.TransportToX( tracker.Row( r.fEndRow ).X(), tracker.Param().ConstBz(), HLTCA_MAX_SIN_PHI) && tParam.Filter( r.fLastY, r.fLastZ, tParam.Err2Y() / 2, tParam.Err2Z() / 2., HLTCA_MAX_SIN_PHI_LOW, true))))
            {
    			float err2Y, err2Z;
    			tracker.GetErrors2( r.fEndRow, tParam, err2Y, err2Z );
    			if (tParam.GetCov(0) < err2Y) tParam.SetCov(0, err2Y);
    			if (tParam.GetCov(2) < err2Z) tParam.SetCov(2, err2Z);
            }
			r.fNHits -= r.fNHitsEndRow;
			r.fStage = 2;
			iRow = r.fEndRow;
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
