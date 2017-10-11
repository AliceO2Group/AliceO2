// $Id: AliHLTTPCGMTrackParam.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
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


#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMMergedTrack.h"
#ifndef HLTCA_STANDALONE
#include "AliExternalTrackParam.h"
#endif
#include "AliHLTTPCCAParam.h"
#include <cmath>
#include <stdlib.h>

GPUd() void AliHLTTPCGMTrackParam::Fit
(
 float* PolinomialFieldBz,
 float x[], float y[], float z[], int rowType[], float alpha[], const AliHLTTPCCAParam &param,
 int &N, float &Alpha, bool UseMeanPt, float maxSinPhi
 ){

  const float kRho = 1.025e-3;//0.9e-3;
  const float kRadLen = 29.532;//28.94;

  AliHLTTPCGMPropagator prop;
  prop.SetMaterial( kRadLen, kRho );
  prop.SetPolynomialFieldBz( PolinomialFieldBz );  
  prop.SetUseMeanMomentum( UseMeanPt );
  prop.SetContinuousTracking( param.GetContinuousTracking() );
  prop.SetMaxSinPhi( maxSinPhi );
  
  int nWays = param.GetNWays();
  int maxN = N;
  for (int iWay = 0;iWay < nWays;iWay++)
  {
    ResetCovariance();

    const bool rejectChi2ThisRound = ( nWays == 1 || iWay == 1 );
    const bool markNonFittedClusters = rejectChi2ThisRound && !(param.HighQPtForward() < fabs(fP[4]));
    const double kDeg2Rad = 3.14159265358979323846/180.;
    const float maxSinForUpdate = CAMath::Sin(70.*kDeg2Rad);
  
    prop.SetTrack( this, Alpha);

    N = 0;
    int iihit;
    for( iihit=0; iihit<maxN; iihit++)
    {
      const int ihit = (iWay & 1) ? (maxN - iihit - 1) : iihit;
      if (rowType[ihit] < 0) continue; // hit is excluded from fit
      
      int err = prop.PropagateToXAlpha(x[ihit], y[ihit], z[ihit], alpha[ihit], iWay & 1 );

      if ( err || CAMath::Abs(prop.GetSinPhi0())>=maxSinForUpdate )
      {
	if (markNonFittedClusters) rowType[ihit] = -(rowType[ihit] + 1); // can not propagate or the angle is too big - mark the cluster and continue w/o update
	continue;
      }
      
      int retVal = prop.Update( y[ihit], z[ihit], rowType[ihit], param, rejectChi2ThisRound);
      if (retVal == 0) { // track is updated
	N++;
      }
      else if (retVal == 2){ // cluster far away form the track
	if (markNonFittedClusters) rowType[ihit] = -(rowType[ihit] + 1);
      }
      else break; // bad chi2 for the whole track, stop the fit
    }
    maxN = iihit;
  }
  if( N<1 || fNDF<1 ){ // just for a case..
    ResetCovariance();
    fNDF = 1;
    N=1;
  }
  Alpha = prop.GetAlpha();
}





GPUd() bool AliHLTTPCGMTrackParam::CheckNumericalQuality() const
{
  //* Check that the track parameters and covariance matrix are reasonable
  bool ok = AliHLTTPCCAMath::Finite(fX) && AliHLTTPCCAMath::Finite( fChi2 ) && AliHLTTPCCAMath::Finite( fNDF );

  const float *c = fC;
  for ( int i = 0; i < 15; i++ ) ok = ok && AliHLTTPCCAMath::Finite( c[i] );
  for ( int i = 0; i < 5; i++ ) ok = ok && AliHLTTPCCAMath::Finite( fP[i] );
  
  if ( c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0 ) ok = 0;
  if ( c[0] > 5. || c[2] > 5. || c[5] > 2. || c[9] > 2. 
       //|| ( CAMath::Abs( QPt() ) > 1.e-2 && c[14] > 2. ) 
       ) ok = 0;

  if ( fabs( fP[2] ) > .999 ) ok = 0;
  if( ok ){
    ok = ok 
      && ( c[1]*c[1]<=c[2]*c[0] )
      && ( c[3]*c[3]<=c[5]*c[0] )
      && ( c[4]*c[4]<=c[5]*c[2] )
      && ( c[6]*c[6]<=c[9]*c[0] )
      && ( c[7]*c[7]<=c[9]*c[2] )
      && ( c[8]*c[8]<=c[9]*c[5] )
      && ( c[10]*c[10]<=c[14]*c[0] )
      && ( c[11]*c[11]<=c[14]*c[2] )
      && ( c[12]*c[12]<=c[14]*c[5] )
      && ( c[13]*c[13]<=c[14]*c[9] );      
  }
  return ok;
}




#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
bool AliHLTTPCGMTrackParam::GetExtParam( AliExternalTrackParam &T, double alpha ) const
{
  //* Convert from AliHLTTPCGMTrackParam to AliExternalTrackParam parameterisation,
  //* the angle alpha is the global angle of the local X axis

  bool ok = CheckNumericalQuality();

  double par[5], cov[15];
  for ( int i = 0; i < 5; i++ ) par[i] = fP[i];
  for ( int i = 0; i < 15; i++ ) cov[i] = fC[i];

  if ( par[2] > .99 ) par[2] = .99;
  if ( par[2] < -.99 ) par[2] = -.99;

  if ( fabs( par[4] ) < 1.e-5 ) par[4] = 1.e-5; // some other software will crash if q/Pt==0
  if ( fabs( par[4] ) > 1./0.08 ) ok = 0; // some other software will crash if q/Pt is too big

  T.Set( (double) fX, alpha, par, cov );
  return ok;
}


 
void AliHLTTPCGMTrackParam::SetExtParam( const AliExternalTrackParam &T )
{
  //* Convert from AliExternalTrackParam parameterisation

  for ( int i = 0; i < 5; i++ ) fP[i] = T.GetParameter()[i];
  for ( int i = 0; i < 15; i++ ) fC[i] = T.GetCovariance()[i];
  fX = T.GetX();
  if ( fP[2] > .999 ) fP[2] = .999;
  if ( fP[2] < -.999 ) fP[2] = -.999;
}
#endif

GPUd() void AliHLTTPCGMTrackParam::RefitTrack(AliHLTTPCGMMergedTrack &track, float* PolinomialFieldBz, float* x, float* y, float* z, int* rowType, float* alpha, const AliHLTTPCCAParam& param)
{
	if( !track.OK() ) return;    

	int nTrackHits = track.NClusters();
	   
	AliHLTTPCGMTrackParam t = track.Param();
	float Alpha = track.Alpha();  
	int nTrackHitsOld = nTrackHits;
	//float ptOld = t.QPt();
	t.Fit( PolinomialFieldBz,
	   x+track.FirstClusterRef(),
	   y+track.FirstClusterRef(),
	   z+track.FirstClusterRef(),
	   rowType+track.FirstClusterRef(),
	   alpha+track.FirstClusterRef(),
	   param, nTrackHits, Alpha );      
	
	if ( fabs( t.QPt() ) < 1.e-4 ) t.QPt() = 1.e-4 ;
	bool okhits = nTrackHits >= TRACKLET_SELECTOR_MIN_HITS(track.Param().QPt());
	bool okqual = t.CheckNumericalQuality();
	bool okphi = fabs( t.SinPhi() ) <= .999;
			
	bool ok = okhits && okqual && okphi;

	//printf("Track %d OUTPUT hits %d -> %d, QPt %f -> %f, ok %d (%d %d %d) chi2 %f chi2ndf %f\n", blanum,  nTrackHitsOld, nTrackHits, ptOld, t.QPt(), (int) ok, (int) okhits, (int) okqual, (int) okphi, t.Chi2(), t.Chi2() / max(1,nTrackHits);
	if (param.HighQPtForward() < fabs(track.Param().QPt()))
	{
		ok = 1;
		nTrackHits = nTrackHitsOld;
		for (int k = 0;k < nTrackHits;k++) if (rowType[k] < 0) rowType[k] = -rowType[k] - 1;
	}
	track.SetOK(ok);
	if (!ok) return;

	if( 1 ){//SG!!!
	  track.SetNClusters( nTrackHits );
	  track.Param() = t;
	  track.Alpha() = Alpha;
	}

	{
	  int ind = track.FirstClusterRef();
	  float alphaa = alpha[ind];
	  float xx = x[ind];
	  float yy = y[ind];
	  float zz = z[ind];
	  float sinA = AliHLTTPCCAMath::Sin( alphaa - track.Alpha());
	  float cosA = AliHLTTPCCAMath::Cos( alphaa - track.Alpha());
	  track.SetLastX( xx*cosA - yy*sinA );
	  track.SetLastY( xx*sinA + yy*cosA );
	  track.SetLastZ( zz );
	}
}

#ifdef HLTCA_GPUCODE

GPUg() void RefitTracks(AliHLTTPCGMMergedTrack* tracks, int nTracks, float* PolinomialFieldBz, float* x, float* y, float* z, int* rowType, float* alpha, AliHLTTPCCAParam* param)
{
	for (int i = get_global_id(0);i < nTracks;i += get_global_size(0))
	{
		AliHLTTPCGMTrackParam::RefitTrack(tracks[i], PolinomialFieldBz, x, y, z, rowType, alpha, *param);
	}
}

#endif


GPUd() bool AliHLTTPCGMTrackParam::Rotate( float alpha, AliHLTTPCGMPhysicalTrackModel &t0, float maxSinPhi )
{
  //* Rotate the coordinate system in XY on the angle alpha

  float cA = CAMath::Cos( alpha );
  float sA = CAMath::Sin( alpha );
  float x0 = t0.X(), y0 = t0.Y(), sinPhi0 = t0.SinPhi(), cosPhi0 = t0.CosPhi();
  float cosPhi =  cosPhi0 * cA + sinPhi0 * sA;
  float sinPhi = -cosPhi0 * sA + sinPhi0 * cA;

  if ( CAMath::Abs( sinPhi ) > maxSinPhi || CAMath::Abs( cosPhi ) < 1.e-2 || CAMath::Abs( cosPhi0 ) < 1.e-2  ) return 0;

  //float J[5][5] = { { j0, 0, 0,  0,  0 }, // Y
  //                    {  0, 1, 0,  0,  0 }, // Z
  //                    {  0, 0, j2, 0,  0 }, // SinPhi
  //                  {  0, 0, 0,  1,  0 }, // DzDs
  //                  {  0, 0, 0,  0,  1 } }; // Kappa

  float j0 = cosPhi0 / cosPhi;
  float j2 = cosPhi / cosPhi0;
  float d[2] = {Y() - y0, SinPhi() - sinPhi0};

  {
    float px = t0.Px();
    float py = t0.Py();
    
    t0.X()  =  x0*cA + y0*sA;
    t0.Y()  = -x0*sA + y0*cA;
    t0.Px() =  px*cA + py*sA;
    t0.Py() = -px*sA + py*cA;
    t0.UpdateValues();
  }
  
  X() = t0.X();
  Y() = t0.Y() + j0*d[0];

  SinPhi() = sinPhi + j2*d[1] ;

  fC[0] *= j0 * j0;
  fC[1] *= j0;
  fC[3] *= j0;
  fC[6] *= j0;
  fC[10] *= j0;

  fC[3] *= j2;
  fC[4] *= j2;
  fC[5] *= j2 * j2;
  fC[8] *= j2;
  fC[12] *= j2;
  if( cosPhi <0 ){ // change direction ( t0 direction is already changed in t0.UpdateValues(); )
    SinPhi() = -SinPhi();
    DzDs() = -DzDs();
    QPt() = -QPt();
    fC[3] = -fC[3];
    fC[4] = -fC[4];
    fC[6] = -fC[6];
    fC[7] = -fC[7];
    fC[10] = -fC[10];
    fC[11] = -fC[11];
  }
  
  return 1;
}
