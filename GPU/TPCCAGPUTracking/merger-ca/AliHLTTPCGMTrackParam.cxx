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
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "Riostream.h"
#ifndef HLTCA_STANDALONE
#include "AliExternalTrackParam.h"
#endif
#include "AliHLTTPCCAParam.h"
#include <cmath>

GPUd() void AliHLTTPCGMTrackParam::Fit
(
 float* PolinomialFieldBz,
 float x[], float y[], float z[], int rowType[], float alpha[], const AliHLTTPCCAParam &param,
 int &N,
 float &Alpha,
 bool UseMeanPt,
 float maxSinPhi
 ){
  int nWays = param.GetNWays();
  int maxN = N;
  for (int iWay = 0;iWay < nWays;iWay++)
  {
    ResetCovariance();
    AliHLTTPCGMTrackMaterialCorrection par;
    const float kRho = 1.025e-3;//0.9e-3;
    const float kRadLen = 29.532;//28.94;
    const float kRhoOverRadLen = kRho / kRadLen;

    AliHLTTPCGMPhysicalTrackModel t0(*this);    
 
    CalculateMaterialCorrection( par, t0, kRhoOverRadLen, kRho, UseMeanPt );

    bool rejectChi2ThisRound = ( nWays == 1 || iWay == 1 );
    bool markNonFittedClusters = rejectChi2ThisRound && !(param.HighQPtForward() < fabs(t0.QPt()));   
    const double kDeg2Rad = 3.14159265358979323846/180.;
    const float maxSinForUpdate = CAMath::Sin(70.*kDeg2Rad);
  
    bool inFlyDirection = 1;
    N = 0;
    int iihit;
    for( iihit=0; iihit<maxN; iihit++)
    {
      const int ihit = (iWay & 1) ? (maxN - iihit - 1) : iihit;
      if (rowType[ihit] < 0) continue; // hit is excluded from fit
      
      int err = PropagateTrack(PolinomialFieldBz, x[ihit], y[ihit], z[ihit], alpha[ihit], param, Alpha, maxSinPhi, UseMeanPt, par, t0, inFlyDirection );      
      
      if ( err || CAMath::Abs(t0.SinPhi())>=maxSinForUpdate )
      {
	// can not propagate or the angle is too big - mark the cluster and continue w/o update
	if (markNonFittedClusters) rowType[ihit] = -(rowType[ihit] + 1);
	continue;
      }
      inFlyDirection = 0;
      int retVal = UpdateTrack( y[ihit], z[ihit], rowType[ihit], param, t0, maxSinPhi, rejectChi2ThisRound);
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
}



GPUd() int AliHLTTPCGMTrackParam::UpdateTrack( float posY, float posZ, int rowType, const AliHLTTPCCAParam &param, AliHLTTPCGMPhysicalTrackModel& t0, float maxSinPhi, bool rejectChi2)
{
  if (fabs(posY - t0.GetY()) > 3 || fabs(posZ - t0.GetZ()) > 3) return 2; 
	
  float 
    c00 = fC[ 0],
    c11 = fC[ 2],
    c20 = fC[ 3],
    c31 = fC[ 7],
    c40 = fC[10];

  //Copy computation of err2? from first propagation (above) for update
  float err2Y, err2Z;
  {
    const float *cy = param.GetParamS0Par(0,rowType);
    const float *cz = param.GetParamS0Par(1,rowType);
    
    float secPhi2 = t0.GetSecPhi()*t0.GetSecPhi();
    const float kZLength = 250.f - 0.275f;
    float zz = param.GetContinuousTracking() ? 125. : fabs( kZLength - fabs(posZ) );
    float zz2 = zz*zz;
    float angleY2 = secPhi2 - 1.f; 
    float angleZ2 = t0.DzDs()*t0.DzDs() * secPhi2 ;

    float cy0 = cy[0] + cy[1]*zz + cy[3]*zz2;
    float cy1 = cy[2] + cy[5]*zz;
    float cy2 = cy[4];
    float cz0 = cz[0] + cz[1]*zz + cz[3]*zz2;
    float cz1 = cz[2] + cz[5]*zz;
    float cz2 = cz[4];
    
    err2Y = fabs( cy0 + angleY2 * ( cy1 + angleY2*cy2 ) );
    err2Z = fabs( cz0 + angleZ2 * ( cz1 + angleZ2*cz2 ) );      
  }
  //if( fNDF==-5) ResetCovariance();
  if ( fNDF==-5 ) { // first measurement, just shift the track there        
    fP[0] = posY;
    fP[1] = posZ;
    SetCov( 0, err2Y );
    SetCov( 1,  0 );
    SetCov( 2, err2Z);
    SetCov( 3,  0 );
    SetCov( 4,  0 );
    SetCov( 5,  1 );
    SetCov( 6,  0 );
    SetCov( 7,  0 );
    SetCov( 8,  0 );
    SetCov( 9,  10 );
    SetCov( 10,  0 );
    SetCov( 11,  0 );
    SetCov( 12,  0 );
    SetCov( 13,  0 );
    SetCov( 14,  10 );
    SetChi2( 0 );
    SetNDF( -3 );   
    return 0;
  }
  
    
  // Filter block
    
  float mS0 = Reciprocal(err2Y + c00);    

  float  z0 = posY - fP[0];
  float  z1 = posZ - fP[1];
  float mS2 = Reciprocal(err2Z + c11);
  
  //printf("hits %d chi2 %f, new %f %f (dy %f dz %f)\n", N, fChi2, mS0 * z0 * z0, mS2 * z1 * z1, z0, z1);
  float tmpCut = param.HighQPtForward() < fabs(t0.GetQPt()) ? 5 : 5; // change to t0
  if (rejectChi2 && (mS0*z0*z0 > tmpCut || mS2*z1*z1 > tmpCut)) return 2;
  fChi2  += mS0*z0*z0;
  fChi2  +=  mS2*z1*z1;
  //SG!!! if (fChi2 / ((fNDF+5)/2 + 1) > 5) return 1;
  //if (fChi2 / ((fNDF+5)/2 + 1) > 5) return 2;
  //SG!!! if( fabs( fP[2] + z0*c20*mS0  ) > maxSinPhi ) return 1;
  
  
    
  // K = CHtS
    
  float k00, k11, k20, k31, k40;
  
  k00 = c00 * mS0;
  k20 = c20 * mS0;
  k40 = c40 * mS0;
  
  
  k11 = c11 * mS2;
  k31 = c31 * mS2;
  
  fNDF  += 2;
    
  fP[0] += k00 * z0;
  fP[1] += k11 * z1;
  fP[2] += k20 * z0;
  fP[3] += k31 * z1;
  fP[4] += k40 * z0;
  
  fC[ 0] -= k00 * c00 ;
  fC[ 2] -= k11 * c11;
  fC[ 3] -= k20 * c00 ;
  fC[ 5] -= k20 * c20 ;
  fC[ 7] -= k31 * c11;
  fC[ 9] -= k31 * c31;
  fC[10] -= k00 * c40 ;
  fC[12] -= k40 * c20 ;
  fC[14] -= k40 * c40 ;
    
  return 0;
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

//*
//*  Multiple scattering and energy losses
//*

GPUd() float AliHLTTPCGMTrackParam::ApproximateBetheBloch( float beta2 )
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula with
  // the density effect taken into account at beta*gamma > 3.5
  // (the approximation is reasonable only for solid materials)
  //------------------------------------------------------------------

  const float log0 = log( float(5940.f));
  const float log1 = log( float(3.5f*5940.f) );

  bool bad = (beta2 >= .999f)||( beta2 < 1.e-8f );

  Assign( beta2, bad, 0.5f);

  float a = beta2 / ( 1.f - beta2 ); 
  float b = 0.5*log(a);
  float d =  0.153e-3 / beta2;
  float c = b - beta2;

  float ret = d*(log0 + b + c );
  float case1 = d*(log1 + c );
  
  Assign( ret, ( a > 3.5*3.5  ), case1);
  Assign( ret,  bad, 0. ); 

  return ret;
}




GPUd() void AliHLTTPCGMTrackParam::CalculateMaterialCorrection( AliHLTTPCGMTrackMaterialCorrection &par, const AliHLTTPCGMPhysicalTrackModel &t0, float RhoOverRadLen,  float Rho, bool NoField, float mass )
{
  //*!

  float qpt = t0.GetQPt();
  if( NoField ) qpt = 1./0.35;

  float w2 = ( 1. + t0.GetDzDs() * t0.GetDzDs() );//==(P/pt)2
  float pti2 = qpt * qpt;
  Assign( pti2, (  pti2 < 1.e-4f ), 1.e-4f );

  float mass2 = mass * mass;
  float beta2 = w2 / ( w2 + mass2 * pti2 );
  
  float p2 = w2 / pti2; // impuls 2

  //par.fBethe = BetheBlochGas( p2/mass2);
  par.fBetheRho = ApproximateBetheBloch( p2 / mass2 )*Rho;
  par.fE = sqrt( p2 + mass2 );
  par.fTheta2 = ( 14.1*14.1/1.e6 ) / ( beta2 * p2 )*RhoOverRadLen;
  par.fEP2 = par.fE / p2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.07; // To be tuned.
  par.fSigmadE2 = knst * par.fEP2 * qpt;
  par.fSigmadE2 = par.fSigmadE2 * par.fSigmadE2;
  
  par.fK22 = par.fTheta2*w2;
  par.fK33 = par.fK22 * w2;
  par.fK43 = 0.;
  par.fK44 =  par.fTheta2* t0.GetDzDs() * t0.GetDzDs() * pti2;
  
  float br=1.e-8f;
  Assign( br, ( par.fBetheRho>1.e-8f ), par.fBetheRho );
  par.fDLMax = 0.3*par.fE * Reciprocal( br );

  par.fEP2*=par.fBetheRho;
  par.fSigmadE2 = par.fSigmadE2*par.fBetheRho+par.fK44;  
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

//
// Fit with 3D field (test)
//

#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)

#include "AliTracker.h"

inline void GetBxByBz( float Alpha, float x, float y, float z, float* PolinomialFieldBz, float B[3] )
{
  const double kCLight = 0.000299792458;

  // get global coordinates
  double cs=AliHLTTPCCAMath::Cos(Alpha), sn=AliHLTTPCCAMath::Sin(Alpha);
  double r[3] = { x*cs - y*sn, x*sn + y*cs, z};
  double bb[3];
  AliTracker::GetBxByBz( r, bb);

  // rotate field to local coordinates

  B[0] = kCLight*(  bb[0]*cs + bb[1]*sn );
  B[1] = kCLight*( -bb[0]*sn + bb[1]*cs );
  B[2] = kCLight*( bb[2] );
  /*
  // at the moment, test with old polynomial Bz field
  B[0] = 0.;
  B[1] = 0.;
  B[2] = AliHLTTPCGMTrackParam::GetBz( x, y, z, PolinomialFieldBz );
  */
}
#else
inline void GetBxByBz( float Alpha, float x, float y, float z, float* PolinomialFieldBz, float B[3] )
{
  B[0] = 0.;
  B[1] = 0.;
  B[2] = AliHLTTPCGMTrackParam::GetBz( x, y, z, PolinomialFieldBz );
}
#endif

GPUd() int AliHLTTPCGMTrackParam::PropagateTrack(float* PolinomialFieldBz, float posX, float posY, float posZ, float posAlpha, const AliHLTTPCCAParam &param, float& Alpha, float maxSinPhi, bool UseMeanPt, AliHLTTPCGMTrackMaterialCorrection& par, AliHLTTPCGMPhysicalTrackModel& t0, bool inFlyDirection)
{
  float sliceAlpha = posAlpha;
  
  if ( fabs( sliceAlpha - Alpha ) > 1.e-4 ) {
    if( !Rotate(  sliceAlpha - Alpha, t0, .999 ) ) return 1;
    Alpha = sliceAlpha;
  }

  float B[3];
  GetBxByBz( Alpha, X(), Y(), param.GetContinuousTracking() ? (Z() > 0 ? 125. : -125.) : Z(), PolinomialFieldBz, B );

  // propagate t0 to t0e
  
  AliHLTTPCGMPhysicalTrackModel t0e(t0);
  float dLp = 0;
  int err = t0e.PropagateToXBxByBz( posX, posY, posZ, B[0], B[1], B[2], dLp );
  if( err ) return 1;
  if( fabs( t0e.SinPhi() ) >= maxSinPhi ) return 1;

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS =  dLp*t0e.Pt();
  float dL =  fabs(dLp*t0e.P());   
  if( inFlyDirection ) dL = -dL;
  
  float bz = B[2];             
  float k  = -t0.QPt()*bz;
  float dx = posX - X();
  float kdx = k*dx; 
  float dxcci = dx * Reciprocal(t0.CosPhi() + t0e.CosPhi());            
      
  float hh = dxcci*t0e.SecPhi()*(2.f+0.5f*kdx*kdx); 
  float h02 = t0.SecPhi()*hh;
  float h04 = -bz*dxcci*hh;
  float h13 = dS;  
  float h24 = -dx*bz;
  
  float d0 = fP[0] - t0.Y();
  float d1 = fP[1] - t0.Z();
  float d2 = fP[2] - t0.SinPhi();
  float d3 = fP[3] - t0.DzDs();
  float d4 = fP[4] - t0.QPt();
	  
  t0 = t0e;

  fX = t0e.X();
  fP[0] = t0e.Y() + d0    + h02*d2         + h04*d4;
  fP[1] = t0e.Z() + d1    + h13*d3;
  fP[2] = t0e.SinPhi() +  d2           + h24*d4;    
  fP[3] = t0e.DzDs() + d3;
  fP[4] = t0e.QPt() + d4;  
  
  float c20 = fC[ 3];
  float c21 = fC[ 4];
  float c22 = fC[ 5];
  float c30 = fC[ 6];
  float c31 = fC[ 7];
  float c32 = fC[ 8];
  float c33 = fC[ 9];
  float c40 = fC[10];
  float c41 = fC[11];
  float c42 = fC[12];
  float c43 = fC[13];
  float c44 = fC[14];
  
  float c20ph04c42 =  c20 + h04*c42;
  float h02c22 = h02*c22;
  float h04c44 = h04*c44;
  
  float n6 = c30 + h02*c32 + h04*c43;
  float n7 = c31 + h13*c33;
  float n10 = c40 + h02*c42 + h04c44;
  float n11 = c41 + h13*c43;
  float n12 = c42 + h24*c44;
      
  fC[8] = c32 + h24*c43;
  
  fC[0]+= h02*h02c22 + h04*h04c44 + float(2.f)*( h02*c20ph04c42  + h04*c40 );
  
  fC[1]+= h02*c21 + h04*c41 + h13*n6;
  fC[6] = n6;
  
  fC[2]+= h13*(c31 + n7);
  fC[7] = n7; 
  
  fC[3] = c20ph04c42 + h02c22  + h24*n10;
  fC[10] = n10;
  
  fC[4] = c21 + h13*c32 + h24*n11;
  fC[11] = n11;
      
  fC[5] = c22 + h24*( c42 + n12 );
  fC[12] = n12;

  // Energy Loss
  
  float &fC22 = fC[5];
  float &fC33 = fC[9];
  float &fC40 = fC[10];
  float &fC41 = fC[11];
  float &fC42 = fC[12];
  float &fC43 = fC[13];
  float &fC44 = fC[14];
  
  float dLmask = 0.f;
  bool maskMS = ( fabs( dL ) < par.fDLMax );
  Assign( dLmask, maskMS, dL );    
  float dLabs = fabs( dLmask); 
  float corr = float(1.f) - par.fEP2* dLmask ;

  float corrInv = 1.f/corr;
  t0.Px()*=corrInv;
  t0.Py()*=corrInv;
  t0.Pz()*=corrInv;
  t0.Pt()*=corrInv;
  t0.P()*=corrInv;
  t0.QPt()*=corr;

  fP[4]*= corr;
  
  fC40 *= corr;
  fC41 *= corr;
  fC42 *= corr;
  fC43 *= corr;
  fC44  = fC44*corr*corr + dLabs*par.fSigmadE2;
  
  //  Multiple Scattering
  
  fC22 += dLabs * par.fK22 * t0.CosPhi()*t0.CosPhi();
  fC33 += dLabs * par.fK33;
  fC43 += dLabs * par.fK43;
  
  return 0;
}
