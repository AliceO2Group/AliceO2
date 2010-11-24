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
#include "AliHLTTPCGMTrackLinearisation.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "Riostream.h"
#include "AliExternalTrackParam.h"
#include "AliHLTTPCCAParam.h"
#include <cmath>


float AliHLTTPCGMTrackParam::fPolinomialFieldBz[6];   // field coefficients


void AliHLTTPCGMTrackParam::Fit
(
 float x[], float y[], float z[], unsigned int rowType[], float alpha[], AliHLTTPCCAParam &param,
 int &N,
 float &Alpha, 
 bool UseMeanPt,
 float maxSinPhi
 ){
  
  const float kRho = 1.025e-3;//0.9e-3;
  const float kRadLen = 29.532;//28.94;
  const float kRhoOverRadLen = kRho / kRadLen;
  
  AliHLTTPCGMTrackLinearisation t0(*this);
 
  const float kZLength = 250.f - 0.275f;
  float trDzDs2 = t0.DzDs()*t0.DzDs();
 
  AliHLTTPCGMTrackFitParam par;
  CalculateFitParameters( par, kRhoOverRadLen, kRho, UseMeanPt );

  int maxN = N;

  bool first = 1;
  N = 0;

  for( int ihit=0; ihit<maxN; ihit++ ){
    
    float sliceAlpha =  alpha[ihit];
    
    if ( fabs( sliceAlpha - Alpha ) > 1.e-4 ) {
      if( !Rotate(  sliceAlpha - Alpha, t0, .999 ) ) break;
      Alpha = sliceAlpha;
    }

    float dL=0;    
    float bz =  GetBz(x[ihit], y[ihit],z[ihit]);
        
    float err2Y, err2Z;

    { // transport block
      
      bz = -bz;

      float ex = t0.CosPhi();
      
      float ey = t0.SinPhi();
      float k  = t0.QPt()*bz;
      float dx = x[ihit] - X();
      float kdx = k*dx;
      float ey1 = kdx + ey;
      
      if( fabs( ey1 ) >= maxSinPhi ) break;

      float ss = ey + ey1;   
      float ex1 = sqrt(1 - ey1*ey1);
      
      float dxBz = dx * bz;
    
      float cc = ex + ex1;  
      float dxcci = dx * Reciprocal(cc);
      float kdx205 = kdx*kdx*0.5f;
      
      float dy = dxcci * ss;      
      float norm2 = float(1.f) + ey*ey1 + ex*ex1;
      float dl = dxcci * sqrt( norm2 + norm2 );

      float dS;    
      { 
	float dSin = float(0.5f)*k*dl;
	float a = dSin*dSin;
	const float k2 = 1.f/6.f;
	const float k4 = 3.f/40.f;
	//const float k6 = 5.f/112.f;
	dS = dl + dl*a*(k2 + a*(k4 ));//+ k6*a) );
      }
      
      float ex1i = Reciprocal(ex1);
      float dz = dS * t0.DzDs();  
      
      dL = -dS * t0.DlDs();
      
      float hh = dxcci*ex1i*(2.f+kdx205); 
      float h2 = hh * t0.SecPhi();
      float h4 = bz*dxcci*hh;

      float d2 = fP[2] - t0.SinPhi();
      float d3 = fP[3] - t0.DzDs();
      float d4 = fP[4] - t0.QPt();
      
      
      fX+=dx;
      fP[0]+= dy     + h2 * d2           +   h4 * d4;
      fP[1]+= dz               + dS * d3;
      fP[2] = ey1 +     d2           + dxBz * d4;    
      
      t0.CosPhi() = ex1;
      t0.SecPhi() = ex1i;
      t0.SinPhi() = ey1;      

      {
	const float *cy = param.GetParamS0Par(0,rowType[ihit]);
	const float *cz = param.GetParamS0Par(1,rowType[ihit]);

	float secPhi2 = ex1i*ex1i;
	float zz = fabs( kZLength - fabs(fP[2]) );	
	float zz2 = zz*zz;
	float angleY2 = secPhi2 - 1.f; 
	float angleZ2 = trDzDs2 * secPhi2 ;

	float cy0 = cy[0] + cy[1]*zz + cy[3]*zz2;
	float cy1 = cy[2] + cy[5]*zz;
	float cy2 = cy[4];
	float cz0 = cz[0] + cz[1]*zz + cz[3]*zz2;
	float cz1 = cz[2] + cz[5]*zz;
	float cz2 = cz[4];
	
	err2Y = fabs( cy0 + angleY2 * ( cy1 + angleY2*cy2 ) );
	err2Z = fabs( cz0 + angleZ2 * ( cz1 + angleZ2*cz2 ) );      
     }


      if ( first ) {
	fP[0] = y[ihit];
	fP[1] = z[ihit];
	SetCov( 0, err2Y );
	SetCov( 1,  0 );
	SetCov( 2, err2Z);
	SetCov( 3,  0 );
	SetCov( 4,  0 );
	SetCov( 5,  1 );
	SetCov( 6,  0 );
	SetCov( 7,  0 );
	SetCov( 8,  0 );
	SetCov( 9,  1 );
	SetCov( 10,  0 );
	SetCov( 11,  0 );
	SetCov( 12,  0 );
	SetCov( 13,  0 );
	SetCov( 14,  10 );
	SetChi2( 0 );
	SetNDF( -3 );
	CalculateFitParameters( par, kRhoOverRadLen, kRho, UseMeanPt );
	first = 0;
	N+=1;
	continue;
      }

      float c20 = fC[3];
      float c21 = fC[4];
      float c22 = fC[5];
      float c30 = fC[6];
      float c31 = fC[7];
      float c32 = fC[8];
      float c33 = fC[9];
      float c40 = fC[10];
      float c41 = fC[11];
      float c42 = fC[12];
      float c43 = fC[13];
      float c44 = fC[14];
      
      float c20ph4c42 =  c20 + h4*c42;
      float h2c22 = h2*c22;
      float h4c44 = h4*c44;
      
      float n6 = c30 + h2*c32 + h4*c43;
      float n7 = c31 + dS*c33;
      float n10 = c40 + h2*c42 + h4c44;
      float n11 = c41 + dS*c43;
      float n12 = c42 + dxBz*c44;
      
      fC[8] = c32 + dxBz * c43;
      
      fC[0]+= h2*h2c22 + h4*h4c44 + float(2.f)*( h2*c20ph4c42  + h4*c40 );
      
      fC[1]+= h2*c21 + h4*c41 + dS*n6;
      fC[6] = n6;
      
      fC[2]+= dS*(c31 + n7);
      fC[7] = n7; 
      
      fC[3] = c20ph4c42 + h2c22  + dxBz*n10;
      fC[10] = n10;
      
      fC[4] = c21 + dS*c32 + dxBz*n11;
      fC[11] = n11;
      
      fC[5] = c22 + dxBz*( c42 + n12 );
      fC[12] = n12;
      
    } // end transport block 

 
    float &fC22 = fC[5];
    float &fC33 = fC[9];
    float &fC40 = fC[10];
    float &fC41 = fC[11];
    float &fC42 = fC[12];
    float &fC43 = fC[13];
    float &fC44 = fC[14];
    
    float 
      c00 = fC[ 0],
      c11 = fC[ 2],
      c20 = fC[ 3],
      c31 = fC[ 7];
    
    
    // MS block  
    
    float dLmask = 0.f;
    bool maskMS = ( fabs( dL ) < par.fDLMax );

    
    // Filter block
    
    float mS0 = Reciprocal(err2Y + c00);
    
    // MS block
    Assign( dLmask, maskMS, dL );
    
    // Filter block
    
    float  z0 = y[ihit] - fP[0];
    float mS2 = Reciprocal(err2Z + c11);
    
    if( fabs( fP[2] + z0*c20*mS0  ) > maxSinPhi ) break;
    
    // MS block
    
    float dLabs = fabs( dLmask); 
    float corr = float(1.f) - par.fEP2* dLmask ;
    
    fP[4]*= corr;
    fC40 *= corr;
    fC41 *= corr;
    fC42 *= corr;
    fC43 *= corr;
    fC44  = fC44*corr*corr + dLabs*par.fSigmadE2;
    
    fC22 += dLabs * par.fK22 * (float(1.f)-fP[2]*fP[2]);
    fC33 += dLabs * par.fK33;
    fC43 += dLabs * par.fK43;
        

    // Filter block
  
    float c40 = fC40;
    
    // K = CHtS
    
    float k00, k11, k20, k31, k40;
    
    k00 = c00 * mS0;
    k20 = c20 * mS0;
    k40 = c40 * mS0;
    fChi2  += mS0*z0*z0;
    fP[0] += k00 * z0;
    fP[2] += k20 * z0;
    fP[4] += k40 * z0;
    fC[ 0] -= k00 * c00 ;
    fC[ 5] -= k20 * c20 ;
    fC[10] -= k00 * c40 ;
    fC[12] -= k40 * c20 ;
    fC[ 3] -= k20 * c00 ;
    fC[14] -= k40 * c40 ;
  
    float  z1 = z[ihit] - fP[1];
    
    k11 = c11 * mS2;
    k31 = c31 * mS2;
    
    fChi2  +=  mS2*z1*z1 ;
    fNDF  += 2;
    N+=1;
    
    fP[1] += k11 * z1;
    fP[3] += k31 * z1;
    
    fC[ 7] -= k31 * c11;
    fC[ 2] -= k11 * c11;
    fC[ 9] -= k31 * c31;    
  } 
}





bool AliHLTTPCGMTrackParam::CheckNumericalQuality() const
{
  //* Check that the track parameters and covariance matrix are reasonable

  bool ok = finite(fX) && finite( fChi2 ) && finite( fNDF );

  const float *c = fC;
  for ( int i = 0; i < 15; i++ ) ok = ok && finite( c[i] );
  for ( int i = 0; i < 5; i++ ) ok = ok && finite( fP[i] );

  if ( c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0 ) ok = 0;
  if ( c[0] > 5. || c[2] > 5. || c[5] > 2. || c[9] > 2. 
       //|| ( CAMath::Abs( QPt() ) > 1.e-2 && c[14] > 2. ) 
       ) ok = 0;

  if ( fabs( fP[2] ) > .999 ) ok = 0;
  if ( fabs( fP[4] ) > 1. / 0.05 ) ok = 0;
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

float AliHLTTPCGMTrackParam::ApproximateBetheBloch( float beta2 )
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


 void AliHLTTPCGMTrackParam::CalculateFitParameters( AliHLTTPCGMTrackFitParam &par, float RhoOverRadLen,  float Rho, bool NoField, float mass )
{
  //*!

  float qpt = fP[4];
  if( NoField ) qpt = 1./0.35;

  float p2 = ( 1. + fP[3] * fP[3] );
  float k2 = qpt * qpt;
  Assign( k2, (  k2 < 1.e-4f ), 1.e-4f );

  float mass2 = mass * mass;
  float beta2 = p2 / ( p2 + mass2 * k2 );
  
  float pp2 = p2 / k2; // impuls 2

  //par.fBethe = BetheBlochGas( pp2/mass2);
  par.fBetheRho = ApproximateBetheBloch( pp2 / mass2 )*Rho;
  par.fE = sqrt( pp2 + mass2 );
  par.fTheta2 = ( 14.1*14.1/1.e6 ) / ( beta2 * pp2 )*RhoOverRadLen;
  par.fEP2 = par.fE / pp2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.07; // To be tuned.
  par.fSigmadE2 = knst * par.fEP2 * qpt;
  par.fSigmadE2 = par.fSigmadE2 * par.fSigmadE2;
  
  float k22 = 1. + fP[3] * fP[3];
  par.fK22 = par.fTheta2*k22;
  par.fK33 = par.fK22 * k22;
  par.fK43 = 0.;
  par.fK44 =  par.fTheta2*fP[3] * fP[3] * k2;
  
  float br=1.e-8f;
  Assign( br, ( par.fBetheRho>1.e-8f ), par.fBetheRho );
  par.fDLMax = 0.3*par.fE * Reciprocal( br );

  par.fEP2*=par.fBetheRho;
  par.fSigmadE2 = par.fSigmadE2*par.fBetheRho+par.fK44;  
}




//*
//* Rotation
//*


bool AliHLTTPCGMTrackParam::Rotate( float alpha, AliHLTTPCGMTrackLinearisation &t0, float maxSinPhi )
{
  //* Rotate the coordinate system in XY on the angle alpha

  float cA = CAMath::Cos( alpha );
  float sA = CAMath::Sin( alpha );
  float x0 = X(), y0 = Y(), sP = t0.SinPhi(), cP = t0.CosPhi();
  float cosPhi = cP * cA + sP * sA;
  float sinPhi = -cP * sA + sP * cA;

  if ( CAMath::Abs( sinPhi ) > maxSinPhi || CAMath::Abs( cosPhi ) < 1.e-2 || CAMath::Abs( cP ) < 1.e-2  ) return 0;

  //float J[5][5] = { { j0, 0, 0,  0,  0 }, // Y
  //                    {  0, 1, 0,  0,  0 }, // Z
  //                    {  0, 0, j2, 0,  0 }, // SinPhi
  //                  {  0, 0, 0,  1,  0 }, // DzDs
  //                  {  0, 0, 0,  0,  1 } }; // Kappa

  float j0 = cP / cosPhi;
  float j2 = cosPhi / cP;
  float d[2] = {Y() - y0, SinPhi() - sP};

  X() = ( x0*cA +  y0*sA );
  Y() = ( -x0*sA +  y0*cA + j0*d[0] );
  t0.CosPhi() = fabs( cosPhi );
  t0.SecPhi() = ( 1./t0.CosPhi() );
  t0.SinPhi() = ( sinPhi );

  SinPhi() = ( sinPhi + j2*d[1] );

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
  if( cosPhi <0 ){ // change direction
    t0.SinPhi() = -sinPhi;
    t0.DzDs() = -t0.DzDs();
    t0.DlDs() = -t0.DlDs();
    t0.QPt() = -t0.QPt();
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




