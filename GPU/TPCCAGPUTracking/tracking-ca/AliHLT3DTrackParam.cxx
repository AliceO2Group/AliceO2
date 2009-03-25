// @(#) $Id$
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
//***************************************************************************

#include "AliHLT3DTrackParam.h"
#include "TMath.h"

ClassImp(AliHLT3DTrackParam)

//* Transport utilities
  
Double_t AliHLT3DTrackParam::GetDStoPoint( Double_t Bz, const Double_t xyz[3], const Double_t *T0 ) const
{
  //* Get DS = Path/Momentum to a certain space point for Bz field

  Double_t q = fSignQ;
  if( !T0 ) T0 = fParam; 
  else q = T0[6];

  const Double_t kCLight = 0.000299792458;
  Double_t bq = Bz*q*kCLight;
  Double_t pt2 = T0[3]*T0[3] + T0[4]*T0[4];
  if( pt2<1.e-4 ) return 0;
  Double_t dx = xyz[0] - T0[0];
  Double_t dy = xyz[1] - T0[1]; 
  Double_t a = dx*T0[3]+dy*T0[4];
  Double_t dS = 0;
  if( TMath::Abs(bq)<1.e-8 ) dS = a/pt2;
  else dS = TMath::ATan2( bq*a, pt2 + bq*(dy*T0[3] -dx*T0[4]) )/bq;
  return dS;
}


void AliHLT3DTrackParam::TransportToDS( Double_t Bz, Double_t DS, Double_t *T0 )
{
  //* Transport the particle on DS = Path/Momentum, for Bz field 

  Double_t tmp[7];
  if( !T0 ){
    T0 = tmp;
    T0[0] = fParam[0];
    T0[1] = fParam[1];
    T0[2] = fParam[2];
    T0[3] = fParam[3];
    T0[4] = fParam[4];
    T0[5] = fParam[5];
    T0[6] = fSignQ;
  }
  const Double_t kCLight = 0.000299792458;
  Bz = Bz*T0[6]*kCLight;
  Double_t bs= Bz*DS;
  Double_t s = TMath::Sin(bs), c = TMath::Cos(bs);
  Double_t sB, cB;
  if( TMath::Abs(bs)>1.e-10){
    sB= s/Bz;
    cB= (1-c)/Bz;
  }else{
    sB = (1. - bs*bs/6.)*DS;
    cB = .5*sB*bs;
  }
    
  Double_t px = T0[3];
  Double_t py = T0[4];
  Double_t pz = T0[5];
  
  Double_t d[6] = { fParam[0]-T0[0], fParam[1]-T0[1], fParam[2]-T0[2], 
		    fParam[3]-T0[3], fParam[4]-T0[4], fParam[5]-T0[5]  };

  T0[0] = T0[0] + sB*px + cB*py;
  T0[1] = T0[1] - cB*px + sB*py;
  T0[2] = T0[2] + DS*pz                       ;
  T0[3] =          c*px + s*py;
  T0[4] =         -s*px + c*py;
  T0[5] = T0[5];
 
  Double_t mJ[6][6] = { {1,0,0,   sB, cB,  0, },
                        {0,1,0,  -cB, sB,  0, },
                        {0,0,1,    0,  0, DS, },
                        {0,0,0,    c,  s,  0, },
                        {0,0,0,   -s,  c,  0, },
                        {0,0,0,    0,  0,  1, }   };

  for( Int_t i=0; i<6; i++){
    fParam[i] = T0[i];
    for( Int_t j=0; j<6; j++) fParam[i] += mJ[i][j]*d[j];
  }

  Double_t mA[6][6];
  for( Int_t k=0,i=0; i<6; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = fCov[k]; 

  Double_t mJC[6][6];
  for( Int_t i=0; i<6; i++ )
    for( Int_t j=0; j<6; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<6; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<6; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      fCov[k] = 0;
      for( Int_t l=0; l<6; l++ ) fCov[k]+=mJC[i][l]*mJ[j][l];
    }
}


//* Fit utilities 

void AliHLT3DTrackParam::InitializeCovarianceMatrix()
{
  //* Initialization of covariance matrix

  for( Int_t i=0; i<21; i++ ) fCov[i] = 0;
  fSignQ = 0;
  fCov[0] = fCov[ 2] = fCov[ 5] = 100.;
  fCov[9] = fCov[14] = fCov[20] = 10000.;
  fChi2 = 0;
  fNDF = -5;
}

void AliHLT3DTrackParam::GetGlueMatrix( const Double_t xyz[3], 
					Double_t G[6], const Double_t *T0  ) const 
{
  //* !

  if( !T0 ) T0 = fParam;

  Double_t dx = xyz[0]-T0[0], dy = xyz[1]-T0[1], dz = xyz[2]-T0[2];
  Double_t px2= T0[3]*T0[3], py2= T0[4]*T0[4], pz2= T0[5]*T0[5];
  Double_t s2 = (dx*dx + dy*dy + dz*dz);
  Double_t p2 = px2 + py2 + pz2;
  if( p2>1.e-4 ) s2/=p2;
  Double_t x = T0[3]*s2;
  Double_t xx= px2*s2, xy= x*T0[4], xz= x*T0[5], yy= py2*s2, yz= T0[4]*T0[5]*s2;
  G[ 0]= xx;
  G[ 1]= xy;   G[ 2]= yy;
  G[ 3]= xz;   G[ 4]= yz;   G[ 5]= pz2*s2;  
}



void AliHLT3DTrackParam::Filter( const Double_t m[3], const Double_t V[6], const Double_t G[6] )
{ 
  //* !
  
  Double_t 
    c00 = fCov[ 0],
    c10 = fCov[ 1], c11 = fCov[ 2],
    c20 = fCov[ 3], c21 = fCov[ 4], c22 = fCov[ 5],
    c30 = fCov[ 6], c31 = fCov[ 7], c32 = fCov[ 8],
    c40 = fCov[10], c41 = fCov[11], c42 = fCov[12],
    c50 = fCov[15], c51 = fCov[16], c52 = fCov[17];
  
  double
    z0 = m[0]-fParam[0],
    z1 = m[1]-fParam[1],
    z2 = m[2]-fParam[2];
  
  Double_t mS[6] = { c00+V[0]+G[0], c10+V[1]+G[1], c11+V[2]+G[2],
		     c20+V[3]+G[3], c21+V[4]+G[4], c22+V[5]+G[5] };
  Double_t mSi[6];
  mSi[0] = mS[4]*mS[4] - mS[2]*mS[5];
  mSi[1] = mS[1]*mS[5] - mS[3]*mS[4];
  mSi[3] = mS[2]*mS[3] - mS[1]*mS[4];
  Double_t det = 1./(mS[0]*mSi[0] + mS[1]*mSi[1] + mS[3]*mSi[3]);
  mSi[0] *= det;
  mSi[1] *= det;
  mSi[3] *= det;
  mSi[2] = ( mS[3]*mS[3] - mS[0]*mS[5] )*det;
  mSi[4] = ( mS[0]*mS[4] - mS[1]*mS[3] )*det;
  mSi[5] = ( mS[1]*mS[1] - mS[0]*mS[2] )*det;
  
  fNDF  += 2;
  fChi2 += ( +(mSi[0]*z0 + mSi[1]*z1 + mSi[3]*z2)*z0
	     +(mSi[1]*z0 + mSi[2]*z1 + mSi[4]*z2)*z1
	     +(mSi[3]*z0 + mSi[4]*z1 + mSi[5]*z2)*z2 );
        
  Double_t k0, k1, k2 ; // k = CHtS
    
  k0 = c00*mSi[0] + c10*mSi[1] + c20*mSi[3];
  k1 = c00*mSi[1] + c10*mSi[2] + c20*mSi[4];
  k2 = c00*mSi[3] + c10*mSi[4] + c20*mSi[5];
    
  fParam[ 0]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [ 0]-= k0*c00 + k1*c10 + k2*c20;
  
  k0 = c10*mSi[0] + c11*mSi[1] + c21*mSi[3];
  k1 = c10*mSi[1] + c11*mSi[2] + c21*mSi[4];
  k2 = c10*mSi[3] + c11*mSi[4] + c21*mSi[5];
  
  fParam[ 1]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [ 1]-= k0*c00 + k1*c10 + k2*c20;
  fCov  [ 2]-= k0*c10 + k1*c11 + k2*c21;
  
  k0 = c20*mSi[0] + c21*mSi[1] + c22*mSi[3];
  k1 = c20*mSi[1] + c21*mSi[2] + c22*mSi[4];
  k2 = c20*mSi[3] + c21*mSi[4] + c22*mSi[5];
  
  fParam[ 2]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [ 3]-= k0*c00 + k1*c10 + k2*c20;
  fCov  [ 4]-= k0*c10 + k1*c11 + k2*c21;
  fCov  [ 5]-= k0*c20 + k1*c21 + k2*c22;
  
  k0 = c30*mSi[0] + c31*mSi[1] + c32*mSi[3];
  k1 = c30*mSi[1] + c31*mSi[2] + c32*mSi[4];
  k2 = c30*mSi[3] + c31*mSi[4] + c32*mSi[5];
  
  fParam[ 3]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [ 6]-= k0*c00 + k1*c10 + k2*c20;
  fCov  [ 7]-= k0*c10 + k1*c11 + k2*c21;
  fCov  [ 8]-= k0*c20 + k1*c21 + k2*c22;
  fCov  [ 9]-= k0*c30 + k1*c31 + k2*c32;
  
  k0 = c40*mSi[0] + c41*mSi[1] + c42*mSi[3];
  k1 = c40*mSi[1] + c41*mSi[2] + c42*mSi[4];
  k2 = c40*mSi[3] + c41*mSi[4] + c42*mSi[5];
    
  fParam[ 4]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [10]-= k0*c00 + k1*c10 + k2*c20;
  fCov  [11]-= k0*c10 + k1*c11 + k2*c21;
  fCov  [12]-= k0*c20 + k1*c21 + k2*c22;
  fCov  [13]-= k0*c30 + k1*c31 + k2*c32;
  fCov  [14]-= k0*c40 + k1*c41 + k2*c42;

  k0 = c50*mSi[0] + c51*mSi[1] + c52*mSi[3];
  k1 = c50*mSi[1] + c51*mSi[2] + c52*mSi[4];
  k2 = c50*mSi[3] + c51*mSi[4] + c52*mSi[5];
  
  fParam[ 5]+= k0*z0  + k1*z1  + k2*z2 ;
  fCov  [15]-= k0*c00 + k1*c10 + k2*c20;
  fCov  [16]-= k0*c10 + k1*c11 + k2*c21;
  fCov  [17]-= k0*c20 + k1*c21 + k2*c22;
  fCov  [18]-= k0*c30 + k1*c31 + k2*c32;
  fCov  [19]-= k0*c40 + k1*c41 + k2*c42;
  fCov  [20]-= k0*c50 + k1*c51 + k2*c52;

  // fit charge

  Double_t px = fParam[3];
  Double_t py = fParam[4];
  Double_t pz = fParam[5];
  
  Double_t p = TMath::Sqrt( px*px + py*py + pz*pz );
  Double_t pi = 1./p;
  Double_t qp = fSignQ*pi;
  Double_t qp3 = qp*pi*pi;
  Double_t 
    c60 = qp3*(c30+c40+c50),
    c61 = qp3*(c31+c41+c51),
    c62 = qp3*(c32+c42+c52);

  k0 = c60*mSi[0] + c61*mSi[1] + c62*mSi[3];
  k1 = c60*mSi[1] + c61*mSi[2] + c62*mSi[4];
  k2 = c60*mSi[3] + c61*mSi[4] + c62*mSi[5];
  
  qp+= k0*z0  + k1*z1  + k2*z2 ;
  if( qp>0 ) fSignQ = 1;
  else if(qp<0 ) fSignQ = -1;
  else fSignQ = 0;
}


//* Other utilities

void AliHLT3DTrackParam::SetDirection( Double_t Direction[3] )
{
  //* Change track direction 

  if( fParam[3]*Direction[0] + fParam[4]*Direction[1] + fParam[5]*Direction[2] >= 0 ) return;

  fParam[3] = -fParam[3];
  fParam[4] = -fParam[4];
  fParam[5] = -fParam[5];
  fSignQ    = -fSignQ;

  fCov[ 6]=-fCov[ 6]; fCov[ 7]=-fCov[ 7]; fCov[ 8]=-fCov[ 8];
  fCov[10]=-fCov[10]; fCov[11]=-fCov[11]; fCov[12]=-fCov[12];
  fCov[15]=-fCov[15]; fCov[16]=-fCov[16]; fCov[17]=-fCov[17];
}


void AliHLT3DTrackParam::RotateCoordinateSystem( Double_t alpha )
{
  //* !

  Double_t cA = TMath::Cos( alpha );
  Double_t sA = TMath::Sin( alpha );
  Double_t x= fParam[0], y= fParam[1], px= fParam[3], py= fParam[4];
  fParam[0] = x*cA + y*sA;
  fParam[1] =-x*sA + y*cA;
  fParam[2] = fParam[2];
  fParam[3] = px*cA + py*sA;
  fParam[4] =-px*sA + py*cA;
  fParam[5] = fParam[5];  

  Double_t mJ[6][6] = { { cA,sA, 0,  0,  0,  0 },
                        {-sA,cA, 0,  0,  0,  0 },
                        {  0, 0, 1,  0,  0,  0 },
                        {  0, 0, 0, cA, sA,  0 },
                        {  0, 0, 0,-sA, cA,  0 },
                        {  0, 0, 0,  0,  0,  1 }  };
                  
  Double_t mA[6][6];
  for( Int_t k=0,i=0; i<6; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = fCov[k]; 

  Double_t mJC[6][6];
  for( Int_t i=0; i<6; i++ )
    for( Int_t j=0; j<6; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<6; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<6; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      fCov[k] = 0;
      for( Int_t l=0; l<6; l++ ) fCov[k]+=mJC[i][l]*mJ[j][l];
    }
}


void AliHLT3DTrackParam::Get5Parameters( Double_t alpha, Double_t T[6], Double_t C[15] ) const
{
  //* !

  AliHLT3DTrackParam t = *this;
  t.RotateCoordinateSystem(alpha);
  Double_t 
    x= t.fParam[0], y= t.fParam[1], z = t.fParam[2], 
    px= t.fParam[3], py= t.fParam[4], pz = t.fParam[5], q = t.fSignQ;

  Double_t p2 = px*px+py*py+pz*pz;
  if( p2<1.e-8 ) p2 = 1;
  Double_t n2 = 1./p2;
  Double_t n = sqrt(n2);

  T[5] = x;
  T[0] = y;
  T[1] = z;
  T[2] = py/px;
  T[3] = pz/px;
  T[4] = q*n;

  Double_t mJ[5][6] = { { -T[2], 1, 0,  0,  0,  0 },
                        { -T[3], 0, 1,  0,  0,  0 },
                        { 0, 0, 0,  -T[2]/px,  1./px,  0 },
                        { 0, 0, 0, -T[3]/px,  0,  1./px },
                        { 0, 0, 0, -T[4]*n2*px, -T[4]*n2*py, -T[4]*n2*pz} };

  Double_t mA[6][6];
  for( Int_t k=0,i=0; i<6; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = t.fCov[k]; 

  Double_t mJC[5][6];
  for( Int_t i=0; i<5; i++ )
    for( Int_t j=0; j<6; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<6; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<5; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      C[k] = 0;
      for( Int_t l=0; l<6; l++ ) C[k]+=mJC[i][l]*mJ[j][l];
    }
}
