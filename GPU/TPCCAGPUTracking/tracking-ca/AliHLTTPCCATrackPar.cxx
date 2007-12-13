// @(#) $Id$
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
//                                                                        *
// Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
//                  for The ALICE HLT Project.                            *
//                                                                        *
// Permission to use, copy, modify and distribute this software and its   *
// documentation strictly for non-commercial purposes is hereby granted   *
// without fee, provided that the above copyright notice appears in all   *
// copies and that both the copyright notice and this permission notice   *
// appear in the supporting documentation. The authors make no claims     *
// about the suitability of this software for any purpose. It is          *
// provided "as is" without express or implied warranty.                  *
//*************************************************************************

#include "AliHLTTPCCATrackPar.h"
#include "TMath.h"

ClassImp(AliHLTTPCCATrackPar);

void AliHLTTPCCATrackPar::Init()
{
  //* Initialization 

  for( Int_t i=0; i<7; i++ ) fP[i] = 0;
  for( Int_t i=0; i<28; i++ ) fC[i] = 0;
  fC[0] = fC[2] = fC[5] = 10000;
  fC[9] = fC[14] = fC[20] = 10000.;
  fC[27] = 10.;
  fChi2 = 0;
  fNDF = -5;
}


void AliHLTTPCCATrackPar::Normalize( Double_t Direction[3] )
{
  //* Normalize the track

  Double_t p2 = fP[3]*fP[3] + fP[4]*fP[4] + fP[5]*fP[5];
  if( p2<1.e-4 ) return;
  Double_t a2 = 1./p2;
  Double_t a = sqrt(a2);

  if( Direction && ( fP[3]*Direction[0] + fP[4]*Direction[1] + fP[5]*Direction[2] < 0 ) ) a = -a;
  
  Double_t ex = fP[3]*a, ey = fP[4]*a, ez = fP[5]*a, qp = fP[6]*a;

  fP[3] = ex;
  fP[4] = ey;
  fP[5] = ez;
  fP[6] = qp;

  Double_t 
    h0 = fC[ 6]*ex + fC[10]*ey + fC[15]*ez,
    h1 = fC[ 7]*ex + fC[11]*ey + fC[16]*ez, 
    h2 = fC[ 8]*ex + fC[12]*ey + fC[17]*ez,
    h3 = fC[ 9]*ex + fC[13]*ey + fC[18]*ez, 
    h4 = fC[13]*ex + fC[14]*ey + fC[19]*ez, 
    h5 = fC[18]*ex + fC[19]*ey + fC[20]*ez,
    h6 = fC[24]*ex + fC[25]*ey + fC[26]*ez,
    d  = h3*ex + h4*ey + h5*ez, 
    hh = h6 - qp*d ;

  fC[ 6]= a*(fC[ 6] -ex*h0); fC[ 7]= a*(fC[ 7] -ex*h1); fC[ 8]= a*(fC[ 8] -ex*h2); 
  fC[10]= a*(fC[10] -ey*h0); fC[11]= a*(fC[11] -ey*h1); fC[12]= a*(fC[12] -ey*h2); 
  fC[15]= a*(fC[15] -ez*h0); fC[16]= a*(fC[16] -ez*h1); fC[17]= a*(fC[17] -ez*h2); 
  fC[21]= a*(fC[21] -qp*h0); fC[22]= a*(fC[22] -qp*h1); fC[23]= a*(fC[23] -qp*h2);     

  fC[ 9]= a2*( fC[ 9] -h3*ex -h3*ex + d*ex*ex );
  fC[13]= a2*( fC[13] -h4*ex -h3*ey + d*ey*ex ); 
  fC[14]= a2*( fC[14] -h4*ey -h4*ey + d*ey*ey );

  fC[18]= a2*( fC[18] -h5*ex -h3*ez + d*ez*ex ); 
  fC[19]= a2*( fC[19] -h5*ey -h4*ez + d*ez*ey ); 
  fC[20]= a2*( fC[20] -h5*ez -h5*ez + d*ez*ez );

  fC[24]= a2*( fC[24] -qp*h3 - hh*ex ); 
  fC[25]= a2*( fC[25] -qp*h4 - hh*ey ); 
  fC[26]= a2*( fC[26] -qp*h5 - hh*ez );
  fC[27]= a2*( fC[27] -qp*h6 - hh*qp  ); 
}


Double_t AliHLTTPCCATrackPar::GetDsToPointBz( Double_t Bz, const Double_t xyz[3], const Double_t *T0 ) const
{
  //* Get dS to a certain space point for Bz field
  const Double_t kCLight = 0.000299792458;
  Double_t bq = Bz*T0[6]*kCLight;
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

Double_t AliHLTTPCCATrackPar::GetDsToPointBz( Double_t Bz, const Double_t xyz[3] ) const 
{
  return GetDsToPointBz( Bz, xyz, fP );
}

void AliHLTTPCCATrackPar::TransportBz( Double_t Bz, Double_t S, Double_t *T0 ) 
{ 
  //* Transport the particle on dS, for Bz field
 
  const Double_t kCLight = 0.000299792458;
  Bz = Bz*kCLight;
  Double_t bs= Bz*S;
  Double_t bqs= bs*T0[6];
  Double_t s = TMath::Sin(bqs), c = TMath::Cos(bqs);
  Double_t sB, cB, dsB, dcB;
  if( TMath::Abs(bqs)>1.e-10){
    sB= s/Bz/T0[6];
    cB= (1-c)/Bz/T0[6];
    dsB = (c*S - sB)/T0[6];
    dcB = (s*S-cB)/T0[6];
  }else{
    sB = (1. - bqs*bqs/6.)*S;
    cB = .5*sB*bqs;
    dsB = - T0[6]*bs*bs/3.*S;
    dcB = .5*(sB*bs - dsB*bqs);
  }
  
  Double_t px = T0[3];
  Double_t py = T0[4];
  Double_t pz = T0[5];
  
  Double_t d[7] = { fP[0]-T0[0], fP[1]-T0[1], fP[2]-T0[2], 
		    fP[3]-T0[3], fP[4]-T0[4], fP[5]-T0[5], fP[6]-T0[6] };

  T0[0] = T0[0] + sB*px + cB*py;
  T0[1] = T0[1] - cB*px + sB*py;
  T0[2] = T0[2] +  S*pz                       ;
  T0[3] =          c*px + s*py;
  T0[4] =         -s*px + c*py;
  T0[5] = T0[5];
  T0[6] = T0[6];

 
  Double_t mJ[7][7] = { {1,0,0,   sB, cB,  0,   dsB*px + dcB*py },
                        {0,1,0,  -cB, sB,  0, - dcB*px + dsB*py },
                        {0,0,1,    0,  0,  S,   0               },
                        {0,0,0,    c,  s,  0,   (-s*px + c*py)*bs },
                        {0,0,0,   -s,  c,  0,   (-c*px - s*py)*bs },
                        {0,0,0,    0,  0,  1,   0                 },
                        {0,0,0,    0,  0,  0,   1                 }  };

  for( Int_t i=0; i<7; i++){
    fP[i] = T0[i];
    for( Int_t j=0; j<7; j++) fP[i] += mJ[i][j]*d[j];
  }

  Double_t mA[7][7];
  for( Int_t k=0,i=0; i<7; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = fC[k]; 

  Double_t mJC[7][7];
  for( Int_t i=0; i<7; i++ )
    for( Int_t j=0; j<7; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<7; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<7; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      fC[k] = 0;
      for( Int_t l=0; l<7; l++ ) fC[k]+=mJC[i][l]*mJ[j][l];
    }
}

void AliHLTTPCCATrackPar::TransportBz( Double_t Bz, Double_t dS ) 
{ 
  //* Transport the particle on dS, for Bz field
  TransportBz( Bz, dS, fP );
}

void AliHLTTPCCATrackPar::GetConnectionMatrix( Double_t B, const Double_t p[3], Double_t G[6], const Double_t *T0  ) const 
{
  //* Calculate connection matrix between track and point p
  if( !G ) return;
  const Double_t kLight = 0.000299792458;
  B*=kLight;
  Double_t dx = p[0]-T0[0], dy = p[1]-T0[1], dz = p[2]-T0[2];
  Double_t px2= T0[3]*T0[3], py2= T0[4]*T0[4], pz2= T0[5]*T0[5];
  //Double_t B2 = B*B;
  Double_t s2 = (dx*dx + dy*dy + dz*dz);
  Double_t p2 = px2 + py2 + pz2;
  if( p2>1.e-4 ) s2/=p2;
  Double_t x = T0[3]*s2;
  Double_t xx= px2*s2, xy= x*T0[4], xz= x*T0[5], yy= py2*s2, yz= T0[4]*T0[5]*s2;
  //Double_t Bxy= B*xy; 
  G[ 0]= xx;
  G[ 1]= xy;   G[ 2]= yy;
  G[ 3]= xz;   G[ 4]= yz;   G[ 5]= pz2*s2;  
  /*
  C[ 0]+= xx;
  C[ 1]+= xy;   C[ 2]+= yy;
  C[ 3]+= xz;   C[ 4]+= yz;   C[ 5]+= pz2*s2;  
  C[ 6]+= Bxy; C[ 7]+= B*yy; C[ 8]+= B*yz;    C[ 9]+=B2*yy;
  C[10]-= B*xx; C[11]-= Bxy; C[12]-=B*xz;     C[13]-=B2*xy; C[14]+=B2*xx;  
  */
}


void AliHLTTPCCATrackPar::Filter( const Double_t m[3], const Double_t V[6], const Double_t V1[6] )
{
  //* !
  Double_t 
    c00 = fC[ 0],
    c10 = fC[ 1], c11 = fC[ 2],
    c20 = fC[ 3], c21 = fC[ 4], c22 = fC[ 5],
    c30 = fC[ 6], c31 = fC[ 7], c32 = fC[ 8],
    c40 = fC[10], c41 = fC[11], c42 = fC[12],
    c50 = fC[15], c51 = fC[16], c52 = fC[17],
    c60 = fC[21], c61 = fC[22], c62 = fC[23];
  
  double
    z0 = m[0]-fP[0],
    z1 = m[1]-fP[1],
    z2 = m[2]-fP[2];
  
  Double_t mS[6] = { c00+V[0]+V1[0], c10+V[1]+V1[1], c11+V[2]+V1[2],
		     c20+V[3]+V1[3], c21+V[4]+V1[4], c22+V[5]+V1[5] };
  Double_t mSi[6];
  mSi[0] = mS[4]*mS[4] - mS[2]*mS[5];
  mSi[1] = mS[1]*mS[5] - mS[3]*mS[4];
  mSi[3] = mS[2]*mS[3] - mS[1]*mS[4];
  Double_t det = (mS[0]*mSi[0] + mS[1]*mSi[1] + mS[3]*mSi[3]);
  if( TMath::Abs(det)<1.e-10 ) return;
  det = 1./det;
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
    
  fP[ 0]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[ 0]-= k0*c00 + k1*c10 + k2*c20;
  
  k0 = c10*mSi[0] + c11*mSi[1] + c21*mSi[3];
  k1 = c10*mSi[1] + c11*mSi[2] + c21*mSi[4];
  k2 = c10*mSi[3] + c11*mSi[4] + c21*mSi[5];
  
  fP[ 1]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[ 1]-= k0*c00 + k1*c10 + k2*c20;
  fC[ 2]-= k0*c10 + k1*c11 + k2*c21;
  
  k0 = c20*mSi[0] + c21*mSi[1] + c22*mSi[3];
  k1 = c20*mSi[1] + c21*mSi[2] + c22*mSi[4];
  k2 = c20*mSi[3] + c21*mSi[4] + c22*mSi[5];
  
  fP[ 2]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[ 3]-= k0*c00 + k1*c10 + k2*c20;
  fC[ 4]-= k0*c10 + k1*c11 + k2*c21;
  fC[ 5]-= k0*c20 + k1*c21 + k2*c22;
  
  k0 = c30*mSi[0] + c31*mSi[1] + c32*mSi[3];
  k1 = c30*mSi[1] + c31*mSi[2] + c32*mSi[4];
  k2 = c30*mSi[3] + c31*mSi[4] + c32*mSi[5];
  
  fP[ 3]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[ 6]-= k0*c00 + k1*c10 + k2*c20;
  fC[ 7]-= k0*c10 + k1*c11 + k2*c21;
  fC[ 8]-= k0*c20 + k1*c21 + k2*c22;
  fC[ 9]-= k0*c30 + k1*c31 + k2*c32;
  
  k0 = c40*mSi[0] + c41*mSi[1] + c42*mSi[3];
  k1 = c40*mSi[1] + c41*mSi[2] + c42*mSi[4];
  k2 = c40*mSi[3] + c41*mSi[4] + c42*mSi[5];
    
  fP[ 4]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[10]-= k0*c00 + k1*c10 + k2*c20;
  fC[11]-= k0*c10 + k1*c11 + k2*c21;
  fC[12]-= k0*c20 + k1*c21 + k2*c22;
  fC[13]-= k0*c30 + k1*c31 + k2*c32;
  fC[14]-= k0*c40 + k1*c41 + k2*c42;

  k0 = c50*mSi[0] + c51*mSi[1] + c52*mSi[3];
  k1 = c50*mSi[1] + c51*mSi[2] + c52*mSi[4];
  k2 = c50*mSi[3] + c51*mSi[4] + c52*mSi[5];
  
  fP[ 5]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[15]-= k0*c00 + k1*c10 + k2*c20;
  fC[16]-= k0*c10 + k1*c11 + k2*c21;
  fC[17]-= k0*c20 + k1*c21 + k2*c22;
  fC[18]-= k0*c30 + k1*c31 + k2*c32;
  fC[19]-= k0*c40 + k1*c41 + k2*c42;
  fC[20]-= k0*c50 + k1*c51 + k2*c52;

  k0 = c60*mSi[0] + c61*mSi[1] + c62*mSi[3];
  k1 = c60*mSi[1] + c61*mSi[2] + c62*mSi[4];
  k2 = c60*mSi[3] + c61*mSi[4] + c62*mSi[5];
  
  fP[ 6]+= k0*z0  + k1*z1  + k2*z2 ;
  fC[21]-= k0*c00 + k1*c10 + k2*c20;
  fC[22]-= k0*c10 + k1*c11 + k2*c21;
  fC[23]-= k0*c20 + k1*c21 + k2*c22;
  fC[24]-= k0*c30 + k1*c31 + k2*c32;
  fC[25]-= k0*c40 + k1*c41 + k2*c42;
  fC[26]-= k0*c50 + k1*c51 + k2*c52;
  fC[27]-= k0*c60 + k1*c61 + k2*c62;
}


void AliHLTTPCCATrackPar::Rotate( Double_t alpha )
{
  //* !
  Double_t cA = TMath::Cos( alpha );
  Double_t sA = TMath::Sin( alpha );
  Double_t x= fP[0], y= fP[1], px= fP[3], py= fP[4];
  fP[0] = x*cA + y*sA;
  fP[1] =-x*sA + y*cA;
  fP[2] = fP[2];
  fP[3] = px*cA + py*sA;
  fP[4] =-px*sA + py*cA;
  fP[5] = fP[5];
  fP[6] = fP[6];

  Double_t mJ[7][7] = { { cA,sA, 0,  0,  0,  0,  0 },
                        {-sA,cA, 0,  0,  0,  0,  0 },
                        {  0, 0, 1,  0,  0,  0,  0 },
                        {  0, 0, 0, cA, sA,  0,  0 },
                        {  0, 0, 0,-sA, cA,  0,  0 },
                        {  0, 0, 0,  0,  0,  1,  0 },
                        {  0, 0, 0,  0,  0,  0,  1 }  };

  Double_t mA[7][7];
  for( Int_t k=0,i=0; i<7; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = fC[k]; 

  Double_t mJC[7][7];
  for( Int_t i=0; i<7; i++ )
    for( Int_t j=0; j<7; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<7; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<7; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      fC[k] = 0;
      for( Int_t l=0; l<7; l++ ) fC[k]+=mJC[i][l]*mJ[j][l];
    }
}


void AliHLTTPCCATrackPar::ConvertTo5( Double_t alpha, Double_t T[], Double_t C[] )
const {
  //* !
  AliHLTTPCCATrackPar t = *this;
  t.Rotate(alpha);
  Double_t 
    x= t.fP[0], y= t.fP[1], z = t.fP[2], 
    ex= t.fP[3], ey= t.fP[4], ez = t.fP[5], qp = t.fP[6];

  Double_t p2 = ex*ex+ey*ey+ez*ez;
  if( p2<1.e-4 ) p2 = 1;
  Double_t n2 = 1./p2;
  Double_t n = sqrt(n2);

  T[5] = x;
  T[0] = y;
  T[1] = z;
  T[2] = ey/ex;
  T[3] = ez/ex;
  T[4] = qp*n;

  Double_t mJ[5][7] = { { -T[2], 1, 0,  0,  0,  0,  0 },
                        { -T[3], 0, 1,  0,  0,  0,  0 },
                        { 0, 0, 0,  -T[2]/ex,  1./ex,  0,  0 },
                        { 0, 0, 0, -T[3]/ex,  0,  1./ex,  0 },
                        { 0, 0, 0, -T[4]*n2*ex, -T[4]*n2*ey, -T[4]*n2*ez, n }};

  Double_t mA[7][7];
  for( Int_t k=0,i=0; i<7; i++)
    for( Int_t j=0; j<=i; j++, k++ ) mA[i][j] = mA[j][i] = t.fC[k]; 

  Double_t mJC[5][7];
  for( Int_t i=0; i<5; i++ )
    for( Int_t j=0; j<7; j++ ){
      mJC[i][j]=0;
      for( Int_t k=0; k<7; k++ ) mJC[i][j]+=mJ[i][k]*mA[k][j];
    }
  
  for( Int_t k=0,i=0; i<5; i++)
    for( Int_t j=0; j<=i; j++, k++ ){
      C[k] = 0;
      for( Int_t l=0; l<7; l++ ) C[k]+=mJC[i][l]*mJ[j][l];
    }
}
