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

ClassImp(AliHLTTPCCATrackPar)

void AliHLTTPCCATrackPar::Init()
{
  //* Initialization 

  for( Int_t i=0; i<7; i++ ) fP[i] = 0;
  for( Int_t i=0; i<28; i++ ) fC[i] = 0;
  fC[0] = fC[2] = fC[5] = 100.;
  fC[9] = fC[14] = fC[20] = 100.;
  fC[27] = 10.;
  fChi2 = 0;
  fNDF = -5;
}


void AliHLTTPCCATrackPar::Normalize( Double_t Direction[3] )
{
  //* Normalize the track directions
  //* (Px,Py,Pz,qp)-> (Px,Py,Pz,qp)/sqrt(Px²+Py²+Pz²)
  //* 

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
  //* Get dS distance to the given space point for Bz field
  //*

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

  //* The Jacobian matrix is:
  //* J[7][7] = { {1,0,0,   sB, cB,  0,   dsB*px + dcB*py },
  //*             {0,1,0,  -cB, sB,  0, - dcB*px + dsB*py },
  //*             {0,0,1,    0,  0,  S,   0               },
  //*             {0,0,0,    c,  s,  0,   (-s*px + c*py)*bs },
  //*             {0,0,0,   -s,  c,  0,   (-c*px - s*py)*bs },
  //*             {0,0,0,    0,  0,  1,   0                 },
  //*             {0,0,0,    0,  0,  0,   1                 }  };
  //*
  //* below the fP=T0+J*(fP-T0) and fC=J*fC*Jt operations are performed
  //*

  Double_t h0 = dsB*px + dcB*py;
  Double_t h1 = - dcB*px + dsB*py ;
  Double_t h3 = (-s*px + c*py)*bs ;
  Double_t h4 = (-c*px - s*py)*bs ;


  fP[0] = T0[0] + d[0] + sB*d[3] + cB*d[4] + h0*d[6];
  fP[1] = T0[1] + d[1] - cB*d[3] + sB*d[4] + h1*d[6];
  fP[2] = T0[2] + d[2] + S*d[5];
  fP[3] = T0[3] + c*d[3] + s*d[4] + h3*d[6];
  fP[4] = T0[4] - s*d[3] + c*d[4] + h4*d[6];
  fP[5] = T0[5] + d[5];
  fP[6] = T0[6] + d[6];
  

  Double_t 
    c00 = fC[ 0],
    c01 = fC[ 1], c02 = fC[ 2],
    c03 = fC[ 3], c04 = fC[ 4], c05 = fC[ 5],
    c06 = fC[ 6], c07 = fC[ 7], c08 = fC[ 8], c09 = fC[ 9], 
    c10 = fC[10], c11 = fC[11], c12 = fC[12], c13 = fC[13], c14 = fC[14], 
    c15 = fC[15], c16 = fC[16], c17 = fC[17], c18 = fC[18], c19 = fC[19], c20 = fC[20], 
    c21 = fC[21], c22 = fC[22], c23 = fC[23], c24 = fC[24], c25 = fC[25], c26 = fC[26], c27 = fC[27];

  fC[ 0] = c00 + 2*c10*cB + c14*cB*cB + 2*c21*h0 + 2*c25*cB*h0 + c27*h0*h0 + 2*c06*sB + 2*c13*cB*sB + 2*c24*h0*sB + c09*sB*sB;
  fC[ 1] = c01 - c06*cB + c21*h1 + c10*sB + sB*(c07 - c09*cB + c24*h1 + c13*sB) 
    + cB*(c11 - c13*cB + c25*h1 + c14*sB) + h0*(c22 - c24*cB + c27*h1 + c25*sB);
  fC[ 2] = c02 - 2*c07*cB + c09*cB*cB + 2*c22*h1 - 2*c24*cB*h1 + c27*h1*h1 
    + 2*c11*sB - 2*c13*cB*sB + 2*c25*h1*sB + c14*sB*sB;
  fC[ 3] = c03 + c12*cB + c23*h0 + c15*S + c19*cB*S + c26*h0*S + c08*sB + c18*S*sB;
  fC[ 4] = c04 + c16*S  - cB*(c08 + c18*S) + h1*(c23 + c26*S) + (c12 + c19*S)*sB;
  fC[ 5] = c05 + S*(2*c17 + c20*S);
  fC[ 6] = c21*h3 + c25*cB*h3 + c27*h0*h3 + c10*s + c14*cB*s + c25*h0*s + c24*h3*sB + c13*s*sB + c*(c06 + c13*cB + c24*h0 + c09*sB);
  fC[ 7] =  c*c07 + c22*h3 + c11*s - cB*(c*c09 + c24*h3 + c13*s) + 
    h1*(c*c24 + c27*h3 + c25*s) + (c*c13 + c25*h3 + c14*s)*sB;
  fC[ 8] = c23*h3 + c12*s + c26*h3*S + c19*s*S + c*(c08 + c18*S);
  fC[ 9] = c*c*c09 + c27*h3*h3 + 2*c*(c24*h3 + c13*s) + s*(2*c25*h3 + c14*s);
  fC[10] = c21*h4 + c25*cB*h4 + c27*h0*h4 - c06*s - c13*cB*s - c24*h0*s 
    + c24*h4*sB - c09*s*sB + c*(c10 + c14*cB + c25*h0 + c13*sB);
  fC[11] =  c22*h4 - c24*cB*h4 + c27*h1*h4 - c07*s + c09*cB*s - c24*h1*s + 
    c25*h4*sB - c13*s*sB + c*(c11 - c13*cB + c25*h1 + c14*sB);
  fC[12] =  c23*h4 - c08*s + c26*h4*S - c18*s*S + c*(c12 + c19*S);
  fC[13] =  c*c*c13 + c27*h3*h4 - s*(c24*h3 - c25*h4 + c13*s) + c*(c25*h3 + c24*h4 - c09*s + c14*s);
  fC[14] = c*c*c14 + 2*c*c25*h4 + c27*h4*h4 - 2*c*c13*s - 2*c24*h4*s + c09*s*s;
  fC[15] = c15 + c19*cB + c26*h0 + c18*sB;
  fC[16] = c16 - c18*cB + c26*h1 + c19*sB;
  fC[17] = c17 + c20*S;
  fC[18] = c*c18 + c26*h3 + c19*s;
  fC[19] = c*c19 + c26*h4 - c18*s;
  fC[20] = c20;
  fC[21] = c21 + c25*cB + c27*h0 + c24*sB;
  fC[22] = c22 - c24*cB + c27*h1 + c25*sB;
  fC[23] = c23 + c26*S;
  fC[24] = c*c24 + c27*h3 + c25*s;
  fC[25] = c*c25 + c27*h4 - c24*s;
  fC[26] = c26;
  fC[27] = c27;
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


void AliHLTTPCCATrackPar::Filter( const Double_t m[], const Double_t V[], const Double_t G[6] )
{
  //* Add the measurement m to the track using the Kalman Filter mathematics
  //* m[3] is the measurement
  //* V[6] is the low-triangular covariance matrix
  //* G[6] is the track->measurement "connection matrix", additional to V[]
  //*

  Double_t 
    c00 = fC[ 0],
    c10 = fC[ 1], c11 = fC[ 2],
    c20 = fC[ 3], c21 = fC[ 4], c22 = fC[ 5],
    c30 = fC[ 6], c31 = fC[ 7], c32 = fC[ 8],
    c40 = fC[10], c41 = fC[11], c42 = fC[12],
    c50 = fC[15], c51 = fC[16], c52 = fC[17],
    c60 = fC[21], c61 = fC[22], c62 = fC[23];
  
  Double_t
    z0 = m[0]-fP[0],
    z1 = m[1]-fP[1],
    z2 = m[2]-fP[2];
  
  Double_t mS[6] = { c00+V[0]+G[0], c10+V[1]+G[1], c11+V[2]+G[2],
		     c20+V[3]+G[3], c21+V[4]+G[4], c22+V[5]+G[5] };
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
  //* Rotate the track parameters on the alpha angle 

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
const 
{
  //* Convert the track parameterisation to {y,z,ty,tz,q/p,x}
  //* The result is stored in T[], corresponding covariance matrix in C[]
  //*
  //* The method is used for debuging
  //*

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
