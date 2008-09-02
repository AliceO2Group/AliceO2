// $Id$
//***************************************************************************
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

#include "AliHLTTPCCATrackParam.h"
#include "TMath.h"
#include "AliExternalTrackParam.h"

//ClassImp(AliHLTTPCCATrackParam)

//
// Circle in XY:
// 
// R  = 1/TMath::Abs(Kappa);
// Xc = X - sin(Phi)/Kappa;
// Yc = Y + cos(Phi)/Kappa;
//



void AliHLTTPCCATrackParam::ConstructXY3( const Float_t x[3], const Float_t y[3], 
					  const Float_t sigmaY2[3], Float_t CosPhi0 )
{
  //* Construct the track in XY plane by 3 points

  Float_t x0 = x[0];
  Float_t y0 = y[0];
  Float_t x1 = x[1] - x0;
  Float_t y1 = y[1] - y0;
  Float_t x2 = x[2] - x0;
  Float_t y2 = y[2] - y0;
  
  Float_t a1 = x1*x1 + y1*y1;
  Float_t a2 = x2*x2 + y2*y2;
  Float_t a = 2*(x1*y2 - y1*x2);
  Float_t lx =  a1*y2 - a2*y1;
  Float_t ly = -a1*x2 + a2*x1;
  Float_t l = TMath::Sqrt(lx*lx + ly*ly);
 
  Float_t li = 1./l;
  Float_t li2 = li*li;
  Float_t li3 = li2*li;
  Float_t cosPhi = ly*li;

  Float_t sinPhi = -lx*li;
  Float_t kappa = a/l;

  Float_t dlx = a2 - a1;     //  D lx / D y0
  Float_t dly = -a;          //  D ly / D y0
  Float_t dA  = 2*(x2 - x1); // D a  / D y0
  Float_t dl = (lx*dlx + ly*dly)*li;
  
  // D sinPhi,kappa / D y0

  Float_t d0[2] = { -(dlx*ly-lx*dly)*ly*li3, (dA*l-a*dl)*li2 };

  // D sinPhi,kappa / D y1 

  dlx = -a2 + 2*y1*y2;
  dly = -2*x2*y1;
  dA  = -2*x2;
  dl = (lx*dlx + ly*dly)*li;

  Float_t d1[2] = { -(dlx*ly-lx*dly)*ly*li3, (dA*l-a*dl)*li2 };

  // D sinPhi,kappa / D y2

  dlx = a1 - 2*y1*y2;
  dly = -2*x1*y2;
  dA  = 2*x1;
  dl = (lx*dlx + ly*dly)*li;

  Float_t d2[2] = { -(dlx*ly-lx*dly)*ly*li3, (dA*l-a*dl)*li2 };
   
  if( CosPhi0*cosPhi <0 ){
    cosPhi = -cosPhi;
    sinPhi = -sinPhi;
    kappa = -kappa;   
    d0[0] = -d0[0];
    d0[1] = -d0[1];
    d1[0] = -d1[0];
    d1[1] = -d1[1];
    d2[0] = -d2[0];
    d2[1] = -d2[1];
  }
  
  X() = x0;  
  Y() = y0;
  SinPhi() = sinPhi;
  Kappa() = kappa;
  CosPhi() = cosPhi;

  Float_t s0 = sigmaY2[0];
  Float_t s1 = sigmaY2[1];
  Float_t s2 = sigmaY2[2];
  
  fC[0] = s0;
  fC[1] = 0;
  fC[2] = 0;

  fC[3] = d0[0]*s0;
  fC[4] = 0;
  fC[5] = d0[0]*d0[0]*s0 + d1[0]*d1[0]*s1 + d2[0]*d2[0]*s2;

  fC[6] = 0;
  fC[7] = 0;
  fC[8] = 0;
  fC[9] = 0;

  fC[10] = d0[1]*s0;
  fC[11] = 0;
  fC[12] = d0[0]*d0[1]*s0 + d1[0]*d1[1]*s1 + d2[0]*d2[1]*s2;
  fC[13] = 0;
  fC[14] = d0[1]*d0[1]*s0 + d1[1]*d1[1]*s1 + d2[1]*d2[1]*s2;
}


Float_t  AliHLTTPCCATrackParam::GetS( Float_t x, Float_t y ) const
{
  //* Get XY path length to the given point

  Float_t k  = GetKappa();
  Float_t ex = GetCosPhi();
  Float_t ey = GetSinPhi();
  x-= GetX();
  y-= GetY();
  Float_t dS = x*ex + y*ey;
  if( TMath::Abs(k)>1.e-4 ) dS = TMath::ATan2( k*dS, 1+k*(x*ey-y*ex) )/k;
  return dS;
}

void  AliHLTTPCCATrackParam::GetDCAPoint( Float_t x, Float_t y, Float_t z,
					   Float_t &xp, Float_t &yp, Float_t &zp ) const
{
  //* Get the track point closest to the (x,y,z)

  Float_t x0 = GetX();
  Float_t y0 = GetY();
  Float_t k  = GetKappa();
  Float_t ex = GetCosPhi();
  Float_t ey = GetSinPhi();
  Float_t dx = x - x0;
  Float_t dy = y - y0; 
  Float_t ax = dx*k+ey;
  Float_t ay = dy*k-ex;
  Float_t a = sqrt( ax*ax+ay*ay );
  xp = x0 + (dx - ey*( (dx*dx+dy*dy)*k - 2*(-dx*ey+dy*ex) )/(a+1) )/a;
  yp = y0 + (dy + ex*( (dx*dx+dy*dy)*k - 2*(-dx*ey+dy*ex) )/(a+1) )/a;
  Float_t s = GetS(x,y);
  zp = GetZ() + GetDzDs()*s;
  if( TMath::Abs(k)>1.e-2 ){
    Float_t dZ = TMath::Abs( GetDzDs()*TMath::TwoPi()/k );
    if( dZ>.1 ){
      zp+= TMath::Nint((z-zp)/dZ)*dZ;    
    }
  }
}

void AliHLTTPCCATrackParam::ConstructXYZ3( const Float_t p0[5], const Float_t p1[5], 
					   const Float_t p2[5], 
					   Float_t CosPhi0, Float_t t0[] )
{      
  //* Construct the track in XYZ by 3 points

  Float_t px[3]   = { p0[0], p1[0], p2[0] };
  Float_t py[3]   = { p0[1], p1[1], p2[1] };
  Float_t pz[3]   = { p0[2], p1[2], p2[2] };
  Float_t ps2y[3] = { p0[3]*p0[3], p1[3]*p1[3], p2[3]*p2[3] };
  Float_t ps2z[3] = { p0[4]*p0[4], p1[4]*p1[4], p2[4]*p2[4] };

  Float_t kold = t0 ?t0[4] :0;
  ConstructXY3( px, py, ps2y, CosPhi0 );

  Float_t pS[3] = { GetS(px[0],py[0]), GetS(px[1],py[1]), GetS(px[2],py[2]) };
  Float_t k = Kappa();
  if( TMath::Abs(k)>1.e-2 ){    
    Float_t dS = TMath::Abs( TMath::TwoPi()/k );
    pS[1]+= TMath::Nint( (pS[0]-pS[1])/dS )*dS; // not more than half turn
    pS[2]+= TMath::Nint( (pS[1]-pS[2])/dS )*dS;
    if( t0 ){
      Float_t dZ = TMath::Abs(t0[3]*dS);
      if( TMath::Abs(dZ)>1. ){
	Float_t dsDz = 1./t0[3];
	if( kold*k<0 ) dsDz = -dsDz;
	Float_t s0 = (pz[0]-t0[1])*dsDz;
	Float_t s1 = (pz[1]-t0[1])*dsDz;
	Float_t s2 = (pz[2]-t0[1])*dsDz;	
	pS[0]+= TMath::Nint( (s0-pS[0])/dS )*dS ;
	pS[1]+= TMath::Nint( (s1-pS[1])/dS )*dS ;
	pS[2]+= TMath::Nint( (s2-pS[2])/dS )*dS ;	
      }
    }
  }

  Float_t s = pS[0] + pS[1] + pS[2];
  Float_t z = pz[0] + pz[1] + pz[2];
  Float_t sz = pS[0]*pz[0] + pS[1]*pz[1] + pS[2]*pz[2];
  Float_t ss = pS[0]*pS[0] + pS[1]*pS[1] + pS[2]*pS[2];
  
  Float_t a = 3*ss-s*s;
  Z() = (z*ss-sz*s)/a; // z0
  DzDs() = (3*sz-z*s)/a; // t = dz/ds
    
  Float_t dz0[3] = {ss - pS[0]*s,ss - pS[1]*s,ss - pS[2]*s };
  Float_t dt [3] = {3*pS[0] - s, 3*pS[1] - s, 3*pS[2] - s };

  fC[2] = (dz0[0]*dz0[0]*ps2z[0] + dz0[1]*dz0[1]*ps2z[1] + dz0[2]*dz0[2]*ps2z[2])/a/a;
  fC[7]= (dz0[0]*dt [0]*ps2z[0] + dz0[1]*dt [1]*ps2z[1] + dz0[2]*dt [2]*ps2z[2])/a/a;  
  fC[9]= (dt [0]*dt [0]*ps2z[0] + dt [1]*dt [1]*ps2z[1] + dt [2]*dt [2]*ps2z[2])/a/a;  
}


Bool_t  AliHLTTPCCATrackParam::TransportToX( Float_t x )
{
  //* Transport the track parameters to X=x 

  Bool_t ret = 1;

  Float_t x0  = X();
  //Float_t y0  = Y();
  Float_t k   = Kappa();
  Float_t ex = CosPhi();
  Float_t ey = SinPhi();
  Float_t dx = x - x0;

  Float_t ey1 = k*dx + ey;
  Float_t ex1;
  if( TMath::Abs(ey1)>1 ){ // no intersection -> check the border    
    ey1 = ( ey1>0 ) ?1 :-1;
    ex1 = 0;
    dx = ( TMath::Abs(k)>1.e-4) ? ( (ey1-ey)/k ) :0;
    
    Float_t ddx = TMath::Abs(x0+dx - x)*k*k;
    Float_t hx[] = {0, -k, 1+ey };
    Float_t sx2 = hx[1]*hx[1]*fC[ 3] + hx[2]*hx[2]*fC[ 5];
    if( ddx*ddx>3.5*3.5*sx2 ) ret = 0; // x not withing the error
    ret = 0; // any case
    return ret;
  }else{
    ex1 = TMath::Sqrt(1 - ey1*ey1);
    if( ex<0 ) ex1 = -ex1;  
  }
  
  Float_t dx2 = dx*dx;
  CosPhi() = ex1;
  Float_t ss = ey+ey1;
  Float_t cc = ex+ex1;  
  Float_t tg = 0;
  if( TMath::Abs(cc)>1.e-4 ) tg = ss/cc; // tan((phi1+phi)/2)
  else ret = 0; 
  Float_t dy = dx*tg;
  Float_t dl = dx*TMath::Sqrt(1+tg*tg);

  if( cc<0 ) dl = -dl;
  Float_t dSin = dl*k/2;
  if( dSin > 1 ) dSin = 1;
  if( dSin <-1 ) dSin = -1;
  Float_t dS = ( TMath::Abs(k)>1.e-4)  ? (2*TMath::ASin(dSin)/k) :dl;
  Float_t dz = dS*DzDs();

  Float_t cci = 0, exi = 0, ex1i = 0;
  if( TMath::Abs(cc)>1.e-4 ) cci = 1./cc;
  else ret = 0;
  if( TMath::Abs(ex)>1.e-4 ) exi = 1./ex;
  else ret = 0;
  if( TMath::Abs(ex1)>1.e-4 ) ex1i = 1./ex1;
  else ret = 0;

  if( !ret ) return ret;
  X() += dx;
  fP[0]+= dy;
  fP[1]+= dz;
  fP[2] = ey1;
  fP[3] = fP[3];
  fP[4] = fP[4];

  Float_t h2 = dx*(1+ ex*ex1 + ey*ey1 )*cci*exi*ex1i;
  Float_t h4 = dx2*(cc + ss*ey1*ex1i )*cci*cci;

  Float_t c00 = fC[0];
  Float_t c10 = fC[1];
  Float_t c11 = fC[2];
  Float_t c20 = fC[3];
  Float_t c21 = fC[4];
  Float_t c22 = fC[5];
  Float_t c30 = fC[6];
  Float_t c31 = fC[7];
  Float_t c32 = fC[8];
  Float_t c33 = fC[9];
  Float_t c40 = fC[10];
  Float_t c41 = fC[11];
  Float_t c42 = fC[12];
  Float_t c43 = fC[13];
  Float_t c44 = fC[14];

  //Float_t H0[5] = { 1,0, h2,  0, h4 };
  //Float_t H1[5] = { 0, 1, 0, dS,  0 };
  //Float_t H2[5] = { 0, 0, 1,  0, dx };
  //Float_t H3[5] = { 0, 0, 0,  1,  0 };
  //Float_t H4[5] = { 0, 0, 0,  0,  1 };


  fC[0]=( c00  + h2*h2*c22 + h4*h4*c44 
	  + 2*( h2*c20 + h4*c40 + h2*h4*c42 )  ); 

  fC[1]= c10 + h2*c21 + h4*c41 + dS*(c30 + h2*c32 + h4*c43);
  fC[2]= c11 + 2*dS*c31 + dS*dS*c33;

  fC[3]= c20 + h2*c22 + h4*c42 + dx*( c40 + h2*c42 + h4*c44);
  fC[4]= c21 + dS*c32 + dx*(c41 + dS*c43);
  fC[5]= c22 +2*dx*c42 + dx2*c44;

  fC[6]= c30 + h2*c32 + h4*c43;
  fC[7]= c31 + dS*c33;
  fC[8]= c32 + dx*c43;
  fC[9]= c33;

  fC[10]= c40 + h2*c42 + h4*c44;
  fC[11]= c41 + dS*c43;
  fC[12]= c42 + dx*c44;
  fC[13]= c43;
  fC[14]= c44;

  return ret;
}



Bool_t AliHLTTPCCATrackParam::Rotate( Float_t alpha )
{
  //* Rotate the coordinate system in XY on the angle alpha

  Bool_t ret = 1;

  Float_t cA = TMath::Cos( alpha );
  Float_t sA = TMath::Sin( alpha );
  Float_t x = X(), y= Y(), sP= SinPhi(), cP= CosPhi();

  X()      =   x*cA +  y*sA;
  Y()      =  -x*sA +  y*cA;
  CosPhi() =  cP*cA + sP*sA;
  SinPhi() = -cP*sA + sP*cA;

  Float_t j0 = 0, j2 = 0;
    
  if( TMath::Abs(CosPhi())>1.e-4 ) j0 = cP/CosPhi(); else ret = 0;
  if( TMath::Abs(cP)>1.e-4 ) j2 = CosPhi()/cP; else ret = 0;

  //Float_t J[5][5] = { { j0, 0, 0,  0,  0 }, // Y
  //                      {  0, 1, 0,  0,  0 }, // Z
  //                      {  0, 0, j2, 0,  0 }, // SinPhi
  //	                  {  0, 0, 0,  1,  0 }, // DzDs
  //	                  {  0, 0, 0,  0,  1 } }; // Kappa

  fC[0]*= j0*j0;
  fC[1]*= j0;
  //fC[3]*= j0;
  fC[6]*= j0;
  fC[10]*= j0;

  //fC[3]*= j2;
  fC[4]*= j2;
  fC[5]*= j2*j2; 
  fC[8]*= j2;
  fC[12]*= j2;
  return ret;
}


void AliHLTTPCCATrackParam::GetExtParam( AliExternalTrackParam &T, Double_t alpha, Double_t Bz ) const
{
  //* Convert to AliExternalTrackParam parameterisation, 
  //* the angle alpha is the global angle of the local X axis 

  Double_t par[5], cov[15];
  for( Int_t i=0; i<5; i++ ) par[i] = fP[i];
  for( Int_t i=0; i<15; i++ ) cov[i] = fC[i];

  if(par[2]>.999 ) par[2]=.999;
  if(par[2]<-.999 ) par[2]=-.999;

  const Double_t kCLight = 0.000299792458;  
  Double_t c = (TMath::Abs(Bz)>1.e-4) ?1./(Bz*kCLight) :1./(5.*kCLight);
  { // kappa => 1/pt
    par[4] *= c;
    cov[10]*= c;
    cov[11]*= c;
    cov[12]*= c;
    cov[13]*= c;
    cov[14]*= c*c;
  }
  if( GetCosPhi()<0 ){ // change direction
    par[2] = -par[2]; // sin phi
    par[3] = -par[3]; // DzDs
    par[4] = -par[4]; // kappa
    cov[3] = -cov[3];
    cov[4] = -cov[4];
    cov[6] = -cov[6];
    cov[7] = -cov[7];
    cov[10] = -cov[10];
    cov[11] = -cov[11];
  }
  T.Set(GetX(),alpha,par,cov);
}

void AliHLTTPCCATrackParam::SetExtParam( const AliExternalTrackParam &T, Double_t Bz )
{
  //* Convert from AliExternalTrackParam parameterisation
  
  for( Int_t i=0; i<5; i++ ) fP[i] = T.GetParameter()[i];
  for( Int_t i=0; i<15; i++ ) fC[i] = T.GetCovariance()[i];
  X() = T.GetX();
  if(SinPhi()>.999 ) SinPhi()=.999;
  if(SinPhi()<-.999 ) SinPhi()=-.999;
  CosPhi() = TMath::Sqrt(1.-SinPhi()*SinPhi());
  const Double_t kCLight = 0.000299792458;  
  Double_t c = Bz*kCLight;
  { // 1/pt -> kappa 
    fP[4] *= c;
    fC[10]*= c;
    fC[11]*= c;
    fC[12]*= c;
    fC[13]*= c;
    fC[14]*= c*c;
  }
}


void AliHLTTPCCATrackParam::Filter( Float_t y, Float_t z, Float_t erry, Float_t errz )
{
  //* Add the y,z measurement with the Kalman filter 

  Float_t 
    c00 = fC[ 0],
    c10 = fC[ 1], c11 = fC[ 2],
    c20 = fC[ 3], c21 = fC[ 4],
    c30 = fC[ 6], c31 = fC[ 7],
    c40 = fC[10], c41 = fC[11];
  
  Float_t
    z0 = y-fP[0],
    z1 = z-fP[1];

  Float_t v[3] = {erry*erry, 0, errz*errz};

  Float_t mS[3] = { c00+v[0], c10+v[1], c11+v[2] };

  Float_t mSi[3];
  Float_t det = (mS[0]*mS[2] - mS[1]*mS[1]);

  if( TMath::Abs(det)<1.e-8 ) return;
  det = 1./det;
  mSi[0] = mS[2]*det;
  mSi[1] = -mS[1]*det;
  mSi[2] = mS[0]*det;
 
  fNDF  += 2;
  fChi2 += ( +(mSi[0]*z0 + mSi[1]*z1 )*z0
	     +(mSi[1]*z0 + mSi[2]*z1 )*z1 );

  // K = CHtS
  
  Float_t k00, k01 , k10, k11, k20, k21, k30, k31, k40, k41;
    
  k00 = c00*mSi[0] + c10*mSi[1]; k01 = c00*mSi[1] + c10*mSi[2];
  k10 = c10*mSi[0] + c11*mSi[1]; k11 = c10*mSi[1] + c11*mSi[2];
  k20 = c20*mSi[0] + c21*mSi[1]; k21 = c20*mSi[1] + c21*mSi[2];
  k30 = c30*mSi[0] + c31*mSi[1]; k31 = c30*mSi[1] + c31*mSi[2] ;
  k40 = c40*mSi[0] + c41*mSi[1]; k41 = c40*mSi[1] + c41*mSi[2] ;

  Float_t sinPhi = fP[2] + k20*z0  + k21*z1  ;
  if( TMath::Abs(sinPhi)>=0.99 ) return;

  fP[ 0]+= k00*z0  + k01*z1 ;
  fP[ 1]+= k10*z0  + k11*z1  ;
  fP[ 2]+= k20*z0  + k21*z1  ;
  fP[ 3]+= k30*z0  + k31*z1  ;
  fP[ 4]+= k40*z0  + k41*z1  ;

    
  fC[ 0]-= k00*c00 + k01*c10 ;
  
  fC[ 1]-= k10*c00 + k11*c10 ;
  fC[ 2]-= k10*c10 + k11*c11 ;

  fC[ 3]-= k20*c00 + k21*c10 ;
  fC[ 4]-= k20*c10 + k21*c11 ;
  fC[ 5]-= k20*c20 + k21*c21 ;

  fC[ 6]-= k30*c00 + k31*c10 ;
  fC[ 7]-= k30*c10 + k31*c11 ;
  fC[ 8]-= k30*c20 + k31*c21 ;
  fC[ 9]-= k30*c30 + k31*c31 ;

  fC[10]-= k40*c00 + k41*c10 ;
  fC[11]-= k40*c10 + k41*c11 ;
  fC[12]-= k40*c20 + k41*c21 ;
  fC[13]-= k40*c30 + k41*c31 ;
  fC[14]-= k40*c40 + k41*c41 ;
    
  if( CosPhi()>=0 ){
    CosPhi() = TMath::Sqrt(1-SinPhi()*SinPhi());
  }else{
    CosPhi() = -TMath::Sqrt(1-SinPhi()*SinPhi());
  }   
    
}
