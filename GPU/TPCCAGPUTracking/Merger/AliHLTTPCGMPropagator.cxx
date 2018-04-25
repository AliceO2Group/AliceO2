// $Id: AliHLTTPCGMPropagator.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
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



#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCGMMergedTrackHit.h"
#include <cmath>


#if defined(GMPropagatorUseFullField)
#include "AliTracker.h"
#include "AliMagF.h"
#endif

GPUd() void  AliHLTTPCGMPropagator::GetBxByBz( float Alpha, float X, float Y, float Z, float B[3] ) const
{
  // get global coordinates

  float cs = AliHLTTPCCAMath::Cos(Alpha);
  float sn = AliHLTTPCCAMath::Sin(Alpha);

#if defined(GMPropagatorUseFullField)
  const double kCLight = 0.000299792458;
  double r[3] = { X*cs - Y*sn, X*sn + Y*cs, Z };
  double bb[3];
  AliTracker::GetBxByBz( r, bb);
  bb[0]*=kCLight;
  bb[1]*=kCLight;
  bb[2]*=kCLight;
  /*
  cout<<"AliTracker::GetBz()= "<<AliTracker::GetBz()<<endl;
  cout<<"AliTracker::UniformField() "<<AliTracker::UniformField()<<endl;
  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  cout<<"Fast field = "<<(void*) fld->GetFastField()<<endl;
  AliMagF::BMap_t  type = fld->GetMapType() ;
  cout<<"Field type: "<<type<<endl;
  //  fMapType==k2BMap_t     
  */
#else
  float  bb[3];
  fField->GetField( X*cs - Y*sn, X*sn + Y*cs, Z, bb);
#endif

  // rotate field to local coordinates

  B[0] =  bb[0]*cs + bb[1]*sn ;
  B[1] = -bb[0]*sn + bb[1]*cs ;
  B[2] =  bb[2] ;
  /*if( fToyMCEvents ){ // special treatment for toy monte carlo
    B[0] = 0;
    B[1] = 0;
    B[2] = fField->GetNominalBz();
}*/
}

GPUd()  float  AliHLTTPCGMPropagator::GetBz( float Alpha, float X, float Y, float Z ) const
{
  if( fToyMCEvents ){ // special treatment for toy monte carlo
    float B[3];
    GetBxByBz(Alpha,X,Y,Z,B);
    return B[2];
  }

  // get global coordinates

  float cs = AliHLTTPCCAMath::Cos(Alpha);
  float sn = AliHLTTPCCAMath::Sin(Alpha);  

#if defined(GMPropagatorUseFullField)
  const double kCLight = 0.000299792458;
  double r[3] = { X*cs - Y*sn, X*sn + Y*cs, Z };
  double bb[3];
  AliTracker::GetBxByBz( r, bb);
  return bb[2] * kCLight;
#else
  return fField->GetFieldBz( X*cs - Y*sn, X*sn + Y*cs, Z);
#endif
}


GPUd() int AliHLTTPCGMPropagator::RotateToAlpha( float newAlpha )
{
  //
  // Rotate the track coordinate system in XY to the angle newAlpha
  // return value is error code (0==no error)
  //

  float cc = CAMath::Cos( newAlpha - fAlpha );
  float ss = CAMath::Sin( newAlpha - fAlpha );

  AliHLTTPCGMPhysicalTrackModel t0 = fT0; 
  
  float x0 = fT0.X();
  float y0 = fT0.Y();
  float px0 = fT0.Px();
  float py0 = fT0.Py();
  //float pt0 = fT0.GetPt();

  if ( CAMath::Abs( fT->GetSinPhi() ) >= fMaxSinPhi  || CAMath::Abs( px0 ) < 1.e-2 ) return -1;
 
  // rotate t0 track
  float px1  =  px0*cc + py0*ss;
  float py1  = -px0*ss + py0*cc;
  
  {
    t0.X()  =  x0*cc + y0*ss;
    t0.Y()  = -x0*ss + y0*cc;
    t0.Px() =  px1;
    t0.Py() =  py1;
    t0.UpdateValues();
  }

  if ( CAMath::Abs( py1 ) > fMaxSinPhi*fT0.GetPt() || CAMath::Abs( px1 ) < 1.e-2  ) return -1;

  // calculate X of rotated track:
  float trackX = x0*cc + ss*fT->Y();
  
  // transport t0 to trackX
  float B[3];
  GetBxByBz( newAlpha, t0.X(), t0.Y(), t0.Z(), B );
  float dLp = 0;
  int err = t0.PropagateToXBxByBz( trackX, B[0], B[1], B[2], dLp );
  if( err ) return -1;
  
  if( fabs( t0.SinPhi() ) >= fMaxSinPhi ) return -1;

  // now t0 is rotated and propagated, all checks are passed

  
  // Rotate track using fT0 for linearisation. After rotation X is not fixed, but has a covariance

  
  //                    Y  Z Sin DzDs q/p
  // Jacobian J0 = { { j0, 0, 0,  0,  0 }, // Y
  //                 {  0, 1, 0,  0,  0 }, // Z
  //                 {  0, 0, j1, 0,  0 }, // SinPhi
  //                 {  0, 0, 0,  1,  0 }, // DzDs
  //                 {  0, 0, 0,  0,  1 }, // q/p
  //                 { j2, 0, 0,  0,  0 } }// X (rotated )
   
  float j0 = cc;
  float j1 = px1 / px0;
  float j2 = ss;
  //float dy = fT->Y() - y0;
  //float ds = fT->SinPhi() - fT0.SinPhi();
  
  fT->X() =  trackX; // == x0*cc + ss*fT->Y()  == t0.X() + j0*dy;
  fT->Y() = -x0*ss  + cc*fT->Y(); //== t0.Y() + j0*dy;
  //fT->SinPhi() = py1/pt0 + j1*ds; // == t0.SinPhi() + j1*ds; // use py1, since t0.SinPhi can have different sign
  fT->SinPhi() = -sqrt(1.f-fT->SinPhi()*fT->SinPhi())*ss + fT->SinPhi()*cc;
    
  // Rotate cov. matrix Cr = J0 x C x J0T. Cr has one more row+column for X:
  float *c = fT->Cov();
  
  float c15 = c[ 0]*j0*j2;  
  float c16 = c[ 1]*j2; 
  float c17 = c[ 3]*j1*j2;
  float c18 = c[ 6]*j2;
  float c19 = c[10]*j2;
  float c20 = c[ 0]*j2*j2;
  
  c[ 0] *= j0 * j0;
  c[ 3] *= j0;
  c[10] *= j0;
  
  c[ 3] *= j1;
  c[ 5] *= j1 * j1;
  c[12] *= j1;

  if( !fFitInProjections ){
    c[ 1] *= j0;
    c[ 6] *= j0;
    c[ 4] *= j1;
    c[ 8] *= j1;
  }
  
  if( t0.SetDirectionAlongX() ){ // change direction if Px < 0
    fT->SinPhi() = -fT->SinPhi();
    fT->DzDs()   = -fT->DzDs();
    fT->QPt()    = -fT->QPt();    
    c[3] = -c[3]; // covariances with SinPhi
    c[4] = -c[4];
    c17 = -c17;
    c[6] = -c[6];// covariances with DzDs
    c[7] = -c[7];
    c18 = -c18;
    c[10] = -c[10];// covariances with QPt
    c[11] = -c[11];
    c19 = -c19;
  } 
  
  // Now fix the X coordinate: so to say, transport track T to fixed X = fT->X().
  // only covariance changes. Use rotated and transported t0 for linearisation
  float j3 = -t0.Py()/t0.Px();
  float j4 = -t0.Pz()/t0.Px();
  float j5 =  t0.QPt()*B[2];

  //                    Y  Z Sin DzDs q/p  X
  // Jacobian J1 = { {  1, 0, 0,  0,  0,  j3 }, // Y 
  //                 {  0, 1, 0,  0,  0,  j4 }, // Z
  //                 {  0, 0, 1,  0,  0,  j5 }, // SinPhi
  //                 {  0, 0, 0,  1,  0,   0 }, // DzDs
  //                 {  0, 0, 0,  0,  1,   0 } }; // q/p    

  float h15 = c15 + c20*j3;
  float h16 = c16 + c20*j4;
  float h17 = c17 + c20*j5;
  
  c[ 0] += j3*(c15 + h15);
  
  c[ 2] += j4*(c16 + h16);

  c[ 3] += c17*j3 + h15*j5;
  c[ 5] += j5*(c17 + h17);

  c[ 7] += c18*j4;
  // c[ 9] = c[ 9];

  c[10] += c19*j3;
  c[12] += c19*j5;
  // c[14] = c[14];

  if( !fFitInProjections ){
    c[ 1] += c16*j3 + h15*j4;
    c[ 4] += c17*j4 + h16*j5;
    c[ 6] += c18*j3;
    c[ 8] += c18*j5;
    c[11] += c19*j4;
    //c[13] = c[13];
  }
  
  fAlpha = newAlpha;
  fT0 = t0;
  
  return 0;
}

GPUd() int AliHLTTPCGMPropagator::PropagateToXAlpha(float posX, float posAlpha, bool inFlyDirection)
{
  
  if ( fabs( posAlpha - fAlpha) > 1.e-4 ) {
    if( RotateToAlpha( posAlpha )!=0 ) return -2;
  }

  float B[3];
  GetBxByBz( fAlpha, fT0.X(), fT0.Y(), fT0.Z(), B );
 
  // propagate fT0 to t0e
  
  AliHLTTPCGMPhysicalTrackModel t0e(fT0);
  float dLp = 0;
  if (t0e.PropagateToXBxByBz( posX, B[0], B[1], B[2], dLp ) && t0e.PropagateToXBzLight( posX, B[2], dLp )) return 1;

  if( fabs( t0e.SinPhi() ) >= fMaxSinPhi ) return -3;

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS =  dLp*t0e.Pt();
  float dL =  fabs(dLp*t0e.P());

  if( inFlyDirection ) dL = -dL;

  float ey = fT0.SinPhi();
  float ex = fT0.CosPhi();
  float exi = fT0.SecPhi();
  float ey1 = t0e.SinPhi();
  float ex1 = t0e.CosPhi();
  float ex1i= t0e.SecPhi();
  
  float bz = B[2];
  float k  = -fT0.QPt()*bz;
  float dx = posX - fT0.X();
  float kdx = k*dx;
  float cc = ex + ex1;
  float cci = 1.f/cc;

  float dxcci = dx * cci;
  float hh = dxcci*ex1i*(1.f + ex*ex1 + ey*ey1 );
  //float hh = dxcci*ex1i*(2.f+0.5f*kdx*kdx);  //DR: Before was like this!

  float j02 = exi*hh;
  float j04 = -bz*dxcci*hh;
  float j13 = dS;
  float j24 = -dx*bz;

  float *p = fT->Par();

  float d0 = p[0] - fT0.Y();
  float d1 = p[1] - fT0.Z();
  float d2 = p[2] - fT0.SinPhi();
  float d3 = p[3] - fT0.DzDs();
  float d4 = p[4] - fT0.QPt();
  
  float newSinPhi = t0e.SinPhi() + d2 + j24*d4;
  if (fT->NDF() >= 15 && fabs(newSinPhi) > HLTCA_MAX_SIN_PHI) return(-4);

  fT0 = t0e;
  fT->X() = t0e.X();
  p[0] = t0e.Y() + d0    + j02*d2         + j04*d4;
  p[1] = t0e.Z() + d1    + j13*d3;
  p[2] = newSinPhi;
  p[3] = t0e.DzDs() + d3;
  p[4] = t0e.QPt() + d4;

  float *c = fT->Cov();

  float c20 = c[ 3];
  float c21 = c[ 4];
  float c22 = c[ 5];

  float c30 = c[ 6];
  float c31 = c[ 7];
  float c32 = c[ 8];
  float c33 = c[ 9];

  float c40 = c[10];
  float c41 = c[11];
  float c42 = c[12];
  float c43 = c[13];
  float c44 = c[14];
  
  if (fFitInProjections)
  {
    float c20ph04c42 =  c20 + j04*c42;
    float j02c22 = j02*c22;
    float j04c44 = j04*c44;
    
    float n6 = c30 + j02*c32 + j04*c43;
    float n7 = c31 + j13*c33;
    float n10 = c40 + j02*c42 + j04c44;
    float n11 = c41 + j13*c43;
    float n12 = c42 + j24*c44;
        
    c[0]+= j02*j02c22 + j04*j04c44 + 2.f*( j02*c20ph04c42  + j04*c40 );
    c[1]+= j02*c21 + j04*c41 + j13*n6;
    c[2]+= j13*(c31 + n7);
    c[3] = c20ph04c42 + j02c22  + j24*n10;
    c[4] = c21 + j13*c32 + j24*n11;
    c[5] = c22 + j24*( c42 + n12 );
    c[6] = n6;
    c[7] = n7; 
    c[8] = c32 + c43*j24;
    c[10] = n10;
    c[11] = n11;
    c[12] = n12;
  }
  else
  {
    float c00 = c[ 0];
    float c10 = c[ 1];
    float c11 = c[ 2];

    float ss = ey + ey1;
    float tg = ss*cci;
    float xx = 1.f - 0.25f*kdx*kdx*( 1.f + tg*tg );
    if( xx<1.e-8 ) return -1;
    xx = CAMath::Sqrt(xx);
    float yy = CAMath::Sqrt(ss*ss+cc*cc);

    float j12 = dx*fT0.DzDs()*tg*(2.f+tg*(ey*exi+ey1*ex1i))/(xx*yy);
    float j14 = 0;
    if( CAMath::Abs(fT0.QPt())>1.e-6 ){
      j14 = (2.f*xx*ex1i*dx/yy-dS)*fT0.DzDs()/fT0.QPt();    
    } else {
      j14 = -fT0.DzDs()*bz*dx*dx*exi*exi*exi
        *( 0.5*ey + (1.f/3.f)*kdx*(1+2.f*ey*ey)*exi*exi
      );
    }
    
    p[1] += j12*d2 + j14*d4;

    float h00 = c00 + c20*j02 + c40*j04;
    //float h01 = c10 + c21*j02 + c41*j04;
    float h02 = c20 + c22*j02 + c42*j04;
    //float h03 = c30 + c32*j02 + c43*j04;
    float h04 = c40 + c42*j02 + c44*j04;

    float h10 = c10 + c20*j12 + c30*j13 + c40*j14;
    float h11 = c11 + c21*j12 + c31*j13 + c41*j14;
    float h12 = c21 + c22*j12 + c32*j13 + c42*j14;
    float h13 = c31 + c32*j12 + c33*j13 + c43*j14;
    float h14 = c41 + c42*j12 + c43*j13 + c44*j14;

    float h20 = c20 + c40*j24;
    float h21 = c21 + c41*j24;
    float h22 = c22 + c42*j24;
    float h23 = c32 + c43*j24;
    float h24 = c42 + c44*j24;
        
    c[ 0] = h00 + h02*j02 + h04*j04;

    c[ 1] = h10 + h12*j02 + h14*j04;
    c[ 2] = h11 + h12*j12 + h13*j13 + h14*j14;

    c[ 3] = h20 + h22*j02 + h24*j04;
    c[ 4] = h21 + h22*j12 + h23*j13 + h24*j14;
    c[ 5] = h22 + h24*j24;

    c[ 6] = c30 + c32*j02 + c43*j04;
    c[ 7] = c31 + c32*j12 + c33*j13 + c43*j14;
    c[ 8] = c32 + c43*j24;
    //c[ 9] = c33;

    c[10] = c40 + c42*j02 + c44*j04;
    c[11] = c41 + c42*j12 + c43*j13 + c44*j14;
    c[12] = c42 + c44*j24;
    //c[13] = c43;
    //c[14] = c44;
  }

  float &fC22 = c[5];
  float &fC33 = c[9];
  float &fC40 = c[10];
  float &fC41 = c[11];
  float &fC42 = c[12];
  float &fC43 = c[13];
  float &fC44 = c[14];

  float dLmask = 0.f;
  bool maskMS = ( fabs( dL ) < fMaterial.fDLMax );
  if( maskMS ) dLmask = dL;
  float dLabs = fabs( dLmask); 

  // Energy Loss

  if( 1||!fToyMCEvents ){  
    //std::cout<<"APPLY ENERGY LOSS!!!"<<std::endl;
    float corr = 1.f - fMaterial.fEP2* dLmask ;
    float corrInv = 1.f/corr;
    fT0.Px()*=corrInv;
    fT0.Py()*=corrInv;
    fT0.Pz()*=corrInv;
    fT0.Pt()*=corrInv;
    fT0.P()*=corrInv;
    fT0.QPt()*=corr;

    p[4]*= corr;
    
    fC40 *= corr;
    fC41 *= corr;
    fC42 *= corr;
    fC43 *= corr;
    fC44  = fC44*corr*corr + dLabs*fMaterial.fSigmadE2;
  } else {
    // std::cout<<"DONT APPLY ENERGY LOSS!!!"<<std::endl;
  }

  //  Multiple Scattering
  
  if( !fToyMCEvents ){ 
    fC22 += dLabs * fMaterial.fK22 * fT0.CosPhi()*fT0.CosPhi();
    fC33 += dLabs * fMaterial.fK33;
    fC43 += dLabs * fMaterial.fK43;
    fC44 += dLabs * fMaterial.fK44;
  }

  return 0;
}

GPUd() int AliHLTTPCGMPropagator::GetPropagatedYZ(float x, float& projY, float& projZ)
{
  float bz = GetBz(fAlpha, fT->X(), fT->Y(), fT->Z());
  float k  = fT0.QPt() * bz;
  float dx = x - fT->X();
  float kdx = k * dx;
  float ex = fT0.CosPhi();
  float ey = fT0.SinPhi();
  float ey1 = kdx + ey;
  if(fabs(ey1) > HLTCA_MAX_SIN_PHI) return 1;
  float ss = ey + ey1;
  float ex1 = sqrt(1.f - ey1 * ey1);
  float cc = ex + ex1;
  float dxcci = dx / cc;
  float dy = dxcci * ss;
  float norm2 = 1.f + ey * ey1 + ex * ex1;
  float dl = dxcci * sqrt(norm2 + norm2);
  float dS;
  {
    float dSin = 0.5f * k*dl;
    float a = dSin * dSin;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    dS = dl + dl * a * (k2 + a * (k4));
  }
  float dz = dS * fT0.DzDs();
  projY = fT->Y() + dy;
  projZ = fT->Z() + dz;
  return 0;
}

/*
GPUd() int AliHLTTPCGMPropagator::PropagateToXAlphaBz(float posX, float posAlpha, bool inFlyDirection)
{
  
  if ( fabs( posAlpha - fAlpha) > 1.e-4 ) {
    if( RotateToAlpha( posAlpha )!=0 ) return -2;
  }

  float Bz = GetBz( fAlpha, fT0.X(), fT0.Y(), fT0.Z() );

  // propagate fT0 to t0e
  
  AliHLTTPCGMPhysicalTrackModel t0e(fT0);
  float dLp = 0;
  if (t0e.PropagateToXBzLight( posX, Bz, dLp )) return 1;
  t0e.UpdateValues();
  if( fabs( t0e.SinPhi() ) >= fMaxSinPhi ) return -3;

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS =  dLp*t0e.Pt();
  float dL =  fabs(dLp*t0e.P());   

  if( inFlyDirection ) dL = -dL;
  
  float k  = -fT0.QPt()*Bz;
  float dx = posX - fT0.X();
  float kdx = k*dx; 
  float dxcci = dx / (fT0.CosPhi() + t0e.CosPhi());            
      
  float hh = dxcci*t0e.SecPhi()*(2.f+0.5f*kdx*kdx); 
  float h02 = fT0.SecPhi()*hh;
  float h04 = -Bz*dxcci*hh;
  float h13 = dS;  
  float h24 = -dx*Bz;

  float *p = fT->Par();

  float d0 = p[0] - fT0.Y();
  float d1 = p[1] - fT0.Z();
  float d2 = p[2] - fT0.SinPhi();
  float d3 = p[3] - fT0.DzDs();
  float d4 = p[4] - fT0.QPt();
	  
  float newSinPhi = t0e.SinPhi() +  d2           + h24*d4;
  if (fT->NDF() >= 15 && fabs(newSinPhi) > HLTCA_MAX_SIN_PHI) return(-4);

  fT0 = t0e;

  fT->X() = t0e.X();
  p[0] = t0e.Y() + d0    + h02*d2         + h04*d4;
  p[1] = t0e.Z() + d1    + h13*d3;
  p[2] = newSinPhi;
  p[3] = t0e.DzDs() + d3;
  p[4] = t0e.QPt() + d4;

  float *c = fT->Cov();
  float c20 = c[ 3];
  float c21 = c[ 4];
  float c22 = c[ 5];
  float c30 = c[ 6];
  float c31 = c[ 7];
  float c32 = c[ 8];
  float c33 = c[ 9];
  float c40 = c[10];
  float c41 = c[11];
  float c42 = c[12];
  float c43 = c[13];
  float c44 = c[14];
  
  float c20ph04c42 =  c20 + h04*c42;
  float h02c22 = h02*c22;
  float h04c44 = h04*c44;
  
  float n6 = c30 + h02*c32 + h04*c43;
  float n7 = c31 + h13*c33;
  float n10 = c40 + h02*c42 + h04c44;
  float n11 = c41 + h13*c43;
  float n12 = c42 + h24*c44;
      
  c[8] = c32 + h24*c43;
  
  c[0]+= h02*h02c22 + h04*h04c44 + 2.f*( h02*c20ph04c42  + h04*c40 );
  
  c[1]+= h02*c21 + h04*c41 + h13*n6;
  c[6] = n6;
  
  c[2]+= h13*(c31 + n7);
  c[7] = n7; 
  
  c[3] = c20ph04c42 + h02c22  + h24*n10;
  c[10] = n10;
  
  c[4] = c21 + h13*c32 + h24*n11;
  c[11] = n11;
      
  c[5] = c22 + h24*( c42 + n12 );
  c[12] = n12;

  // Energy Loss
  
  float &fC22 = c[5];
  float &fC33 = c[9];
  float &fC40 = c[10];
  float &fC41 = c[11];
  float &fC42 = c[12];
  float &fC43 = c[13];
  float &fC44 = c[14];

  float dLmask = 0.f;
  bool maskMS = ( fabs( dL ) < fMaterial.fDLMax );
  if( maskMS ) dLmask = dL;
  float dLabs = fabs( dLmask); 
  float corr = 1.f - fMaterial.fEP2* dLmask ;

  float corrInv = 1.f/corr;
  fT0.Px()*=corrInv;
  fT0.Py()*=corrInv;
  fT0.Pz()*=corrInv;
  fT0.Pt()*=corrInv;
  fT0.P()*=corrInv;
  fT0.QPt()*=corr;

  p[4]*= corr;
  
  fC40 *= corr;
  fC41 *= corr;
  fC42 *= corr;
  fC43 *= corr;
  fC44  = fC44*corr*corr + dLabs*fMaterial.fSigmadE2;
  
  //  Multiple Scattering
  
  fC22 += dLabs * fMaterial.fK22 * fT0.CosPhi()*fT0.CosPhi();
  fC33 += dLabs * fMaterial.fK33;
  fC43 += dLabs * fMaterial.fK43;
  fC44 += dLabs * fMaterial.fK44;
  
  return 0;
}
*/

GPUd() void AliHLTTPCGMPropagator::GetErr2(float& err2Y, float& err2Z, const AliHLTTPCCAParam &param, float posZ, int iRow, short clusterState)
{
  if (fSpecialErrors) param.GetClusterErrors2( iRow, posZ, fT0.GetSinPhi(), fT0.DzDs(), err2Y, err2Z );
  else param.GetClusterRMS2( iRow, posZ, fT0.GetSinPhi(), fT0.DzDs(), err2Y, err2Z );

  if (clusterState & AliHLTTPCGMMergedTrackHit::flagEdge) {err2Y += 0.35;err2Z += 0.15;}
  if (clusterState & AliHLTTPCGMMergedTrackHit::flagSingle) {err2Y += 0.2;err2Z += 0.2;}
  if (clusterState & (AliHLTTPCGMMergedTrackHit::flagSplitPad | AliHLTTPCGMMergedTrackHit::flagShared | AliHLTTPCGMMergedTrackHit::flagSingle)) {err2Y += 0.03;err2Y *= 3;}
  if (clusterState & (AliHLTTPCGMMergedTrackHit::flagSplitTime | AliHLTTPCGMMergedTrackHit::flagShared | AliHLTTPCGMMergedTrackHit::flagSingle)) {err2Z += 0.03;err2Z *= 3;}
  fStatErrors.GetOfflineStatisticalErrors(err2Y, err2Z, fT0.SinPhi(), fT0.DzDs(), clusterState);
}

GPUd() int AliHLTTPCGMPropagator::Update( float posY, float posZ, int iRow, const AliHLTTPCCAParam &param, short clusterState, bool rejectChi2, bool refit )
{
  float *fC = fT->Cov();
  float *fP = fT->Par();

  float err2Y, err2Z;
  GetErr2(err2Y, err2Z, param, posZ, iRow, clusterState);
  
  if ( fT->NDF()==-5 ) { // first measurement: no need to filter, as the result is known in advance. just set it. 
    fT->ResetCovariance();
    if (refit)
    {
        fC[14] = CAMath::Max(0.5f, fabs(fP[4]));
        fC[5] = CAMath::Max(0.2f, fabs(fP[2]) / 2);
        fC[9] = CAMath::Max(0.5f, fabs(fP[3]) / 2);
    }
    fP[ 0] = posY;
    fP[ 1] = posZ;
    fC[ 0] = err2Y;
    fC[ 2] = err2Z;
    fT->NDF() = -3;   
    return 0;
  }
        
  float d00= fC[ 0]; float d01= fC[ 1]; float d02= fC[ 3]; float d03= fC[ 6]; float d04= fC[10];
  float d10= fC[ 1]; float d11= fC[ 2]; float d12= fC[ 4]; float d13= fC[ 7]; float d14= fC[11];

  float z0 = posY - fP[0];
  float z1 = posZ - fP[1];

  float w0, w1, w2, chiY, chiZ;
  if (fFitInProjections)
  {
    w0 = 1./(err2Y + d00);
    w1 = 0;
    w2 = 1./(err2Z + d11);
    chiY = w0*z0*z0;
    chiZ = w2*z1*z1;
  }
  else
  {
    w0 = d11 + err2Z, w1 = d10, w2 = d00 + err2Y;
    { // Invert symmetric matrix
      float det = w0*w2 - w1*w1;
      if( CAMath::Abs(det)<1.e-10 ) return -1;
      det = 1./det;    
      w0 =  w0*det;
      w1 = -w1*det;
      w2 =  w2*det;
    }
    chiY = CAMath::Abs( (w0*z0 + w1*z1 ) * z0 );
    chiZ = CAMath::Abs( (w1*z0 + w2*z1 ) * z1 );
  }
  float dChi2 = chiY + chiZ;
  //printf("hits %d chi2 %f, new %f %f (dy %f dz %f)\n", N, fChi2, chiY, chiZ, z0, z1);
  if (fSpecialErrors && rejectChi2 && RejectCluster(chiY, chiZ, clusterState)) return 2; //DRTOTO get rid of stupid specialerror
 
  fT->Chi2() += dChi2;
  fT->NDF() += 2;

  if (fFitInProjections)
  {
    float k00 = d00 * w0;
    float k20 = d02 * w0;
    float k40 = d04 * w0;
    float k11 = d11 * w2;
    float k31 = d13 * w2;
    fP[0] += k00 * z0;
    fP[1] += k11 * z1;
    fP[2] += k20 * z0;
    fP[3] += k31 * z1;
    fP[4] += k40 * z0;

    fC[ 0] -= k00 * d00 ;
    fC[ 2] -= k11 * d11;
    fC[ 3] -= k20 * d00 ;
    fC[ 5] -= k20 * d02 ;
    fC[ 7] -= k31 * d11;
    fC[ 9] -= k31 * d13;
    fC[10] -= k00 * d04 ;
    fC[12] -= k40 * d02 ;
    fC[14] -= k40 * d04 ;
  }
  else
  {  
    float k00= d00*w0 + d01*w1;   float k01= d00*w1 + d10*w2;
    float k10= d01*w0 + d11*w1;   float k11= d01*w1 + d11*w2;  
    float k20= d02*w0 + d12*w1;   float k21= d02*w1 + d12*w2;
    float k30= d03*w0 + d13*w1;   float k31= d03*w1 + d13*w2; 
    float k40= d04*w0 + d14*w1;   float k41= d04*w1 + d14*w2;
    
    fP[0]+= k00*z0 + k01*z1;
    fP[1]+= k10*z0 + k11*z1;
    fP[2]+= k20*z0 + k21*z1;
    fP[3]+= k30*z0 + k31*z1;
    fP[4]+= k40*z0 + k41*z1;

    fC[0]-= k00*d00 + k01*d10;
    
    fC[2]-= k10*d01 + k11*d11;

    fC[3]-= k20*d00 + k21*d10;
    fC[5]-= k20*d02 + k21*d12;

    fC[7]-= k30*d01 + k31*d11;
    fC[9]-= k30*d03 + k31*d13;

    fC[10]-= k40*d00 + k41*d10;
    fC[12]-= k40*d02 + k41*d12;
    fC[14]-= k40*d04 + k41*d14;

    if( !fFitInProjections ){
      
      fC[1]-= k10*d00 + k11*d10;
      
      fC[4]-= k20*d01 + k21*d11;
      
      fC[6]-= k30*d00 + k31*d10;
      fC[8]-= k30*d02 + k31*d12;
      
      
      fC[11]-= k40*d01 + k41*d11;
      fC[13]-= k40*d03 + k41*d13;
    }
  }
  return 0;
}

//*
//*  Multiple scattering and energy losses
//*

GPUd() float AliHLTTPCGMPropagator::ApproximateBetheBloch( float beta2 )
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula with
  // the density effect taken into account at beta*gamma > 3.5
  // (the approximation is reasonable only for solid materials)
  //------------------------------------------------------------------

  const float log0 = log( 5940.f );
  const float log1 = log( 3.5f*5940.f );

  bool bad = (beta2 >= .999f)||( beta2 < 1.e-8f );

  if( bad ) beta2 = 0.5f;

  float a = beta2 / ( 1.f - beta2 ); 
  float b = 0.5*log(a);
  float d =  0.153e-3 / beta2;
  float c = b - beta2;

  float ret = d*(log0 + b + c );
  float case1 = d*(log1 + c );
  
  if( a > 3.5*3.5  ) ret = case1;
  if( bad ) ret = 0. ; 

  return ret;
}


GPUd() void AliHLTTPCGMPropagator::CalculateMaterialCorrection()
{
  //*!
  
  const float mass = 0.13957;
  
  float qpt = fT0.GetQPt();
  if (fabs(qpt) > 20) qpt = 20;

  float w2 = ( 1. + fT0.GetDzDs() * fT0.GetDzDs() );//==(P/pt)2
  float pti2 = qpt * qpt;
  if( pti2 < 1.e-4f ) pti2 = 1.e-4f;

  float mass2 = mass * mass;
  float beta2 = w2 / ( w2 + mass2 * pti2 );
  
  float p2 = w2 / pti2; // impuls 2
  float betheRho = ApproximateBetheBloch( p2 / mass2 )*fMaterial.fRho;
  float E = sqrt( p2 + mass2 );
  float theta2 = ( 14.1*14.1/1.e6 ) / ( beta2 * p2 )*fMaterial.fRhoOverRadLen;

  fMaterial.fEP2 = E / p2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.07; // To be tuned.
  fMaterial.fSigmadE2 = knst * fMaterial.fEP2 * qpt;
  fMaterial.fSigmadE2 = fMaterial.fSigmadE2 * fMaterial.fSigmadE2;
  
  fMaterial.fK22 = theta2*w2;
  fMaterial.fK33 = fMaterial.fK22 * w2;
  fMaterial.fK43 = 0.;
  fMaterial.fK44 = theta2* fT0.GetDzDs() * fT0.GetDzDs() * pti2;
  
  float  br = ( betheRho>1.e-8f ) ?betheRho :1.e-8f;
  fMaterial.fDLMax = 0.3* E / br ;
  fMaterial.fEP2*= betheRho;
  fMaterial.fSigmadE2 = fMaterial.fSigmadE2*betheRho;// + fMaterial.fK44;
}


GPUd() void AliHLTTPCGMPropagator::Mirror(bool inFlyDirection) 
{
  // mirror the track and the track approximation to the point which has the same X, but located on the other side of trajectory
  float B[3];
  GetBxByBz(  fAlpha, fT0.X(), fT0.Y(), fT0.Z(), B );
  float Bz = B[2];
  if( fabs(Bz)<1.e-8 ) Bz = 1.e-8;

  float dy = - 2.f*fT0.Q()*fT0.Px()/Bz;  
  float dS; // path in XY
  {
    float chord = dy; // chord to the extrapolated point == |dy|*sign(x direction)
    float sa = -fT0.CosPhi(); //  sin( half of the rotation angle ) ==  (chord/2) / radius
  
    // dS = (Pt/b)*2*arcsin( sa )
    //    = (Pt/b)*2*sa*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )
    //    =       chord*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )   
    
    float sa2 = sa*sa;
    const float k2 = 1./6.;
    const float k4 = 3./40.;
    //const float k6 = 5.f/112.f;
    dS =  chord + chord*sa2*(k2 + k4*sa2);
    //dS = sqrt(pt2)/b*2.*AliHLTTPCCAMath::ASin( sa );
  }

  if( fT0.SinPhi()<0.f ) dS = -dS;
    
  fT0.X() = fT0.X();
  fT0.Y() = fT0.Y() + dy;
  fT0.Z() = fT0.Z() + fT0.DzDs()*dS;
  fT0.Px() = fT0.Px(); // should be positive
  fT0.Py() = -fT0.Py();
  fT0.Pz() = -fT0.Pz();
  fT0.Q()  = -fT0.Q();
  fT0.UpdateValues();
  fT0.SetDirectionAlongX(); // not needed

  fT->X() = fT0.X();
  fT->Y() = fT->Y()+dy;
  fT->Z() = fT->Z() + fT0.DzDs()*dS;
  fT->SinPhi() = -fT->SinPhi();
  fT->DzDs()   = -fT->DzDs();
  fT->QPt()    = -fT->QPt();

  float *c = fT->Cov();
  
  c[3] = -c[3];
  c[4] = -c[4];
  c[6] = -c[6];
  c[7] = -c[7];
  c[10] = -c[10];
  c[11] = -c[11];

  // Energy Loss
  
  if( 1||!fToyMCEvents ){
 
    // std::cout<<"MIRROR: APPLY ENERGY LOSS!!!"<<std::endl;
  
    float dL =  fabs(dS*fT0.GetDlDs());
    
    if( inFlyDirection ) dL = -dL;
    
    float &fC40 = c[10];
    float &fC41 = c[11];
    float &fC42 = c[12];
    float &fC43 = c[13];
    float &fC44 = c[14];
    
    float dLmask = 0.f;
    bool maskMS = ( fabs( dL ) < fMaterial.fDLMax );
    if( maskMS ) dLmask = dL;
    float dLabs = fabs( dLmask); 
    float corr = 1.f - fMaterial.fEP2* dLmask ;
    
    float corrInv = 1.f/corr;
    fT0.Px()*=corrInv;
    fT0.Py()*=corrInv;
    fT0.Pz()*=corrInv;
    fT0.Pt()*=corrInv;
    fT0.P()*=corrInv;
    fT0.QPt()*=corr;

    fT->QPt()*= corr;
  
    fC40 *= corr;
    fC41 *= corr;
    fC42 *= corr;
    fC43 *= corr;
    fC44  = fC44*corr*corr + dLabs*fMaterial.fSigmadE2; 
  } else {
    // std::cout<<"MIRROR: DONT APPLY ENERGY LOSS!!!"<<std::endl;
  }

}
