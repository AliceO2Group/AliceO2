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

#define GMPropagatorUseFullField

#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCCAParam.h"
#include <cmath>


#if defined(GMPropagatorUseFullField) & !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
#include "AliTracker.h"
#include "AliMagF.h"
#endif


GPUd() inline void  AliHLTTPCGMPropagator::GetBxByBz( float Alpha, float X, float Y, float Z, float B[3] ) const
{

  if( fContinuousTracking ) Z =  ( Z > 0 ? 125. : -125.);

  // get global coordinates

  float cs = AliHLTTPCCAMath::Cos(Alpha);
  float sn = AliHLTTPCCAMath::Sin(Alpha);

#if defined(GMPropagatorUseFullField) & !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
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
}

GPUd()  float  AliHLTTPCGMPropagator::GetBz( float Alpha, float X, float Y, float Z ) const
{

  if( fContinuousTracking ) Z =  ( Z > 0 ? 125. : -125.);

  // get global coordinates

  float cs = AliHLTTPCCAMath::Cos(Alpha);
  float sn = AliHLTTPCCAMath::Sin(Alpha);

#if defined(GMPropagatorUseFullField) & !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
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
    t0.UpdateValues(); // if px1<0, t0 direction is changed here: t0.GetPx()==-px1, etc.
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
  
  if( px1 <0 ){ // change direction ( t0 direction is already changed in t0.UpdateValues(); )
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


/*
// OLD
GPUd() int AliHLTTPCGMPropagator::RotateToAlpha( float newAlpha )
{
  //
  // Rotate the track coordinate system in XY to the angle newAlpha
  // return value is error (0==no error)
  //
  
  float cc = CAMath::Cos( newAlpha - fAlpha );
  float ss = CAMath::Sin( newAlpha - fAlpha );
  float x0 = fT0.X();
  float y0 = fT0.Y();
  float px = fT0.Px();
  float py = fT0.Py();    
  float pxe  =  px*cc + py*ss;
  float pye  = -px*ss + py*cc;

  if ( CAMath::Abs( pye ) > fMaxSinPhi*fT0.GetPt() || CAMath::Abs( px ) < 1.e-2 || CAMath::Abs( pxe ) < 1.e-2  ) return -1;

  //
  // after rotation the track has to be moved to X=fT0.X()
  //  
  // Jacobian = { { j0, 0, 0,  0,  0 }, // Y
  //              {  0, 1, 0,  0,  0 }, // Z
  //              {  0, 0, j2, 0,  0 }, // SinPhi
  //              {  0, 0, 0,  1,  0 }, // DzDs
  //              {  0, 0, 0,  0,  1 } }; // Kappa

  float j0 = px / pxe;
  float j2 = pxe / px;
  float d0 = fT->Y() - y0;
  float d2 = fT->SinPhi() - fT0.SinPhi();

  // rotate fT0 track
  {
    fT0.X()  =  x0*cc + y0*ss;
    fT0.Y()  = -x0*ss + y0*cc;
    fT0.Px() =  pxe;
    fT0.Py() =  pye;
    fT0.UpdateValues();
  }
  
  fT->X() = fT0.X();
  fT->Y() = fT0.Y() + j0*d0;
  fT->SinPhi() = fT0.SinPhi() + j2*d2;

  float *c = fT->Cov();
  c[ 0] *= j0 * j0;
  c[ 1] *= j0;
  c[ 3] *= j0;
  c[ 6] *= j0;
  c[10] *= j0;

  c[ 3] *= j2;
  c[ 4] *= j2;
  c[ 5] *= j2 * j2;
  c[ 8] *= j2;
  c[12] *= j2;

  fAlpha = newAlpha;
  
  if( pxe <0 ){ // change direction ( fT0 direction is already changed in fT0.UpdateValues(); )
    fT->SinPhi() = -fT->SinPhi();
    fT->DzDs()   = -fT->DzDs();
    fT->QPt()    = -fT->QPt();
    c[3] = -c[3];
    c[4] = -c[4];
    c[6] = -c[6];
    c[7] = -c[7];
    c[10] = -c[10];
    c[11] = -c[11];
  }
  
  return 0;
}
*/

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
  if (t0e.PropagateToXBxByBz( posX, B[0], B[1], B[2], dLp )) return(1);
  if( fabs( t0e.SinPhi() ) >= fMaxSinPhi ) return -3;

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS =  dLp*t0e.Pt();
  float dL =  fabs(dLp*t0e.P());   

  if( inFlyDirection ) dL = -dL;
  
  float bz = B[2];             
  float k  = -fT0.QPt()*bz;
  float dx = posX - fT0.X();
  float kdx = k*dx; 
  float dxcci = dx / (fT0.CosPhi() + t0e.CosPhi());            
      
  float hh = dxcci*t0e.SecPhi()*(2.f+0.5f*kdx*kdx); 
  float h02 = fT0.SecPhi()*hh;
  float h04 = -bz*dxcci*hh;
  float h13 = dS;  
  float h24 = -dx*bz;

  float *p = fT->Par();

  float d0 = p[0] - fT0.Y();
  float d1 = p[1] - fT0.Z();
  float d2 = p[2] - fT0.SinPhi();
  float d3 = p[3] - fT0.DzDs();
  float d4 = p[4] - fT0.QPt();
  
  float newSinPhi = t0e.SinPhi() +  d2           + h24*d4;;
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
  
  c[0]+= h02*h02c22 + h04*h04c44 + float(2.f)*( h02*c20ph04c42  + h04*c40 );
  
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
  float corr = float(1.f) - fMaterial.fEP2* dLmask ;

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
  
  return 0;
}

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
  
  c[0]+= h02*h02c22 + h04*h04c44 + float(2.f)*( h02*c20ph04c42  + h04*c40 );
  
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
  float corr = float(1.f) - fMaterial.fEP2* dLmask ;

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
  
  return 0;
}

GPUd() void AliHLTTPCGMPropagator::GetErr2(float& err2Y, float& err2Z, const AliHLTTPCCAParam &param, float posZ, int iRow)
{
  param.GetClusterErrors2( iRow,  fContinuousTracking ? 125.:posZ, fT0.GetSinPhi(),fT0.GetCosPhi(),fT0.DzDs(), err2Y, err2Z );  
}

GPUd() int AliHLTTPCGMPropagator::Update( float posY, float posZ, int iRow, const AliHLTTPCCAParam &param, bool rejectChi2 )
{
  float *fC = fT->Cov();
  float *fP = fT->Par();
  
  float 
    c00 = fC[ 0],
    c11 = fC[ 2],
    c20 = fC[ 3],
    c31 = fC[ 7],
    c40 = fC[10];

  float err2Y, err2Z;
  GetErr2(err2Y, err2Z, param, posZ, iRow);
  
  if ( fT->NDF()==-5 ) { // first measurement: no need to filter, as the result is known in advance. just set it. 
    fT->ResetCovariance();
    fP[ 0] = posY;
    fP[ 1] = posZ;
    fC[ 0] = err2Y;
    fC[ 2] = err2Z;
    fT->NDF() = -3;   
    return 0;
  }  
    
  // Filter block
    
  float mS0 = 1./(err2Y + c00);    

  float  z0 = posY - fP[0];
  float  z1 = posZ - fP[1];
  float mS2 = 1./(err2Z + c11);
  
  //printf("hits %d chi2 %f, new %f %f (dy %f dz %f)\n", N, fChi2, mS0 * z0 * z0, mS2 * z1 * z1, z0, z1);
  //float tmpCut = param.HighQPtForward() < fabs(fT0.GetQPt()) ? 25 : 25; // change to fT0
  //if (rejectChi2 && (mS0*z0*z0 > tmpCut || mS2*z1*z1 > tmpCut)) return 2;
  fT->Chi2()  += mS0*z0*z0 + mS2*z1*z1;
  //SG!!! if( fabs( fP[2] + z0*c20*mS0  ) > fMaxSinPhi ) return 1;
    
    
  // K = CHtS
     
  float k00, k11, k20, k31, k40;
  
  k00 = c00 * mS0;
  k20 = c20 * mS0;
  k40 = c40 * mS0;
  
  
  k11 = c11 * mS2;
  k31 = c31 * mS2;
  
  fT->NDF()  += 2;
  
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

  const float log0 = log( float(5940.f));
  const float log1 = log( float(3.5f*5940.f) );

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
  if( fUseMeanMomentum ) qpt = 1./0.35;
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
  fMaterial.fSigmadE2 = fMaterial.fSigmadE2*betheRho + fMaterial.fK44;
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
  fT0.Px() = fT0.Px();
  fT0.Py() = -fT0.Py();
  fT0.Pz() = -fT0.Pz();
  fT0.Q()  = -fT0.Q();
  fT0.UpdateValues();

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
  float corr = float(1.f) - fMaterial.fEP2* dLmask ;

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
   
}
