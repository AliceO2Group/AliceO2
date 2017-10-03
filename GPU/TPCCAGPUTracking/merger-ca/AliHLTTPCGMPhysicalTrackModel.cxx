// $Id: AliHLTTPCGMPhysicalTrackModel.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
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

#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCCAMath.h"

inline GPUd() int AliHLTTPCGMPhysicalTrackModel::PropagateToXBzLight( float x,  float Bz, float &dLp )
{
  //
  // transport the track to X=x in magnetic field B = ( 0, 0, Bz[kG*0.000299792458] )
  // dLp is a return value == path length / track momentum [cm/(GeV/c)]
  // the method returns error code (0 == no error)
  //
  // Additional values are not recalculated, UpdateValues() has to be called afterwards!!
  //
  
  float b = fQ*Bz;
  float pt2 = fPx*fPx + fPy*fPy;  
  float dx = x - fX;
  float pye = fPy - dx*b; // extrapolated py
  float pxe2 = pt2 - pye*pye;

  if( fPx<1.e-3f || pxe2<1.e-6f ) return -1; // can not transport to x=x  
  
  float pxe = AliHLTTPCCAMath::Sqrt( pxe2 ); // extrapolated px
  float pti = 1.f/AliHLTTPCCAMath::Sqrt(pt2);
  
  float ty = ( fPy+pye ) / ( fPx+pxe );  
  float dy = dx*ty;
  float dS; // path in XY
  {
    float chord = dx*AliHLTTPCCAMath::Sqrt( 1.f + ty*ty ); // chord to the extrapolated point == sqrt(dx^2+dy^2)*sign(dx)
    float sa = 0.5*chord*b*pti; //  sin( half of the rotation angle ) ==  (chord/2) / radius
  
    // dS = (Pt/b)*2*arcsin( sa )
    //    = (Pt/b)*2*sa*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )
    //    =       chord*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )   
  
    float sa2 = sa*sa;
    const float k2 = 1./6.;
    const float k4 = 3./40.;
    //const float k6 = 5.f/112.f;
    dS =  chord + chord*sa2*(k2 + k4*sa2);
  }

  dLp = pti*dS; // path in XYZ / p == path in XY / pt

  float dz = fPz * dLp;

  fX = x;
  fY+=dy;
  fZ+=dz;
  fPx = pxe;
  fPy = pye;
  //fPz = fPz;
  //fQ = fQ;
  return 0;
}



GPUd() int AliHLTTPCGMPhysicalTrackModel::PropagateToXBxByBz( float x,  float y,  float z,
							      float Bx, float By, float Bz,							      
							      float &dLp )
{
  //
  // transport the track to X=x in magnetic field B = ( Bx, By, Bz )[kG*0.000299792458] 
  // xyzPxPyPz as well as all the additional values will change. No need to call UpdateValues() afterwards.
  // the method returns error code (0 == no error)
  //
  
  dLp = 0.;
   
  if(0){ // simple transport in Bz for test proposes
    if( fabs(x-X())<1.e-8f ) return 0;
    if( PropagateToXBzLight( x, Bz, dLp ) !=0 ) return -1;
    UpdateValues(); 
    return 0;
  }
  
  // Rotate to the system where Bx=By=0.

  float bt = AliHLTTPCCAMath::Sqrt(Bz*Bz + By*By);
  float bb = AliHLTTPCCAMath::Sqrt(Bx*Bx + By*By + Bz*Bz);

  float c1=1.f, s1=0.f;
  float c2=1.f, s2=0.f;

  if( bt > 1.e-4f) {
    c1=Bz/bt; s1= By/bt;
    c2=bt/bb; s2=-Bx/bb;
  }

  // rotation matrix: first around x, then around y'
  // after the first rotation: Bx'==Bx, By'==0, Bz'==Bt, X'==X
  // after the second rotation: Bx''==0, By''==0, Bz''==B, X'' axis is as close as possible to the original X

  //  
  //     ( c2 0 s2 )   ( 1  0   0 )
  // R = (  0 1 0  ) X ( 0 c1 -s1 )
  //     (-s2 0 c2 )   ( 0 s1  c1 )
  //
  
  float R0[3] = { c2, s1*s2, c1*s2 };
  float R1[3] = {  0,    c1,   -s1 };	 
  float R2[3] = {-s2, s1*c2, c1*c2 };


  // parameters and the extrapolation point in the rotated coordinate system
  {
    float lx = fX, ly = fY, lz=fZ, lpx = fPx, lpy = fPy, lpz = fPz;
 
    fX = R0[0]*lx + R0[1]*ly + R0[2]*lz;
    fY = R1[0]*lx + R1[1]*ly + R1[2]*lz;
    fZ = R2[0]*lx + R2[1]*ly + R2[2]*lz;

    fPx = R0[0]*lpx + R0[1]*lpy + R0[2]*lpz;
    fPy = R1[0]*lpx + R1[1]*lpy + R1[2]*lpz;
    fPz = R2[0]*lpx + R2[1]*lpy + R2[2]*lpz;
  }
  
  float xe = R0[0]*x + R0[1]*y + R0[2]*z;

  // transport in rotated coordinate system to X''=xe:

  if( PropagateToXBzLight( xe, bb, dLp )!=0 ) return -1;

  // rotate coordinate system back to the original R{-1}==R.999f {T}
  {  
    float lx = fX, ly = fY, lz=fZ, lpx = fPx, lpy = fPy, lpz = fPz;
 
    fX = R0[0]*lx + R1[0]*ly + R2[0]*lz;
    fY = R0[1]*lx + R1[1]*ly + R2[1]*lz;
    fZ = R0[2]*lx + R1[2]*ly + R2[2]*lz;

    fPx = R0[0]*lpx + R1[0]*lpy + R2[0]*lpz;
    fPy = R0[1]*lpx + R1[1]*lpy + R2[1]*lpz;
    fPz = R0[2]*lpx + R1[2]*lpy + R2[2]*lpz;
  }

  // a small (hopefully) additional step to X=x. Perhaps it may be replaced by linear extrapolation.
  
  float ddLp = 0;
  if( PropagateToXBzLight( x, Bz, ddLp ) !=0 ) return -1;
  
  dLp+=ddLp;

  UpdateValues();
  
  return 0;
}


GPUd() int AliHLTTPCGMPhysicalTrackModel::PropagateToLpBz( float Lp, float Bz )
{
  // Lp is path length L over track momentum p in [cm/GeV], Bz in kG*clight
  //
  // it is a copy of AliExternalTrackParam: ghelix3 routine.
  //
  // the method returns error code (0 == no error)
  //
  
  float qfield = fQ*Bz;

  float step = Lp;

  const float kOvSqSix = AliHLTTPCCAMath::Sqrt(1./6.);
  
  float px = fPx;
  float py = fPy;
  float pz = fPz;
    
  float tet = qfield*step;

  float tsint, sintt, sint, cos1t; 
  if (CAMath::Abs(tet) > 0.03) {
     sint  = CAMath::Sin(tet);
     sintt = sint/tet;
     tsint = (tet - sint)/tet;
     float t=CAMath::Sin(0.5*tet);
     cos1t = 2.f*t*t/tet;
  } else {
     tsint = tet*tet/6.;
     sintt = (1.f-tet*kOvSqSix)*(1.f+tet*kOvSqSix); // 1.- tsint;
     sint  = tet*sintt;
     cos1t = 0.5f*tet; 
  }

  float f1 = step*sintt;
  float f2 = step*cos1t;
  float f3 = step*tsint;
  float f4 = -tet*cos1t;
  float f5 = sint;

  fX += f1*px - f2*py;
  fY += f1*py + f2*px;
  fZ += f1*pz + f3*pz;

  fPx += f4*px - f5*py;
  fPy += f4*py + f5*px;

  UpdateValues();
 
  return 0;
}

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

GPUd() void AliHLTTPCGMPhysicalTrackModel::Print() const
{
#if !defined(HLTCA_GPUCODE)
  std::cout<<"AliHLTTPCGMPhysicalTrackModel:  x "<<fX<<" y "<<fY<<" z "<<fZ<<" px "<<fPx<<" py "<<fPy<<" pz "<<fPz<<" q "<<fQ<<std::endl;
#endif
}
