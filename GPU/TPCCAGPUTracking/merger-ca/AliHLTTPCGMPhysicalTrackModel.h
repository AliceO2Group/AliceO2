// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPhysicalTrackModel_H
#define AliHLTTPCGMPhysicalTrackModel_H

#include "AliHLTTPCGMTrackParam.h"


/**
 * @class AliHLTTPCGMPhysicalTrackModel
 *
 * AliHLTTPCGMPhysicalTrackModel class is a trajectory in physical parameterisation (X,Y,Z,Px,PY,Pz,Q)
 * without covariance matrix. Px>0 and Q is {-1,+1} (no uncharged tracks).
 *
 * It is used to linearise transport equations for AliHLTTPCGMTrackParam trajectory during (re)fit.
 *
 */

class AliHLTTPCGMPhysicalTrackModel
{

  public:

  GPUd() AliHLTTPCGMPhysicalTrackModel()
    : fX(0.f), fY(0.f), fZ(0.f), fPx(1.e4f), fPy(0.f), fPz(0.f), fQ(1.f), fSinPhi( 0. ), fCosPhi( 1. ), fSecPhi( 1. ), fDzDs( 0. ), fDlDs( 0.), fQPt( 0. ), fP(fPx), fPt(fPx)
  {};
  
  GPUd() AliHLTTPCGMPhysicalTrackModel( const AliHLTTPCGMTrackParam &t );

  GPUd() void Set( const AliHLTTPCGMTrackParam &t );   
  GPUd() void Set( float X, float Y, float Z, float Px, float Py, float Pz, float Q);
    
  GPUd() float& X() { return fX; }
  GPUd() float& Y() { return fY; }
  GPUd() float& Z() { return fZ; }
  GPUd() float& Px() { return fPx; }
  GPUd() float& Py() { return fPy; }
  GPUd() float& Pz() { return fPz; }
  GPUd() float& Q() { return fQ; }

  GPUd() float& SinPhi() { return fSinPhi; }
  GPUd() float& CosPhi() { return fCosPhi; }
  GPUd() float& SecPhi() { return fSecPhi; }
  GPUd() float& DzDs() { return fDzDs; }
  GPUd() float& DlDs() { return fDlDs; }
  GPUd() float& QPt() { return fQPt; }
  GPUd() float& P() { return fP; }
  GPUd() float& Pt() { return fPt; }

  GPUd() float GetX() const { return fX; }
  GPUd() float GetY() const { return fY; }
  GPUd() float GetZ() const { return fZ; }
  GPUd() float GetPx() const { return fPx; }
  GPUd() float GetPy() const { return fPy; }
  GPUd() float GetPz() const { return fPz; }
  GPUd() float GetQ() const { return fQ; }

  GPUd() float GetSinPhi() const { return fSinPhi; }
  GPUd() float GetCosPhi() const { return fCosPhi; }
  GPUd() float GetSecPhi() const { return fSecPhi; }
  GPUd() float GetDzDs() const { return fDzDs; }
  GPUd() float GetDlDs() const { return fDlDs; }
  GPUd() float GetQPt() const { return fQPt; }
  GPUd() float GetP() const { return fP; }
  GPUd() float GetPt() const { return fPt; }

  GPUd() int PropagateToXBzLight( float x, float Bz, float &dLp );
  
  GPUd() int PropagateToXBxByBz( float x,  float y,  float z,
				 float Bx, float By, float Bz,				   
				 float &dLp );
  
  GPUd() int PropagateToLpBz( float Lp, float Bz );

  GPUd() void UpdateValues();

  GPUd() void Print() const;

  GPUd() float GetMirroredY( float Bz ) const ;
  
 private:

  // physical parameters of the trajectory

  float fX; // X
  float fY; // Y
  float fZ; // Z
  float fPx; // Px, >0
  float fPy; // Py
  float fPz; // Pz
  float fQ; // charge, +-1

  // some additional variables needed for GMTrackParam transport
  
  float fSinPhi; // SinPhi = Py/Pt
  float fCosPhi; // CosPhi = abs(Px)/Pt
  float fSecPhi; // 1/cos(phi) = Pt/abs(Px)    
  float fDzDs;   // DzDs = Pz/Pt
  float fDlDs;   // DlDs = P/Pt
  float fQPt;    // QPt = q/Pt
  float fP;    // momentum 
  float fPt;    // Pt momentum 
};


GPUd() inline void AliHLTTPCGMPhysicalTrackModel::Set( const AliHLTTPCGMTrackParam &t )
{
  float pti = fabs(t.GetQPt());
  if( pti < 1.e-4 ) pti = 1.e-4; // set 10000 GeV momentum for straight track
  fQ = (t.GetQPt()>=0) ?1.f :-1.f; // only charged tracks are considered
  fX = t.GetX();
  fY = t.GetY();
  fZ = t.GetZ();
  
  fPt = 1./pti;  
  fSinPhi = t.GetSinPhi();
  if( fSinPhi >  .999f ) fSinPhi = .999f;
  if( fSinPhi < -.999f ) fSinPhi = -.999f;
  fCosPhi = sqrt( (1. - fSinPhi)*(1.+fSinPhi) );  
  fSecPhi = 1./fCosPhi;
  fDzDs = t.GetDzDs();
  fDlDs = sqrt(1.f+fDzDs*fDzDs);
  fP = fPt*fDlDs;  

  fPy = fPt*fSinPhi;
  fPx = fPt*fCosPhi;
  fPz = fPt*fDzDs;
  fQPt = fQ*pti;
}

GPUd() inline AliHLTTPCGMPhysicalTrackModel::AliHLTTPCGMPhysicalTrackModel( const AliHLTTPCGMTrackParam &t )
: fX(0.f), fY(0.f), fZ(0.f), fPx(1.e4f), fPy(0.f), fPz(0.f), fQ(1.f), fSinPhi( 0. ), fCosPhi( 1. ), fSecPhi( 1. ), fDzDs( 0. ), fDlDs( 0.), fQPt( 0. ), fP(fPx), fPt(fPx)
{
  Set(t);
}


GPUd() inline void AliHLTTPCGMPhysicalTrackModel::Set( float X, float Y, float Z, float Px, float Py, float Pz, float Q)
{
  fX = X; fY = Y; fZ = Z; fPx = Px; fPy = Py; fPz = Pz;
  fQ = (Q>=0) ?1 :-1;
  UpdateValues();
}


GPUd() inline void AliHLTTPCGMPhysicalTrackModel::UpdateValues()
{
  if( fPx<0.f ){ // should not happen, change direction of the movenment
    fPx = -fPx;
    fPy = -fPy;
    fPz = -fPz;
    fQ = -fQ;
  }
  if( fPx<1.e-8f ) fPx = 1.e-8f;  
  fPt = sqrt( fPx*fPx + fPy*fPy );
  float pti = 1.f/fPt;
  fP = sqrt(fPx*fPx + fPy*fPy + fPz*fPz );
  fSinPhi = fPy*pti;
  fCosPhi = fPx*pti;
  fSecPhi = 1.f/fCosPhi;
  fDzDs = fPz*pti;
  fDlDs = fP*pti;
  fQPt = fQ*pti;
}

GPUd() inline float AliHLTTPCGMPhysicalTrackModel::GetMirroredY( float Bz ) const
{
  // get Y of the point which has the same X, but located on the other side of trajectory
  if( fabs(Bz)<1.e-8 ) Bz = 1.e-8;
  return fY - 2.f*fQ*fPx/Bz;
}

#endif
