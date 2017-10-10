//-*- Mode: C++ -*-
// $Id: AliHLTTPCGMPropagator.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPropagator_H
#define AliHLTTPCGMPropagator_H

#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"

class AliHLTTPCGMTrackParam;
class AliHLTTPCCAParam;

/**
 * @class AliHLTTPCGMPropagator
 *
 */

class AliHLTTPCGMPropagator
{
public:

  AliHLTTPCGMPropagator();

  struct MaterialCorrection {
    MaterialCorrection() : fRadLen(29.532), fRho(1.025e-3), fRhoOverRadLen(fRho/fRadLen), 
			   fDLMax(0.f), fEP2(0.f), fSigmadE2(0.f), fK22(0.f), fK33(0.f), fK43(0.f), fK44(0.f) {}

    float fRadLen, fRho, fRhoOverRadLen,
      fDLMax, fEP2, fSigmadE2, fK22, fK33, fK43, fK44; // precalculated values for MS and EnergyLoss correction
  };

  GPUd() void SetMaterial( float radLen, float rho );
  GPUd() void SetPolynomialFieldBz( const float *fieldBz );

  GPUd() void SetUseMeanMomentum( bool Flag ){ fUseMeanMomentum = Flag; CalculateMaterialCorrection(); }
  GPUd() void SetContinuousTracking( bool Flag ){ fContinuousTracking = Flag; }
  GPUd() void SetMaxSinPhi( float maxSinPhi ){ fMaxSinPhi = maxSinPhi; }
  
  GPUd() void SetTrack( AliHLTTPCGMTrackParam *track, float Alpha ); 
    
  GPUd() int RotateToAlpha( float newAlpha );
  
  GPUd() int PropagateToXAlpha( float posX, float posY, float posZ, float posAlpha, bool inFlyDirection );

  GPUd() int Update( float posY, float posZ, int rowType, const AliHLTTPCCAParam &param, bool rejectChi2 );  


  GPUd() float GetBz( float Alpha, float X, float Y, float Z ) const;
  GPUd() void  GetBxByBz( float Alpha, float X, float Y, float Z, float B[3] ) const;

  GPUd() float GetAlpha() const { return fAlpha; }
  GPUd() float GetQPt0() const { return fT0.GetQPt(); }
  GPUd() float GetSinPhi0() const { return fT0.GetSinPhi(); }

private:

  GPUd() void CalculateMaterialCorrection();
  GPUd() static float ApproximateBetheBloch( float beta2 );

  AliHLTTPCGMTrackParam *fT;
  float fAlpha; // rotation angle of the track coordinate system
  AliHLTTPCGMPhysicalTrackModel fT0;
  MaterialCorrection fMaterial;
  bool fUseMeanMomentum;//
  bool fContinuousTracking; // take field at the mean TPC Z
  float fMaxSinPhi;
  float fPolynomialFieldBz[6];
};

GPUd() inline AliHLTTPCGMPropagator::AliHLTTPCGMPropagator()
   : fT(0), fAlpha(0), fT0(), fMaterial(),
     fUseMeanMomentum(0), fContinuousTracking(0), fMaxSinPhi(.999)
{
  for( int i=0; i<6; i++ ) fPolynomialFieldBz[i] = 0.f;
}

GPUd() inline void AliHLTTPCGMPropagator::SetMaterial( float radLen, float rho )
{
  fMaterial.fRho = rho;
  fMaterial.fRadLen = radLen;
  fMaterial.fRhoOverRadLen = (radLen>1.e-4) ?rho/radLen : 0.;
  CalculateMaterialCorrection();
}

GPUd() inline void AliHLTTPCGMPropagator::SetPolynomialFieldBz( const float *field )
{
  if( !field ) return;
  for( int i=0; i<6; i++ ) fPolynomialFieldBz[i] = field[i];
}

GPUd() inline void AliHLTTPCGMPropagator::SetTrack( AliHLTTPCGMTrackParam *track, float Alpha )
{
  fT = track;
  if( !fT ) return;
  fT0.Set(*fT);
  fAlpha = Alpha;
  CalculateMaterialCorrection();
}

GPUd() inline float AliHLTTPCGMPropagator::GetBz( float /*Alpha*/, float x, float y, float z ) const
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = fPolynomialFieldBz;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}


#endif 
