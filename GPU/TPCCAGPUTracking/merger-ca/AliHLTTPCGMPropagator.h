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
#include "AliHLTTPCGMPolynomialField.h"
#include "AliHLTTPCGMOfflineStatisticalErrors.h"

class AliHLTTPCGMTrackParam;
class AliHLTTPCCAParam;

/**
 * @class AliHLTTPCGMPropagator
 *
 */

class AliHLTTPCGMPropagator
{
public:

  GPUd() AliHLTTPCGMPropagator();

  struct MaterialCorrection {
    GPUd() MaterialCorrection() : fRadLen(29.532), fRho(1.025e-3), fRhoOverRadLen(fRho/fRadLen), 
			   fDLMax(0.f), fEP2(0.f), fSigmadE2(0.f), fK22(0.f), fK33(0.f), fK43(0.f), fK44(0.f) {}

    float fRadLen, fRho, fRhoOverRadLen,
      fDLMax, fEP2, fSigmadE2, fK22, fK33, fK43, fK44; // precalculated values for MS and EnergyLoss correction
  };

  GPUd() void SetMaterial( float radLen, float rho );

  GPUd() void SetPolynomialField( const AliHLTTPCGMPolynomialField* field ){ fField = field; }

  GPUd() void SetFitInProjections( bool Flag ){ fFitInProjections = Flag; }
  GPUd() void SetToyMCEventsFlag( bool Flag ){ fToyMCEvents = Flag; }
  GPUd() void SetSpecialErrors( bool Flag ){ fSpecialErrors = Flag; }

  GPUd() void SetMaxSinPhi( float maxSinPhi ){ fMaxSinPhi = maxSinPhi; }
  
  GPUd() void SetTrack( AliHLTTPCGMTrackParam *track, float Alpha ); 
  GPUd() void ResetT0 () { if (!fT) return; fT0.Set(*fT);}
    
  GPUd() int RotateToAlpha( float newAlpha );
  
  GPUd() int PropagateToXAlpha( float posX, float posAlpha, bool inFlyDirection );

  //  GPUd() int PropagateToXAlphaBz( float posX, float posAlpha, bool inFlyDirection );

  GPUd() int Update( float posY, float posZ, int iRow, const AliHLTTPCCAParam &param, short clusterState, bool rejectChi2, bool refit );  
  GPUd() int RejectCluster(float chiY, float chiZ, unsigned char clusterState)
  {
    if (chiY > 9.f || chiZ > 9.f) return 2;
    if ((chiY > 6.25f || chiZ > 6.25f) && (clusterState & (AliHLTTPCGMMergedTrackHit::flagSplit | AliHLTTPCGMMergedTrackHit::flagShared))) return 2;
    if ((chiY > 1.f || chiZ > 6.25f) && (clusterState & (AliHLTTPCGMMergedTrackHit::flagEdge | AliHLTTPCGMMergedTrackHit::flagSingle))) return 2;
    return 0;
  }
      

  GPUd() float GetBz( float Alpha, float X, float Y, float Z ) const;
  GPUd() void  GetBxByBz( float Alpha, float X, float Y, float Z, float B[3] ) const;
  
  GPUd() void GetErr2( float& err2Y, float& err2Z, const AliHLTTPCCAParam &param, float posZ, int iRow, short clusterState);

  GPUd() float GetAlpha() const { return fAlpha; }
  GPUd() float GetQPt0() const { return fT0.GetQPt(); }
  GPUd() float GetSinPhi0() const { return fT0.GetSinPhi(); }
  GPUd() float GetCosPhi0() const { return fT0.GetCosPhi(); }
  GPUd() void Mirror(bool inFlyDirection);
  GPUd() float GetMirroredYModel() const;
  GPUd() float GetMirroredYTrack() const;
  GPUd() int GetPropagatedYZ(float x, float& projY, float& projZ);
  GPUd() bool GetFitInProjections() const {return fFitInProjections;}
  
  GPUd() AliHLTTPCGMPhysicalTrackModel& Model() {return fT0;}
  GPUd() void CalculateMaterialCorrection();
  GPUd() void SetStatErrorCurCluster(AliHLTTPCGMMergedTrackHit* c) {fStatErrors.SetCurCluster(c);}

private:

  GPUd() static float ApproximateBetheBloch( float beta2 );

  const AliHLTTPCGMPolynomialField* fField;
  AliHLTTPCGMTrackParam *fT;
  float fAlpha; // rotation angle of the track coordinate system
  AliHLTTPCGMPhysicalTrackModel fT0;
  MaterialCorrection fMaterial;
  bool fSpecialErrors;
  bool fFitInProjections; // fit (Y,SinPhi,QPt) and (Z,DzDs) paramteres separatelly
  bool fToyMCEvents; // events are simulated with simple home-made simulation
  float fMaxSinPhi;
  
  AliHLTTPCGMOfflineStatisticalErrors fStatErrors;
};

GPUd() inline AliHLTTPCGMPropagator::AliHLTTPCGMPropagator()
: fField(0), fT(0), fAlpha(0), fT0(), fMaterial(),
  fSpecialErrors(0), fFitInProjections(1), fToyMCEvents(0), fMaxSinPhi(HLTCA_MAX_SIN_PHI), fStatErrors()
{
}

GPUd() inline void AliHLTTPCGMPropagator::SetMaterial( float radLen, float rho )
{
  fMaterial.fRho = rho;
  fMaterial.fRadLen = radLen;
  fMaterial.fRhoOverRadLen = (radLen>1.e-4) ?rho/radLen : 0.;
  CalculateMaterialCorrection();
}

GPUd() inline void AliHLTTPCGMPropagator::SetTrack( AliHLTTPCGMTrackParam *track, float Alpha )
{
  fT = track;
  if( !fT ) return;
  fT0.Set(*fT);
  fAlpha = Alpha;
  CalculateMaterialCorrection();
}

GPUd() inline float AliHLTTPCGMPropagator::GetMirroredYModel() const
{
  float Bz = GetBz( fAlpha, fT0.GetX(), fT0.GetY(), fT0.GetZ() );
  return fT0.GetMirroredY( Bz );
}

GPUd() inline float AliHLTTPCGMPropagator::GetMirroredYTrack() const
{
  if( !fT ) return -1.E10;
  float Bz = GetBz( fAlpha, fT->GetX(), fT->GetY(), fT->GetZ() );  
  return fT->GetMirroredY( Bz ); 
}

#endif 
