//-*- Mode: C++ -*-
// $Id: AliHLTTPCGMTrackParam.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMTRACKPARAM_H
#define ALIHLTTPCGMTRACKPARAM_H

#include "AliHLTTPCCAMath.h"

class AliHLTTPCGMBorderTrack;
class AliExternalTrackParam;
class AliHLTTPCCAParam;
class AliHLTTPCGMMergedTrack;
class AliHLTTPCGMPhysicalTrackModel;

/**
 * @class AliHLTTPCGMTrackParam
 *
 * AliHLTTPCGMTrackParam class describes the track parametrisation
 * which is used by the AliHLTTPCGMTracker slice tracker.
 *
 */
class AliHLTTPCGMTrackParam
{
public:

  struct AliHLTTPCGMTrackMaterialCorrection {
    //float fBethe, fE, fTheta2, fEP2, fSigmadE2, fK22, fK33, fK43, fK44;// parameters
    float fDLMax, fBetheRho, fE, fTheta2, fEP2, fSigmadE2, fK22, fK33, fK43, fK44;// parameters
  };
    
  GPUd() float& X()      { return fX;    }
  GPUd() float& Y()      { return fP[0]; }
  GPUd() float& Z()      { return fP[1]; }
  GPUd() float& SinPhi() { return fP[2]; }
  GPUd() float& DzDs()   { return fP[3]; }
  GPUd() float& QPt()    { return fP[4]; }
  
  GPUhd() float GetX()      const { return fX; }
  GPUhd() float GetY()      const { return fP[0]; }
  GPUhd() float GetZ()      const { return fP[1]; }
  GPUd() float GetSinPhi() const { return fP[2]; }
  GPUd() float GetDzDs()   const { return fP[3]; }
  GPUd() float GetQPt()    const { return fP[4]; }

  GPUd() float GetKappa( float Bz ) const { return -fP[4]*Bz; }

  GPUd() void SetX( float v ){ fX = v; }

  GPUd() float *Par() { return fP; }
  GPUd() const float *GetPar() const { return fP; }
  GPUd() float GetPar( int i) const { return(fP[i]); }
  GPUd() void SetPar( int i, float v ) { fP[i] = v; }

  GPUd() float& Chi2()  { return fChi2; }
  GPUd() int&   NDF()  { return fNDF; }
  
  GPUd() float Err2Y()      const { return fC[0]; }
  GPUd() float Err2Z()      const { return fC[2]; }
  GPUd() float Err2SinPhi() const { return fC[5]; }
  GPUd() float Err2DzDs()   const { return fC[9]; }
  GPUd() float Err2QPt()    const { return fC[14]; }
  
  GPUd() float GetChi2()   const { return fChi2; }
  GPUd() int   GetNDF()    const { return fNDF; }

  GPUd() float GetCosPhi() const { return sqrt( float(1.) - GetSinPhi()*GetSinPhi() ); }

  GPUd() float GetErr2Y()      const { return fC[0]; }
  GPUd() float GetErr2Z()      const { return fC[2]; }
  GPUd() float GetErr2SinPhi() const { return fC[5]; }
  GPUd() float GetErr2DzDs()   const { return fC[9]; }
  GPUd() float GetErr2QPt()    const { return fC[14]; }

  GPUd() float *Cov() { return fC; }

  GPUd() const float *GetCov() const { return fC; }
  GPUd() float GetCov(int i) const {return fC[i]; }


  GPUd() void SetCov( int i, float v ) { fC[i] = v; }
  GPUd() void SetChi2( float v )  {  fChi2 = v; }
  GPUd() void SetNDF( int v )   { fNDF = v; }

  GPUd() void ResetCovariance();

  GPUd() static float ApproximateBetheBloch( float beta2 );

  GPUd() static void CalculateMaterialCorrection( AliHLTTPCGMTrackMaterialCorrection &par, const AliHLTTPCGMPhysicalTrackModel &t0, float RhoOverRadLen,  float Rho,  bool NoField=0, float mass = 0.13957 );

  GPUd() bool CheckNumericalQuality() const ;

  GPUd() void Fit
  (
   float* PolinomialFieldBz,
   float x[], float y[], float z[], int rowType[], float alpha[], const AliHLTTPCCAParam &param,
   int &N, float &Alpha, 
   bool UseMeanPt = 0,
   float maxSinPhi = .999
   );
  
  GPUd() bool Rotate( float alpha, AliHLTTPCGMPhysicalTrackModel &t0, float maxSinPhi = .999 );

  GPUhd() static float GetBz( float x, float y, float z, float* PolinomialFieldBz );
  GPUhd() float GetBz(float* PolinomialFieldBz ) const{ return GetBz( fX, fP[0], fP[1], PolinomialFieldBz );}

  GPUd() static float Reciprocal( float x ){ return 1./x; }
  GPUd() static void Assign( float &x, bool mask, float v ){
    if( mask ) x = v;
  }

  GPUd() static void Assign( int &x, bool mask, int v ){
    if( mask ) x = v;
  }
  
  GPUd() int PropagateTrack(float* PolinomialFieldBz, float posX, float posY, float posZ, float posAlpha, const AliHLTTPCCAParam &param, float& Alpha, float maxSinPhi, bool UseMeanPt, AliHLTTPCGMTrackMaterialCorrection& par, AliHLTTPCGMPhysicalTrackModel& t0, bool inFlyDirection );

  GPUd() int UpdateTrack( float posY, float posZ, int rowType, const AliHLTTPCCAParam &param, AliHLTTPCGMPhysicalTrackModel& t0, float maxSinPhi, bool rejectChi2);
  
  GPUd() static void RefitTrack(AliHLTTPCGMMergedTrack &track, float* PolinomialFieldBz, float* x, float* y, float* z, int* rowType, float* alpha, const AliHLTTPCCAParam& param);

#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
  bool GetExtParam( AliExternalTrackParam &T, double alpha ) const;
  void SetExtParam( const AliExternalTrackParam &T );
#endif

  private:
  
    float fX;      // x position
    float fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
    float fC[15];  // the covariance matrix for Y,Z,SinPhi,..
    float fChi2;   // the chi^2 value
    int   fNDF;    // the Number of Degrees of Freedom
};

inline float AliHLTTPCGMTrackParam::GetBz( float x, float y, float z, float* PolinomialFieldBz ) 
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = PolinomialFieldBz;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

GPUd() inline void AliHLTTPCGMTrackParam::ResetCovariance()
{
  fC[ 0] = 100.;
  fC[ 1] = 0.;  fC[ 2] = 100.;
  fC[ 3] = 0.;  fC[ 4] = 0.;  fC[ 5] = 1.;
  fC[ 6] = 0.;  fC[ 7] = 0.;  fC[ 8] = 0.; fC[ 9] = 1.;
  fC[10] = 0.;  fC[11] = 0.;  fC[12] = 0.; fC[13] = 0.; fC[14] = 10.;
  fChi2 = 0;
  fNDF = -5;
}

#endif //ALIHLTTPCCATRACKPARAM_H
