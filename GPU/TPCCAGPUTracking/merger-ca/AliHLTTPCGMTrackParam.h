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
#include "AliHLTTPCGMMergedTrackHit.h"

class AliHLTTPCGMBorderTrack;
class AliExternalTrackParam;
class AliHLTTPCCAParam;
class AliHLTTPCGMPhysicalTrackModel;
class AliHLTTPCGMPolynomialField;
class AliHLTTPCGMMergedTrack;

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

  GPUd() float& X()      { return fX;    }
  GPUd() float& Y()      { return fP[0]; }
  GPUd() float& Z()      { return fP[1]; }
  GPUd() float& SinPhi() { return fP[2]; }
  GPUd() float& DzDs()   { return fP[3]; }
  GPUd() float& QPt()    { return fP[4]; }
  GPUd() float& ZOffset() {return fZOffset;}
  
  GPUhd() float GetX()      const { return fX; }
  GPUhd() float GetY()      const { return fP[0]; }
  GPUhd() float GetZ()      const { return fP[1]; }
  GPUd() float GetSinPhi() const { return fP[2]; }
  GPUd() float GetDzDs()   const { return fP[3]; }
  GPUd() float GetQPt()    const { return fP[4]; }
  GPUd() float GetZOffset() const {return fZOffset;}

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

  GPUd() float GetMirroredY( float Bz ) const;

  GPUd() void ResetCovariance();

  GPUd() bool CheckNumericalQuality(float overrideCovYY = -1.) const ;
  GPUd() bool CheckCov() const ;

  GPUd() bool Fit(const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrackHit* clusters, const AliHLTTPCCAParam &param, int &N, int &NTolerated, float &Alpha, int attempt = 0, float maxSinPhi = HLTCA_MAX_SIN_PHI);
  GPUd() void MarkClusters(AliHLTTPCGMMergedTrackHit* clusters, int ihitFirst, int ihitLast, int wayDirection, unsigned char state)
  {
    clusters[ihitFirst].fState |= state; while (ihitFirst != ihitLast) {ihitFirst += wayDirection; clusters[ihitFirst].fState |= state;}
  }
  GPUd() void UnmarkClusters(AliHLTTPCGMMergedTrackHit* clusters, int ihitFirst, int ihitLast, int wayDirection, unsigned char state)
  {
    clusters[ihitFirst].fState &= ~state; while (ihitFirst != ihitLast) {ihitFirst += wayDirection; clusters[ihitFirst].fState &= ~state;}
  }
  
  GPUd() bool Rotate( float alpha );
  GPUd() void ShiftZ(const AliHLTTPCGMPolynomialField* field, const AliHLTTPCGMMergedTrackHit* clusters, const AliHLTTPCCAParam &param, int N);

  GPUd() static float Reciprocal( float x ){ return 1./x; }
  GPUd() static void Assign( float &x, bool mask, float v ){
    if( mask ) x = v;
  }

  GPUd() static void Assign( int &x, bool mask, int v ){
    if( mask ) x = v;
  }
  
  GPUd() static void RefitTrack(AliHLTTPCGMMergedTrack &track, const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrackHit* clusters, const AliHLTTPCCAParam& param);
  
  struct AliHLTTPCCAOuterParam {
    float fX, fAlpha;
    float fP[5];
    float fC[15];
  };
  GPUd() const AliHLTTPCCAOuterParam& OuterParam() const {return fOuterParam;}

#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
  bool GetExtParam( AliExternalTrackParam &T, double alpha ) const;
  void SetExtParam( const AliExternalTrackParam &T );
#endif
      
  GPUd() void ConstrainSinPhi(float limit = HLTCA_MAX_SIN_PHI)
  {
    if (fP[2] > limit) fP[2] = limit;
    else if (fP[2] < -limit) fP[2] = -limit;
  }

  private:
  
    float fX;      // x position
    float fZOffset;
    float fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
    float fC[15];  // the covariance matrix for Y,Z,SinPhi,..
    float fChi2;   // the chi^2 value
    int   fNDF;    // the Number of Degrees of Freedom
    AliHLTTPCCAOuterParam fOuterParam;
};

GPUd() inline void AliHLTTPCGMTrackParam::ResetCovariance()
{
  fC[ 0] = 100.;
  fC[ 1] = 0.;  fC[ 2] = 100.;
  fC[ 3] = 0.;  fC[ 4] = 0.;  fC[ 5] = 1.;
  fC[ 6] = 0.;  fC[ 7] = 0.;  fC[ 8] = 0.; fC[ 9] = 10.;
  fC[10] = 0.;  fC[11] = 0.;  fC[12] = 0.; fC[13] = 0.; fC[14] = 10.;
  fChi2 = 0;
  fNDF = -5;
}

GPUd() inline float AliHLTTPCGMTrackParam::GetMirroredY( float Bz ) const
{
  // get Y of the point which has the same X, but located on the other side of trajectory
  float qptBz = GetQPt()*Bz;
  float cosPhi2 = 1.f - GetSinPhi()*GetSinPhi();
  if( fabs(qptBz)<1.e-8 ) qptBz = 1.e-8;
  if( cosPhi2<0.f ) cosPhi2 = 0.f;
  return GetY() - 2.f*sqrt(cosPhi2)/qptBz;
}

#endif //ALIHLTTPCCATRACKPARAM_H
