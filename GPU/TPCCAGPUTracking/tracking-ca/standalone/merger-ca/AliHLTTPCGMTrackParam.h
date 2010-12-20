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

#include "TMath.h"

class AliHLTTPCGMTrackLinearisation;
class AliHLTTPCGMBorderTrack;
class AliExternalTrackParam;
class AliHLTTPCCAParam;

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

  struct AliHLTTPCGMTrackFitParam {
    //float fBethe, fE, fTheta2, fEP2, fSigmadE2, fK22, fK33, fK43, fK44;// parameters
    float fDLMax, fBetheRho, fE, fTheta2, fEP2, fSigmadE2, fK22, fK33, fK43, fK44;// parameters
  };
    
  float& X()      { return fX;    }
  float& Y()      { return fP[0]; }
  float& Z()      { return fP[1]; }
  float& SinPhi() { return fP[2]; }
  float& DzDs()   { return fP[3]; }
  float& QPt()    { return fP[4]; }
  
  float GetX()      const { return fX; }
  float GetY()      const { return fP[0]; }
  float GetZ()      const { return fP[1]; }
  float GetSinPhi() const { return fP[2]; }
  float GetDzDs()   const { return fP[3]; }
  float GetQPt()    const { return fP[4]; }

  float GetKappa( float Bz ) const { return -fP[4]*Bz; }

  void SetX( float v ){ fX = v; }

  float *Par() { return fP; }
  const float *GetPar() const { return fP; }
  float GetPar( int i) const { return(fP[i]); }
  void SetPar( int i, float v ) { fP[i] = v; }

  float& Chi2()  { return fChi2; }
  int&   NDF()  { return fNDF; }
  
  float Err2Y()      const { return fC[0]; }
  float Err2Z()      const { return fC[2]; }
  float Err2SinPhi() const { return fC[5]; }
  float Err2DzDs()   const { return fC[9]; }
  float Err2QPt()    const { return fC[14]; }
  
  float GetChi2()   const { return fChi2; }
  int   GetNDF()    const { return fNDF; }

  float GetCosPhi() const { return sqrt( float(1.) - GetSinPhi()*GetSinPhi() ); }

  float GetErr2Y()      const { return fC[0]; }
  float GetErr2Z()      const { return fC[2]; }
  float GetErr2SinPhi() const { return fC[5]; }
  float GetErr2DzDs()   const { return fC[9]; }
  float GetErr2QPt()    const { return fC[14]; }

  float *Cov() { return fC; }

  const float *GetCov() const { return fC; }
  float GetCov(int i) const {return fC[i]; }


  void SetCov( int i, float v ) { fC[i] = v; }
  void SetChi2( float v )  {  fChi2 = v; }
  void SetNDF( int v )   { fNDF = v; }
  

  static float ApproximateBetheBloch( float beta2 );

  void CalculateFitParameters( AliHLTTPCGMTrackFitParam &par,float RhoOverRadLen,  float Rho,  bool NoField=0, float mass = 0.13957 );
    

  bool CheckNumericalQuality() const ;

  void Fit
  (
   float x[], float y[], float z[], unsigned int rowType[], float alpha[], AliHLTTPCCAParam &param,
   int &N, float &Alpha, 
   bool UseMeanPt = 0,
   float maxSinPhi = .999
   );
  
  bool Rotate( float alpha, AliHLTTPCGMTrackLinearisation &t0, float maxSinPhi = .999 );
  

  static float fPolinomialFieldBz[6];   // field coefficients
  static float GetBz( float x, float y, float z );
  float GetBz() const{ return GetBz( fX, fP[0], fP[1] );}

  static float Reciprocal( float x ){ return 1./x; }
  static void Assign( float &x, bool mask, float v ){
    if( mask ) x = v;
  }

  static void Assign( int &x, bool mask, int v ){
    if( mask ) x = v;
  }

  bool GetExtParam( AliExternalTrackParam &T, double alpha ) const;
  void SetExtParam( const AliExternalTrackParam &T );

  private:
  
    float fX;      // x position
    float fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
    float fC[15];  // the covariance matrix for Y,Z,SinPhi,..
    float fChi2;   // the chi^2 value
    int   fNDF;    // the Number of Degrees of Freedom
};


inline  float AliHLTTPCGMTrackParam::GetBz( float x, float y, float z ) 
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = fPolinomialFieldBz;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

#endif //ALIHLTTPCCATRACKPARAM_H
