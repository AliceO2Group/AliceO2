//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project           * 
//* ALICE Experiment at CERN, All rights reserved.                            *
//* See cxx source for full Copyright notice                                  *


#ifndef ALIHLTTPCCATRACKPARAM_H
#define ALIHLTTPCCATRACKPARAM_H

#include "Rtypes.h"

class AliHLTTPCCATrackFitParam
{
public:
  Float_t fBethe, fE,fTheta2, fEP2, fSigmadE2, fK22,fK33,fK43,fK44;
};

/**
 * @class AliHLTTPCCATrackParam
 *
 * AliHLTTPCCATrackParam class describes the track parametrisation
 * which is used by the AliHLTTPCCATracker slice tracker.
 *
 */
class AliHLTTPCCATrackParam
{
 public:

  AliHLTTPCCATrackParam(): fX(0),fCosPhi(1),fChi2(0), fNDF(0){}
  //virtual ~AliHLTTPCCATrackParam(){}

  Float_t &X()     { return fX;    }
  Float_t &Y()     { return fP[0]; }
  Float_t &Z()     { return fP[1]; }
  Float_t &SinPhi(){ return fP[2]; }
  Float_t &DzDs()  { return fP[3]; }
  Float_t &Kappa() { return fP[4]; }
  Float_t &CosPhi(){ return fCosPhi; }
  Float_t &Chi2()  { return fChi2; }
  Int_t   &NDF()   { return fNDF; }

  Float_t &Err2Y()      { return fC[0]; }
  Float_t &Err2Z()      { return fC[2]; }
  Float_t &Err2SinPhi() { return fC[5]; }
  Float_t &Err2DzDs()   { return fC[9]; }
  Float_t &Err2Kappa()  { return fC[14]; }

  Float_t GetX()      const { return fX; }
  Float_t GetY()      const { return fP[0]; }
  Float_t GetZ()      const { return fP[1]; }
  Float_t GetSinPhi() const { return fP[2]; }
  Float_t GetDzDs()   const { return fP[3]; }
  Float_t GetKappa()  const { return fP[4]; }
  Float_t GetCosPhi() const { return fCosPhi; }
  Float_t GetChi2()   const { return fChi2; }
  Int_t   GetNDF()    const { return fNDF; }

  Float_t GetErr2Y()      const { return fC[0]; }
  Float_t GetErr2Z()      const { return fC[2]; }
  Float_t GetErr2SinPhi() const { return fC[5]; }
  Float_t GetErr2DzDs()   const { return fC[9]; }
  Float_t GetErr2Kappa()  const { return fC[14]; }

  Float_t *Par(){ return fP; }
  Float_t *Cov(){ return fC; }

  const Float_t *GetPar() const { return fP; }
  const Float_t *GetCov() const { return fC; }

  void ConstructXY3( const Float_t x[3], const Float_t y[3], const Float_t sigmaY2[3], Float_t CosPhi0 );

  void ConstructXYZ3( const Float_t p0[5], const Float_t p1[5], const Float_t p2[5], 
		      Float_t CosPhi0, Float_t t0[]=0 );

  Float_t GetS( Float_t x, Float_t y ) const;

  void GetDCAPoint( Float_t x, Float_t y, Float_t z,
		    Float_t &px, Float_t &py, Float_t &pz ) const;
  
  Bool_t TransportToX( Float_t X );
  Bool_t TransportToXWithMaterial( Float_t X, AliHLTTPCCATrackFitParam &par );
  Bool_t TransportToXWithMaterial( Float_t X, Float_t Bz );
  Bool_t Rotate( Float_t alpha );

  Bool_t Filter2( Float_t y, Float_t z, Float_t err2Y, Float_t err2Z );
  void FilterY( Float_t y, Float_t erry );
  void FilterZ( Float_t z, Float_t errz );

  static Float_t ApproximateBetheBloch( Float_t beta2 );
  void CalculateFitParameters( AliHLTTPCCATrackFitParam &par, Float_t Bz, Float_t mass = 0.13957 );
  Bool_t CorrectForMeanMaterial( Float_t xOverX0,  Float_t xTimesRho, AliHLTTPCCATrackFitParam &par );
  void Print() const;

 private:

  Float_t fX;      // x position
  Float_t fCosPhi; // cosPhi
  Float_t fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, Kappa
  Float_t fC[15];  // the covariance matrix for Y,Z,SinPhi,..
  Float_t fChi2;   // the chi^2 value
  Int_t   fNDF;    // the Number of Degrees of Freedom

  //ClassDef(AliHLTTPCCATrackParam, 0);

};


#endif
