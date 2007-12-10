//-*- Mode: C++ -*-
// @(#) $Id$

/* This file is property of and copyright by the ALICE HLT Project        * 
 * ALICE Experiment at CERN, All rights reserved.                         *
 * See cxx source for full Copyright notice                               */

#ifndef ALIHLTTPCCATRACKPAR_H
#define ALIHLTTPCCATRACKPAR_H

#include "Rtypes.h"

/**
 * @class AliHLTTPCCATrackPar
 */
class AliHLTTPCCATrackPar {
 public:

  AliHLTTPCCATrackPar(): fChi2(0), fNDF(0){;}

  Double_t *Par(){ return fP; }
  Double_t *Cov(){ return fC; }
  Double_t &Chi2(){ return fChi2; }
  Int_t &NDF(){ return fNDF; }

  void Init();
  void Normalize( Double_t Direction[3]=0 );

  void TransportBz( Double_t Bz, Double_t S );
  void TransportBz( Double_t Bz, Double_t S, Double_t *T0 );

  Double_t GetDsToPointBz( Double_t Bz, const Double_t xyz[3] ) const;
  Double_t GetDsToPointBz( Double_t Bz, const Double_t xyz[3], const Double_t *T0 ) const;

  void TransportBz( Double_t Bz, const Double_t xyz[3] ){ 
    TransportBz( Bz,GetDsToPointBz(Bz, xyz)) ; 
  }

  void TransportBz( Double_t Bz, const Double_t xyz[3], Double_t *T0 ){ 
    TransportBz( Bz,GetDsToPointBz(Bz, xyz, T0), T0) ; 
  }

  void TransportBz( Double_t Bz, Double_t x, Double_t y, Double_t z ){ 
    Double_t xyz[3] = {x,y,z};
    TransportBz(Bz, xyz);
  }

  void GetConnectionMatrix( Double_t Bz, const Double_t p[3], Double_t G[6], const Double_t *T0  ) const ;

  void GetConnectionMatrix( Double_t Bz, const Double_t p[3], Double_t G[6] ) const {
    GetConnectionMatrix( Bz, p, G, fP );
  }

  void Filter( const Double_t m[3], const Double_t V[6], const Double_t V1[6] );
  void Rotate( Double_t alpha );
  void ConvertTo5( Double_t alpha, Double_t T[], Double_t C[] ) const;

 private:

  Double_t fP[7];  // parameters:  X, Y, Z, ex, ey, ez, q/P
  Double_t fC[28]; // Covariance matrix
  Double_t fChi2;  // Chi^2
  Int_t fNDF;      // NDF

  ClassDef(AliHLTTPCCATrackPar, 0);

};


#endif
