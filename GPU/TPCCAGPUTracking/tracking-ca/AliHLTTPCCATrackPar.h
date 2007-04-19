//-*- Mode: C++ -*-
// @(#) $Id$
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
//                                                                        *
// Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
//                  for The ALICE HLT Project.                            *
//                                                                        *
// Permission to use, copy, modify and distribute this software and its   *
// documentation strictly for non-commercial purposes is hereby granted   *
// without fee, provided that the above copyright notice appears in all   *
// copies and that both the copyright notice and this permission notice   *
// appear in the supporting documentation. The authors make no claims     *
// about the suitability of this software for any purpose. It is          *
// provided "as is" without express or implied warranty.                  *
//*************************************************************************


#ifndef ALIHLTTPCCATRACKPAR_H
#define ALIHLTTPCCATRACKPAR_H

#include "Rtypes.h"

/**
 * @class AliHLTTPCCATrackPar
 *
 * AliHLTTPCCATrackPar class describes the track parametrisation
 * which is used by the AliHLTTPCCATracker slice tracker.
 * The class is under development.
 *
 */
class AliHLTTPCCATrackPar 
{
 public:

  AliHLTTPCCATrackPar(): fChi2(0), fNDF(0){}
  virtual ~AliHLTTPCCATrackPar(){}

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

  void TransportBz( Double_t Bz, const Double_t xyz[3] )
  { 
    TransportBz( Bz,GetDsToPointBz(Bz, xyz)) ; 
  }

  void TransportBz( Double_t Bz, const Double_t xyz[3], Double_t *T0 )
  { 
    TransportBz( Bz,GetDsToPointBz(Bz, xyz, T0), T0) ; 
  }

  void TransportBz( Double_t Bz, Double_t x, Double_t y, Double_t z )
  { 
    Double_t xyz[3] = {x,y,z};
    TransportBz(Bz, xyz);
  }

  void GetConnectionMatrix( Double_t Bz, const Double_t p[3], Double_t G[6], const Double_t *T0  ) const ;

  void GetConnectionMatrix( Double_t Bz, const Double_t p[3], Double_t G[6] ) const 
  {
    GetConnectionMatrix( Bz, p, G, fP );
  }

  void Filter( const Double_t m[3], const Double_t V[6], const Double_t V1[6] );
  void Rotate( Double_t alpha );
  void ConvertTo5( Double_t alpha, Double_t T[], Double_t C[] ) const;

 private:

  Double_t fP[7];  // track parameters:  X, Y, Z, ex, ey, ez, q/P
  Double_t fC[28]; // the covariance matrix in the low-triangular form
  Double_t fChi2;  // the chi^2 value
  Int_t fNDF;      // the Number of Degrees of Freedom

  ClassDef(AliHLTTPCCATrackPar, 0);

};


#endif
