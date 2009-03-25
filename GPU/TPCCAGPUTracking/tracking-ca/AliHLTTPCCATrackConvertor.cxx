// $Id: AliHLTTPCCATrackConvertor.cxx 27042 2008-07-02 12:06:02Z richterm $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          * 
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
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


#include "AliHLTTPCCATrackConvertor.h"
#include "AliExternalTrackParam.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCAMath.h"


void AliHLTTPCCATrackConvertor::GetExtParam( const AliHLTTPCCATrackParam &T1, AliExternalTrackParam &T2, Double_t alpha, Double_t Bz )
{
  //* Convert from AliHLTTPCCATrackParam to AliExternalTrackParam parameterisation, 
  //* the angle alpha is the global angle of the local X axis 

  Double_t par[5], cov[15];
  for( Int_t i=0; i<5; i++ ) par[i] = T1.GetPar()[i];
  for( Int_t i=0; i<15; i++ ) cov[i] = T1.GetCov()[i];

  if(par[2]>.99 ) par[2]=.99;
  if(par[2]<-.99 ) par[2]=-.99;

  { // kappa => 1/pt
    const Double_t kCLight = 0.000299792458;  
    Double_t c = 1.e4;
    if( CAMath::Abs(Bz)>1.e-4 ) c = 1./(Bz*kCLight);
    par[4] *= c;
    cov[10]*= c;
    cov[11]*= c;
    cov[12]*= c;
    cov[13]*= c;
    cov[14]*= c*c;
  }
  if( T1.GetCosPhi()<0 ){ // change direction
    par[2] = -par[2]; // sin phi
    par[3] = -par[3]; // DzDs
    par[4] = -par[4]; // kappa
    cov[3] = -cov[3];
    cov[4] = -cov[4];
    cov[6] = -cov[6];
    cov[7] = -cov[7];
    cov[10] = -cov[10];
    cov[11] = -cov[11];
  }
  T2.Set( (Double_t) T1.GetX(),alpha,par,cov);
}

void AliHLTTPCCATrackConvertor::SetExtParam( AliHLTTPCCATrackParam &T1, const AliExternalTrackParam &T2, Double_t Bz )
{
  //* Convert from AliExternalTrackParam parameterisation
  
  for( Int_t i=0; i<5; i++ ) T1.SetPar( i, T2.GetParameter()[i] );
  for( Int_t i=0; i<15; i++ ) T1.SetCov( i, T2.GetCovariance()[i] );
  T1.SetX( T2.GetX() );
  if(T1.SinPhi()>.99 ) T1.SetSinPhi( .99 );
  if(T1.SinPhi()<-.99 ) T1.SetSinPhi( -.99 );
  T1.SetCosPhi( CAMath::Sqrt(1.-T1.SinPhi()*T1.SinPhi()));
  const Double_t kCLight = 0.000299792458;  
  Double_t c = Bz*kCLight;
  { // 1/pt -> kappa 
    T1.SetPar( 4, T1.Par()[4]*c );
    T1.SetCov( 10, T1.Cov()[10]*c );
    T1.SetCov( 11, T1.Cov()[10]*c );
    T1.SetCov( 12, T1.Cov()[10]*c );
    T1.SetCov( 13, T1.Cov()[10]*c );
    T1.SetCov( 14, T1.Cov()[10]*c*c );
  }
}

