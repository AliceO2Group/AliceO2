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


bool AliHLTTPCCATrackConvertor::GetExtParam( const AliHLTTPCCATrackParam &T1, AliExternalTrackParam &T2, double alpha )
{
  //* Convert from AliHLTTPCCATrackParam to AliExternalTrackParam parameterisation,
  //* the angle alpha is the global angle of the local X axis

  bool ok = T1.CheckNumericalQuality();

  double par[5], cov[15];
  for ( int i = 0; i < 5; i++ ) par[i] = T1.GetPar()[i];
  for ( int i = 0; i < 15; i++ ) cov[i] = T1.GetCov()[i];

  if ( par[2] > .99 ) par[2] = .99;
  if ( par[2] < -.99 ) par[2] = -.99;

  if ( T1.GetSignCosPhi() < 0 ) { // change direction
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

  if ( CAMath::Abs( par[4] ) < 1.e-5 ) par[4] = 1.e-5; // some other software will crash if q/Pt==0
  if ( CAMath::Abs( par[4] ) > 0.08 ) ok = 0; // some other software will crash if q/Pt is too big

  T2.Set( ( double ) T1.GetX(), alpha, par, cov );

  return ok;
}

void AliHLTTPCCATrackConvertor::SetExtParam( AliHLTTPCCATrackParam &T1, const AliExternalTrackParam &T2 )
{
  //* Convert from AliExternalTrackParam parameterisation

  for ( int i = 0; i < 5; i++ ) T1.SetPar( i, T2.GetParameter()[i] );
  for ( int i = 0; i < 15; i++ ) T1.SetCov( i, T2.GetCovariance()[i] );
  T1.SetX( T2.GetX() );
  if ( T1.SinPhi() > .99 ) T1.SetSinPhi( .99 );
  if ( T1.SinPhi() < -.99 ) T1.SetSinPhi( -.99 );
  T1.SetSignCosPhi( 1 );
}

