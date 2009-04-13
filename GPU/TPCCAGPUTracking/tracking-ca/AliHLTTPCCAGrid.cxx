// $Id$
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



#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCAMath.h"
#include <iostream>

GPUd() void AliHLTTPCCAGrid::Create( float yMin, float yMax, float zMin, float zMax, unsigned int n )
{
  //* Create the grid

  fYMin = CAMath::Min( yMin, yMax );
  fYMax = CAMath::Max( yMin, yMax ) + .1;
  fZMin = CAMath::Min( zMin, zMax );
  fZMax = CAMath::Max( zMin, zMax ) + .1;
  fNy = ( unsigned int ) CAMath::Sqrt( CAMath::Abs( ( float ) n ) );
  fNy = CAMath::Max( fNy, 1 );
  fNz = fNy;
  fN = fNy * fNz;
  fStepYInv = ( fYMax - fYMin );
  fStepZInv = ( fZMax - fZMin );
  //int ky = (fNy>1) ?fNy-1 :1;
  //int kz = (fNz>1) ?fNz-1 :1;
  fStepYInv =  ( fStepYInv > 1.e-4 ) ? fNy / fStepYInv : 1;
  fStepZInv =  ( fStepZInv > 1.e-4 ) ? fNz / fStepZInv : 1;
}

GPUd() void AliHLTTPCCAGrid::Create( float yMin, float yMax, float zMin, float zMax, float sy, float sz )
{
  //* Create the grid

  fYMin = CAMath::Min( yMin, yMax );
  fYMax = CAMath::Max( yMin, yMax ) + .1;
  fZMin = CAMath::Min( zMin, zMax );
  fZMax = CAMath::Max( zMin, zMax ) + .1;
  fStepYInv = 1. / sy;
  fStepZInv = 1. / sz;

  fNy = ( unsigned int ) ( ( fYMax - fYMin ) * fStepYInv + 1 );
  fNz = ( unsigned int ) ( ( fZMax - fZMin ) * fStepZInv + 1 );
  fYMax = fYMin + fNy * sy;
  fZMax = fZMin + fNz * sz;
  fN = fNy * fNz;
}

GPUd() unsigned int AliHLTTPCCAGrid::GetBin( float Y, float Z ) const
{
  //* get the bin pointer

  int bbY = ( int ) CAMath::FMulRZ( Y - fYMin, fStepYInv );
  int bbZ = ( int ) CAMath::FMulRZ( Z - fZMin, fStepZInv );
  if ( bbY < 0 ) bbY = 0;
  else if ( bbY >= ( int )fNy ) bbY = fNy - 1;
  if ( bbZ < 0 ) bbZ = 0;
  else if ( bbZ >= ( int )fNz ) bbZ = fNz - 1;
  int bin = CAMath::Mul24( bbZ, fNy ) + bbY;
  return ( unsigned int ) bin;
}

GPUd() void AliHLTTPCCAGrid::GetBin( float Y, float Z, unsigned int &bY, unsigned int &bZ ) const
{
  //* get the bin pointer

  int bbY = ( int ) ( ( Y - fYMin ) * fStepYInv );
  int bbZ = ( int ) ( ( Z - fZMin ) * fStepZInv );

  if ( bbY < 0 ) bbY = 0;
  else if ( bbY >= ( int )fNy ) bbY = fNy - 1;
  if ( bbZ < 0 ) bbZ = 0;
  else if ( bbZ >= ( int )fNz ) bbZ = fNz - 1;
  bY = ( unsigned int ) bbY;
  bZ = ( unsigned int ) bbZ;
}
