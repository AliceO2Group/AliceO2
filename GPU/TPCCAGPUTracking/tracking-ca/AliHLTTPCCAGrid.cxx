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

GPUd() void AliHLTTPCCAGrid::Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, UInt_t n )
{
  //* Create the grid
  
  fYMin = CAMath::Min(yMin,yMax);
  fYMax = CAMath::Max(yMin,yMax);
  fZMin = CAMath::Min(zMin,zMax);
  fZMax = CAMath::Max(zMin,zMax);
  fNy = (UInt_t) CAMath::Sqrt( CAMath::Abs( (Float_t) n ) );
  fNy = CAMath::Max(fNy,1);
  fNz = fNy;
  fN = fNy*fNz;
  fStepYInv = (fYMax - fYMin);
  fStepZInv = (fZMax - fZMin);
  //Int_t ky = (fNy>1) ?fNy-1 :1;
  //Int_t kz = (fNz>1) ?fNz-1 :1;
  fStepYInv =  ( fStepYInv>1.e-4 ) ?fNy/fStepYInv :1;
  fStepZInv =  ( fStepZInv>1.e-4 ) ?fNz/fStepZInv :1;
}

GPUd() void AliHLTTPCCAGrid::Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, Float_t sy, Float_t sz )
{
  //* Create the grid
  
  fYMin = CAMath::Min(yMin,yMax);
  fYMax = CAMath::Max(yMin,yMax);
  fZMin = CAMath::Min(zMin,zMax);
  fZMax = CAMath::Max(zMin,zMax);
  fStepYInv = 1./sy;
  fStepZInv = 1./sz;

  fNy = (UInt_t) ( (fYMax - fYMin)*fStepYInv + 1 );
  fNz = (UInt_t) ( (fZMax - fZMin)*fStepZInv + 1 );
  fYMax = fYMin + fNy*sy; 
  fZMax = fZMin + fNz*sz; 
  fN = fNy*fNz;
}

GPUd() UInt_t AliHLTTPCCAGrid::GetBin( Float_t Y, Float_t Z ) const
{
  //* get the bin pointer
  
  Int_t bbY = (Int_t) CAMath::FMulRZ( Y-fYMin, fStepYInv );
  Int_t bbZ = (Int_t) CAMath::FMulRZ( Z-fZMin, fStepZInv );
  if( bbY<0 ) bbY = 0;
  else if( bbY>=(Int_t)fNy ) bbY = fNy - 1;  
  if( bbZ<0 ) bbZ = 0;
  else if( bbZ>=(Int_t)fNz ) bbZ = fNz - 1;
  Int_t bin = CAMath::Mul24(bbZ,fNy) + bbY;    
  return (UInt_t) bin;
}

GPUd() void AliHLTTPCCAGrid::GetBin( Float_t Y, Float_t Z, UInt_t &bY, UInt_t &bZ ) const
{
  //* get the bin pointer

  Int_t bbY = (Int_t) ( (Y-fYMin)*fStepYInv );
  Int_t bbZ = (Int_t) ( (Z-fZMin)*fStepZInv );  
  
  if( bbY<0 ) bbY = 0;
  else if( bbY>=(Int_t)fNy ) bbY = fNy - 1;  
  if( bbZ<0 ) bbZ = 0;
  else if( bbZ>=(Int_t)fNz ) bbZ = fNz - 1;
  bY = (UInt_t) bbY;
  bZ = (UInt_t) bbZ;
}
