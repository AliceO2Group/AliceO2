// $Id$
//***************************************************************************
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
//***************************************************************************

#include "AliHLTTPCCAGrid.h"
#include "TMath.h"

AliHLTTPCCAGrid::AliHLTTPCCAGrid(const AliHLTTPCCAGrid&)
  :fGrid(0),fNy(0),fNz(0),fN(0),
   fYMin(0),fYMax(0),fZMin(0),fZMax(0),fStepYInv(0),fStepZInv(0)
{
  //* dummy
}

AliHLTTPCCAGrid& AliHLTTPCCAGrid::operator=(const AliHLTTPCCAGrid&)
{
  //* dummy
  return *this;
}

void AliHLTTPCCAGrid::Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, Int_t n )
{
  //* Create the grid
  
  fYMin = TMath::Min(yMin,yMax);
  fYMax = TMath::Max(yMin,yMax);
  fZMin = TMath::Min(zMin,zMax);
  fZMax = TMath::Max(zMin,zMax);
  fNy = fNz = (Int_t) TMath::Sqrt( (Float_t) n );
  fNy = TMath::Max(fNy,1);
  fNz = TMath::Max(fNz,1);
  fN = fNy*fNz;
  if( fGrid ) delete[] fGrid;
  fGrid = new void*[fN];
  for( Int_t i=0; i<fN; i++ ) fGrid[i] = 0;
  fStepYInv = (fYMax - fYMin);
  fStepZInv = (fZMax - fZMin);
  fStepYInv =  ( fStepYInv>1.e-4 ) ?(fNy-1)/fStepYInv :0;
  fStepZInv =  ( fStepZInv>1.e-4 ) ?(fNz-1)/fStepZInv :0;
}

void **AliHLTTPCCAGrid::Get( Float_t Y, Float_t Z ) const
{
  //* get the bin pointer

  Int_t yBin = (Int_t) ( (Y-fYMin)*fStepYInv );
  Int_t zBin = (Int_t) ( (Z-fZMin)*fStepZInv );
  if( yBin<0 ) yBin = 0;
  else if( yBin>=fNy ) yBin = fNy - 1;
  if( zBin<0 ) zBin = 0;
  else if( zBin>=fNz ) zBin = fNz - 1;
  return fGrid + zBin*fNy + yBin;
}
