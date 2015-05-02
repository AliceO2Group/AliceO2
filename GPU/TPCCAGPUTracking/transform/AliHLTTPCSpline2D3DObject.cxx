//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

/** @file   AliHLTTPCSpline2D3D.cxx
    @author Sergey Gorbubnov
    @date   
    @brief 
*/

#include "AliHLTTPCSpline2D3DObject.h"

  
#include <iostream>
#include <iomanip>

using namespace std;

ClassImp(AliHLTTPCSpline2D3DObject);

void AliHLTTPCSpline2D3DObject::Init( Float_t minA, Int_t  nBinsA, Float_t  stepA, Float_t  minB, Int_t  nBinsB, Float_t  stepB )
{
  //
  // Initialisation
  //

  fNA = nBinsA;
  fNB = nBinsB;
  fMinA = minA;
  fMinB = minB;

  fStepA = stepA;
  fStepB = stepB;
  fXYZ.Set(fNA*fNB);
}


