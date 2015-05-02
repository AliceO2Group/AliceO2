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

/** @file   AliHLTTPCFastTransformObject.cxx
    @author Sergey Gorbubnov
    @date   
    @brief 
*/


#include "AliHLTTPCFastTransformObject.h"
 
#include <iostream>
#include <iomanip>

using namespace std;

ClassImp(AliHLTTPCFastTransformObject); //ROOT macro for the implementation of ROOT specific class methods


AliHLTTPCFastTransformObject::AliHLTTPCFastTransformObject()
:
  fVersion(0),
  fLastTimeBin(600),
  fTimeSplit1(100),
  fTimeSplit2(500),
  fAlignment(0)
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt      
  Reset();
}



void  AliHLTTPCFastTransformObject::Reset()
{
  // Deinitialisation

  for( Int_t i=0; i<fkNSec; i++){
    for( Int_t j=0; j<fkNRows; j++ ){
      for( Int_t k=0; k<3; k++ ){
	fSplines[i][j][k].Init(0.,0,0.,0.,0,0.);
      }
    }
  }
  fAlignment.Set(0);
}



