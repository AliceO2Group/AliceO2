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
 
ClassImp(AliHLTTPCFastTransformObject); //ROOT macro for the implementation of ROOT specific class methods


AliHLTTPCFastTransformObject::AliHLTTPCFastTransformObject()
  :
  TObject(),
  fVersion(0),
  fLastTimeBin(0),
  fTimeSplit1(0),
  fTimeSplit2(0),
  fAlignment(0)
{
  // constructor
  
  Reset();
}



void  AliHLTTPCFastTransformObject::Reset()
{
  // Deinitialisation
  fLastTimeBin = 0.;
  fTimeSplit1 = 0.;
  fTimeSplit2 = 0.;  
  for( Int_t i=0; i<fkNSec*fkNRows*3; i++) fSplines[i].Reset();
  fAlignment.Set(0);
}

AliHLTTPCFastTransformObject::AliHLTTPCFastTransformObject( const AliHLTTPCFastTransformObject &o )
  :
  TObject( o ),
  fVersion(0),
  fLastTimeBin(o.fLastTimeBin),
  fTimeSplit1(o.fTimeSplit1),
  fTimeSplit2(o.fTimeSplit2),
  fAlignment(o.fAlignment)
{ 
  // constructor    
  for( Int_t i=0; i<fkNSec*fkNRows*3; i++){
    fSplines[i] = o.fSplines[i];
  }
}

AliHLTTPCFastTransformObject& AliHLTTPCFastTransformObject::operator=( const AliHLTTPCFastTransformObject &o)
{
  // assignment operator
   new (this) AliHLTTPCFastTransformObject( o );
   return *this;
}

