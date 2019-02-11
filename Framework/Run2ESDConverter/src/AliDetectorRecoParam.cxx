/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/


///////////////////////////////////////////////////////////////////////////////
//
// Origin Marian.Ivanov@cern.ch                                                                 //
// Base Class for detector Reconstruction Parameters                         //
//                                                                           //  
// 
// AliDetectorRecoParam are identified according fEventSpecie - event specie(s)
// for which the object is valid.  
// The owner of the AliDetectorRecoParam is the AliRecoParam
// More than one RecoParam per recon can be registered.
//          
//  
//
///////////////////////////////////////////////////////////////////////////////


#include "AliDetectorRecoParam.h"

ClassImp(AliDetectorRecoParam)


AliDetectorRecoParam::AliDetectorRecoParam():
  TNamed(),
fEventSpecie(AliRecoParam::kDefault)
{
  //
  // default constructor
  //
}

AliDetectorRecoParam::~AliDetectorRecoParam(){
  //
  // destructor
  //
}

