/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
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

//-------------------------------------------------------------------------
//     Base class for ESD and AOD VZERO data
//     Author: Cvetan Cheshkov
//     cvetan.cheshkov@cern.ch 2/02/2011
//-------------------------------------------------------------------------

#include "AliVVZERO.h"
#include "AliLog.h"

ClassImp(AliVVZERO)

//__________________________________________________________________________
AliVVZERO::AliVVZERO(const AliVVZERO& source) :
  TObject(source) { } // Copy constructor

//__________________________________________________________________________
AliVVZERO& AliVVZERO::operator=(const AliVVZERO& source)
{
  // Assignment operator
  //
  if (this!=&source) { 
    TObject::operator=(source); 
  }
  
  return *this; 
}

//__________________________________________________________________________
Bool_t AliVVZERO::OutOfRange(Int_t i, const char* s, Int_t upper) const
{
  // checks if i is a valid index.
  // s = name of calling method
  if (i > upper || i < 0) {
    AliInfo(Form("%s: Index %d out of range",s,i));
    return kTRUE;
  }
  return kFALSE;
}

//__________________________________________________________________________
Float_t AliVVZERO::GetVZEROEtaMin(Int_t channel)
{
  // The method returns
  // the lower eta limit of a given channel
  Float_t eta[8] = {-3.7,-3.2,-2.7,-2.2,4.5,3.9,3.4,2.8};
  return eta[channel/8];
}

//__________________________________________________________________________
Float_t AliVVZERO::GetVZEROEtaMax(Int_t channel)
{
  // The method returns
  // the upper eta limit of a given channel
  Float_t eta[8] = {-3.2,-2.7,-2.2,-1.7,5.1,4.5,3.9,3.4};
  return eta[channel/8];
}
