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
//     Base class for ESD and AOD AD data
//     Author: Michal Broz
//     michal.broz@cern.ch
//-------------------------------------------------------------------------

#include "AliVAD.h"
#include "AliLog.h"

ClassImp(AliVAD)

//__________________________________________________________________________
AliVAD::AliVAD(const AliVAD& source) :
  TObject(source) { } // Copy constructor

//__________________________________________________________________________
AliVAD& AliVAD::operator=(const AliVAD& source)
{
  // Assignment operator
  //
  if (this!=&source) { 
    TObject::operator=(source); 
  }
  
  return *this; 
}

//__________________________________________________________________________
Bool_t AliVAD::OutOfRange(Int_t i, const char* s, Int_t upper) const
{
  // checks if i is a valid index.
  // s = name of calling method
  if (i > upper || i < 0) {
    AliInfo(Form("%s: Index %d out of range",s,i));
    return kTRUE;
  }
  return kFALSE;
}

