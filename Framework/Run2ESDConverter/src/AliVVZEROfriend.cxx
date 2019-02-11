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
///
/// This is a class for containing all the VZERO DDL raw data
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include "AliVVZEROfriend.h"
#include <TObject.h>

//_____________________________________________________________________________
AliVVZEROfriend::AliVVZEROfriend(const AliVVZEROfriend& vzerofriend)
  : TObject(vzerofriend)
{
  // copy constructor
}

//_____________________________________________________________________________
AliVVZEROfriend& AliVVZEROfriend::operator = (const AliVVZEROfriend& vzerofriend)
{
  // assignment operator
  if(&vzerofriend == this) return *this;
  TObject::operator=(vzerofriend);
  return *this;
}

ClassImp(AliVVZEROfriend)
