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
#include "AliVCaloTrigger.h"

/// \cond CLASSIMP
ClassImp(AliVCaloTrigger)
/// \endcond
  
  AliVCaloTrigger::AliVCaloTrigger(const AliVCaloTrigger& vtrg) :
    TNamed(vtrg) { ; }

AliVCaloTrigger& AliVCaloTrigger::operator=(const AliVCaloTrigger& vtrg)
{ 
  if (this!=&vtrg) { 
    TObject::operator=(vtrg); 
  }
  
  return *this; 
}


void AliVCaloTrigger::Copy(TObject &obj) const 
{	
  TNamed::Copy(obj);
}
