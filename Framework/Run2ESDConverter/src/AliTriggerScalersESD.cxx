/*************************************************************************
* Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
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
//  Class to define the ALICE Trigger Scalers  
//
//  For each trigger class there are six scalers:
//
//    LOCB       L0 triggers before any vetos 
//    LOCA       L0 triggers after all vetos 
//    L1CB       L1 triggers before any vetos 
//    L1CA       L1 triggers after all vetos 
//    L2CB       L2 triggers before any vetos 
//    L2CA       L2 triggers after all vetos 
//    LMCB       LM triggers before any vetos 
//    LMCA       LM triggers after all vetos 
//
//////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>

#include "AliLog.h"
#include "AliTriggerScalersESD.h"

using std::endl;
using std::cout;
ClassImp( AliTriggerScalersESD )

//_____________________________________________________________________________
AliTriggerScalersESD::AliTriggerScalersESD(): 
  TObject(),
  fClassIndex(0),
  fLOCB(0),     
  fLOCA(0),     
  fL1CB(0),     
  fL1CA(0),     
  fL2CB(0),     
  fL2CA(0),      
  fLMCB(0),     
  fLMCA(0)      
{
  // Default constructor
}

//_____________________________________________________________________________
AliTriggerScalersESD::AliTriggerScalersESD( UChar_t classIndex, ULong64_t LOCB, ULong64_t LOCA,        
                                      ULong64_t L1CB, ULong64_t L1CA, ULong64_t L2CB, ULong64_t L2CA ):   
  TObject(),
  fClassIndex( classIndex ),
  fLOCB(LOCB),     
  fLOCA(LOCA),     
  fL1CB(L1CB),     
  fL1CA(L1CA),     
  fL2CB(L2CB),     
  fL2CA(L2CA),      
  fLMCB(0), 
  fLMCA(0)      
{
  // Default constructor for L0,L1,L2 levels
}
//_____________________________________________________________________________
AliTriggerScalersESD::AliTriggerScalersESD( UChar_t classIndex, ULong64_t LOCB, ULong64_t LOCA,        
                                      ULong64_t L1CB, ULong64_t L1CA, ULong64_t L2CB, ULong64_t L2CA, ULong64_t LMCB, ULong64_t LMCA ):   
  TObject(),
  fClassIndex( classIndex ),
  fLOCB(LOCB),     
  fLOCA(LOCA),     
  fL1CB(L1CB),     
  fL1CA(L1CA),     
  fL2CB(L2CB),     
  fL2CA(L2CA),      
  fLMCB(LMCB),     
  fLMCA(LMCA)      
{
  // Constructor for LM,L0,L1,L2 levels
}
//_____________________________________________________________________________
AliTriggerScalersESD::AliTriggerScalersESD(UChar_t classIndex,ULong64_t* s64):
TObject(),
fClassIndex( classIndex ),
fLOCB(s64[0]),     
fLOCA(s64[1]),     
fL1CB(s64[2]),     
fL1CA(s64[3]),     
fL2CB(s64[4]),     
fL2CA(s64[5]),
fLMCB(0),
fLMCA(0)
{
 // Construct from array
}
//_____________________________________________________________________________
AliTriggerScalersESD::AliTriggerScalersESD(const AliTriggerScalersESD& scal ):   
  TObject(scal),
  fClassIndex( scal.fClassIndex ),
  fLOCB(scal.fLOCB),     
  fLOCA(scal.fLOCA),     
  fL1CB(scal.fL1CB),     
  fL1CA(scal.fL1CA),     
  fL2CB(scal.fL2CB),     
  fL2CA(scal.fL2CA),      
  fLMCB(scal.fLMCB),     
  fLMCA(scal.fLMCA)      
{
  // Copy constructor
}
//_____________________________________________________________________________
AliTriggerScalersESD& AliTriggerScalersESD::operator=(const AliTriggerScalersESD& scal)
{
if(&scal == this) return *this;
((TObject *)this)->operator=(scal);

  fClassIndex = scal.fClassIndex;
  fLOCB = scal.fLOCB;     
  fLOCA = scal.fLOCA;     
  fL1CB = scal.fL1CB;     
  fL1CA = scal.fL1CA;     
  fL2CB = scal.fL2CB;     
  fL2CA = scal.fL2CA;
  fLMCB = scal.fLMCB;     
  fLMCA = scal.fLMCA;

return *this;
}
//____________________________________________________________________________
void AliTriggerScalersESD::GetAllScalers(ULong64_t *scalers) const
{
 scalers[0]=fLOCB;
 scalers[1]=fLOCA;
 scalers[2]=fL1CB;
 scalers[3]=fL1CA;
 scalers[4]=fL2CB;
 scalers[5]=fL2CA;
}
void AliTriggerScalersESD::GetAllScalersM012(ULong64_t *scalers) const
{
 scalers[2]=fLOCB;
 scalers[3]=fLOCA;
 scalers[4]=fL1CB;
 scalers[5]=fL1CA;
 scalers[6]=fL2CB;
 scalers[7]=fL2CA;
 scalers[0]=fLMCB;
 scalers[1]=fLMCA;
}
//_____________________________________________________________________________
void AliTriggerScalersESD::Print( const Option_t* ) const
{
   // Print
  cout << "Trigger Scalers for Class: " << (Int_t)fClassIndex << endl;
  cout << "  LOCB: " << fLOCB << " LOCA: " << fLOCA; //<< endl;
  cout << "  L1CB: " << fL1CB << " L1CA: " << fL1CA; //<< endl;
  cout << "  L2CB: " << fL2CB << " L2CA: " << fL2CA << endl;
  cout << "  LMCB: " << fLMCB << " LMCA: " << fLMCA << endl;
}
