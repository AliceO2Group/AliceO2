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
/// This is a class for containing T0 time corrected by SPD vertex and amplitude
///
///////////////////////////////////////////////////////////////////////////////

#include "AliESDTZEROfriend.h"

ClassImp(AliESDTZEROfriend)

//_____________________________________________________________________________
AliESDTZEROfriend::AliESDTZEROfriend():
  TObject()
{
  for(int i = 0;i<24;i++)fT0time[i] = fT0ampQTC[i] = fT0ampLEDminCFD[i] = 0;
} 
AliESDTZEROfriend::AliESDTZEROfriend(const AliESDTZEROfriend &tzero ) :
  TObject(tzero)
{
  // copy constuctor
  for(int i = 0;i<24;i++){
    fT0time[i] = tzero.fT0time[i]; 
    fT0ampQTC[i] = tzero.fT0ampQTC[i];
    fT0ampLEDminCFD[i] = tzero.fT0ampLEDminCFD[i];
  }
}

AliESDTZEROfriend& AliESDTZEROfriend::operator=(const AliESDTZEROfriend& tzero){
  // assigmnent operator
  if(this!=&tzero) {
    TObject::operator=(tzero);
    for(int i = 0;i<24;i++){
      fT0time[i] = tzero.fT0time[i]; 
      fT0ampQTC[i] = tzero.fT0ampQTC[i];
      fT0ampLEDminCFD[i] = tzero.fT0ampLEDminCFD[i];
    }
  } 
  return *this;
}

void AliESDTZEROfriend::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDTZEROfriend *robj = dynamic_cast<AliESDTZEROfriend*>(&obj);
  if(!robj)return; // not an AliESDTZEROfriend
  *robj = *this;

}
void AliESDTZEROfriend::Reset()
{
  // Reset the contents of the object
    for(int i = 0;i<24;i++)
      fT0time[i]= fT0ampQTC[i] = fT0ampLEDminCFD[i] =0 ;
    
}


