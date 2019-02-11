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

// *
// *
// *
// * this class defines the TOF object to be stored
// * in OCDB on a run-by-run basis in order to have the status
// * of TOF trigger inputs. it stores 32 bit masks for each crate
// * 
// *
// *
// *

#include "AliTOFTriggerMask.h"

Int_t AliTOFTriggerMask::fPowerMask[24];

ClassImp(AliTOFTriggerMask)
//_________________________________________________________

AliTOFTriggerMask::AliTOFTriggerMask() :
  TObject(),
  fTriggerMask()
{
  /*
   * default constructor
   */

  for (Int_t iddl = 0; iddl < 72; iddl++) fTriggerMask[iddl] = 0;

  fPowerMask[0] = 1;
  for(Int_t i=1;i <= 23;i++){
      fPowerMask[i] = fPowerMask[i-1]*2;
  }

}

//_________________________________________________________

AliTOFTriggerMask::~AliTOFTriggerMask()
{
  /*
   * default destructor
   */

}

//_________________________________________________________

AliTOFTriggerMask::AliTOFTriggerMask(const AliTOFTriggerMask &source) :
  TObject(source),
  fTriggerMask()
{
  /*
   * copy constructor
   */

  for (Int_t iddl = 0; iddl < 72; iddl++) fTriggerMask[iddl] = source.fTriggerMask[iddl];
}

//_________________________________________________________

AliTOFTriggerMask &
AliTOFTriggerMask::operator=(const AliTOFTriggerMask &source)
{
  /*
   * operator=
   */

  if (this == &source) return *this;
  TObject::operator=(source);
  
  for (Int_t iddl = 0; iddl < 72; iddl++) fTriggerMask[iddl] = source.fTriggerMask[iddl];

  return *this;
}

//_________________________________________________________

void
AliTOFTriggerMask::SetTriggerMaskArray(UInt_t *array)
{
  /*
   * set trigger mask array
   */

  for (Int_t iddl = 0; iddl < 72; iddl++) fTriggerMask[iddl] = array[iddl];
}
//_________________________________________________________

Int_t AliTOFTriggerMask::GetNumberMaxiPadOn() {
  Int_t n=0;
  for(Int_t j=0;j<72;j++) 
    for(Int_t i=22;i>=0;i--) 
      n += (fTriggerMask[j]%fPowerMask[i+1])/fPowerMask[i];
  return n;
};
//_________________________________________________________
void AliTOFTriggerMask::SetON(Int_t icrate,Int_t ich){
  if(ich < 24 && icrate < 72 && !IsON(icrate,ich)) fTriggerMask[icrate] += fPowerMask[ich];
}
//_________________________________________________________
Bool_t AliTOFTriggerMask::IsON(Int_t icrate,Int_t ich){
  if(ich < 24 && icrate < 72) return (fTriggerMask[icrate] & fPowerMask[ich]);
  else return kFALSE;
}
//_________________________________________________________

TH2F *AliTOFTriggerMask::GetHistoMask() {
  TH2F *h = new TH2F("hTOFTriggerMask","TOF trigger mask;crate;MaxiPad",72,0,72,23,0,23);
  for(Int_t j=0;j<72;j++) 
    for(Int_t i=22;i>=0;i--) 
      h->SetBinContent(j+1,i+1,(fTriggerMask[j]%fPowerMask[i+1])/fPowerMask[i]);
  return h;
};
//_________________________________________________________
void AliTOFTriggerMask::ResetMask() {
  for (Int_t iddl = 0; iddl < 72; iddl++) fTriggerMask[iddl] = 0;
}
