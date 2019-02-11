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
// This class which defines the trigger bunch-crossing mask
//
//
///////////////////////////////////////////////////////////////////////////////
#include <Riostream.h>
#include <TObjArray.h>
#include <TObjString.h>

#include "AliTriggerBCMask.h"
#include "AliLog.h"

using std::endl;
using std::cout;
ClassImp(AliTriggerBCMask)

//_____________________________________________________________________________
AliTriggerBCMask::AliTriggerBCMask():
  TNamed()
{
  // Default constructor
  for (Int_t i = 0; i < kNBytesPerBCMask; i++) fBCMask[i] = 0;
}

//_____________________________________________________________________________
AliTriggerBCMask::AliTriggerBCMask( TString & name ):
  TNamed( name, name )
{
  // Constructor
  for (Int_t i = 0; i < kNBytesPerBCMask; i++) fBCMask[i] = 255;
}

//_____________________________________________________________________________
AliTriggerBCMask::AliTriggerBCMask( TString & name, TString & mask ):
  TNamed( name, mask )
{
  // Constructor
  CreateMask(mask);
}

//_____________________________________________________________________________
AliTriggerBCMask::AliTriggerBCMask( const char* name, const char* mask ):
  TNamed( name, mask )
{
  // Constructor
  SetMask(mask);
}

//_____________________________________________________________________________
AliTriggerBCMask::~AliTriggerBCMask() 
{ 
  // Destructor
}
//_____________________________________________________________________________
AliTriggerBCMask::AliTriggerBCMask( const AliTriggerBCMask& mask ):
  TNamed( mask )
{
   // Copy constructor
  for (Int_t i = 0; i < kNBytesPerBCMask; i++) fBCMask[i] = mask.fBCMask[i];
}

//______________________________________________________________________________
AliTriggerBCMask& AliTriggerBCMask::operator=(const AliTriggerBCMask& mask)
{
   // AliTriggerBCMask assignment operator.

   if (this != &mask) {
      TNamed::operator=(mask);
      for (Int_t i = 0; i < kNBytesPerBCMask; i++) fBCMask[i] = mask.fBCMask[i];
   }
   return *this;
}

//_____________________________________________________________________________
Bool_t AliTriggerBCMask::GetMask( UShort_t index) const
{
  // Return true or false whenever the mask is active
  // for the bunch-crossing # = index
  UShort_t position = index/8;
  if (position >= kNBytesPerBCMask) return kFALSE;
  UChar_t offset = index%8;
  return (fBCMask[position] & (0x1 << offset));
}

//_____________________________________________________________________________
void AliTriggerBCMask::Print( const Option_t* opt) const
{
   // Print
  cout << "Trigger bunch-crossing mask:" << endl;
  cout << "  Name:                      " << GetName() << endl;
  cout << "  Mask:                      " << GetTitle() << endl;

  if (strcmp(opt,"bits") == 0) {
    cout << "  Bits:                      " << endl;
    for (UShort_t i = 0; i < kNBits; i++) {
      if (GetMask(i)) {
	cout << "1";
      }
      else {
	cout << "0";
      }
    }
    cout << endl;
  }
}

Bool_t AliTriggerBCMask::SetMask (const char *mask)
{
  // Wrapper used in pileup generators
  // Call directly CreateMask method.
  SetTitle(mask);
  return CreateMask(fTitle);
}

//_____________________________________________________________________________
Bool_t AliTriggerBCMask::CreateMask(TString &mask)
{
  // (re)creates the bc mask bit pattern
  // according to the bc string.
  // The string has the following syntax:
  //   "25L 25(2H2LH 3(23HL))"
  //        - H/h -> 1  L/l -> 0
  //        - spaces, new lines are white characters
  // The method returns kTRUE in case of successful
  // parsing and kFALSE otherwise.

  for (Int_t i = 0; i < kNBytesPerBCMask; i++) fBCMask[i] = 255;

  mask.ReplaceAll("("," ( ");
  mask.ReplaceAll(")"," ) ");
  mask.ReplaceAll("H"," H ");
  mask.ReplaceAll("h"," H ");
  mask.ReplaceAll("L"," L ");
  mask.ReplaceAll("l"," L ");
  TObjArray *tokens = mask.Tokenize(" \t");
  if (tokens->GetEntriesFast() == 0) {
    delete tokens;
    return kTRUE;
  }

  TBits bits(kNBits);
  Int_t index = 0, ibit = 0, level = 0;
  if ((!Bcm2Bits(tokens,index,bits,ibit,level)) ||
      (index != tokens->GetEntriesFast())) {
    AliError("Invalid bunch-crossing mask syntax. Empty mask produced.");
    delete tokens;
    return kFALSE;
  }

  delete tokens;

  if (ibit != kNBits) {
    AliError(Form("Incomplete bunch-crossing mask. Only the first %d bits are filled.",ibit));
    return kFALSE;
  }

  bits.Get(fBCMask);

  return kTRUE;
}

Bool_t AliTriggerBCMask::Bcm2Bits(TObjArray *tokens, Int_t &index, TBits &bits, Int_t &ibit, Int_t &level) const
{

  level++;
  Int_t repetion = 1;

  while(1) {
    if (index == tokens->GetEntriesFast()) {
      if (level > 1) {
	AliError("Missing )");
	return kFALSE;
      }
      break;
    }
    TString st = ((TObjString*)tokens->At(index))->String();
    if (st.CompareTo("H") == 0) {
      for (Int_t i = 0; i < repetion; i++) bits.SetBitNumber(ibit++,kTRUE);
      repetion = 1;
      index++;
    }
    else if (st.CompareTo("L") == 0) {
      for (Int_t i = 0; i < repetion; i++) bits.SetBitNumber(ibit++,kFALSE);
      repetion = 1;
      index++;
    }
    else if (st.IsDigit()) {
      repetion = st.Atoi();
      index++;
    }
    else if (st.CompareTo("(") == 0) {
      index++;
      Int_t ibit1 = ibit;
      if (!Bcm2Bits(tokens,index,bits,ibit,level)) {
	return kFALSE;
      }
      Int_t ibit2 = ibit;
      for (Int_t i = 0; i < (repetion-1); i++) {
	for (Int_t j = ibit1; j < ibit2; j++) {
	  bits.SetBitNumber(ibit++,bits.TestBitNumber(j));
	}
      }
      repetion = 1;
    }
    else if (st.CompareTo(")") == 0) {
      index++;
      if (level <= 1) {
	AliError("Incorrectly placed )");
	return kFALSE;
      }
      break;
    }
    else {
      AliError(Form("Incorrect BC mask field: %s",st.Data()));
      return kFALSE;
    }
  }
  level--;
  return kTRUE;

}
//_____________________________________________________________________________
UShort_t AliTriggerBCMask::GetNUnmaskedBCs() const
{
  UShort_t nBCs=0;
  for (Int_t i=0; i<kNBits; i++){
    if (!GetMask(i)) nBCs++;
  }
  return nBCs;
}
