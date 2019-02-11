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

/*
 

 

Author: R. GUERNANE LPSC Grenoble CNRS/IN2P3
*/

#include "AliESDCalofriend.h"
#include "AliLog.h"

#include "TArrayI.h"
#include "Riostream.h"
#include <cstdlib>

ClassImp(AliESDCalofriend)

namespace
{
  const Int_t kNSamples = 64; //
}

//_______________
AliESDCalofriend::AliESDCalofriend() : TObject(),
fNEntries(0),
fCurrent(-1),
fId(0x0),
fType(0x0),
fNSamples(0x0),
fSamples(new TArrayI())
{
  // Def ctor
}

//_______________
AliESDCalofriend::AliESDCalofriend(const AliESDCalofriend& src) : TObject(src),
fNEntries(0),
fCurrent(-1),
fId(0x0),
fType(0x0),
fNSamples(0x0),
fSamples(new TArrayI())
{
  // Copy ctor
  src.Copy(*this);
}

//_______________
AliESDCalofriend::~AliESDCalofriend()
{
  //
  if (fNEntries) DeAllocate();
  
  delete fSamples; fSamples = 0x0;
}

//_______________
void AliESDCalofriend::DeAllocate()
{
  //
  delete [] fId;       fId       = 0x0;
  delete [] fType;     fType     = 0x0;     
  delete [] fNSamples; fNSamples = 0x0;
  
  fNEntries =  0;
  fCurrent  = -1;
  
  fSamples->Reset();
}

//_______________
AliESDCalofriend& AliESDCalofriend::operator=(const AliESDCalofriend& src)
{
  //
  if (this != &src) src.Copy(*this);
  
  return *this;
}

//_______________
void AliESDCalofriend::Copy(TObject &obj) const 
{	
  //
  TObject::Copy(obj);
  
  AliESDCalofriend& dest = static_cast<AliESDCalofriend&>(obj);
  
  if (dest.fNEntries) dest.DeAllocate();
  
  dest.Allocate(fNEntries);
  
  for (Int_t i = 0; i < fNEntries; i++) {
    Int_t samples[kNSamples];
    for (Int_t j = 0; j < kNSamples; j++) samples[j] = fSamples->At(kNSamples * i + j);
    
    dest.Add(fId[i], fType[i], fNSamples[i], samples);
  }	
}

//_______________
void AliESDCalofriend::Allocate(Int_t size)
{
  //
  if (!size) return;
  
  fNEntries = size;
  
  fId       = new Int_t[fNEntries];
  fType     = new Int_t[fNEntries];
  fNSamples = new Int_t[fNEntries];
  
  for (Int_t i = 0; i < fNEntries; i++) {
    fId[i]       = -1;
    fType[i]     = -1;
    fNSamples[i] = 0;
  }
  
  fSamples->Set(fNEntries * kNSamples);
  fCurrent = -1;
}

//_______________
Bool_t AliESDCalofriend::Add(Int_t id, Int_t type, Int_t nsamples, Int_t samples[])
{
  //
  fCurrent++;
  
        fId[fCurrent] = id;
      fType[fCurrent] = type;
  fNSamples[fCurrent] = nsamples;
  
  if (nsamples > kNSamples) {
    AliError(Form("Should not have more than %d samples",kNSamples));
    return kFALSE;
  }
  
//   printf("curr: %d nsamples: %d fNSamples: %d\n",fCurrent,nsamples,fNSamples[fCurrent]);
  
  for (Int_t i = 0; i < fNSamples[fCurrent]; i++) {
    fSamples->AddAt(samples[i], kNSamples * fCurrent + i);
//     printf("Adding value %d at %d\n",samples[i], kNSamples * fCurrent + i);
  }
  
//   for (int i=0;fSamples->GetSize();i++) {
//     printf("array content at %d is %d\n",i,fSamples->GetAt(i));
//   }
  
  
  fDict.insert(std::make_pair(std::make_pair(id,type), fCurrent));
  
  return kTRUE;
}

//_______________
Bool_t AliESDCalofriend::Next()
{
  //
  if (fCurrent >= fNEntries - 1 || !fNEntries) return kFALSE;
  
  fCurrent++;
  
  return kTRUE;
}

//_______________
void AliESDCalofriend::GetId(Int_t& idx) const
{
  //
  if (fCurrent == -1) {idx = -1; return;}
  
  idx = fId?fId[fCurrent]:0;
}

//_______________
Int_t AliESDCalofriend::GetId() const
{
  //
  if (fCurrent == -1) return -1;
  
  return fId?fId[fCurrent]:0;
}

//_______________
void AliESDCalofriend::GetType(Int_t& type) const
{
  //
  if (fCurrent == -1) {type = -1; return;}
  
  type = fType?fType[fCurrent]:0;
}

//_______________
Int_t AliESDCalofriend::GetType() const
{
  //
  if (fCurrent == -1) return -1;
  
  return fType?fType[fCurrent]:0;
}

//_______________
void AliESDCalofriend::GetEntry(Int_t id, Int_t type)
{
  //
  if (!fNEntries || fDict.find(std::make_pair(id,type)) == fDict.end() ) {
    AliError("No entry!!!");
    fCurrent = -1;
  }

  fCurrent = fDict[std::make_pair(id,type)];
}

//_______________
void AliESDCalofriend::GetNSamples(Int_t& ntimes) const
{
  //
  if (fCurrent == -1) return;
  
  ntimes = fNSamples?fNSamples[fCurrent]:0;
}

//_______________
Int_t AliESDCalofriend::GetNSamples() const
{
  //
  if (fCurrent == -1) return 0;
  
  return fNSamples?fNSamples[fCurrent]:0;
}

//_______________
void AliESDCalofriend::GetSamples(Int_t samples[]) const
{
  //
  if (fCurrent == -1) return;
  
  if (fNSamples && fSamples) {
    for (Int_t i = 0; i < fNSamples[fCurrent]; i++) samples[i] = fSamples->At(kNSamples * fCurrent + i);
  }
}

//_______________
void AliESDCalofriend::Print(const Option_t* /*opt*/) const
{
  //
  if (fCurrent == -1) return;
  if (!fId)           return;
  if (!fType)         return;
  if (!fNSamples)     return;
  if (!fSamples)      return;
  
  printf("============\n");
  printf("\t(ID: %5d TYPE: %d)\n", fId[fCurrent], fType[fCurrent]);
  printf("\t%d SAMPLES# (", fNSamples[fCurrent]); 
  for (Int_t i = 0; i < fNSamples[fCurrent]; i++) printf("%2d ",fSamples->At(kNSamples * fCurrent + i));
  printf(")\n");
}



