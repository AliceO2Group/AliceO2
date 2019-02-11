/**************************************************************************
 * Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
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

#include "AliRefArray.h"
#include <string.h>

ClassImp(AliRefArray)

//____________________________________________________________________
AliRefArray::AliRefArray() : fNElems(0),fRefSize(0),fElems(0),fRefInd(0),fRefBuff(0)
{
  // default constructor
}
 
//____________________________________________________________________
AliRefArray::AliRefArray(UInt_t nelem,UInt_t depth) : 
  TObject(),fNElems(nelem),fRefSize(depth),fElems(0),fRefInd(0),fRefBuff(0)
{
  // constructor
  fNElems = nelem;
  // create array with nelem initial referres
  if (fNElems<1) fNElems = 1;
  fElems   = new Int_t[fNElems];   
  if (fRefSize>0) {
    fRefInd  = new UInt_t[fRefSize];  
    fRefBuff = new UInt_t[fRefSize];  
  }
  Reset();
  //
}
 
//____________________________________________________________________
AliRefArray::AliRefArray(const AliRefArray& src) :
  TObject(src),
  fNElems(src.fNElems),
  fRefSize(src.fRefSize),
  fElems(new Int_t[fNElems]),
  fRefInd(new UInt_t[fRefSize]),
  fRefBuff(new UInt_t[fRefSize])
{
  //
  // create a copy 
  //
  memcpy(fElems,src.fElems,fNElems*sizeof(Int_t));
  memcpy(fRefInd,src.fRefInd,fRefSize*sizeof(UInt_t));
  memcpy(fRefBuff,src.fRefBuff,fRefSize*sizeof(UInt_t));
}
 
//____________________________________________________________________
AliRefArray& AliRefArray::operator=(const AliRefArray& src)
{
  // create a copy with useful info (skip unused slots)
  if(&src != this) {
    TObject::operator=(src);
    fNElems = src.fNElems;
    fRefSize=0;
    if (fElems) delete[] fElems;
    if (fRefInd) delete[] fRefInd;
    if (fRefBuff) delete[] fRefBuff;
    fElems = 0;
    fRefInd = 0;
    fRefBuff = 0;
    if (src.fRefInd) {
      fRefSize = src.fRefInd[0];
      fRefInd  = new UInt_t[fRefSize];
      fRefBuff = new UInt_t[fRefSize];
      memcpy(fRefInd, src.fRefInd, fRefSize*sizeof(UInt_t));
      memcpy(fRefBuff,src.fRefBuff,fRefSize*sizeof(UInt_t));
    }
    if (fNElems) {
      fElems   = new Int_t[fNElems];   
      memcpy(fElems,src.fElems,fNElems*sizeof(Int_t));
    }
  }
  return *this;
  //
}

//____________________________________________________________________
AliRefArray::~AliRefArray() 
{
  // destructor
  delete[] fElems;
  delete[] fRefBuff;
  delete[] fRefInd;
}

//____________________________________________________________________
void AliRefArray::Expand(UInt_t size)
{
  // expand the size
  if (size<fNElems) {
    if (size>0) {printf("The size can be only increased\n");return;}
    else size = (fNElems<<2) + 1;
  }
  else if (size==fNElems) return;
  Int_t *tmpArr = new Int_t[size];
  memcpy(tmpArr,fElems,fNElems*sizeof(Int_t));
  memset(tmpArr+fNElems,0,(size-fNElems)*sizeof(UInt_t));
  delete[] fElems;
  fElems  = tmpArr;
  fNElems = size;
}

//____________________________________________________________________
void AliRefArray::Reset()
{
  // reset references
  if (fNElems) memset(fElems,0,fNElems*sizeof(Int_t));
  if (fRefSize) {
    memset(fRefInd,0,fRefSize*sizeof(UInt_t));
    memset(fRefBuff,0,fRefSize*sizeof(UInt_t));
    fRefInd[0] = 1;
  }
}

//____________________________________________________________________
void AliRefArray::ExpandReferences(Int_t addSize)
{
  // add extra slots
  if (addSize<3) addSize = 3;
  UInt_t oldSize = fRefSize;
  fRefSize += addSize;
  UInt_t*   buff = new UInt_t[fRefSize];
  UInt_t*   ind  = new UInt_t[fRefSize];
  if (fRefBuff) memcpy(buff, fRefBuff, oldSize*sizeof(UInt_t)); // copy current content
  if (fRefInd)  memcpy(ind,  fRefInd,  oldSize*sizeof(UInt_t));
  memset(buff+oldSize,0,addSize*sizeof(UInt_t));
  memset(ind +oldSize,0,addSize*sizeof(UInt_t));
  delete[] fRefBuff; fRefBuff = buff;
  delete[] fRefInd;  fRefInd  = ind;
  if (!oldSize) fRefInd[0] = 1;
}

//____________________________________________________________________
void AliRefArray::Print(Option_t*) const
{
  // reset references
  for (UInt_t i=0;i<fNElems;i++) {
    printf("Entry%4d: ",i);
    Int_t ref;
    if (!(ref=fElems[i])) {printf("None\n"); continue;}
    if (fElems[i]<0)      {printf("%d\n",-(1+ref));   continue;}
    do { printf("%d ",fRefBuff[ref]-1); }    while((ref=fRefInd[ref])); printf("\n");
  }
}

//____________________________________________________________________
void AliRefArray::AddReferences(UInt_t from, UInt_t *refs, UInt_t nref)
{
  // add nodes to the references of "from"
  if (nref==1) {AddReference(from, refs[0]); return;}
  if (!nref) return;
  //
  if (from>=fNElems) Expand(from+1);
  UInt_t chk = nref + (fElems[from]<0); // if <0, need to transfer to indices the only existing reference
  if      (!fRefInd) ExpandReferences(chk+1);
  else if ( fRefInd[0]+chk >= fRefSize ) ExpandReferences(chk);
  UInt_t &freeSlot = fRefInd[0];
  // if there is already single ref, transfer it to indices
  Int_t ref = fElems[from];
  if (ref<0) { fRefInd[freeSlot]=0; fRefBuff[freeSlot] = -ref; ref = fElems[from] = freeSlot++; }
  //
  while(fRefInd[ref]) ref=fRefInd[ref]; // find last index of last entry for cluster from
  if (fElems[from]) fRefInd[ref] = freeSlot;           // not a first entry, register it in the indices
  else              fElems[from] = freeSlot;           // first entry, register it in the refs
  for (UInt_t ir=0;ir<nref;ir++) {
    if (!ir && !fElems[from]) fElems[from] = freeSlot;
    else ref = fRefInd[ref] = freeSlot;
    fRefBuff[freeSlot++] = refs[ir]+1;
  }
}
