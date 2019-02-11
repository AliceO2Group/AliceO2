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

/* $Id$ */

//____________________________________________________________________
//                                                                          
// Base class for caches of per-strip information.
// This is used to index a strip. 
// Data stored depends on derived class. 
// This class provides some common infra-structure.
// Derived classes sould define Reset, and operator(). 
//
#include "AliFMDMap.h"		// ALIFMDMAP_H
#include "AliLog.h"
//#include <TClass.h>
//#include <TBuffer.h>
#include <TFile.h>
#include <TList.h>
#include <TStreamerInfo.h>

//____________________________________________________________________
ClassImp(AliFMDMap)
#if 0
  ; // This is here to keep Emacs for indenting the next line
#endif

//____________________________________________________________________
AliFMDMap::AliFMDMap(UShort_t maxDet, 
		     UShort_t maxRing, 
		     UShort_t maxSec, 
		     UShort_t maxStr)
  : fMaxDetectors(maxDet), 
    fMaxRings(maxRing), 
    fMaxSectors(maxSec), 
    fMaxStrips(maxStr)
{
  // Construct a map
  //
  // Parameters:
  //     maxDet       Maximum # of detectors
  //     maxRinf      Maximum # of rings
  //     maxSec       Maximum # of sectors
  //     maxStr       Maximum # of strips
  SetBit(kNeedUShort, kFALSE);
}

//____________________________________________________________________
AliFMDMap::AliFMDMap(const AliFMDMap& other)
  : TObject(other), 
    fMaxDetectors(other.fMaxDetectors), 
    fMaxRings(other.fMaxRings),
    fMaxSectors(other.fMaxSectors),
    fMaxStrips(other.fMaxStrips)
{
  SetBit(kNeedUShort, other.TestBit(kNeedUShort));
}

//____________________________________________________________________
void
AliFMDMap::CheckNeedUShort(TFile* file) 
{
  if (!file) return;
  TObject* o = file->GetStreamerInfoList()->FindObject("AliFMDMap");
  if (!o) return;
  TStreamerInfo* info = static_cast<TStreamerInfo*>(o);
  if (info->GetClassVersion() == 2) SetBit(kNeedUShort);
}
//____________________________________________________________________
void
AliFMDMap::Index2CoordsOld(Int_t     idx, 
			   UShort_t& det, 
			   Char_t&   ring, 
			   UShort_t& sec, 
			   UShort_t& str) const
{
  UShort_t rng;
  str  = idx % fMaxStrips;
  sec  = (idx / fMaxStrips) % fMaxSectors;
  rng  = (idx / fMaxStrips / fMaxSectors) % fMaxRings;
  det  = (idx / fMaxStrips / fMaxSectors / fMaxRings) % fMaxDetectors + 1;
  ring = (rng == 0 ? 'I' : 'O');
}

//____________________________________________________________________
void
AliFMDMap::Index2Coords(Int_t     idx, 
			UShort_t& det, 
			Char_t&   ring, 
			UShort_t& sec, 
			UShort_t& str) const
{
  UShort_t nStr;
  Int_t    i   = idx;
  if      (i >= kFMD3Base)  { det  = 3;  i -= kFMD3Base; }
  else if (i >= kFMD2Base)  { det  = 2;  i -= kFMD2Base; }
  else                      { det  = 1;  i -= kFMD1Base; } 
  if      (i >= kBaseOuter) { ring = 'O';i -= kBaseOuter; nStr = kNStripOuter; }
  else                      { ring = 'I';                 nStr = kNStripInner; }
  sec  = i / nStr;
  str  = i % nStr;
}

//____________________________________________________________________
void
AliFMDMap::CalcCoords(Int_t     idx, 
		      UShort_t& det, 
		      Char_t&   ring, 
		      UShort_t& sec, 
		      UShort_t& str) const
{
  if (fMaxDetectors == 0) {
    Index2Coords(idx, det, ring, sec, str);
  }
  else {
    Index2CoordsOld(idx, det, ring, sec, str);
  }
}

//____________________________________________________________________
Int_t 
AliFMDMap::Coords2IndexOld(UShort_t det, Char_t ring, UShort_t sec, 
			   UShort_t str) const
{
  // Check that the index supplied is OK.   Returns true index, or -1
  // on error. 
  if (det < 1) return -1;
  UShort_t ringi = (ring == 'I' ||  ring == 'i' ? 0 : 1);
  Int_t idx = 
    (str + fMaxStrips * (sec + fMaxSectors * (ringi + fMaxRings * (det-1))));
  if (TestBit(kNeedUShort)) idx = UShort_t(idx);
  if (idx < 0 || idx >= fMaxDetectors * fMaxRings * fMaxSectors * fMaxStrips) 
    return -1;
  return idx;
}

//____________________________________________________________________
Int_t 
AliFMDMap::Coords2Index(UShort_t det, Char_t ring, UShort_t sec, 
			UShort_t str) const
{
  // Check that the index supplied is OK.   Returns true index, or -1
  // on error. 
  UShort_t irg  = (ring == 'I' || ring == 'i' ? kInner : 
		   (ring == 'O' || ring == 'o' ? kOuter  : kOuter+1));
  if (irg > kOuter) return -1;
    
  Int_t idx = 0;
  switch (det) { 
  case 1: idx = kFMD1Base;  if (irg > 0) return -1; break;
  case 2: idx = kFMD2Base + irg * kBaseOuter; break;
  case 3: idx = kFMD3Base + irg * kBaseOuter; break;
  default: return -1;
  }
  UShort_t nSec = (irg == 0 ?  kNSectorInner :  kNSectorOuter);
  if (sec >= nSec) return -1;
  UShort_t nStr = (irg == 0 ? kNStripInner : kNStripOuter);
  if (str >= nStr) return -1;
  idx += nStr * sec + str;

  return idx;
}

//____________________________________________________________________
Int_t 
AliFMDMap::CheckIndex(UShort_t det, Char_t ring, UShort_t sec, 
		      UShort_t str) const
{
  // Check that the index supplied is OK.   Returns true index, or -1
  // on error. 
  if (fMaxDetectors == 0)
    return Coords2Index(det, ring, sec, str);
  return Coords2IndexOld(det, ring, sec, str);
}


    
//____________________________________________________________________
Int_t 
AliFMDMap::CalcIndex(UShort_t det, Char_t ring, UShort_t sec, UShort_t str) const
{
  // Calculate index into storage from arguments. 
  // 
  // Parameters: 
  //     det       Detector #
  //     ring      Ring ID
  //     sec       Sector # 
  //     str       Strip # 
  //
  // Returns appropriate index into storage 
  //
  Int_t idx = CheckIndex(det, ring, sec, str);
  if (idx < 0) {
    UShort_t ringi = (ring == 'I' ||  ring == 'i' ? 0 : 
		      (ring == 'O' || ring == 'o' ? 1 : 2));
    AliFatal(Form("Index FMD%d%c[%2d,%3d] out of bounds, "
		  "in particular the %s index ", 
		  det, ring, sec, str, 
		  (det > fMaxDetectors ? "Detector" : 
		   (ringi >= fMaxRings ? "Ring" : 
		    (sec >= fMaxSectors ? "Sector" : "Strip")))));
    return 0;
  }
  return idx;
}

#define INCOMP_OP(self, other, OP) do {					\
  AliWarning("Incompatible sized AliFMDMap");				\
  UShort_t maxDet = TMath::Min(self->MaxDetectors(), other.MaxDetectors()); \
  UShort_t maxRng = TMath::Min(self->MaxRings(),     other.MaxRings());	\
  UShort_t maxSec = TMath::Min(self->MaxSectors(),   other.MaxSectors()); \
  UShort_t maxStr = TMath::Min(self->MaxStrips(),    other.MaxStrips()); \
  for (UShort_t d = 1; d <= maxDet; d++) {				\
    UShort_t nRng = TMath::Min(UShort_t(d == 1 ? 1 : 2), maxRng);	\
    for (UShort_t q = 0; q < nRng; q++) {                               \
      Char_t   r    = (q == 0 ? 'I' : 'O');                             \
      UShort_t nSec = TMath::Min(UShort_t(q == 0 ?  20 :  40), maxSec);	\
      UShort_t nStr = TMath::Min(UShort_t(q == 0 ? 512 : 256), maxStr);	\
      for (UShort_t s = 0; s < nSec; s++) {				\
        for (UShort_t t = 0; t < nStr; t++) {				\
	  Int_t idx1 = self->CalcIndex(d, r, s, t);			\
	  Int_t idx2 = other.CalcIndex(d, r, s, t);			\
	  if (idx1 < 0 || idx2 < 0) {					\
	    AliWarning("Index out of bounds");				\
	    continue;							\
	  }								\
	  if (self->IsFloat())						\
	    self->AtAsFloat(idx1) OP other.AtAsFloat(idx2);		\
	  else if (self->IsInt())					\
	    self->AtAsInt(idx1) OP other.AtAsInt(idx2);			\
	  else if (self->IsUShort())					\
	    self->AtAsUShort(idx1) OP other.AtAsUShort(idx2);		\
	  else if (self->IsBool())					\
	    self->AtAsBool(idx1) OP other.AtAsBool(idx2);		\
	}								\
      }									\
    }									\
  }									\
  } while (false)

#define COMP_OP(self,other,OP) do {					\
    for (Int_t i = 0; i < self->MaxIndex(); i++) {			\
      if (self->IsFloat())						\
	self->AtAsFloat(i) OP other.AtAsFloat(i);			\
      else if (self->IsInt())						\
	self->AtAsInt(i) OP other.AtAsInt(i);				\
      else if (self->IsUShort())					\
	self->AtAsUShort(i) OP other.AtAsUShort(i);			\
      else if (self->IsBool())						\
	self->AtAsBool(i) OP other.AtAsBool(i);				\
    } } while (false)

//__________________________________________________________
AliFMDMap&
AliFMDMap::operator*=(const AliFMDMap& other)
{
  // Right multiplication assignment operator 
  if(fMaxDetectors!= other.fMaxDetectors||
     fMaxRings    != other.fMaxRings    ||
     fMaxSectors  != other.fMaxSectors  ||
     fMaxStrips   != other.fMaxStrips   ||
     MaxIndex()   != other.MaxIndex()) {
    INCOMP_OP(this, other, *=);
    return *this;
  }
  COMP_OP(this, other, *=);
  return *this;
}

//__________________________________________________________
AliFMDMap&
AliFMDMap::operator/=(const AliFMDMap& other)
{
  // Right division assignment operator 
  if(fMaxDetectors!= other.fMaxDetectors||
     fMaxRings    != other.fMaxRings    ||
     fMaxSectors  != other.fMaxSectors  ||
     fMaxStrips   != other.fMaxStrips   ||
     MaxIndex()   != other.MaxIndex()) {
    INCOMP_OP(this, other, /=);
    return *this;
  }
  COMP_OP(this, other, /=);
  return *this;
}

//__________________________________________________________
AliFMDMap&
AliFMDMap::operator+=(const AliFMDMap& other)
{
  // Right addition assignment operator 
  if(fMaxDetectors!= other.fMaxDetectors||
     fMaxRings    != other.fMaxRings    ||
     fMaxSectors  != other.fMaxSectors  ||
     fMaxStrips   != other.fMaxStrips   ||
     MaxIndex()   != other.MaxIndex()) {
    INCOMP_OP(this, other, +=);
    return *this;
  }
  COMP_OP(this, other, +=);
  return *this;
}

//__________________________________________________________
AliFMDMap&
AliFMDMap::operator-=(const AliFMDMap& other)
{
  // Right subtraction assignment operator 
  if(fMaxDetectors!= other.fMaxDetectors||
     fMaxRings    != other.fMaxRings    ||
     fMaxSectors  != other.fMaxSectors  ||
     fMaxStrips   != other.fMaxStrips   ||
     MaxIndex()   != other.MaxIndex()) {
    INCOMP_OP(this, other, +=);
    return *this;
  }
  COMP_OP(this, other, +=);
  return *this;
}

//__________________________________________________________
Bool_t
AliFMDMap::ForEach(ForOne& algo) const
{
  // Assignment operator 
  Bool_t ret = kTRUE;
  for (Int_t i = 0; i < this->MaxIndex(); i++) { 
    UShort_t d, s, t;
    Char_t r;
    CalcCoords(i, d, r, s, t);
    Bool_t rr = kTRUE;
    if (IsFloat()) 
      rr = algo.operator()(d, r, s, t, this->AtAsFloat(i));
    else if (IsInt()) 
      rr = algo.operator()(d, r, s, t, this->AtAsInt(i));
    else if (IsUShort()) 
      rr = algo.operator()(d, r, s, t, this->AtAsUShort(i));
    else if (IsBool()) 
      rr = algo.operator()(d, r, s, t, this->AtAsBool(i));
    if (!rr) {
      ret = kFALSE;
      break;
    }
  }
  return ret;
}

//__________________________________________________________
void
AliFMDMap::Print(Option_t* option) const
{
  // Print contents of map
  if (!option || option[0] == '\0') TObject::Print();
  Printer p(option);
  ForEach(p);
  printf("\n");
}

//===================================================================
AliFMDMap::Printer::Printer(const char* format)
  : fFormat(format), fOldD(0), fOldR('-'), fOldS(1024) 
{}

//___________________________________________________________________
AliFMDMap::Printer::Printer(const Printer& p) 
  : AliFMDMap::ForOne(p),
    fFormat(p.fFormat), 
    fOldD(p.fOldD), 
    fOldR(p.fOldR), 
    fOldS(p.fOldS) 
{}
//___________________________________________________________________
void
AliFMDMap::Printer::PrintHeadings(UShort_t d, Char_t r, UShort_t s, UShort_t t) 
{
  if (d != fOldD) { 
    fOldD = d;
    fOldR = '-';
    if (d != 0) printf("\n");
    printf("FMD%d", fOldD);
  }
  if (r != fOldR) {
    fOldR = r;
    fOldS = 1024;
    printf("\n %s ring", (r == 'I' ? "Inner" : "Outer"));
  }
  if (s != fOldS) { 
    fOldS = s;
    printf("\n  Sector %2d", fOldS);
  }
  if (t % 4 == 0) printf("\n   %3d-%3d ", t, t+3);
}
//___________________________________________________________________
Bool_t
AliFMDMap::Printer::operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			       Float_t m)
{
  PrintHeadings(d, r, s, t);
  printf(fFormat, m);
  return kTRUE;
}
//___________________________________________________________________
Bool_t
AliFMDMap::Printer::operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			       Int_t m)
{
  PrintHeadings(d, r, s, t);
  printf(fFormat, m);
  return kTRUE;
}
//___________________________________________________________________
Bool_t
AliFMDMap::Printer::operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			       UShort_t m)
{
  PrintHeadings(d, r, s, t);
  printf(fFormat, m);
  return kTRUE;
}
//___________________________________________________________________
Bool_t
AliFMDMap::Printer::operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			       Bool_t m)
{
  PrintHeadings(d, r, s, t);
  printf(fFormat, int(m));
  return kTRUE;
}

#if 0
//___________________________________________________________________
void AliFMDMap::Streamer(TBuffer &R__b)
{
  // Stream an object of class AliFMDMap.
  // This is overridden so that we can know the version of the object
  // that we are reading in.  In this way, we can fix problems that
  // might occur in the class. 
  if (R__b.IsReading()) {
    // read the class version from the buffer
    UInt_t R__s, R__c;
    Version_t version = R__b.ReadVersion(&R__s, &R__c, this->Class());
    TFile *file = (TFile*)R__b.GetParent();
    if (file && file->GetVersion() < 30000) version = -1; 
    AliFMDMap::Class()->ReadBuffer(R__b, this, version, R__s, R__c);
    if (version == 2) SetBit(kNeedUShort);
  } else {
    AliFMDMap::Class()->WriteBuffer(R__b, this);
  }
}
#endif

//___________________________________________________________________
//
// EOF
//
