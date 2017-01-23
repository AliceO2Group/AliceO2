#ifndef ALIHLTTRDTRACKLETWORD_H
#define ALIHLTTRDTRACKLETWORD_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: AliHLTTRDTrackletWord.h 27496 2008-07-22 08:35:45Z cblume $ */

//-----------------------------------
//
// TRD tracklet word (as from FEE)
// only 32-bit of information + detector ID
//
//----------------------------------

#include "AliTRDgeometry.h"
#include "AliTRDpadPlane.h"

class AliTRDtrackletWord;
class AliTRDtrackletMCM;

class AliHLTTRDTrackletWord {
 public:
  AliHLTTRDTrackletWord(UInt_t trackletWord = 0);
  AliHLTTRDTrackletWord(UInt_t trackletWord, Int_t hcid, Int_t id);
  AliHLTTRDTrackletWord(const AliHLTTRDTrackletWord &rhs);
  AliHLTTRDTrackletWord(const AliTRDtrackletWord &rhs);
  AliHLTTRDTrackletWord(const AliTRDtrackletMCM &rhs);
  ~AliHLTTRDTrackletWord();
  AliHLTTRDTrackletWord& operator=(const AliHLTTRDTrackletWord &rhs);
  AliHLTTRDTrackletWord& operator=(const AliTRDtrackletMCM &rhs);

  // ----- Override operators < and > to enable tracklet sorting by HCId -----
  bool operator<(const AliHLTTRDTrackletWord &t) const { return (GetHCId() < t.GetHCId()); }
  bool operator>(const AliHLTTRDTrackletWord &t) const { return (GetHCId() > t.GetHCId()); }

  // ----- Getters for contents of tracklet word -----
  Int_t GetYbin() const;
  Int_t GetdY() const;
  Int_t GetZbin() const { return ((fTrackletWord >> 20) & 0xf); }
  Int_t GetPID() const { return ((fTrackletWord >> 24) & 0xff); }

  Int_t GetROB() const;
  Int_t GetMCM() const;
  Int_t GetId() const { return fId; }

  // ----- Getters for offline corresponding values -----
  Bool_t CookPID() { return kFALSE; }
  Double_t GetPID(Int_t /* is */) const { return (Double_t) GetPID()/256.; }
  Int_t GetDetector() const { return fHCId / 2; }
  Int_t GetHCId() const { return fHCId; }
  Float_t GetdYdX() const { return (GetdY() * 140e-4 / 3.); }
  Float_t GetX() const { return fgGeo->GetTime0((fHCId%12)/2); }
  Float_t GetY() const { return (GetYbin() * 160e-4); }
  Float_t GetZ() const { return fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowPos(GetZbin()) -
      fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowSize(GetZbin())  * .5; }
  Float_t GetLocalZ() const { return GetZ() - fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowPos((((fHCId/12) % 5) != 2) ? 8 : 6); }

  UInt_t GetTrackletWord() const { return fTrackletWord; }

  void SetTrackletWord(UInt_t trackletWord) { fTrackletWord = trackletWord; }
  void SetDetector(Int_t id) { fHCId = 2 * id + (GetYbin() < 0 ? 0 : 1); }
  void SetId(Int_t id) { fId = id; }
  void SetHCId(Int_t id) { fHCId = id; }

 protected:
  Int_t fId;              // index in tracklet array
  Int_t fHCId;            // half-chamber ID
  UInt_t fTrackletWord;   // tracklet word: PID | Z | deflection length | Y
                          //          bits:   8   4            7          13
  static AliTRDgeometry *fgGeo;  // pointer to TRD geometry for coordinate calculations

};

#endif
