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

#ifdef HLTCA_BUILD_ALIROOT_LIB
#include "AliTRDgeometry.h"
#include "AliTRDpadPlane.h"
#else
class AliTRDgeometry {};
#endif

class AliTRDtrackletWord;
class AliTRDtrackletMCM;

class AliHLTTRDTrackletWord {
 public:
  AliHLTTRDTrackletWord(unsigned int trackletWord = 0);
  AliHLTTRDTrackletWord(unsigned int trackletWord, int hcid, int id, int *label = 0x0);
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
  int GetYbin() const;
  int GetdY() const;
  int GetZbin() const { return ((fTrackletWord >> 20) & 0xf); }
  int GetPID() const { return ((fTrackletWord >> 24) & 0xff); }

  int GetROB() const;
  int GetMCM() const;

  int GetId() const { return fId; }
  const int* GetLabels() const { return fLabel; }
  int GetLabel(int i=0) const { return fLabel[i];}
  
  // ----- Getters for offline corresponding values -----
  bool CookPID() { return false; }
  double GetPID(int /* is */) const { return (double) GetPID()/256.; }
  int GetDetector() const { return fHCId / 2; }
  int GetHCId() const { return fHCId; }
  float GetdYdX() const { return (GetdY() * 140e-4 / 3.); }
#ifdef HLTCA_BUILD_ALIROOT_LIB
  float GetX() const { return fgGeo->GetTime0((fHCId%12)/2); }
  float GetY() const { return (GetYbin() * 160e-4); }
  float GetZ() const { return fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowPos(GetZbin()) -
      fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowSize(GetZbin())  * .5; }
  float GetLocalZ() const { return GetZ() - fgGeo->GetPadPlane((fHCId % 12) / 2, (fHCId/12) % 5)->GetRowPos((((fHCId/12) % 5) != 2) ? 8 : 6); }
#else
  float GetX() const { return 0; }
  float GetY() const { return 0; }
  float GetZ() const { return 0; }
  float GetLocalZ() const { return 0; }
#endif
  unsigned int GetTrackletWord() const { return fTrackletWord; }

  void SetTrackletWord(unsigned int trackletWord) { fTrackletWord = trackletWord; }
  void SetDetector(int id) { fHCId = 2 * id + (GetYbin() < 0 ? 0 : 1); }
  void SetId(int id) { fId = id; }
  void SetLabel(const int *label) { for (int i=3;i--;) fLabel[i] = label[i]; }
  void SetLabel(int i, int label) { fLabel[i] = label; }
  void SetHCId(int id) { fHCId = id; }

 protected:
  int fId;                      // index in tracklet array
  int fLabel[3];                // MC label
  int fHCId;                    // half-chamber ID
  unsigned int fTrackletWord;   // tracklet word: PID | Z | deflection length | Y
                                //          bits:   8   4            7          13
  static AliTRDgeometry *fgGeo; // pointer to TRD geometry for coordinate calculations

};

#endif
