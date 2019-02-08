#ifndef ALIGPUTRDTRACKLETWORD_H
#define ALIGPUTRDTRACKLETWORD_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: AliGPUTRDTrackletWord.h 27496 2008-07-22 08:35:45Z cblume $ */

//-----------------------------------
//
// TRD tracklet word (as from FEE)
// only 32-bit of information + detector ID
//
//----------------------------------

class AliTRDtrackletWord;
class AliTRDtrackletMCM;

#include "AliGPUTPCDef.h"

class AliGPUTRDTrackletWord {
 public:
  GPUd() AliGPUTRDTrackletWord(unsigned int trackletWord = 0);
  GPUd() AliGPUTRDTrackletWord(unsigned int trackletWord, int hcid, int id);
  GPUdDefault() AliGPUTRDTrackletWord(const AliGPUTRDTrackletWord &rhs) CON_DEFAULT;
  GPUdDefault() AliGPUTRDTrackletWord& operator=(const AliGPUTRDTrackletWord &rhs) CON_DEFAULT;
  GPUdDefault() ~AliGPUTRDTrackletWord() CON_DEFAULT;
#ifndef GPUCA_GPUCODE_DEVICE
  AliGPUTRDTrackletWord(const AliTRDtrackletWord &rhs);
  AliGPUTRDTrackletWord(const AliTRDtrackletMCM &rhs);
  AliGPUTRDTrackletWord& operator=(const AliTRDtrackletMCM &rhs);
#endif

  // ----- Override operators < and > to enable tracklet sorting by HCId -----
  GPUd() bool operator<(const AliGPUTRDTrackletWord &t) const { return (GetHCId() < t.GetHCId()); }
  GPUd() bool operator>(const AliGPUTRDTrackletWord &t) const { return (GetHCId() > t.GetHCId()); }
  GPUd() bool operator<=(const AliGPUTRDTrackletWord &t) const { return (GetHCId() < t.GetHCId()) || (GetHCId() == t.GetHCId()); }

  // ----- Getters for contents of tracklet word -----
  GPUd() int GetYbin() const;
  GPUd() int GetdY() const;
  GPUd() int GetZbin() const { return ((fTrackletWord >> 20) & 0xf); }
  GPUd() int GetPID() const { return ((fTrackletWord >> 24) & 0xff); }

  GPUd() int GetId() const { return fId; }

  // ----- Getters for offline corresponding values -----
  GPUd() double GetPID(int /* is */) const { return (double) GetPID()/256.f; }
  GPUd() int GetDetector() const { return fHCId / 2; }
  GPUd() int GetHCId() const { return fHCId; }
  GPUd() float GetdYdX() const { return (GetdY() * 140e-4f / 3.f); }
  GPUd() float GetY() const { return (GetYbin() * 160e-4f); }
  GPUd() unsigned int GetTrackletWord() const { return fTrackletWord; }

  GPUd() void SetTrackletWord(unsigned int trackletWord) { fTrackletWord = trackletWord; }
  GPUd() void SetDetector(int id) { fHCId = 2 * id + (GetYbin() < 0 ? 0 : 1); }
  GPUd() void SetId(int id) { fId = id; }
  GPUd() void SetHCId(int id) { fHCId = id; }

 protected:
  int fId;                      // index in tracklet array
  int fHCId;                    // half-chamber ID
  unsigned int fTrackletWord;   // tracklet word: PID | Z | deflection length | Y
                                //          bits:   8   4            7          13

};

#endif
