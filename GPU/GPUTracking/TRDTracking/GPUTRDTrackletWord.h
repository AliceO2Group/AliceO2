// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackletWord.h
/// \brief TRD Tracklet word for GPU tracker - 32bit tracklet info + half chamber ID + index

/// \author Ole Schmidt

#ifndef GPUTRDTRACKLETWORD_H
#define GPUTRDTRACKLETWORD_H

#include "GPUDef.h"

class AliTRDtrackletWord;
class AliTRDtrackletMCM;

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDTrackletWord
{
 public:
  GPUd() GPUTRDTrackletWord(unsigned int trackletWord = 0);
  GPUd() GPUTRDTrackletWord(unsigned int trackletWord, int hcid, int id);
  GPUdDefault() GPUTRDTrackletWord(const GPUTRDTrackletWord& rhs) CON_DEFAULT;
  GPUdDefault() GPUTRDTrackletWord& operator=(const GPUTRDTrackletWord& rhs) CON_DEFAULT;
  GPUdDefault() ~GPUTRDTrackletWord() CON_DEFAULT;
#ifndef GPUCA_GPUCODE_DEVICE
  GPUTRDTrackletWord(const AliTRDtrackletWord& rhs);
  GPUTRDTrackletWord(const AliTRDtrackletMCM& rhs);
  GPUTRDTrackletWord& operator=(const AliTRDtrackletMCM& rhs);
#endif

  // ----- Override operators < and > to enable tracklet sorting by HCId -----
  GPUd() bool operator<(const GPUTRDTrackletWord& t) const { return (GetHCId() < t.GetHCId()); }
  GPUd() bool operator>(const GPUTRDTrackletWord& t) const { return (GetHCId() > t.GetHCId()); }
  GPUd() bool operator<=(const GPUTRDTrackletWord& t) const { return (GetHCId() < t.GetHCId()) || (GetHCId() == t.GetHCId()); }

  // ----- Getters for contents of tracklet word -----
  GPUd() int GetYbin() const;
  GPUd() int GetdY() const;
  GPUd() int GetZbin() const { return ((mTrackletWord >> 20) & 0xf); }
  GPUd() int GetPID() const { return ((mTrackletWord >> 24) & 0xff); }

  GPUd() int GetId() const { return mId; }

  // ----- Getters for offline corresponding values -----
  GPUd() double GetPID(int /* is */) const { return (double)GetPID() / 256.f; }
  GPUd() int GetDetector() const { return mHCId / 2; }
  GPUd() int GetHCId() const { return mHCId; }
  GPUd() float GetdYdX() const { return (GetdY() * 140e-4f / 3.f); }
  GPUd() float GetY() const { return (GetYbin() * 160e-4f); }
  GPUd() unsigned int GetTrackletWord() const { return mTrackletWord; }

  GPUd() void SetTrackletWord(unsigned int trackletWord) { mTrackletWord = trackletWord; }
  GPUd() void SetDetector(int id) { mHCId = 2 * id + (GetYbin() < 0 ? 0 : 1); }
  GPUd() void SetId(int id) { mId = id; }
  GPUd() void SetHCId(int id) { mHCId = id; }

 protected:
  int mId;                    // index in tracklet array
  int mHCId;                  // half-chamber ID
  unsigned int mTrackletWord; // tracklet word: PID | Z | deflection length | Y
                              //          bits:   8   4            7          13
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDTRACKLETWORD_H
