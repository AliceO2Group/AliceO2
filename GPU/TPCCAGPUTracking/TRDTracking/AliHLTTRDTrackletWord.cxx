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

/* $Id: AliHLTTRDTrackletWord.cxx 28397 2008-09-02 09:33:00Z cblume $ */

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  A tracklet word in HLT - adapted from AliTRDtrackletWord              //
//                                                                        //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "AliHLTTRDTrackletWord.h"
#include <new>


GPUd() AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(unsigned int trackletWord) :
  fId(-1),
  fHCId(-1),
  fTrackletWord(trackletWord)
{
}

GPUd() AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(unsigned int trackletWord, int hcid, int id) :
  fId(id),
  fHCId(hcid),
  fTrackletWord(trackletWord)
{
}

GPUd() AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliHLTTRDTrackletWord &rhs) :
  fId(rhs.fId),
  fHCId(rhs.fHCId),
  fTrackletWord(rhs.fTrackletWord)
{
}

#ifdef GPUCA_ALIROOT_LIB
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"


AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletWord &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletMCM &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
}

AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliTRDtrackletMCM &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

#endif

GPUd() AliHLTTRDTrackletWord::~AliHLTTRDTrackletWord()
{

}

GPUd() AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliHLTTRDTrackletWord &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

GPUd() int AliHLTTRDTrackletWord::GetYbin() const {
  // returns (signed) value of Y
  if (fTrackletWord & 0x1000) {
    return -((~(fTrackletWord-1)) & 0x1fff);
  }
  else {
    return (fTrackletWord & 0x1fff);
  }
}

GPUd() int AliHLTTRDTrackletWord::GetdY() const
{
  // returns (signed) value of the deflection length
  if (fTrackletWord & (1 << 19)) {
    return -((~((fTrackletWord >> 13) - 1)) & 0x7f);
  }
  else {
    return ((fTrackletWord >> 13) & 0x7f);
  }
}
