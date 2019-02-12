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

/* $Id: AliGPUTRDTrackletWord.cxx 28397 2008-09-02 09:33:00Z cblume $ */

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  A tracklet word for GPU tracker- adapted from AliTRDtrackletWord      //
//                                                                        //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "AliGPUTRDTrackletWord.h"
#ifndef __OPENCL__
#include <new>
#endif


GPUd() AliGPUTRDTrackletWord::AliGPUTRDTrackletWord(unsigned int trackletWord) :
  fId(-1),
  fHCId(-1),
  fTrackletWord(trackletWord)
{
}

GPUd() AliGPUTRDTrackletWord::AliGPUTRDTrackletWord(unsigned int trackletWord, int hcid, int id) :
  fId(id),
  fHCId(hcid),
  fTrackletWord(trackletWord)
{
}

#ifndef GPUCA_GPUCODE_DEVICE
#ifdef GPUCA_ALIROOT_LIB
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"

AliGPUTRDTrackletWord::AliGPUTRDTrackletWord(const AliTRDtrackletWord &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
}

AliGPUTRDTrackletWord::AliGPUTRDTrackletWord(const AliTRDtrackletMCM &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
}

AliGPUTRDTrackletWord& AliGPUTRDTrackletWord::operator=(const AliTRDtrackletMCM &rhs)
{
  this->~AliGPUTRDTrackletWord();
  new(this) AliGPUTRDTrackletWord(rhs);
  return *this;
}

#endif //GPUCA_ALIROOT_LIB
#endif //GPUCA_GPUCODE_DEVICE

GPUd() int AliGPUTRDTrackletWord::GetYbin() const {
  // returns (signed) value of Y
  if (fTrackletWord & 0x1000) {
    return -((~(fTrackletWord-1)) & 0x1fff);
  }
  else {
    return (fTrackletWord & 0x1fff);
  }
}

GPUd() int AliGPUTRDTrackletWord::GetdY() const
{
  // returns (signed) value of the deflection length
  if (fTrackletWord & (1 << 19)) {
    return -((~((fTrackletWord >> 13) - 1)) & 0x7f);
  }
  else {
    return ((fTrackletWord >> 13) & 0x7f);
  }
}
