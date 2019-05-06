// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackletWord.cxx
/// \author Ole Schmidt

#include "GPUTRDTrackletWord.h"
using namespace GPUCA_NAMESPACE::gpu;
#ifndef __OPENCL__
#include <new>
#endif

GPUd() GPUTRDTrackletWord::GPUTRDTrackletWord(unsigned int trackletWord) : mId(-1), mHCId(-1), mTrackletWord(trackletWord)
{
}

GPUd() GPUTRDTrackletWord::GPUTRDTrackletWord(unsigned int trackletWord, int hcid, int id) : mId(id), mHCId(hcid), mTrackletWord(trackletWord) {}

#ifndef GPUCA_GPUCODE_DEVICE
#ifdef GPUCA_ALIROOT_LIB
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"

GPUTRDTrackletWord::GPUTRDTrackletWord(const AliTRDtrackletWord& rhs) : mId(-1), mHCId(rhs.GetHCId()), mTrackletWord(rhs.GetTrackletWord())
{
}

GPUTRDTrackletWord::GPUTRDTrackletWord(const AliTRDtrackletMCM& rhs) : mId(-1), mHCId(rhs.GetHCId()), mTrackletWord(rhs.GetTrackletWord()) {}

GPUTRDTrackletWord& GPUTRDTrackletWord::operator=(const AliTRDtrackletMCM& rhs)
{
  this->~GPUTRDTrackletWord();
  new (this) GPUTRDTrackletWord(rhs);
  return *this;
}

#endif // GPUCA_ALIROOT_LIB
#endif // GPUCA_GPUCODE_DEVICE

GPUd() int GPUTRDTrackletWord::GetYbin() const
{
  // returns (signed) value of Y
  if (mTrackletWord & 0x1000) {
    return -((~(mTrackletWord - 1)) & 0x1fff);
  } else {
    return (mTrackletWord & 0x1fff);
  }
}

GPUd() int GPUTRDTrackletWord::GetdY() const
{
  // returns (signed) value of the deflection length
  if (mTrackletWord & (1 << 19)) {
    return -((~((mTrackletWord >> 13) - 1)) & 0x7f);
  } else {
    return ((mTrackletWord >> 13) & 0x7f);
  }
}
