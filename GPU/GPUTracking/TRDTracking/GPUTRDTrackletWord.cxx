// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackletWord.cxx
/// \author Ole Schmidt

#include "GPUTRDTrackletWord.h"
using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_TPC_GEOMETRY_O2

GPUd() GPUTRDTrackletWord::GPUTRDTrackletWord(uint32_t trackletWord) : mHCId(-1), mTrackletWord(trackletWord)
{
}
GPUd() GPUTRDTrackletWord::GPUTRDTrackletWord(uint32_t trackletWord, int32_t hcid) : mHCId(hcid), mTrackletWord(trackletWord) {}

#ifdef GPUCA_ALIROOT_LIB
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"

GPUTRDTrackletWord::GPUTRDTrackletWord(const AliTRDtrackletWord& rhs) : mHCId(rhs.GetHCId()), mTrackletWord(rhs.GetTrackletWord())
{
}
GPUTRDTrackletWord::GPUTRDTrackletWord(const AliTRDtrackletMCM& rhs) : mHCId(rhs.GetHCId()), mTrackletWord(rhs.GetTrackletWord()) {}

GPUTRDTrackletWord& GPUTRDTrackletWord::operator=(const AliTRDtrackletMCM& rhs)
{
  this->~GPUTRDTrackletWord();
  new (this) GPUTRDTrackletWord(rhs);
  return *this;
}

#endif // GPUCA_ALIROOT_LIB

GPUd() int32_t GPUTRDTrackletWord::GetYbin() const
{
  // returns (signed) value of Y
  if (mTrackletWord & 0x1000) {
    return -((~(mTrackletWord - 1)) & 0x1fff);
  } else {
    return (mTrackletWord & 0x1fff);
  }
}

GPUd() int32_t GPUTRDTrackletWord::GetdYbin() const
{
  // returns (signed) value of the deflection length
  if (mTrackletWord & (1 << 19)) {
    return -((~((mTrackletWord >> 13) - 1)) & 0x7f);
  } else {
    return ((mTrackletWord >> 13) & 0x7f);
  }
}

#endif // !GPUCA_TPC_GEOMETRY_O2
