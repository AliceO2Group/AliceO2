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

#include <iostream>
#include "DataFormatsEMCAL/CompressedTriggerData.h"

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::CompressedTRU& tru)
{
  stream << "TRU " << tru.mTRUIndex << ": Fired " << (tru.mFired ? "yes" : "no") << ", time " << (tru.mFired ? std::to_string(static_cast<int>(tru.mTriggerTime)) : "Undefined") << ", number of patches " << tru.mNumberOfPatches;
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::CompressedTriggerPatch& patch)
{
  stream << "Patch " << patch.mPatchIndexInTRU << " in TRU " << patch.mTRUIndex << ": Time " << patch.mTime << ", ADC " << patch.mADC;
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::CompressedL0TimeSum& timesum)
{
  stream << "FastOR " << timesum.mIndex << ": " << timesum.mTimesum << " ADC counts";
  return stream;
}