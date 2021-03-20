// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsMCH/Digit.h"
#include <cmath>

namespace o2::mch
{

bool closeEnough(double x, double y, double eps = 1E-6)
{
  return std::fabs(x - y) <= eps * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
}

Digit::Digit(int detid, int pad, unsigned long adc, int32_t time, uint16_t nSamples)
  : mTFtime(time), mNofSamples(nSamples), mDetID(detid), mPadID(pad), mADC(adc)
{
  setSaturated(false);
}

void Digit::setNofSamples(uint16_t n)
{
  uint16_t sat = mNofSamples & 0x8000;
  mNofSamples = (n & 0x7FFF) + sat;
}

void Digit::setSaturated(bool sat)
{
  if (sat) {
    mNofSamples |= 0x8000;
  } else {
    mNofSamples &= 0x7FFF;
  }
}

bool Digit::operator==(const Digit& other) const
{
  return mDetID == other.mDetID &&
         mPadID == other.mPadID &&
         mADC == other.mADC &&
         mTFtime == other.mTFtime &&
         mNofSamples == other.mNofSamples;
}

} // namespace o2::mch
