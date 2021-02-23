// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/Digit.h"
#include <cmath>

#define MCH_TF_TIME_VALID_BIT 2

namespace o2::mch
{

bool closeEnough(double x, double y, double eps = 1E-6)
{
  return std::fabs(x - y) <= eps * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
}

Digit::Digit(int detid, int pad, unsigned long adc, int32_t time, uint16_t nSamples)
  : mTFtime(time), mTimeFlags(0), mDetID(detid), mPadID(pad), mADC(adc), mNofSamples(nSamples)
{
}

void Digit::setTimeValid(bool valid)
{
  if (valid) {
    mTimeFlags |= (1 << MCH_TF_TIME_VALID_BIT);
  } else {
    mTimeFlags &= ~(1 << MCH_TF_TIME_VALID_BIT);
  }
}

bool Digit::isTimeValid() const
{
  return ((mTimeFlags & (1 << MCH_TF_TIME_VALID_BIT)) != 0);
}

void Digit::setTFindex(uint8_t idx)
{
  // reset the two least significant bits first
  mTimeFlags &= 0xFC;
  // then set them to the new value
  mTimeFlags |= (idx & 0x3);
}

uint8_t Digit::getTFindex() const
{
  return (mTimeFlags & 0x3);
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
