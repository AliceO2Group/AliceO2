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
#include <fmt/format.h>
#include <iostream>

namespace o2::mch
{

std::ostream& operator<<(std::ostream& os, const o2::mch::Digit& d)
{
  os << fmt::format("DetID {:4d} PadId {:6d} ADC {:10d} TFtime {:10d} NofSamples {:5d} {}",
                    d.getDetID(), d.getPadID(), d.getADC(), d.getTime(), d.getNofSamples(),
                    d.isSaturated() ? "(S)" : "");
  return os;
}

bool closeEnough(double x, double y, double eps = 1E-6)
{
  return std::fabs(x - y) <= eps * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
}

Digit::Digit(int detid, int pad, uint32_t adc, int32_t time, uint16_t nSamples, bool saturated)
  : mTFtime(time), mNofSamples(nSamples), mIsSaturated(saturated), mDetID(detid), mPadID(pad), mADC(adc)
{
  setNofSamples(nSamples);
}

uint16_t Digit::getNofSamples() const
{
  return mNofSamples;
}

bool Digit::isSaturated() const
{
  return mIsSaturated;
}

void Digit::setNofSamples(uint16_t n)
{
  constexpr uint64_t max10bits = (static_cast<uint64_t>(1) << 10);
  if (static_cast<uint64_t>(n) >= max10bits) {
    throw std::invalid_argument("mch digit nofsamples must fit within 10 bits");
  }
  mNofSamples = n;
}

void Digit::setSaturated(bool sat)
{
  mIsSaturated = sat;
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
