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

#include "MCHCalibration/PedestalChannel.h"
#include <cmath>
#include <fmt/core.h>
#include <iostream>

namespace o2::mch::calibration

{
double PedestalChannel::getRms() const
{
  return mEntries > 0 ? std::sqrt(mVariance / mEntries) : std::numeric_limits<double>::max();
}

bool PedestalChannel::isValid() const
{
  return dsChannelId.isValid();
}

std::string PedestalChannel::asString() const
{
  return fmt::format("{} entries {:8d} mean {:7.2f} mVariance {:7.2f} rms {:7.2f}",
                     dsChannelId.asString(), mEntries, mPedestal, mVariance, getRms());
}

std::ostream& operator<<(std::ostream& os, const PedestalChannel& c)
{
  os << c.asString();
  return os;
}

} // namespace o2::mch::calibration
