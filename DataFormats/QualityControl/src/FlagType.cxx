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

#include "DataFormatsQualityControl/FlagType.h"
#include "DataFormatsQualityControl/FlagTypeFactory.h"

#include <iostream>
#include <tuple>

namespace o2::quality_control
{

FlagType::FlagType()
{
  *this = FlagTypeFactory::Invalid();
}

std::ostream& operator<<(std::ostream& os, FlagType const& my)
{
  os << "Flag Reason: id - " << my.mId << ", name - " << my.mName << ", bad - " << (my.mBad ? "true" : "false");
  return os;
}
bool FlagType::operator==(const FlagType& rhs) const
{
  return std::tie(mId, mName, mBad) == std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagType::operator!=(const FlagType& rhs) const
{
  return std::tie(mId, mName, mBad) != std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagType::operator<(const FlagType& rhs) const
{
  return std::tie(mId, mName, mBad) < std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagType::operator>(const FlagType& rhs) const
{
  return std::tie(mId, mName, mBad) > std::tie(rhs.mId, rhs.mName, rhs.mBad);
}

} // namespace o2::quality_control
