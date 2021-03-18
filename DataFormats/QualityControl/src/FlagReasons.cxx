// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsQualityControl/FlagReasons.h"

#include <iostream>
#include <tuple>

namespace o2::quality_control
{

std::ostream& operator<<(std::ostream& os, FlagReason const& my)
{
  os << "Flag Reason: id - " << my.mId << ", name - " << my.mName << ", bad - " << (my.mBad ? "true" : "false");
  return os;
}
bool FlagReason::operator==(const FlagReason& rhs) const
{
  return std::tie(mId, mName, mBad) == std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagReason::operator!=(const FlagReason& rhs) const
{
  return std::tie(mId, mName, mBad) != std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagReason::operator<(const FlagReason& rhs) const
{
  return std::tie(mId, mName, mBad) < std::tie(rhs.mId, rhs.mName, rhs.mBad);
}
bool FlagReason::operator>(const FlagReason& rhs) const
{
  return std::tie(mId, mName, mBad) > std::tie(rhs.mId, rhs.mName, rhs.mBad);
}

} // namespace o2::quality_control