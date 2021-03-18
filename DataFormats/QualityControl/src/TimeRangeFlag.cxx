// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsQualityControl/TimeRangeFlag.h"

#include <iostream>
#include <tuple>

namespace o2::quality_control
{

TimeRangeFlag::TimeRangeFlag(time_type start, time_type end, flag_type flag, std::string comment, std::string source)
  : mInterval(start, end), mFlag(flag), mComment(comment), mSource(source)
{
  if (mInterval.isInvalid()) {
    throw std::runtime_error("TimeRangeFlag start time '" + std::to_string(mInterval.getMin()) + "' is larger than end time '" + std::to_string(mInterval.getMax()) + "'");
  }
}

bool TimeRangeFlag::operator==(const TimeRangeFlag& rhs) const
{
  return std::tie(mInterval, mFlag, mComment, mSource) == std::tie(rhs.mInterval, rhs.mFlag, rhs.mComment, rhs.mSource);
}

bool TimeRangeFlag::operator<(const TimeRangeFlag& rhs) const
{
  // We don't use the comparison mechanism in Bracket,
  // because std::set which is used in TRFCollection assumes that a < b, a > b <=> a == b.
  // Using relation operators in Bracket would break insertion and merging.
  return std::tie(static_cast<const time_type&>(mInterval.getMin()), static_cast<const time_type&>(mInterval.getMax()), mFlag, mComment, mSource) < std::tie(static_cast<const time_type&>(rhs.mInterval.getMin()), static_cast<const time_type&>(rhs.mInterval.getMax()), rhs.mFlag, rhs.mComment, rhs.mSource);
}

bool TimeRangeFlag::operator>(const TimeRangeFlag& rhs) const
{
  // we don't use the comparison mechanism in Bracket,
  // because std::set which is used in TRFCollection assumes that a < b, a > b <=> a == b
  return std::tie(static_cast<const time_type&>(mInterval.getMin()), static_cast<const time_type&>(mInterval.getMax()), mFlag, mComment, mSource) > std::tie(static_cast<const time_type&>(rhs.mInterval.getMin()), static_cast<const time_type&>(rhs.mInterval.getMax()), rhs.mFlag, rhs.mComment, rhs.mSource);
}

void TimeRangeFlag::streamTo(std::ostream& output) const
{
  output << "TimeRangeFlag:\n";
  output << "- Start: " << mInterval.getMin() << "\n";
  output << "- End: " << mInterval.getMax() << "\n";
  output << "- " << mFlag << "\n";
  output << "- Comment: " << mComment << "\n";
  output << "- Source: " << mSource;
}

std::ostream& operator<<(std::ostream& output, const TimeRangeFlag& data)
{
  data.streamTo(output);
  return output;
}

} // namespace o2::quality_control
