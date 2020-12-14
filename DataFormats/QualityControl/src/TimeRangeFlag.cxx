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

ClassImp(o2::quality_control::TimeRangeFlag);

namespace o2::quality_control
{

TimeRangeFlag::TimeRangeFlag(time_type start, time_type end, flag_type flag, std::string comment, std::string source)
  : mStart(start), mEnd(end), mFlag(flag), mComment(comment), mSource(source)
{
  if (mStart > mEnd) {
    throw std::runtime_error("TimeRangeFlag start time '" + std::to_string(mStart) + "' is larger than end time '" + std::to_string(mEnd) + "'");
  }
}

bool TimeRangeFlag::operator==(const TimeRangeFlag& other) const
{
  return std::tie(mStart, mEnd, mFlag, mComment, mSource) == std::tie(other.mStart, other.mEnd, other.mFlag, other.mComment, other.mSource);
}

bool TimeRangeFlag::operator<(const TimeRangeFlag& rhs) const
{
  return std::tie(mStart, mEnd, mFlag, mComment, mSource) < std::tie(rhs.mStart, rhs.mEnd, rhs.mFlag, rhs.mComment, rhs.mSource);
}

bool TimeRangeFlag::operator>(const TimeRangeFlag& rhs) const
{
  return std::tie(mStart, mEnd, mFlag, mComment, mSource) > std::tie(rhs.mStart, rhs.mEnd, rhs.mFlag, rhs.mComment, rhs.mSource);
}

void TimeRangeFlag::streamTo(std::ostream& output) const
{
  output << "TimeRangeFlag:\n";
  output << "- Start: " << mStart << "\n";
  output << "- End: " << mEnd << "\n";
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
