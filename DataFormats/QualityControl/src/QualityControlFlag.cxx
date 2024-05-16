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

#include "DataFormatsQualityControl/QualityControlFlag.h"

#include <iostream>
#include <tuple>

namespace o2::quality_control
{

QualityControlFlag::QualityControlFlag(time_type start, time_type end, FlagType flag, std::string comment, std::string source)
  : mInterval(start, end), mFlag(flag), mComment(comment), mSource(source)
{
  if (mInterval.isInvalid()) {
    throw std::runtime_error("QualityControlFlag start time '" + std::to_string(mInterval.getMin()) + "' is larger than end time '" + std::to_string(mInterval.getMax()) + "'");
  }
}

bool QualityControlFlag::operator==(const QualityControlFlag& rhs) const
{
  return std::tie(mInterval, mFlag, mComment, mSource) == std::tie(rhs.mInterval, rhs.mFlag, rhs.mComment, rhs.mSource);
}

bool QualityControlFlag::operator<(const QualityControlFlag& rhs) const
{
  // We don't use the comparison mechanism in Bracket,
  // because std::set which is used in TRFCollection assumes that a < b, a > b <=> a == b.
  // Using relation operators in Bracket would break insertion and merging.
  return std::tie(static_cast<const time_type&>(mInterval.getMin()), static_cast<const time_type&>(mInterval.getMax()), mFlag, mComment, mSource) < std::tie(static_cast<const time_type&>(rhs.mInterval.getMin()), static_cast<const time_type&>(rhs.mInterval.getMax()), rhs.mFlag, rhs.mComment, rhs.mSource);
}

bool QualityControlFlag::operator>(const QualityControlFlag& rhs) const
{
  // we don't use the comparison mechanism in Bracket,
  // because std::set which is used in TRFCollection assumes that a < b, a > b <=> a == b
  return std::tie(static_cast<const time_type&>(mInterval.getMin()), static_cast<const time_type&>(mInterval.getMax()), mFlag, mComment, mSource) > std::tie(static_cast<const time_type&>(rhs.mInterval.getMin()), static_cast<const time_type&>(rhs.mInterval.getMax()), rhs.mFlag, rhs.mComment, rhs.mSource);
}

void QualityControlFlag::streamTo(std::ostream& output) const
{
  output << "QualityControlFlag:\n";
  output << "- Start: " << mInterval.getMin() << "\n";
  output << "- End: " << mInterval.getMax() << "\n";
  output << "- " << mFlag << "\n";
  output << "- Comment: " << mComment << "\n";
  output << "- Source: " << mSource;
}

std::ostream& operator<<(std::ostream& output, const QualityControlFlag& data)
{
  data.streamTo(output);
  return output;
}

} // namespace o2::quality_control
