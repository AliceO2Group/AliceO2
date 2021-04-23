// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 include
#include "DataFormatsQualityControl/TimeRangeFlagCollection.h"

#include <iostream>

namespace o2::quality_control
{

TimeRangeFlagCollection::TimeRangeFlagCollection(std::string name, std::string detector)
  : mName(name), mDetID(detector) {}

void TimeRangeFlagCollection::insert(TimeRangeFlag&& trf)
{
  mTimeRangeFlags.insert(std::move(trf));
}
void TimeRangeFlagCollection::insert(const TimeRangeFlag& trf)
{
  mTimeRangeFlags.insert(trf);
}

size_t TimeRangeFlagCollection::size() const
{
  return mTimeRangeFlags.size();
}

void TimeRangeFlagCollection::merge(TimeRangeFlagCollection& other)
{
  if (mDetID != other.mDetID) {
    // We assume that one TimeRangeFlagCollection should correspond to one detector at most.
    // However, if this becomes annoying, we can reconsider it.
    throw std::runtime_error("The detector ID of the target collection '" + mDetID + "' is different than the other '" + mDetID);
  }
  mTimeRangeFlags.merge(other.mTimeRangeFlags);
}

void TimeRangeFlagCollection::merge(const TimeRangeFlagCollection& other)
{
  TimeRangeFlagCollection otherCopy{other};
  merge(otherCopy);
}

TimeRangeFlagCollection::collection_t::const_iterator TimeRangeFlagCollection::begin() const
{
  return mTimeRangeFlags.begin();
}
TimeRangeFlagCollection::collection_t::const_iterator TimeRangeFlagCollection::end() const
{
  return mTimeRangeFlags.end();
}

void TimeRangeFlagCollection::streamTo(std::ostream& output) const
{
  output << "TimeRangeFlagCollection '" << mName << "' for detector '" << mDetID << "':"
         << "\n";
  for (const auto& trf : mTimeRangeFlags) {
    output << trf << "\n";
  }
}

std::ostream& operator<<(std::ostream& output, const TimeRangeFlagCollection& data)
{
  data.streamTo(output);
  return output;
}

const std::string& TimeRangeFlagCollection::getName() const
{
  return mName;
}
const std::string& TimeRangeFlagCollection::getDetector() const
{
  return mDetID;
}

} // namespace o2::quality_control