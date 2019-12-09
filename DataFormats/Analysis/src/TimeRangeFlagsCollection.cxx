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
#include "Framework/Logger.h"
#include "DataFormatsAnalysis/TimeRangeFlagsCollection.h"

using namespace o2::analysis;

templateClassImp(TimeRangeFlagsCollection);

template <typename time_type, typename bitmap_type>
TimeRangeFlags<time_type, bitmap_type>* TimeRangeFlagsCollection<time_type, bitmap_type>::addTimeRangeFlags(DetID detID, time_type start, time_type end, bitmap_type reasons)
{
  if (const auto* range = findTimeRangeFlags(detID, start)) {
    const std::string reasonsString = range->collectMaskReasonNames();
    O2ERROR("Start time %llu already in range [%llu, %llu]: %s",
            start, range->getStart(), range->getEnd(), reasonsString.data());
    return nullptr;
  }

  if (const auto* range = findTimeRangeFlags(detID, end)) {
    const std::string reasonsString = range->collectMaskReasonNames();
    O2ERROR("End time %llu already in range [%llu, %llu]: %s",
            end, range->getStart(), range->getEnd(), reasonsString.data());
    return nullptr;
  }

  return &mTimeRangeFlagsCollection[detID.getID()].emplace_back(start, end, reasons);
}

template <typename time_type, typename bitmap_type>
const TimeRangeFlags<time_type, bitmap_type>* TimeRangeFlagsCollection<time_type, bitmap_type>::findTimeRangeFlags(DetID detID, time_type time) const
{
  for (const auto& timeRange : mTimeRangeFlagsCollection[detID.getID()]) {
    if (timeRange == time) {
      return &timeRange;
    }
  }
  return nullptr;
}

template <typename time_type, typename bitmap_type>
void TimeRangeFlagsCollection<time_type, bitmap_type>::streamTo(std::ostream& output) const
{
  for (DetID::ID id = 0; id < mTimeRangeFlagsCollection.size(); ++id) {
    const auto& timeRangeDet = mTimeRangeFlagsCollection[id];
    if (!timeRangeDet.size()) {
      continue;
    }
    const DetID detID(id);
    output << fmt::format("{:=^106}\n", fmt::format("{} {} {}", "| Time range flags", detID.getName(), " |"));
    output << fmt::format("{:>20} - {:>20} : {:>60}\n", "start", "end", "flag mask");
    for (const auto& range : mTimeRangeFlagsCollection[id]) {
      range.streamTo(output);
    }
  }
}

template class TimeRangeFlagsCollection<uint64_t>;
