// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// System includes
#include <fmt/ostream.h>

// O2 includes
#include "Framework/Logger.h"
#include "DataFormatsAnalysis/FlagReasons.h"
#include "DataFormatsAnalysis/TimeRangeFlagsCollection.h"

using namespace o2::analysis;

templateClassImp(TimeRangeFlags);

template <typename time_type, typename bitmap_type>
TimeRangeFlags<time_type, bitmap_type>::TimeRangeFlags(time_type start, time_type end, bitmap_type reasons)
  : mStart(start),
    mEnd(end),
    fFlags(reasons)
{
}

template <typename time_type, typename bitmap_type>
TimeRangeFlags<time_type, bitmap_type>::TimeRangeFlags(time_type time)
  : mStart(time),
    mEnd(time),
    fFlags()
{
}

template <typename time_type, typename bitmap_type>
void TimeRangeFlags<time_type, bitmap_type>::setReason(size_t reason)
{
  const auto& flags = FlagReasons::instance();

  const auto numberOfReasons = flags.getReasonCollection().size();
  if ( reason >= numberOfReasons) {
    O2ERROR("Reason {} outside of reason collection size ({})", reason, numberOfReasons);
    return;
  }

  SETBIT(fFlags, reason);
}

template <typename time_type, typename bitmap_type>
std::string TimeRangeFlags<time_type, bitmap_type>::collectMaskReasonNames() const
{
  const auto& flags = FlagReasons::instance();
  const auto& collection = flags.getReasonCollection();
  const auto numberOfReasons = collection.size();

  std::string reasons;
  for (size_t iReason = 0; iReason < numberOfReasons; ++iReason) {
    if (hasReason(iReason)) {
      if (reasons.length()) {
        reasons += " | ";
      }
      reasons += collection[iReason];
    }
  }

  return reasons;
}

template <typename time_type, typename bitmap_type>
void TimeRangeFlags<time_type, bitmap_type>::streamTo(std::ostream& output) const
{
  fmt::print("{:>20} - {:>20} : {:>60}\n", mStart,  mEnd, collectMaskReasonNames());
}
// default implementations
template class TimeRangeFlags<uint64_t>;
