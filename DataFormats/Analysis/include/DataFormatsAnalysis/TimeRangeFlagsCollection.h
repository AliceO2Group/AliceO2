// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ANALYSIS_TIMERANGEMASK
#define ALICEO2_ANALYSIS_TIMERANGEMASK

/// \file TimeRangeFlagsCollection.h
/// \brief classes for defining time ranges with a certain mask to be able to cut on
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de

// System includes
#include <iostream>
#include <vector>
#include <string>

// ROOT includes
#include "Rtypes.h"

// O2 includes
#include "DataFormatsAnalysis/TimeRangeFlags.h"

namespace o2
{
namespace analysis
{

/// \class TimeRangeFlagsCollection
/// A Class for keeping several time ranges with mask of type TimeRangeFlags
template <typename time_type, typename bitmap_type = uint16_t>
class TimeRangeFlagsCollection
{
 public:
  TimeRangeFlagsCollection() = default;

  /// add a time range with mask reason
  TimeRangeFlags<time_type, bitmap_type>* addTimeRangeFlags(time_type start, time_type end, bitmap_type reasons = {});

  /// find range that contains 'time'
  const TimeRangeFlags<time_type, bitmap_type>* findTimeRangeFlags(time_type time) const;

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const TimeRangeFlagsCollection& data)
  {
    data.streamTo(output);
    return output;
  }

  /// print function
  virtual void print(/*Option_t* option = ""*/) const { std::cout << *this; }

 private:
  std::vector<TimeRangeFlags<time_type, bitmap_type>> mTimeRangeFlagsCollection{}; ///< flag reasons

  ClassDefNV(TimeRangeFlagsCollection, 1);
};

} // namespace analysis
} // namespace o2
#endif
