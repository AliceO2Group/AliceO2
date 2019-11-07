// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ANALYSIS_TIMERANGEFLAGS
#define ALICEO2_ANALYSIS_TIMERANGEFLAGS

/// \file TimeRangeFlagsCollection.h
/// \brief classes for defining time ranges with a certain mask to be able to cut on
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de

// System includes
#include <iostream>
#include <vector>
#include <string>

// ROOT includes
#include "Rtypes.h"

namespace o2
{
namespace analysis
{

/// \class TimeRangeFlags
/// A Class for assiciating a bit mask with a time range
template <typename time_type, typename bitmap_type = uint16_t>
class TimeRangeFlags
{
 public:
  TimeRangeFlags() = default;
  TimeRangeFlags(TimeRangeFlags const&) = default;
  TimeRangeFlags(time_type start, time_type end, bitmap_type reasons = 0);
  TimeRangeFlags(time_type time);

  void setReason(size_t reason);
  void resetReason(size_t reason) { CLRBIT(fFlags, reason); }
  Bool_t hasReason(size_t reason) const { return TESTBIT(fFlags, reason); }

  time_type getStart() const { return mStart; }
  time_type getEnd() const { return mEnd; }

  bitmap_type getFlags() const { return fFlags; }

  /// equal operator
  ///
  /// defined as range is contained in
  bool operator==(const TimeRangeFlags& other) const
  {
    return other.mStart >= mStart && other.mEnd <= mEnd;
  }

  /// create a string of the
  std::string collectMaskReasonNames() const;

  /// equal operator
  ///
  /// defined as time is contained in range
  bool operator==(const time_type time) const
  {
    return time >= mStart && time <= mEnd;
  }

  /// smaller comparison
  bool operator<(const TimeRangeFlags& other) const
  {
    return mEnd < other.mStart;
  }

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const TimeRangeFlags& data)
  {
    data.streamTo(output);
    return output;
  }

  /// print function
  virtual void print(/*Option_t* option = ""*/) const { std::cout << *this; }

 private:
  time_type mStart{};   ///< start time of masked range
  time_type mEnd{};     ///< end time of masked range
  bitmap_type fFlags{}; ///< bitmap of masking reasons

  ClassDefNV(TimeRangeFlags, 1);
};

} // namespace analysis
} // namespace o2
#endif
