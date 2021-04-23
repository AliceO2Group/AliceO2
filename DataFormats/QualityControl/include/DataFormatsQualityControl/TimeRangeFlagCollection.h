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

/// \file TimeRangeFlagCollection.h
/// \brief classes for defining time ranges with a certain mask to be able to cut on
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de

// System includes
#include <iosfwd>

// ROOT includes
#include "Rtypes.h"

// O2 includes
#include "DataFormatsQualityControl/TimeRangeFlag.h"

// STL
#include <set>

namespace o2
{
namespace quality_control
{

/// \class TimeRangeFlagCollection
/// A Class for keeping several time ranges of type TimeRangeFlag
class TimeRangeFlagCollection
{
 public:
  using collection_t = std::set<TimeRangeFlag>;

  explicit TimeRangeFlagCollection(std::string name, std::string detector = "TST");

  void insert(TimeRangeFlag&&);
  void insert(const TimeRangeFlag&);

  size_t size() const;

  // moves all non-repeating TimeRangeFlags from other to this
  void merge(TimeRangeFlagCollection& other);
  // add all non-repeating TimeRangeFlags from other to this.
  void merge(const TimeRangeFlagCollection& other);

  collection_t::const_iterator begin() const;
  collection_t::const_iterator end() const;

  const std::string& getName() const;
  const std::string& getDetector() const;

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const TimeRangeFlagCollection& data);

 private:
  std::string mDetID;     // three letter detector code
  std::string mName = ""; // some description of the collection, e.g. "Raw data checks", "QA Expert masks"
  // with std::set we can sort the flags in time and have merge() for granted.
  collection_t mTimeRangeFlags;

  ClassDefNV(TimeRangeFlagCollection, 1);
};

} // namespace quality_control
} // namespace o2
#endif
