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

/// \file TimeRangeFlag.h
/// \brief Class to define a time range of a flag type
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

// System includes
#include <iosfwd>
#include <string>

// ROOT includes
#include <Rtypes.h>

#include "DataFormatsQualityControl/FlagReasons.h"

namespace o2
{
namespace quality_control
{

/// \class TimeRangeFlag
/// A Class for associating a bit mask with a time range
class TimeRangeFlag
{
 public:
  using time_type = uint64_t;
  using flag_type = FlagReason;

  TimeRangeFlag() = default;
  TimeRangeFlag(TimeRangeFlag const&) = default;
  TimeRangeFlag(time_type start, time_type end, flag_type flag, std::string comment = "", std::string source = "Unknown");

  time_type getStart() const { return mStart; }
  time_type getEnd() const { return mEnd; }
  flag_type getFlag() const { return mFlag; }
  const std::string& getComment() const { return mComment; }
  const std::string& getSource() const { return mSource; }

  void setStart(time_type start) { mStart = start; }
  void setEnd(time_type end) { mEnd = end; }
  void setFlag(flag_type flag) { mFlag = flag; }
  void setComment(const std::string& comment) { mComment = comment; }
  void setSource(const std::string& source) { mSource = source; }

  /// equal operator
  bool operator==(const TimeRangeFlag& other) const;

  /// comparison operators
  bool operator<(const TimeRangeFlag& rhs) const;
  bool operator>(const TimeRangeFlag& rhs) const;

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const TimeRangeFlag& data);

 private:
  time_type mStart = -1;                          ///< start time of masked range
  time_type mEnd = -1;                            ///< end time of masked range
  flag_type mFlag = FlagReasonFactory::Invalid(); ///< flag reason
  std::string mComment = "";                      ///< optional comment, which may extend the reason
  std::string mSource = "Unknown";                ///< optional (but encouraged) source of the flag (e.g. Qc Check name)

  ClassDefNV(TimeRangeFlag, 1);
};

} // namespace quality_control
} // namespace o2
#endif
