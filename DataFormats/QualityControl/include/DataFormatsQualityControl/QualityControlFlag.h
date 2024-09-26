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

#ifndef O2_QUALITYCONTROL_QCFLAG_H
#define O2_QUALITYCONTROL_QCFLAG_H

/// \file QualityControlFlag.h
/// \brief Class to define a flag type with a time range and comments
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

// System includes
#include <iosfwd>
#include <string>

// ROOT includes
#include <Rtypes.h>

#include <MathUtils/detail/Bracket.h>

#include "DataFormatsQualityControl/FlagType.h"
#include "DataFormatsQualityControl/FlagTypeFactory.h"

namespace o2
{
namespace quality_control
{

/// \class QualityControlFlag
/// A Class for associating a bit mask with a time range
class QualityControlFlag
{
 public:
  using time_type = uint64_t;
  using RangeInterval = o2::math_utils::detail::Bracket<time_type>;

  QualityControlFlag() = default;
  QualityControlFlag(QualityControlFlag const&) = default;
  QualityControlFlag(time_type start, time_type end, FlagType flag, std::string comment = "", std::string source = "Unknown");

  time_type getStart() const { return mInterval.getMin(); }
  time_type getEnd() const { return mInterval.getMax(); }
  RangeInterval& getInterval() { return mInterval; }
  const RangeInterval getInterval() const { return mInterval; }
  FlagType getFlag() const { return mFlag; }
  const std::string& getComment() const { return mComment; }
  const std::string& getSource() const { return mSource; }

  void setStart(time_type start) { mInterval.setMin(start); }
  void setEnd(time_type end) { mInterval.setMax(end); }
  void setInterval(RangeInterval interval) { mInterval = interval; }
  void setFlag(FlagType flag) { mFlag = flag; }
  void setComment(const std::string& comment) { mComment = comment; }
  void setSource(const std::string& source) { mSource = source; }

  /// equal operator
  bool operator==(const QualityControlFlag& rhs) const;

  /// comparison operators
  bool operator<(const QualityControlFlag& rhs) const;
  bool operator>(const QualityControlFlag& rhs) const;

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const QualityControlFlag& data);

 private:
  RangeInterval mInterval = {};    ///< time interval of the masked range
  FlagType mFlag;                  ///< flag reason
  std::string mComment = "";       ///< optional comment, which may extend the reason
  std::string mSource = "Unknown"; ///< optional (but encouraged) source of the flag (e.g. Qc Check name)

  ClassDefNV(QualityControlFlag, 1);
};

} // namespace quality_control
} // namespace o2
#endif // O2_QUALITYCONTROL_QCFLAG_H
