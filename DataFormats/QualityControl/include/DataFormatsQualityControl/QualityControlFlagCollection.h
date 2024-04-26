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

#ifndef O2_QUALITYCONTROL_QCFLAGCOLLECTION_H
#define O2_QUALITYCONTROL_QCFLAGCOLLECTION_H
/// \file QualityControlFlagCollection.h
/// \brief classes for defining time ranges with a certain mask to be able to cut on
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de

// System includes
#include <iosfwd>

// ROOT includes
#include "Rtypes.h"

// O2 includes
#include "MathUtils/detail/Bracket.h"
#include "DataFormatsQualityControl/QualityControlFlag.h"

// STL
#include <set>

namespace o2
{
namespace quality_control
{

/// \class QualityControlFlagCollection
/// A Class for keeping several time ranges of type QualityControlFlag
class QualityControlFlagCollection
{
 public:
  using collection_t = std::set<QualityControlFlag>;
  using time_type = uint64_t;
  using RangeInterval = o2::math_utils::detail::Bracket<time_type>;

  explicit QualityControlFlagCollection(std::string name, std::string detector = "TST", RangeInterval validityRange = {},
                                        int runNumber = 0, std::string periodName = "Invalid", std::string passName = "Invalid",
                                        std::string provenance = "qc");

  void insert(QualityControlFlag&&);
  void insert(const QualityControlFlag&);

  size_t size() const;

  // moves all non-repeating QualityControlFlags from other to this
  void merge(QualityControlFlagCollection& other);
  // add all non-repeating QualityControlFlags from other to this.
  void merge(const QualityControlFlagCollection& other);

  collection_t::const_iterator begin() const;
  collection_t::const_iterator end() const;

  const std::string& getName() const;
  const std::string& getDetector() const;
  int getRunNumber() const;
  const std::string& getPeriodName() const;
  const std::string& getPassName() const;
  const std::string& getProvenance() const;

  time_type getStart() const { return mValidityRange.getMin(); }
  time_type getEnd() const { return mValidityRange.getMax(); }
  RangeInterval& getInterval() { return mValidityRange; }

  void setStart(time_type start) { mValidityRange.setMin(start); }
  void setEnd(time_type end) { mValidityRange.setMax(end); }
  void setInterval(RangeInterval interval) { mValidityRange = interval; }

  /// write data to ostream
  void streamTo(std::ostream& output) const;
  /// Read data from instream
  void streamFrom(std::istream& input);

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const QualityControlFlagCollection& data);

 private:
  std::string mDetID; // three letter detector code
  std::string mName;  // some description of the collection, e.g. "Raw data checks", "QA Expert masks"
  // with std::set we can sort the flags in time and have merge() for granted.
  collection_t mQualityControlFlags;
  RangeInterval mValidityRange; // we need a validity range to e.g. state that there are no TRFs for given time interval
  int mRunNumber;
  std::string mPeriodName;
  std::string mPassName;
  std::string mProvenance;

  ClassDefNV(QualityControlFlagCollection, 1);
};

} // namespace quality_control
} // namespace o2

#endif // O2_QUALITYCONTROL_QCFLAGCOLLECTION_H
