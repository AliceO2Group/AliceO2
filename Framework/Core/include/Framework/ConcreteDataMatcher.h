// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CONCRETEDATAMATCHER_H_
#define O2_FRAMEWORK_CONCRETEDATAMATCHER_H_

#include "Headers/DataHeader.h"

namespace o2::framework
{

/// This matches the same kind of data, but it does not take into account
/// further data type specific subdivisions (i.e. on subspec).
struct ConcreteDataTypeMatcher {
  header::DataOrigin origin;
  header::DataDescription description;

  ConcreteDataTypeMatcher(header::DataOrigin origin_,
                          header::DataDescription description_)
    : origin(origin_),
      description(description_)
  {
  }

  ConcreteDataTypeMatcher(ConcreteDataTypeMatcher const& other) = default;
  ConcreteDataTypeMatcher(ConcreteDataTypeMatcher&& other) noexcept = default;
  ConcreteDataTypeMatcher& operator=(ConcreteDataTypeMatcher const& other) = default;
  ConcreteDataTypeMatcher& operator=(ConcreteDataTypeMatcher&& other) noexcept = default;

  bool operator==(ConcreteDataTypeMatcher const& that) const
  {
    return origin == that.origin && description == that.description;
  }
};

/// This fully qualifies data geometry.
///
/// There is two more degree of freedom allowed by the Data Model:
///
/// * Some further chunking of the data (e.g. when it gets populated
///   from readout, where multiple superpages can share the same subSpec).
/// * Some aggregation of the data timewise.
struct ConcreteDataMatcher {
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec;

  ConcreteDataMatcher(header::DataOrigin origin_,
                      header::DataDescription description_,
                      header::DataHeader::SubSpecificationType subSpec_)
    : origin(origin_),
      description(description_),
      subSpec(subSpec_)
  {
  }
  ConcreteDataMatcher(ConcreteDataMatcher const& other) = default;
  ConcreteDataMatcher(ConcreteDataMatcher&& other) noexcept = default;
  ConcreteDataMatcher& operator=(ConcreteDataMatcher const& other) = default;
  ConcreteDataMatcher& operator=(ConcreteDataMatcher&& other) noexcept = default;

  /// Two DataDescription are the same if and only
  /// if every component is the same.
  bool operator==(ConcreteDataMatcher const& that) const
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec;
  }
};

} // namespace o2::framework
#endif
