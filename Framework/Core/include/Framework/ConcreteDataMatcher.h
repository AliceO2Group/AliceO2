// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_ConcreteDataMatcher_H_INCLUDED
#define o2_framework_ConcreteDataMatcher_H_INCLUDED

#include "Headers/DataHeader.h"

namespace o2
{
namespace framework
{

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

} // namespace framework
} // namespace o2
#endif
