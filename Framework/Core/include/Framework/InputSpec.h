// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTSPEC_H
#define FRAMEWORK_INPUTSPEC_H

#include "Framework/Lifetime.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataDescriptorMatcher.h"

#include <string>
#include <ostream>
#include <variant>

namespace o2
{
namespace framework
{

/// A selector for some kind of data being processed, either in
/// input or in output. This can be used, for example to match
/// specific payloads in a timeframe.
struct InputSpec {
  /// Create a fully qualified InputSpec
  InputSpec(std::string binding_, header::DataOrigin origin_, header::DataDescription description_, header::DataHeader::SubSpecificationType subSpec_, enum Lifetime lifetime_ = Lifetime::Timeframe);
  /// Create a fully qualified InputSpec (alternative syntax)
  InputSpec(std::string binding_, ConcreteDataMatcher const& dataType, enum Lifetime lifetime_ = Lifetime::Timeframe);
  /// Create a fully qualified InputSpec where the subSpec is 0
  InputSpec(std::string binding_, header::DataOrigin origin_, header::DataDescription description_, enum Lifetime lifetime_ = Lifetime::Timeframe);
  /// Create an InputSpec which does not check for the subSpec.
  InputSpec(std::string binding_, ConcreteDataTypeMatcher const& dataType, enum Lifetime lifetime_ = Lifetime::Timeframe);
  InputSpec(std::string binding, data_matcher::DataDescriptorMatcher &&matcher);

  std::string binding;
  std::variant<ConcreteDataMatcher, data_matcher::DataDescriptorMatcher> matcher;
  enum Lifetime lifetime;

  friend std::ostream& operator<<(std::ostream& stream, InputSpec const& arg);
  bool operator==(InputSpec const& that) const;
};

} // namespace framework
} // namespace o2
#endif
