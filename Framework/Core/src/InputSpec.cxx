// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
#include "Framework/InputSpec.h"
#include "Framework/DataSpecUtils.h"

#include <variant>

namespace o2
{
namespace framework
{

InputSpec::InputSpec(std::string binding_, ConcreteDataMatcher const& concrete, enum Lifetime lifetime_)
  : binding{binding_},
    matcher{concrete},
    lifetime{lifetime_}
{
}

InputSpec::InputSpec(std::string binding_, header::DataOrigin origin_, header::DataDescription description_, header::DataHeader::SubSpecificationType subSpec_, enum Lifetime lifetime_)
  : binding{binding_},
    matcher{ConcreteDataMatcher{origin_, description_, subSpec_}},
    lifetime{lifetime_}
{
}

InputSpec::InputSpec(std::string binding_, header::DataOrigin origin_, header::DataDescription description_, enum Lifetime lifetime_)
  : binding{binding_},
    matcher{ConcreteDataMatcher{origin_, description_, 0}},
    lifetime{lifetime_}
{
}

InputSpec::InputSpec(std::string binding_, ConcreteDataTypeMatcher const& dataType, enum Lifetime lifetime_)
  : binding{binding_},
    matcher{DataSpecUtils::dataDescriptorMatcherFrom(dataType)},
    lifetime{lifetime_}
{
}

InputSpec::InputSpec(std::string binding_, data_matcher::DataDescriptorMatcher&& matcher_)
  : binding{binding_},
    matcher{matcher_},
    lifetime{Lifetime::Timeframe}
{
}

bool InputSpec::operator==(InputSpec const& that) const
{
  return this->matcher == that.matcher && this->lifetime == that.lifetime;
}

} // namespace framework
} // namespace o2
