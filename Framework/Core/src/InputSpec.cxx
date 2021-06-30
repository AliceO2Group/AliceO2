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
//
#include "Framework/InputSpec.h"
#include "Framework/DataSpecUtils.h"

#include <variant>
#include <vector>

namespace o2
{
namespace framework
{

InputSpec::InputSpec(std::string binding_,
                     ConcreteDataMatcher const& concrete,
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{concrete},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec::InputSpec(std::string binding_,
                     header::DataOrigin origin_,
                     header::DataDescription description_,
                     header::DataHeader::SubSpecificationType subSpec_,
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{ConcreteDataMatcher{origin_, description_, subSpec_}},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec::InputSpec(std::string binding_,
                     header::DataOrigin origin_,
                     header::DataDescription description_,
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{ConcreteDataMatcher{origin_, description_, 0}},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec::InputSpec(std::string binding_,
                     header::DataOrigin const& origin_,
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{DataSpecUtils::dataDescriptorMatcherFrom(origin_)},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec::InputSpec(std::string binding_,
                     ConcreteDataTypeMatcher const& dataType,
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{DataSpecUtils::dataDescriptorMatcherFrom(dataType)},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec::InputSpec(std::string binding_,
                     data_matcher::DataDescriptorMatcher&& matcher_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{matcher_},
    lifetime{Lifetime::Timeframe},
    metadata{metadata_}
{
}

bool InputSpec::operator==(InputSpec const& that) const
{
  return this->matcher == that.matcher && this->lifetime == that.lifetime;
}

} // namespace framework
} // namespace o2
