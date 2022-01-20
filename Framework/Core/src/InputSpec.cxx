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
#include <regex>

namespace o2::framework
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
                     enum Lifetime lifetime_,
                     std::vector<ConfigParamSpec> const& metadata_)
  : binding{binding_},
    matcher{matcher_},
    lifetime{lifetime_},
    metadata{metadata_}
{
}

InputSpec InputSpec::fromString(std::string s)
{
  std::regex word_regex("(\\w+)");
  auto words = std::sregex_iterator(s.begin(), s.end(), word_regex);
  if (std::distance(words, std::sregex_iterator()) != 3) {
    throw runtime_error_f("Malformed input spec metadata: %s", s.c_str());
  }
  std::vector<std::string> data;
  for (auto i = words; i != std::sregex_iterator(); ++i) {
    data.emplace_back(i->str());
  }
  char origin[4];
  char description[16];
  std::memcpy(&origin, data[1].c_str(), 4);
  std::memcpy(&description, data[2].c_str(), 16);
  return InputSpec{data[0], header::DataOrigin{origin}, header::DataDescription{description}};
}

bool InputSpec::operator==(InputSpec const& that) const
{
  return this->matcher == that.matcher && this->lifetime == that.lifetime;
}

void updateInputList(std::vector<InputSpec>& list, InputSpec&& input)
{
  auto locate = std::find_if(list.begin(), list.end(), [&](InputSpec& entry) { return entry.binding == input.binding; });
  if (locate != list.end()) {
    // amend entry
    auto& entryMetadata = locate->metadata;
    entryMetadata.insert(entryMetadata.end(), input.metadata.begin(), input.metadata.end());
    std::sort(entryMetadata.begin(), entryMetadata.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name < b.name; });
    auto new_end = std::unique(entryMetadata.begin(), entryMetadata.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name == b.name; });
    entryMetadata.erase(new_end, entryMetadata.end());
  } else {
    // add entry
    list.emplace_back(std::move(input));
  }
}

} // namespace o2::framework
