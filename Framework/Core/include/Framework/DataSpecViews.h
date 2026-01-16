// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATASPECVIEWS_H_
#define O2_FRAMEWORK_DATASPECVIEWS_H_

#include "Framework/DataSpecUtils.h"
#include <ranges>

namespace o2::framework::views
{
static auto partial_match_filter(auto what)
{
  return std::views::filter([what](auto const& t) -> bool { return DataSpecUtils::partialMatch(t, what); });
}

static auto exclude_by_name(std::string name)
{
  return std::views::filter([name](auto const& t) -> bool { return t.name != name; });
}

static auto filter_not_matching(auto const& provided)
{
  return std::views::filter([&provided](auto const& input) { return std::none_of(provided.begin(), provided.end(), [&input](auto const& output) { return DataSpecUtils::match(input, output); }); });
}

static auto filter_matching(auto const& provided)
{
  return std::views::filter([&provided](auto const& input) { return std::any_of(provided.begin(), provided.end(), [&input](auto const& output) { return DataSpecUtils::match(input, output); }); });
}

static auto filter_string_params_with(std::string match)
{
  return std::views::filter([match](auto const& param) {
    return (param.type == VariantType::String) && (param.name.find(match) != std::string::npos);
  });
}

static auto filter_string_params_starts_with(std::string match)
{
  return std::views::filter([match](auto const& param) {
    return (param.type == VariantType::String) && (param.name.starts_with(match));
  });
}

static auto input_to_output_specs()
{
  return std::views::transform([](auto const& input) {
    auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
    return OutputSpec{concrete.origin, concrete.description, concrete.subSpec, input.lifetime, input.metadata};
  });
}

static auto params_to_input_specs()
{
  return std::views::transform([](auto const& param) {
    return DataSpecUtils::fromMetadataString(param.defaultValue.template get<std::string>());
  });
}
} // namespace o2::framework::views
//
namespace o2::framework::sinks
{
template <class Container>
struct append_to {
  Container& c;
  // ends the pipeline, returns the container
  template <std::ranges::input_range R>
  friend Container& operator|(R&& r, append_to self)
  {
    std::ranges::copy(r, std::back_inserter(self.c));
    return self.c;
  }
};

template <class Container>
struct update_input_list {
  Container& c;
  // ends the pipeline, returns the container
  template <std::ranges::input_range R>
  friend Container& operator|(R&& r, update_input_list self)
  {
    for (auto const& item : r) {
      auto copy = item;
      DataSpecUtils::updateInputList(self.c, std::move(copy));
    }
    return self.c;
  }
};

template <class Container>
struct update_output_list {
  Container& c;
  // ends the pipeline, returns the container
  template <std::ranges::input_range R>
  friend Container& operator|(R&& r, update_output_list self)
  {
    for (auto const& item : r) {
      auto copy = item;
      DataSpecUtils::updateOutputList(self.c, std::move(copy));
    }
    return self.c;
  }
};

} // namespace o2::framework::sinks

#endif // O2_FRAMEWORK_DATASPECVIEWS_H_
