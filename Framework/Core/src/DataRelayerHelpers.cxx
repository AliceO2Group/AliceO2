// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataRelayerHelpers.h"
#include "Framework/DataDescriptorMatcher.h"
#include <stdexcept>

using namespace o2::framework::data_matcher;

namespace o2::framework
{

namespace
{
DataDescriptorMatcher fromConcreteMatcher(ConcreteDataMatcher const& matcher)
{
  return DataDescriptorMatcher{
    DataDescriptorMatcher::Op::And,
    StartTimeValueMatcher{ContextRef{0}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{matcher.origin.as<std::string>()},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{matcher.description.as<std::string>()},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          SubSpecificationTypeValueMatcher{matcher.subSpec})))};
}
} // namespace

std::vector<size_t>
  DataRelayerHelpers::createDistinctRouteIndex(std::vector<InputRoute> const& routes)
{
  std::vector<size_t> result;
  for (size_t ri = 0; ri < routes.size(); ++ri) {
    auto& route = routes[ri];
    if (route.timeslice == 0) {
      result.push_back(ri);
    }
  }
  return result;
}

/// This converts from InputRoute to the associated DataDescriptorMatcher.
std::vector<DataDescriptorMatcher>
  DataRelayerHelpers::createInputMatchers(std::vector<InputRoute> const& routes)
{
  std::vector<DataDescriptorMatcher> result;

  for (auto& route : routes) {
    if (auto pval = std::get_if<ConcreteDataMatcher>(&route.matcher.matcher)) {
      result.emplace_back(fromConcreteMatcher(*pval));
    } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&route.matcher.matcher)) {
      result.push_back(*matcher);
    } else {
      throw std::runtime_error("Unsupported InputSpec type");
    }
  }

  return result;
}

} // namespace o2::framework
