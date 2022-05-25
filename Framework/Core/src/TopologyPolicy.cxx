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
#include "Framework/DataProcessorSpec.h"
#include "Framework/TopologyPolicy.h"
#include <string>
#include <regex>

namespace o2::framework
{

struct TopologyPolicyHelpers {
  static TopologyPolicy::DataProcessorMatcher matchAll();
  static TopologyPolicy::DataProcessorMatcher matchByName(std::string const& name);
  static TopologyPolicy::DataProcessorMatcher matchByRegex(std::string const& re);
  static TopologyPolicy::DependencyChecker dataDependency();
  static TopologyPolicy::DependencyChecker alwaysDependent();
};

TopologyPolicy::DataProcessorMatcher TopologyPolicyHelpers::matchAll()
{
  return [](DataProcessorSpec const& spec) {
    return true;
  };
}

TopologyPolicy::DataProcessorMatcher TopologyPolicyHelpers::matchByName(std::string const& name)
{
  return [name](DataProcessorSpec const& spec) {
    return spec.name == name;
  };
}

TopologyPolicy::DataProcessorMatcher TopologyPolicyHelpers::matchByRegex(std::string const& re)
{
  return [re](DataProcessorSpec const& spec) -> bool {
    const std::regex matcher(re);
    // Check if regex applies
    std::cmatch m;
    return std::regex_match(spec.name.data(), m, matcher);
  };
}

bool dataDeps(DataProcessorSpec const& a, DataProcessorSpec const& b)
{
  for (size_t ii = 0; ii < a.inputs.size(); ++ii) {
    for (size_t oi = 0; oi < b.outputs.size(); ++oi) {
      try {
        if (DataSpecUtils::match(a.inputs[ii], b.outputs[oi])) {
          return true;
        }
      } catch (...) {
        continue;
      }
    }
  }
  return false;
};

TopologyPolicy::DependencyChecker TopologyPolicyHelpers::dataDependency()
{
  return [](DataProcessorSpec const& a, DataProcessorSpec const& b) {
    return dataDeps(a, b);
  };
}

TopologyPolicy::DependencyChecker TopologyPolicyHelpers::alwaysDependent()
{
  return [](DataProcessorSpec const& dependent, DataProcessorSpec const& ancestor) {
    if (dependent.name == ancestor.name) {
      return false;
    }
    if (ancestor.name == "internal-dpl-injected-dummy-sink") {
      return false;
    }
    const std::regex matcher(".*output-proxy.*");
    // Check if regex applies
    std::cmatch m;
    if (std::regex_match(ancestor.name.data(), m, matcher) && std::regex_match(ancestor.name.data(), m, matcher)) {
      return dataDeps(dependent, ancestor);
    }
    return true;
  };
}

std::vector<TopologyPolicy> TopologyPolicy::createDefaultPolicies()
{
  return {
    {TopologyPolicyHelpers::matchByRegex(".*output-proxy.*"), TopologyPolicyHelpers::alwaysDependent()},
    {TopologyPolicyHelpers::matchAll(), TopologyPolicyHelpers::dataDependency()}};
}

} // namespace o2::framework
