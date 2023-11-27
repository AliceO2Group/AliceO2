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

/// Return true if a depends on b, i.e. if any of the inputs of a
/// is satisfied by any of the outputs of b.
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
}

bool expendableDataDeps(DataProcessorSpec const& a, DataProcessorSpec const& b)
{
  // We never put anything behind the dummy sink.
  if (b.name.find("internal-dpl-injected-dummy-sink") != std::string::npos) {
    return false;
  }
  if (a.name.find("internal-dpl-injected-dummy-sink") != std::string::npos) {
    return true;
  }
  /// If there is an actual dependency between a and b, we return true.
  if (dataDeps(a, b)) {
    return true;
  }
  // If we are here we do not have any data dependency,
  // however we strill consider a dependent on b if
  // a has the "expendable" label and b does not.
  auto checkExpendable = [](DataProcessorLabel const& label) {
    if (label.value == "expendable") {
      return true;
    }
    return false;
  };
  // A task marked as expendable or resilient can be put after an expendable task
  auto checkResilient = [](DataProcessorLabel const& label) {
    if (label.value == "resilient") {
      return true;
    }
    return false;
  };
  bool isBExpendable = std::find_if(b.labels.begin(), b.labels.end(), checkExpendable) != b.labels.end();
  bool isAExpendable = std::find_if(a.labels.begin(), a.labels.end(), checkExpendable) != a.labels.end();
  bool bResilient = std::find_if(b.labels.begin(), b.labels.end(), checkResilient) != b.labels.end();

  // If none is expendable. We simply return false and sort as usual.
  if (!isAExpendable && !isBExpendable) {
    LOGP(debug, "Neither {} nor {} are expendable. No dependency beyond data deps.", a.name, b.name);
    return false;
  }
  // If both are expendable. We return false and sort as usual.
  if (isAExpendable && isBExpendable) {
    LOGP(debug, "Both {} and {} are expendable. No dependency.", a.name, b.name);
    return false;
  }

  // If b is expendable but b is resilient, we can keep the same order.
  if (isAExpendable && bResilient) {
    LOGP(debug, "{} is expendable but b is resilient, no need to add an unneeded dependency", a.name, a.name, b.name);
    return false;
  }
  // If a is expendable we consider it as if there was a dependency from a to b,
  // however we still need to check if there is not one already from b to a.
  if (isAExpendable) {
    LOGP(debug, "{} is expendable. Checking if there is a dependency from {} to {}.", a.name, b.name, a.name);
    return !dataDeps(b, a);
  }
  // b is expendable and a is not. We are fine with no dependency.
  return false;
};

TopologyPolicy::DependencyChecker TopologyPolicyHelpers::dataDependency()
{
  return [](DataProcessorSpec const& a, DataProcessorSpec const& b) {
    return expendableDataDeps(a, b);
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
