// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataProcessorSpec.h"
#include "Framework/TopologyPolicy.h"
#include <string>

namespace o2::framework
{

struct TopologyPolicyHelpers {
  static TopologyPolicy::DataProcessorMatcher matchAll();
  static TopologyPolicy::DataProcessorMatcher matchByName(std::string const& name);
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

TopologyPolicy::DependencyChecker TopologyPolicyHelpers::dataDependency()
{
  return [](DataProcessorSpec const& a, DataProcessorSpec const& b) {
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
}

TopologyPolicy::DependencyChecker TopologyPolicyHelpers::alwaysDependent()
{
  return [](DataProcessorSpec const& dependent, DataProcessorSpec const& ancestor) {
    if (dependent.name == ancestor.name) {
      return false;
    }
    return true;
  };
}

std::vector<TopologyPolicy> TopologyPolicy::createDefaultPolicies()
{
  return {
    {TopologyPolicyHelpers::matchByName("dpl-output-proxy"), TopologyPolicyHelpers::alwaysDependent()},
    {TopologyPolicyHelpers::matchAll(), TopologyPolicyHelpers::dataDependency()}};
}

} // namespace o2::framework
