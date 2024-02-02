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
#include "Framework/Signpost.h"
#include <string>
#include <regex>

O2_DECLARE_DYNAMIC_LOG(topology);

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
  O2_SIGNPOST_ID_GENERATE(sid, topology);
  O2_SIGNPOST_START(topology, sid, "expendableDataDeps", "Checking if %s depends on %s", a.name.c_str(), b.name.c_str());
  // We never put anything behind the dummy sink.
  if (b.name.find("internal-dpl-injected-dummy-sink") != std::string::npos) {
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "false. %s is dummy sink and it nothing can depend on it.", b.name.c_str());
    return false;
  }
  if (a.name.find("internal-dpl-injected-dummy-sink") != std::string::npos) {
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "true. %s is dummy sink and it nothing can depend on it.", a.name.c_str());
    return true;
  }
  /// If there is an actual dependency between a and b, we return true.
  if (dataDeps(a, b)) {
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "true. %s has a data dependency on %s", a.name.c_str(), b.name.c_str());
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
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "false. Neither %s nor %s are expendable. No dependency beyond data deps.",
                    a.name.c_str(), b.name.c_str());
    return false;
  }
  // If both are expendable. We return false and sort as usual.
  if (isAExpendable && isBExpendable) {
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "false. Both %s and %s are expendable. No dependency.",
                    a.name.c_str(), b.name.c_str());
    return false;
  }

  // If b is expendable but b is resilient, we can keep the same order.
  if (isAExpendable && bResilient) {
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "false. %s is expendable but %s is resilient, no need to add an unneeded dependency",
                    a.name.c_str(), b.name.c_str());
    return false;
  }
  // If a is expendable we consider it as if there was a dependency from a to b,
  // however we still need to check if there is not one already from b to a.
  if (isAExpendable) {
    bool hasDependency = dataDeps(b, a);
    O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "%s is expendable. %s from %s to %s => %s.",
                    a.name.c_str(), hasDependency ? "There is however an inverse dependency" : "No inverse dependency", b.name.c_str(), a.name.c_str(),
                    !hasDependency ? "true" : "false");
    return !hasDependency;
  }
  // b is expendable and a is not. We are fine with no dependency.
  O2_SIGNPOST_END(topology, sid, "expendableDataDeps", "false. %s is expendable but %s is not. No need to add an unneeded dependency.",
                  b.name.c_str(), a.name.c_str());
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
    O2_SIGNPOST_ID_GENERATE(sid, topology);
    O2_SIGNPOST_START(topology, sid, "alwaysDependent", "Checking if %s depends on %s", dependent.name.c_str(), ancestor.name.c_str());
    if (dependent.name == ancestor.name) {
      O2_SIGNPOST_END(topology, sid, "alwaysDependent", "false. %s and %s are the same.", dependent.name.c_str(), ancestor.name.c_str());
      return false;
    }
    if (ancestor.name == "internal-dpl-injected-dummy-sink") {
      O2_SIGNPOST_END(topology, sid, "alwaysDependent", "false. %s is a dummy sink.", ancestor.name.c_str());
      return false;
    }
    const std::regex matcher(".*output-proxy.*");
    // Check if regex applies
    std::cmatch m;
    bool isAncestorOutputProxy = std::regex_match(ancestor.name.data(), m, matcher);
    // For now dependent is always an output proxy.
    assert(std::regex_match(dependent.name.data(), m, matcher));
    bool isAncestorExpendable = std::find_if(ancestor.labels.begin(), ancestor.labels.end(), [](DataProcessorLabel const& label) {
                                  return label.value == "expendable";
                                }) != ancestor.labels.end();

    bool isDependentResilient = std::find_if(dependent.labels.begin(), dependent.labels.end(), [](DataProcessorLabel const& label) {
                                  return label.value == "resilient";
                                }) != dependent.labels.end();
    bool isAncestorResilient = std::find_if(ancestor.labels.begin(), ancestor.labels.end(), [](DataProcessorLabel const& label) {
                                 return label.value == "resilient";
                               }) != ancestor.labels.end();

    if (!isDependentResilient && isAncestorExpendable) {
      O2_SIGNPOST_END(topology, sid, "alwaysDependent", "false. Ancestor %s is expendable while %s is non-resilient output proxy (dependent).",
                      ancestor.name.c_str(), dependent.name.c_str());
      return false;
    }

    if (isAncestorOutputProxy || (!isDependentResilient && isAncestorResilient)) {
      bool hasDependency = dataDeps(dependent, ancestor);
      O2_SIGNPOST_END(topology, sid, "alwaysDependent", "%s. Dependent %s %s a dependency on ancestor %s.",
                      hasDependency ? "true" : "false", dependent.name.c_str(), hasDependency ? "has" : "has not", ancestor.name.c_str());
      return hasDependency;
    }
    O2_SIGNPOST_END(topology, sid, "alwaysDependent", "true by default. Ancestor %s is not an output proxy.", ancestor.name.c_str());
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
