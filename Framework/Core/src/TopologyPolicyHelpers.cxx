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

#include "Framework/TopologyPolicyHelpers.h"
#include "Framework/TopologyPolicy.h"

namespace o2::framework
{
namespace
{
void describeDataProcessorSpec(std::ostream& stream, DataProcessorSpec const& spec)
{
  stream << spec.name;
  if (!spec.labels.empty()) {
    stream << "(";
    bool first = false;
    for (auto& label : spec.labels) {
      stream << (first ? "" : ",") << label.value;
      first = true;
    }
    stream << ")";
  }
}
} // namespace

auto TopologyPolicyHelpers::buildEdges(WorkflowSpec& physicalWorkflow) -> std::vector<std::pair<int, int>>
{
  std::vector<TopologyPolicy> topologyPolicies = TopologyPolicy::createDefaultPolicies();
  std::vector<TopologyPolicy::DependencyChecker> dependencyCheckers;
  dependencyCheckers.reserve(physicalWorkflow.size());

  for (auto& spec : physicalWorkflow) {
    for (auto& policy : topologyPolicies) {
      if (policy.matcher(spec)) {
        dependencyCheckers.push_back(policy.checkDependency);
        break;
      }
    }
  }
  assert(dependencyCheckers.size() == physicalWorkflow.size());
  // check if DataProcessorSpec at i depends on j
  auto checkDependencies = [&workflow = physicalWorkflow,
                            &dependencyCheckers](int i, int j) {
    TopologyPolicy::DependencyChecker& checker = dependencyCheckers[i];
    return checker(workflow[i], workflow[j]);
  };
  std::vector<std::pair<int, int>> edges;
  for (size_t i = 0; i < physicalWorkflow.size() - 1; ++i) {
    for (size_t j = i; j < physicalWorkflow.size(); ++j) {
      if (i == j && checkDependencies(i, j)) {
        throw std::runtime_error(physicalWorkflow[i].name + " depends on itself");
      }
      bool both = false;
      if (checkDependencies(i, j)) {
        edges.emplace_back(j, i);
        both = true;
      }
      if (checkDependencies(j, i)) {
        edges.emplace_back(i, j);
        if (both) {
          std::ostringstream str;
          describeDataProcessorSpec(str, physicalWorkflow[i]);
          str << " has circular dependency with ";
          describeDataProcessorSpec(str, physicalWorkflow[j]);
          str << ":\n";
          for (auto x : {i, j}) {
            str << physicalWorkflow[x].name << ":\n";
            str << "inputs:\n";
            for (auto& input : physicalWorkflow[x].inputs) {
              str << "- " << input << " " << (int)input.lifetime << "\n";
            }
            str << "outputs:\n";
            for (auto& output : physicalWorkflow[x].outputs) {
              str << "- " << output << " " << (int)output.lifetime << "\n";
            }
          }
          throw std::runtime_error(str.str());
        }
      }
    }
  }
  return edges;
};
} // namespace o2::framework
