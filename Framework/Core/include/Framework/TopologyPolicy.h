// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TOPOLOGYPOLICY_H_
#define O2_FRAMEWORK_TOPOLOGYPOLICY_H_

#include <functional>
#include <vector>

namespace o2::framework
{
struct DataProcessorSpec;

struct TopologyPolicy {
  using DataProcessorMatcher = std::function<bool(DataProcessorSpec const& device)>;
  using DependencyChecker = std::function<bool(DataProcessorSpec const& dependent, DataProcessorSpec const& ascendant)>;
  DataProcessorMatcher matcher;
  DependencyChecker checkDependency;
  static std::vector<TopologyPolicy> createDefaultPolicies();
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_TOPOLOGYPOLICY_H_
