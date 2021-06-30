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

#include "Framework/DispatchPolicy.h"
#include "Framework/DeviceSpec.h"
#include <functional>
#include <iostream>

namespace o2
{
namespace framework
{

/// By default the DispatchPolicy matches any Device and messages are sent
/// after computation
std::vector<DispatchPolicy> DispatchPolicy::createDefaultPolicies()
{
  return {DispatchPolicy{"dispatch-all-after-computation", [](DeviceSpec const&) { return true; }, DispatchPolicy::DispatchOp::AfterComputation}};
}

DispatchPolicy::TriggerMatcher DispatchPolicy::defaultDispatchPolicy()
{
  return DispatchPolicy::TriggerMatcher{[](Output const&) -> bool { return true; }};
}

std::ostream& operator<<(std::ostream& oss, DispatchPolicy::DispatchOp const& val)
{
  switch (val) {
    case DispatchPolicy::DispatchOp::AfterComputation:
      oss << "after computation";
      break;
    case DispatchPolicy::DispatchOp::WhenReady:
      oss << "when ready";
      break;
  };
  return oss;
}

} // namespace framework
} // namespace o2
