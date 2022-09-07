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

#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/InputRecord.h"
#include "Framework/DeviceSpec.h"
#include <functional>
#include <iostream>

namespace o2::framework
{

/// By default the CompletionPolicy matches any Device and only runs a
/// computation when all the inputs are there.
std::vector<CompletionPolicy>
  CompletionPolicy::createDefaultPolicies()
{
  return {
    CompletionPolicyHelpers::defineByNameOrigin("internal-dpl-aod-writer", "TFN", CompletionOp::Consume),
    CompletionPolicyHelpers::defineByNameOrigin("internal-dpl-aod-writer", "TFF", CompletionOp::Consume),
    CompletionPolicyHelpers::consumeWhenAny("internal-dpl-injected-dummy-sink", [](DeviceSpec const& s) { return s.name == "internal-dpl-injected-dummy-sink"; }),
    CompletionPolicyHelpers::consumeWhenAll()};
}

std::ostream& operator<<(std::ostream& oss, CompletionPolicy::CompletionOp const& val)
{
  switch (val) {
    case CompletionPolicy::CompletionOp::Consume:
      oss << "consume";
      break;
    case CompletionPolicy::CompletionOp::Process:
      oss << "process";
      break;
    case CompletionPolicy::CompletionOp::Wait:
      oss << "wait";
      break;
    case CompletionPolicy::CompletionOp::Discard:
      oss << "discard";
      break;
    case CompletionPolicy::CompletionOp::ConsumeExisting:
      oss << "consumeExisting";
      break;
    case CompletionPolicy::CompletionOp::ConsumeAndRescan:
      oss << "consumeAndRescan";
      break;
    case CompletionPolicy::CompletionOp::Retry:
      oss << "retry";
      break;
  };
  return oss;
}

} // namespace o2::framework
