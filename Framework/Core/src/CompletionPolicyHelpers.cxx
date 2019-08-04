// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/InputRecord.h"
#include "Framework/DeviceSpec.h"
#include "Framework/PartRef.h"
#include "Framework/CompilerBuiltins.h"

#include <gsl/span>

#include <cassert>

namespace o2
{
namespace framework
{

CompletionPolicy CompletionPolicyHelpers::defineByName(std::string const& name, CompletionPolicy::CompletionOp op)
{
  auto matcher = [name](DeviceSpec const& device) -> bool {
    return device.name == name;
  };
  auto callback = [op](gsl::span<PartRef const> const& inputs) -> CompletionPolicy::CompletionOp {
    return op;
  };
  switch (op) {
    case CompletionPolicy::CompletionOp::Consume:
      return CompletionPolicy{"always-consume", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Process:
      return CompletionPolicy{"always-process", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Wait:
      return CompletionPolicy{"always-wait", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Discard:
      return CompletionPolicy{"always-discard", matcher, callback};
      break;
  }
  O2_BUILTIN_UNREACHABLE();
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAll()
{
  auto matcher = [](DeviceSpec const&) -> bool { return true; };
  auto callback = [](gsl::span<PartRef const> const& inputs) -> CompletionPolicy::CompletionOp {
    for (auto& input : inputs) {
      if (input.header == nullptr && input.payload == nullptr) {
        return CompletionPolicy::CompletionOp::Wait;
      }
    }
    return CompletionPolicy::CompletionOp::Consume;
  };
  return CompletionPolicy{"consume-all", matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAny()
{
  auto matcher = [](DeviceSpec const&) -> bool { return true; };
  auto callback = [](gsl::span<PartRef const> const& inputs) -> CompletionPolicy::CompletionOp {
    for (auto& input : inputs) {
      if (input.header != nullptr && input.payload != nullptr) {
        return CompletionPolicy::CompletionOp::Consume;
      }
    }
    return CompletionPolicy::CompletionOp::Wait;
  };
  return CompletionPolicy{"consume-any", matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::processWhenAny()
{
  auto matcher = [](DeviceSpec const&) -> bool { return true; };
  auto callback = [](gsl::span<PartRef const> const& inputs) -> CompletionPolicy::CompletionOp {
    size_t present = 0;
    for (auto& input : inputs) {
      if (input.header != nullptr) {
        present++;
      }
    }
    if (present == inputs.size()) {
      return CompletionPolicy::CompletionOp::Consume;
    } else if (present == 0) {
      return CompletionPolicy::CompletionOp::Wait;
    }
    return CompletionPolicy::CompletionOp::Process;
  };
  return CompletionPolicy{"process-any", matcher, callback};
}

} // namespace framework
} // namespace o2
