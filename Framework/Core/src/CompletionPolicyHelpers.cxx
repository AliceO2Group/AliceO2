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
#include "Framework/DeviceSpec.h"
#include "Framework/CompilerBuiltins.h"

#include <cassert>
#include <regex>

namespace o2
{
namespace framework
{

CompletionPolicy CompletionPolicyHelpers::defineByName(std::string const& name, CompletionPolicy::CompletionOp op)
{
  auto matcher = [name](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(name));
  };
  auto callback = [op](CompletionPolicy::InputSet) -> CompletionPolicy::CompletionOp {
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

CompletionPolicy CompletionPolicyHelpers::consumeWhenAll(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](CompletionPolicy::InputSet inputs) -> CompletionPolicy::CompletionOp {
    for (auto& input : inputs) {
      if (input.header == nullptr) {
        return CompletionPolicy::CompletionOp::Wait;
      }
    }
    return CompletionPolicy::CompletionOp::Consume;
  };
  return CompletionPolicy{name, matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAny(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](CompletionPolicy::InputSet inputs) -> CompletionPolicy::CompletionOp {
    for (auto& input : inputs) {
      if (input.header != nullptr) {
        return CompletionPolicy::CompletionOp::Consume;
      }
    }
    return CompletionPolicy::CompletionOp::Wait;
  };
  return CompletionPolicy{name, matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::processWhenAny(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](CompletionPolicy::InputSet inputs) -> CompletionPolicy::CompletionOp {
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
  return CompletionPolicy{name, matcher, callback};
}

} // namespace framework
} // namespace o2
