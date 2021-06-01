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
#include "Framework/InputSpan.h"
#include "Framework/DeviceSpec.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Logger.h"

#include <cassert>
#include <regex>

namespace o2
{
namespace framework
{

CompletionPolicy CompletionPolicyHelpers::defineByNameOrigin(std::string const& name, std::string const& origin, CompletionPolicy::CompletionOp op)
{
  auto matcher = [name](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(name));
  };

  auto originReceived = std::make_shared<std::vector<uint64_t>>();

  auto callback = [originReceived, origin, op](InputSpan const& inputRefs) -> CompletionPolicy::CompletionOp {
    // update list of the start times of inputs with origin @origin
    for (auto& ref : inputRefs) {
      if (ref.header != nullptr) {
        auto header = CompletionPolicyHelpers::getHeader<o2::header::DataHeader>(ref);
        if (header->dataOrigin.str == origin) {
          auto startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->startTime;
          auto it = std::find(originReceived->begin(), originReceived->end(), startTime);
          if (it == originReceived->end()) {
            originReceived->emplace_back(startTime);
          }
        }
      }
    }

    // find out if all inputs which are not of origin @origin have a corresponding entry in originReceived
    // if one is missing then we have to wait
    for (auto& ref : inputRefs) {
      if (ref.header != nullptr) {
        auto header = CompletionPolicyHelpers::getHeader<o2::header::DataHeader>(ref);
        if (header->dataOrigin.str != origin) {
          auto startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->startTime;
          auto it = std::find(originReceived->begin(), originReceived->end(), startTime);
          if (it == originReceived->end()) {
            LOGP(INFO, "Have to wait until message of origin {} with startTime {} has arrived.", origin, startTime);
            return CompletionPolicy::CompletionOp::Wait;
          }
        }
      }
    }
    return op;
  };
  return CompletionPolicy{"wait-origin", matcher, callback};

  O2_BUILTIN_UNREACHABLE();
}

CompletionPolicy CompletionPolicyHelpers::defineByName(std::string const& name, CompletionPolicy::CompletionOp op)
{
  auto matcher = [name](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(name));
  };
  auto callback = [op](InputSpan const&) -> CompletionPolicy::CompletionOp {
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
  auto callback = [](InputSpan const& inputs) -> CompletionPolicy::CompletionOp {
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
  auto callback = [](InputSpan const& inputs) -> CompletionPolicy::CompletionOp {
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
  auto callback = [](InputSpan const& inputs) -> CompletionPolicy::CompletionOp {
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
