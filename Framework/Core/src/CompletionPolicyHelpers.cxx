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

#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/InputSpan.h"
#include "Framework/DeviceSpec.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Logger.h"

#include <cassert>
#include <regex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
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
            LOGP(info, "Have to wait until message of origin {} with startTime {} has arrived.", origin, startTime);
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
      return consumeWhenAny(name.c_str(), matcher);
      break;
    case CompletionPolicy::CompletionOp::ConsumeExisting:
      return CompletionPolicy{"consume-existing", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Process:
      return CompletionPolicy{"always-process", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Wait:
      return CompletionPolicy{"always-wait", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Discard:
      return CompletionPolicy{"always-discard", matcher, callback, false};
      break;
    case CompletionPolicy::CompletionOp::ConsumeAndRescan:
      return CompletionPolicy{"always-rescan", matcher, callback};
      break;
    case CompletionPolicy::CompletionOp::Retry:
      return CompletionPolicy{"retry", matcher, callback};
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

CompletionPolicy CompletionPolicyHelpers::consumeWhenAllOrdered(const char* name, CompletionPolicy::Matcher matcher)
{
  auto nextTimeSlice = std::make_shared<long int>(0);
  auto callback = [nextTimeSlice](InputSpan const& inputs) -> CompletionPolicy::CompletionOp {
    for (auto& input : inputs) {
      if (input.header == nullptr) {
        return CompletionPolicy::CompletionOp::Wait;
      }
      long int startTime = framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(input)->startTime;
      if (startTime == 0) {
        LOGP(info, "startTime is 0, which means we have the first message, so we can process it.");
        *nextTimeSlice = 0;
      }
      if (framework::DataRefUtils::isValid(input) && startTime != *nextTimeSlice) {
        return CompletionPolicy::CompletionOp::Retry;
      }
    }
    (*nextTimeSlice)++;
    return CompletionPolicy::CompletionOp::ConsumeAndRescan;
  };
  return CompletionPolicy{name, matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAllOrdered(std::string matchName)
{
  auto matcher = [matchName](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(matchName));
  };
  return consumeWhenAllOrdered(matcher);
}

CompletionPolicy CompletionPolicyHelpers::consumeExistingWhenAny(const char* name, CompletionPolicy::Matcher matcher)
{
  return CompletionPolicy{
    name,
    matcher,
    [](InputSpan const& inputs, std::vector<InputSpec> const& specs) -> CompletionPolicy::CompletionOp {
      size_t present = 0;
      size_t current = 0;
      size_t withPayload = 0;
      size_t sporadic = 0;
      size_t maxSporadic = 0;
      size_t i = 0;
      for (auto& input : inputs) {
        auto& spec = specs[i++];
        if (spec.lifetime == Lifetime::Sporadic) {
          maxSporadic++;
        }
        if (input.header != nullptr) {
          present++;
          if (spec.lifetime == Lifetime::Sporadic) {
            sporadic++;
          }
        }
        if (input.payload != nullptr) {
          withPayload++;
        }
        current++;
      }
      // * In case we have all inputs but the sporadic ones: Consume, since we do not know if the sporadic ones
      // will ever come.
      // * In case we have only sporadic inputs: Consume, since we do not know if we already Consumed
      // the non sporadic ones above.
      // * In case we do not have payloads: Wait
      // * In all other cases we consume what is there, but we wait for the non sporadic ones to be complete
      //   (i.e. we wait for present + maxSporadic).
      if (present - sporadic + maxSporadic == inputs.size()) {
        return CompletionPolicy::CompletionOp::Consume;
      } else if (present - sporadic == 0) {
        return CompletionPolicy::CompletionOp::Consume;
      } else if (withPayload == 0) {
        return CompletionPolicy::CompletionOp::Wait;
      }
      return CompletionPolicy::CompletionOp::ConsumeAndRescan;
    }

  };
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
  return CompletionPolicy{name, matcher, callback, false};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAny(std::string matchName)
{
  auto matcher = [matchName](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(matchName));
  };
  return consumeWhenAny(matcher);
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

} // namespace o2::framework
#pragma GCC diagnostic pop
