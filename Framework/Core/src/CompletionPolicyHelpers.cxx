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
#include "Framework/TimesliceIndex.h"
#include "Framework/TimingInfo.h"
#include "DecongestionService.h"
#include "Framework/Signpost.h"

#include <cassert>
#include <regex>

O2_DECLARE_DYNAMIC_LOG(completion);

namespace o2::framework
{

CompletionPolicy CompletionPolicyHelpers::defineByNameOrigin(std::string const& name, std::string const& origin, CompletionPolicy::CompletionOp op)
{
  auto matcher = [name](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(name));
  };

  auto originReceived = std::make_shared<std::vector<uint64_t>>();

  auto callback = [originReceived, origin, op](InputSpan const& inputRefs, std::vector<InputSpec> const&, ServiceRegistryRef&) -> CompletionPolicy::CompletionOp {
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
  auto callback = [op](InputSpan const&, std::vector<InputSpec> const& specs, ServiceRegistryRef& ref) -> CompletionPolicy::CompletionOp {
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
  auto callback = [](InputSpan const& inputs, std::vector<InputSpec> const& specs, ServiceRegistryRef& ref) -> CompletionPolicy::CompletionOp {
    assert(inputs.size() == specs.size());
    O2_SIGNPOST_ID_GENERATE(sid, completion);
    O2_SIGNPOST_START(completion, sid, "consumeWhenAll", "Completion policy invoked");

    size_t si = 0;
    int sporadicCount = 0;
    int timeframeCount = 0;
    int missingSporadicCount = 0;
    bool needsProcessing = false;
    size_t currentTimeslice = -1;
    for (auto& input : inputs) {
      assert(si < specs.size());
      auto& spec = specs[si++];
      sporadicCount += spec.lifetime == Lifetime::Sporadic ? 1 : 0;
      timeframeCount += spec.lifetime == Lifetime::Timeframe ? 1 : 0;
      // If we are missing something which is not sporadic, we wait.
      if (input.header == nullptr && spec.lifetime != Lifetime::Sporadic) {
        O2_SIGNPOST_END(completion, sid, "consumeWhenAll", "Completion policy returned %{public}s due to missing input %lu", "Wait", si);
        return CompletionPolicy::CompletionOp::Wait;
      }
      // If we are missing something which is sporadic, we wait until we are sure it will not come.
      if (input.header == nullptr && spec.lifetime == Lifetime::Sporadic) {
        O2_SIGNPOST_EVENT_EMIT(completion, sid, "consumeWhenAll", "Missing sporadic found for route index %lu", si);
        missingSporadicCount += 1;
      }
      // If we have a header, we use it to determine the current timesliceIsTimer
      // (unless this is a timer which does not enter the oldest possible timeslice).
      if (input.header != nullptr && currentTimeslice == -1) {
        auto const* dph = framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(input);
        if (dph && !TimingInfo::timesliceIsTimer(dph->startTime)) {
          currentTimeslice = dph->startTime;
          O2_SIGNPOST_EVENT_EMIT(completion, sid, "consumeWhenAll", "currentTimeslice %lu from route index %lu", currentTimeslice, si);
        }
      }
      // If we have a header, we need to process it if it is not a condition object.
      if (input.header != nullptr && spec.lifetime != Lifetime::Condition) {
        needsProcessing = true;
      }
    }
    // If some sporadic inputs are missing, we wait for them util we are sure they will not come,
    // i.e. until the oldest possible timeslice is beyond the timeslice of the input.
    auto& timesliceIndex = ref.get<TimesliceIndex>();
    auto oldestPossibleTimeslice = timesliceIndex.getOldestPossibleInput().timeslice.value;

    if (missingSporadicCount && currentTimeslice >= oldestPossibleTimeslice) {
      O2_SIGNPOST_END(completion, sid, "consumeWhenAll", "Completion policy returned %{public}s for timeslice %lu > oldestPossibleTimeslice %lu", "Retry", currentTimeslice, oldestPossibleTimeslice);
      return CompletionPolicy::CompletionOp::Retry;
    }

    // No need to process if we have only sporadic inputs and they are all missing.
    if (needsProcessing && (sporadicCount > 0) && (missingSporadicCount == sporadicCount) && (timeframeCount == 0)) {
      O2_SIGNPOST_END(completion, sid, "consumeWhenAll", "Completion policy returned %{public}s for timeslice %lu", "Discard", currentTimeslice);
      return CompletionPolicy::CompletionOp::Discard;
    }
    auto consumes = (needsProcessing || sporadicCount == 0);
    O2_SIGNPOST_END(completion, sid, "consumeWhenAll", "Completion policy returned %{public}s for timeslice %lu", consumes ? "Consume" : "Discard", currentTimeslice);
    return consumes ? CompletionPolicy::CompletionOp::Consume : CompletionPolicy::CompletionOp::Discard;
  };
  return CompletionPolicy{name, matcher, callback};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAllOrdered(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callbackFull = [](InputSpan const& inputs, std::vector<InputSpec> const&, ServiceRegistryRef& ref) -> CompletionPolicy::CompletionOp {
    auto& decongestionService = ref.get<DecongestionService>();
    decongestionService.orderedCompletionPolicyActive = true;
    for (auto& input : inputs) {
      if (input.header == nullptr) {
        return CompletionPolicy::CompletionOp::Wait;
      }
      long int startTime = framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(input)->startTime;
      if (startTime == 0) {
        LOGP(debug, "startTime is 0, which means we have the first message, so we can process it.");
        decongestionService.nextTimeslice = 0;
      }
      if (framework::DataRefUtils::isValid(input) && startTime != decongestionService.nextTimeslice) {
        return CompletionPolicy::CompletionOp::Retry;
      }
    }
    decongestionService.nextTimeslice++;
    return CompletionPolicy::CompletionOp::ConsumeAndRescan;
  };
  return CompletionPolicy{name, matcher, callbackFull};
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
    [](InputSpan const& inputs, std::vector<InputSpec> const& specs, ServiceRegistryRef&) -> CompletionPolicy::CompletionOp {
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
    }};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAny(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](InputSpan const& inputs, std::vector<InputSpec> const&, ServiceRegistryRef& ref) -> CompletionPolicy::CompletionOp {
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

CompletionPolicy CompletionPolicyHelpers::consumeWhenAnyWithAllConditions(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](InputSpan const& inputs, std::vector<InputSpec> const& specs, ServiceRegistryRef&) -> CompletionPolicy::CompletionOp {
    bool canConsume = false;
    bool hasConditions = false;
    bool conditionMissing = false;
    size_t timeslice = -1;
    static size_t timesliceOK = -1; // FIXME: This breaks start/stop/start, since it must be reset!
                                    // FIXME: Also, this just checks the max timeslice that was already consumed.
                                    // In case timeslices do not come in order, we might have consumed a later
                                    // condition object, but not the one for the current time slice.
                                    // But I don't see any possibility to handle this in a better way.

    // Iterate on all specs and all inputs simultaneously
    for (size_t i = 0; i < inputs.size(); ++i) {
      char const* header = inputs.header(i);
      auto& spec = specs[i];
      // In case a condition object is not there, we need to wait.
      if (header != nullptr) {
        canConsume = true;
      }
      if (spec.lifetime == Lifetime::Condition) {
        hasConditions = true;
        if (header == nullptr) {
          conditionMissing = true;
        }
      }
    }
    if (canConsume || conditionMissing) {
      for (auto it = inputs.begin(), end = inputs.end(); it != end; ++it) {
        for (auto const& ref : it) {
          if (!framework::DataRefUtils::isValid(ref)) {
            continue;
          }
          auto const* dph = framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
          if (dph && !TimingInfo::timesliceIsTimer(dph->startTime)) {
            timeslice = dph->startTime;
            break;
          }
        }
        if (timeslice != -1) {
          break;
        }
      }
    }

    // If there are no conditions, just consume.
    if (!hasConditions) {
      canConsume = true;
    } else if (conditionMissing && (timeslice == -1 || timesliceOK == -1 || timeslice > timesliceOK)) {
      return CompletionPolicy::CompletionOp::Wait;
    }

    if (canConsume && timeslice != -1 && (timeslice > timesliceOK || timesliceOK == -1)) {
      timesliceOK = timeslice;
    }
    return canConsume ? CompletionPolicy::CompletionOp::Consume : CompletionPolicy::CompletionOp::Wait;
  };
  return CompletionPolicy{name, matcher, callback, false};
}

CompletionPolicy CompletionPolicyHelpers::consumeWhenAnyWithAllConditions(std::string matchName)
{
  auto matcher = [matchName](DeviceSpec const& device) -> bool {
    return std::regex_match(device.name.begin(), device.name.end(), std::regex(matchName));
  };
  return consumeWhenAnyWithAllConditions(matcher);
}

CompletionPolicy CompletionPolicyHelpers::processWhenAny(const char* name, CompletionPolicy::Matcher matcher)
{
  auto callback = [](InputSpan const& inputs, std::vector<InputSpec> const&, ServiceRegistryRef& ref) -> CompletionPolicy::CompletionOp {
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
