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

#include "Framework/DataProcessingHeader.h"
#include "Framework/InputSpec.h"
#include "Framework/LifetimeHelpers.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/Logger.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/CallbackService.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/VariableContextHelpers.h"
#include "Framework/DataTakingContext.h"
#include "Framework/InputRecord.h"
#include "Framework/FairMQDeviceProxy.h"

#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/Stack.h"
#include "CommonConstants/LHCConstants.h"
#include "MemoryResources/MemoryResources.h"
#include <typeinfo>
#include <TError.h>
#include <TMemFile.h>
#include <curl/curl.h>

#include <fairmq/Device.h>

#include <cstdlib>
#include <random>

using namespace o2::header;
using namespace fair;

namespace o2::framework
{

namespace
{
size_t getCurrentTime()
{
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}
} // namespace

ExpirationHandler::Creator LifetimeHelpers::dataDrivenCreation()
{
  return [](ChannelIndex, TimesliceIndex&) -> TimesliceSlot {
    return {TimesliceSlot::ANY};
  };
}

ExpirationHandler::Creator LifetimeHelpers::enumDrivenCreation(size_t start, size_t end, size_t step, size_t inputTimeslice, size_t maxInputTimeslices, size_t maxRepetitions)
{
  auto last = std::make_shared<size_t>(start + inputTimeslice * step);
  auto repetition = std::make_shared<size_t>(0);

  return [end, step, last, maxInputTimeslices, maxRepetitions, repetition](ChannelIndex channelIndex, TimesliceIndex& index) -> TimesliceSlot {
    for (size_t si = 0; si < index.size(); si++) {
      if (*last > end) {
        LOGP(debug, "Last greater than end");
        return TimesliceSlot{TimesliceSlot::INVALID};
      }
      auto slot = TimesliceSlot{si};
      if (index.isValid(slot) == false) {
        TimesliceId timestamp{*last};
        *repetition += 1;
        if (*repetition % maxRepetitions == 0) {
          *last += step * maxInputTimeslices;
        }
        LOGP(debug, "Associating timestamp {} to slot {}", timestamp.value, slot.index);
        index.associate(timestamp, slot);
        // We know that next association will bring in last
        // so we can state this will be the latest possible input for the channel
        // associated with this.
        LOG(debug) << "Oldest possible input is " << *last;
        auto newOldest = index.setOldestPossibleInput({*last}, channelIndex);
        index.updateOldestPossibleOutput();
        return slot;
      }
    }

    LOGP(debug, "No slots available");
    return TimesliceSlot{TimesliceSlot::INVALID};
  };
}

ExpirationHandler::Creator LifetimeHelpers::timeDrivenCreation(std::chrono::microseconds period)
{
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<uint64_t> dist(0, period.count() * 0.9);

  // We start with a random offset to avoid all the devices
  // send their first message at the same time, bring down
  // the QC machine.
  // We reduce the first interval rather than increasing it
  // to avoid having a triggered timer which appears to be in
  // the future.
  size_t start = getCurrentTime() - dist(e1) - period.count() * 0.1;
  auto last = std::make_shared<decltype(start)>(start);
  // FIXME: should create timeslices when period expires....
  return [last, period](ChannelIndex channelIndex, TimesliceIndex& index) -> TimesliceSlot {
    // Nothing to do if the time has not expired yet.
    auto current = getCurrentTime();
    auto delta = current - *last;
    if (delta < period.count()) {
      auto newOldest = index.setOldestPossibleInput({*last}, channelIndex);
      index.updateOldestPossibleOutput();
      return TimesliceSlot{TimesliceSlot::INVALID};
    }
    // We first check if the current time is not already present
    // FIXME: this should really be done by query matching? Ok
    //        for now to avoid duplicate entries.
    for (size_t i = 0; i < index.size(); ++i) {
      TimesliceSlot slot{i};
      if (index.isValid(slot) == false) {
        continue;
      }
      auto& variables = index.getVariablesForSlot(slot);
      if (VariableContextHelpers::getTimeslice(variables).value == current) {
        auto newOldest = index.setOldestPossibleInput({*last}, channelIndex);
        index.updateOldestPossibleOutput();
        return TimesliceSlot{TimesliceSlot::INVALID};
      }
    }
    // If we are here the timer has expired and a new slice needs
    // to be created.
    *last = current;
    data_matcher::VariableContext newContext;
    newContext.put({0, static_cast<uint64_t>(current)});
    newContext.commit();
    auto [action, slot] = index.replaceLRUWith(newContext, TimesliceId{current});
    switch (action) {
      case TimesliceIndex::ActionTaken::ReplaceObsolete:
      case TimesliceIndex::ActionTaken::ReplaceUnused:
        index.associate(TimesliceId{current}, slot);
        break;
      case TimesliceIndex::ActionTaken::DropInvalid:
      case TimesliceIndex::ActionTaken::DropObsolete:
      case TimesliceIndex::ActionTaken::Wait:
        break;
    }

    auto newOldest = index.setOldestPossibleInput({*last}, channelIndex);
    index.updateOldestPossibleOutput();
    return slot;
  };
}

ExpirationHandler::Checker LifetimeHelpers::expireNever()
{
  return [](ServiceRegistryRef, int64_t, InputSpan const&) -> bool { return false; };
}

ExpirationHandler::Checker LifetimeHelpers::expireAlways()
{
  return [](ServiceRegistryRef, int64_t, InputSpan const&) -> bool { return true; };
}

ExpirationHandler::Checker LifetimeHelpers::expireIfPresent(std::vector<InputRoute> const& routes, ConcreteDataMatcher)
{
  // find all the input routes which have timeframe data
  // and store it in a vector for use inside the lambda
  std::vector<InputRecord::InputPos> inputPositions;
  std::vector<InputRecord::InputPos> optionalPositions;
  size_t index = 0;
  for (auto& route : routes) {
    if (route.timeslice != 0) {
      continue;
    }
    if (route.matcher.lifetime != Lifetime::Optional) {
      LOGP(debug, "Lifetime of input route {} is not optional at position {}", route.matcher.binding, index);
      inputPositions.push_back({index});
    } else {
      LOGP(debug, "Lifetime of input route {} is optional at position {}", route.matcher.binding, index);
      optionalPositions.push_back({index});
    }
    index++;
  }

  return [inputPositions, optionalPositions, routes](ServiceRegistryRef, int64_t, InputSpan const& span) -> bool {
    // Check if timeframe data is fully present.
    // If yes, we expire the optional data.
    // If not, we continue to wait for the data.
    size_t requiredCount = 0;
    size_t optionalCount = 0;
    for (auto& inputPos : inputPositions) {
      auto ref = InputRecord::getByPos(routes, span, inputPos.index, 0);
      if (ref.header != nullptr) {
        requiredCount++;
      }
    }
    for (auto& inputPos : optionalPositions) {
      auto ref = InputRecord::getByPos(routes, span, inputPos.index, 0);
      if (ref.header != nullptr) {
        optionalCount++;
      }
    }
    LOGP(debug, "ExpireIfPresent: allRequired={}/{}, allOptional={}/{}", requiredCount, inputPositions.size(), optionalCount, optionalPositions.size());
    return (requiredCount == inputPositions.size()) && (optionalCount != optionalPositions.size());
  };
}

ExpirationHandler::Creator LifetimeHelpers::uvDrivenCreation(int requestedLoopReason, DeviceState& state)
{
  return [requestedLoopReason, &state](ChannelIndex, TimesliceIndex& index) -> TimesliceSlot {
    /// Not the expected loop reason, return an invalid slot.
    if ((state.loopReason & requestedLoopReason) == 0) {
      LOGP(debug, "No expiration due to a loop event. Requested: {:b}, reported: {:b}, matching: {:b}",
           requestedLoopReason,
           state.loopReason,
           requestedLoopReason & state.loopReason);
      return TimesliceSlot{TimesliceSlot::INVALID};
    }
    auto current = getCurrentTime();

    // We first check if the current time is not already present
    // FIXME: this should really be done by query matching? Ok
    //        for now to avoid duplicate entries.
    for (size_t i = 0; i < index.size(); ++i) {
      TimesliceSlot slot{i};
      if (index.isValid(slot) == false) {
        continue;
      }
      auto& variables = index.getVariablesForSlot(slot);
      if (VariableContextHelpers::getTimeslice(variables).value == current) {
        return TimesliceSlot{TimesliceSlot::INVALID};
      }
    }

    LOGP(debug, "Record was expired due to a loop event. Requested: {:b}, reported: {:b}, matching: {:b}",
         requestedLoopReason,
         state.loopReason,
         requestedLoopReason & state.loopReason);

    // If we are here the loop has triggered with the expected
    // event so we need to create a slot.
    data_matcher::VariableContext newContext;
    newContext.put({0, static_cast<uint64_t>(current)});
    newContext.commit();
    auto [action, slot] = index.replaceLRUWith(newContext, TimesliceId{current});
    switch (action) {
      case TimesliceIndex::ActionTaken::ReplaceObsolete:
      case TimesliceIndex::ActionTaken::ReplaceUnused:
        index.associate(TimesliceId{current}, slot);
        break;
      case TimesliceIndex::ActionTaken::DropInvalid:
      case TimesliceIndex::ActionTaken::DropObsolete:
      case TimesliceIndex::ActionTaken::Wait:
        break;
    }
    return slot;
  };
}

ExpirationHandler::Checker LifetimeHelpers::expireTimed(std::chrono::microseconds period)
{
  auto start = getCurrentTime();
  auto last = std::make_shared<decltype(start)>(start);
  return [last, period](ServiceRegistryRef, int64_t, InputSpan const&) -> bool {
    auto current = getCurrentTime();
    auto delta = current - *last;
    if (delta > period.count()) {
      *last = current;
      return true;
    }
    return false;
  };
}

/// Does nothing. Use this for cases where you do not want to do anything
/// when records expire. This is the default behavior for data (which never
/// expires via this mechanism).
ExpirationHandler::Handler LifetimeHelpers::doNothing()
{
  return [](ServiceRegistryRef, PartRef&, data_matcher::VariableContext&) -> void { return; };
}

// We simply put everything
size_t readToBuffer(void* p, size_t size, size_t nmemb, void* userdata)
{
  if (nmemb == 0) {
    return 0;
  }
  if (size == 0) {
    return 0;
  }
  auto* buffer = (std::vector<char>*)userdata;
  size_t oldSize = buffer->size();
  buffer->resize(oldSize + nmemb * size);
  memcpy(buffer->data() + oldSize, p, nmemb * size);
  return size * nmemb;
}

// We simply put everything in a stringstream and read it afterwards.
size_t readToMessage(void* p, size_t size, size_t nmemb, void* userdata)
{
  if (nmemb == 0) {
    return 0;
  }
  if (size == 0) {
    return 0;
  }
  auto* buffer = (o2::pmr::vector<char>*)userdata;
  size_t oldSize = buffer->size();
  buffer->resize(oldSize + nmemb * size);
  memcpy(buffer->data() + oldSize, p, nmemb * size);
  return size * nmemb;
}

ExpirationHandler::Handler
  LifetimeHelpers::fetchFromFairMQ(InputSpec const& spec,
                                   std::string const& channelName)
{
  return [spec, channelName](ServiceRegistryRef services, PartRef& ref, data_matcher::VariableContext&) -> void {
    auto& rawDeviceService = services.get<RawDeviceService>();
    auto device = rawDeviceService.device();

    // Receive parts and put them in the PartRef
    // we know this is not blocking because we were polled
    // on the channel.
    fair::mq::Parts parts;
    device->Receive(parts, channelName, 0);
    ref.header = std::move(parts.At(0));
    ref.payload = std::move(parts.At(1));
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromQARegistry()
{
  return [](ServiceRegistryRef, PartRef&, data_matcher::VariableContext&) -> void {
    throw runtime_error("fetchFromQARegistry: Not yet implemented");
    return;
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromObjectRegistry()
{
  return [](ServiceRegistryRef, PartRef&, data_matcher::VariableContext&) -> void {
    throw runtime_error("fetchFromObjectRegistry: Not yet implemented");
    return;
  };
}

/// Enumerate entries on every invokation.
ExpirationHandler::Handler LifetimeHelpers::enumerate(ConcreteDataMatcher const& matcher, std::string const& sourceChannel,
                                                      int64_t orbitOffset, int64_t orbitMultiplier)
{
  using counter_t = int64_t;
  auto counter = std::make_shared<counter_t>(0);
  return [matcher, counter, sourceChannel, orbitOffset, orbitMultiplier](ServiceRegistryRef services, PartRef& ref, data_matcher::VariableContext& variables) -> void {
    // Get the ChannelIndex associated to a given channel name
    auto& deviceProxy = services.get<FairMQDeviceProxy>();
    auto channelIndex = deviceProxy.getInputChannelIndexByName(sourceChannel);
    // We should invoke the handler only once.
    assert(!ref.header);
    assert(!ref.payload);

    auto timestamp = VariableContextHelpers::getTimeslice(variables).value;
    LOGP(debug, "Enumerating record");
    DataHeader dh;
    dh.dataOrigin = matcher.origin;
    dh.dataDescription = matcher.description;
    dh.subSpecification = matcher.subSpec;
    dh.payloadSize = sizeof(counter_t);
    dh.payloadSerializationMethod = gSerializationMethodNone;
    dh.tfCounter = timestamp;
    dh.firstTForbit = timestamp * orbitMultiplier + orbitOffset;
    DataProcessingHeader dph{timestamp, 1};
    services.get<CallbackService>()(CallbackService::Id::NewTimeslice, dh, dph);

    variables.put({data_matcher::FIRSTTFORBIT_POS, dh.firstTForbit});
    variables.put({data_matcher::TFCOUNTER_POS, dh.tfCounter});
    variables.put({data_matcher::RUNNUMBER_POS, dh.runNumber});
    variables.put({data_matcher::STARTTIME_POS, dph.startTime});
    variables.put({data_matcher::CREATIONTIME_POS, dph.creation});

    auto&& transport = deviceProxy.getInputChannel(channelIndex)->Transport();
    auto channelAlloc = o2::pmr::getTransportAllocator(transport);
    auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
    ref.header = std::move(header);

    auto payload = transport->CreateMessage(sizeof(counter_t));
    *(counter_t*)payload->GetData() = *counter;
    ref.payload = std::move(payload);
    (*counter)++;

  };
}

/// Create a dummy message with the provided ConcreteDataMatcher
ExpirationHandler::Handler LifetimeHelpers::dummy(ConcreteDataMatcher const& matcher, std::string const& sourceChannel)
{
  using counter_t = int64_t;
  auto counter = std::make_shared<counter_t>(0);
  auto f = [matcher, counter, sourceChannel](ServiceRegistryRef services, PartRef& ref, data_matcher::VariableContext& variables) -> void {
    // We should invoke the handler only once.
    assert(!ref.header);
    assert(!ref.payload);
    // Get the ChannelIndex associated to a given channel name
    auto& deviceProxy = services.get<FairMQDeviceProxy>();
    auto channelIndex = deviceProxy.getInputChannelIndexByName(sourceChannel);

    auto timestamp = VariableContextHelpers::getTimeslice(variables).value;
    DataHeader dh;
    dh.dataOrigin = matcher.origin;
    dh.dataDescription = matcher.description;
    dh.subSpecification = matcher.subSpec;
    dh.payloadSize = 0;
    dh.payloadSerializationMethod = gSerializationMethodNone;

    {
      auto pval = std::get_if<uint32_t>(&variables.get(data_matcher::FIRSTTFORBIT_POS));
      if (pval == nullptr) {
        dh.firstTForbit = -1;
      } else {
        dh.firstTForbit = *pval;
      }
    }
    {
      auto pval = std::get_if<uint32_t>(&variables.get(data_matcher::TFCOUNTER_POS));
      if (pval == nullptr) {
        dh.tfCounter = timestamp;
      } else {
        dh.tfCounter = *pval;
      }
    }

    DataProcessingHeader dph{timestamp, 1};

    auto&& transport = deviceProxy.getInputChannel(channelIndex)->Transport();
    auto channelAlloc = o2::pmr::getTransportAllocator(transport);
    auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
    ref.header = std::move(header);
    auto payload = transport->CreateMessage(0);
    ref.payload = std::move(payload);
  };
  return f;
}

// Life is too short. LISP rules.
#define STREAM_ENUM(x) \
  case Lifetime::x:    \
    oss << #x;         \
    break;
std::ostream& operator<<(std::ostream& oss, Lifetime const& val)
{
  switch (val) {
    STREAM_ENUM(Timeframe)
    STREAM_ENUM(Condition)
    STREAM_ENUM(QA)
    STREAM_ENUM(Transient)
    STREAM_ENUM(Timer)
    STREAM_ENUM(Enumeration)
    STREAM_ENUM(Signal)
    STREAM_ENUM(Optional)
    STREAM_ENUM(OutOfBand)
  };
  return oss;
}

} // namespace o2::framework
