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
#include "Framework/RootSerializationSupport.h"
#include "Framework/DataRelayer.h"

#include "Framework/CompilerBuiltins.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/Logger.h"
#include "Framework/PartRef.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/Signpost.h"
#include "Framework/RoutingIndices.h"
#include "Framework/VariableContextHelpers.h"
#include "Framework/FairMQDeviceProxy.h"
#include "DataProcessingStatus.h"
#include "DataRelayerHelpers.h"
#include "InputRouteHelpers.h"
#include "Framework/LifetimeHelpers.h"

#include "Headers/DataHeaderHelpers.h"
#include "Framework/Formatters.h"

#include <Monitoring/Metric.h>
#include <Monitoring/Monitoring.h>

#include <fairmq/Channel.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gsl/span>
#include <numeric>
#include <string>

using namespace o2::framework::data_matcher;
using DataHeader = o2::header::DataHeader;
using DataProcessingHeader = o2::framework::DataProcessingHeader;
using Verbosity = o2::monitoring::Verbosity;

namespace o2::framework
{

constexpr int INVALID_INPUT = -1;

// 128 is just some reasonable numer
// The number should really be tuned at runtime for each processor.
constexpr int DEFAULT_PIPELINE_LENGTH = 128;

DataRelayer::DataRelayer(const CompletionPolicy& policy,
                         std::vector<InputRoute> const& routes,
                         monitoring::Monitoring& metrics,
                         TimesliceIndex& index)
  : mMetrics{metrics},
    mTimesliceIndex{index},
    mCompletionPolicy{policy},
    mDistinctRoutesIndex{DataRelayerHelpers::createDistinctRouteIndex(routes)},
    mInputMatchers{DataRelayerHelpers::createInputMatchers(routes)},
    mMaxLanes{InputRouteHelpers::maxLanes(routes)}
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  if (policy.configureRelayer == nullptr) {
    char* defaultPipelineLengthTxt = getenv("DPL_DEFAULT_PIPELINE_LENGTH");
    int defaultPipelineLength = defaultPipelineLengthTxt ? std::stoi(defaultPipelineLengthTxt) : DEFAULT_PIPELINE_LENGTH;
    setPipelineLength(defaultPipelineLength);
  } else {
    policy.configureRelayer(*this);
  }

  // The queries are all the same, so we only have width 1
  auto numInputTypes = mDistinctRoutesIndex.size();
  sQueriesMetricsNames.resize(numInputTypes * 1);
  mMetrics.send({(int)numInputTypes, "data_queries/h", Verbosity::Debug});
  mMetrics.send({(int)1, "data_queries/w", Verbosity::Debug});
  for (size_t i = 0; i < numInputTypes; ++i) {
    sQueriesMetricsNames[i] = std::string("data_queries/") + std::to_string(i);
    char buffer[128];
    assert(mDistinctRoutesIndex[i] < routes.size());
    mInputs.push_back(routes[mDistinctRoutesIndex[i]].matcher);
    auto& matcher = routes[mDistinctRoutesIndex[i]].matcher;
    DataSpecUtils::describe(buffer, 127, matcher);
    mMetrics.send({fmt::format("{} ({})", buffer, mInputs.back().lifetime), sQueriesMetricsNames[i], Verbosity::Debug});
  }
}

TimesliceId DataRelayer::getTimesliceForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  auto& variables = mTimesliceIndex.getVariablesForSlot(slot);
  return VariableContextHelpers::getTimeslice(variables);
}

DataRelayer::ActivityStats DataRelayer::processDanglingInputs(std::vector<ExpirationHandler> const& expirationHandlers,
                                                              ServiceRegistryRef services, bool createNew)
{
  LOGP(debug, "DataRelayer::processDanglingInputs");
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  auto& deviceProxy = services.get<FairMQDeviceProxy>();

  ActivityStats activity;
  /// Nothing to do if nothing can expire.
  if (expirationHandlers.empty()) {
    LOGP(debug, "DataRelayer::processDanglingInputs: No expiration handlers");
    return activity;
  }
  // Create any slot for the time based fields
  std::vector<TimesliceSlot> slotsCreatedByHandlers;
  if (createNew) {
    LOGP(debug, "Creating new slot");
    for (auto& handler : expirationHandlers) {
      LOGP(debug, "handler.creator for {}", handler.name);
      auto channelIndex = deviceProxy.getInputChannelIndex(handler.routeIndex);
      slotsCreatedByHandlers.push_back(handler.creator(channelIndex, mTimesliceIndex));
    }
  }
  // Count how many slots are not invalid
  auto validSlots = 0;
  for (auto slot : slotsCreatedByHandlers) {
    if (slot.index == TimesliceSlot::INVALID) {
      continue;
    }
    validSlots++;
  }
  if (validSlots > 0) {
    activity.newSlots++;
    LOGP(debug, "DataRelayer::processDanglingInputs: {} slots created by handler", validSlots);
  } else {
    LOGP(debug, "DataRelayer::processDanglingInputs: no slots created by handler");
  }
  // Outer loop, we process all the records because the fact that the record
  // expires is independent from having received data for it.
  int headerPresent = 0;
  int payloadPresent = 0;
  int noCheckers = 0;
  int badSlot = 0;
  int checkerDenied = 0;
  for (size_t ti = 0; ti < mTimesliceIndex.size(); ++ti) {
    TimesliceSlot slot{ti};
    if (mTimesliceIndex.isValid(slot) == false) {
      continue;
    }
    assert(mDistinctRoutesIndex.empty() == false);
    auto& variables = mTimesliceIndex.getVariablesForSlot(slot);
    auto timestamp = VariableContextHelpers::getTimeslice(variables);
    // We iterate on all the hanlders checking if they need to be expired.
    for (size_t ei = 0; ei < expirationHandlers.size(); ++ei) {
      auto& expirator = expirationHandlers[ei];
      // We check that no data is already there for the given cell
      // it is enough to check the first element
      auto& part = mCache[ti * mDistinctRoutesIndex.size() + expirator.routeIndex.value];
      if (part.size() > 0 && part.header(0) != nullptr) {
        headerPresent++;
        continue;
      }
      if (part.size() > 0 && part.payload(0) != nullptr) {
        payloadPresent++;
        continue;
      }
      // We check that the cell can actually be expired.
      if (!expirator.checker) {
        noCheckers++;
        continue;
      }
      if (slotsCreatedByHandlers[ei] != slot) {
        badSlot++;
        continue;
      }

      auto getPartialRecord = [&cache = mCache, numInputTypes = mDistinctRoutesIndex.size()](int li) -> gsl::span<MessageSet const> {
        auto offset = li * numInputTypes;
        assert(cache.size() >= offset + numInputTypes);
        auto const start = cache.data() + offset;
        auto const end = cache.data() + offset + numInputTypes;
        return {start, end};
      };

      auto partial = getPartialRecord(ti);
      // TODO: get the data ref from message model
      auto getter = [&partial](size_t idx, size_t part) {
        if (partial[idx].size() > 0 && partial[idx].header(part).get()) {
          auto header = partial[idx].header(part).get();
          auto payload = partial[idx].payload(part).get();
          return DataRef{nullptr,
                         reinterpret_cast<const char*>(header->GetData()),
                         reinterpret_cast<char const*>(payload ? payload->GetData() : nullptr),
                         payload ? payload->GetSize() : 0};
        }
        return DataRef{};
      };
      auto nPartsGetter = [&partial](size_t idx) {
        return partial[idx].size();
      };
      InputSpan span{getter, nPartsGetter, static_cast<size_t>(partial.size())};
      // Setup the input span

      if (expirator.checker(services, timestamp.value, span) == false) {
        checkerDenied++;
        continue;
      }

      assert(ti * mDistinctRoutesIndex.size() + expirator.routeIndex.value < mCache.size());
      assert(expirator.handler);
      PartRef newRef;
      expirator.handler(services, newRef, variables);
      part.reset(std::move(newRef));
      activity.expiredSlots++;

      mTimesliceIndex.markAsDirty(slot, true);
      assert(part.header(0) != nullptr);
      assert(part.payload(0) != nullptr);
    }
  }
  LOGP(debug, "DataRelayer::processDanglingInputs headerPresent:{}, payloadPresent:{}, noCheckers:{}, badSlot:{}, checkerDenied:{}",
       headerPresent, payloadPresent, noCheckers, badSlot, checkerDenied);
  return activity;
}

/// This does the mapping between a route and a InputSpec. The
/// reason why these might diffent is that when you have timepipelining
/// you have one route per timeslice, even if the type is the same.
size_t matchToContext(void const* data,
                      std::vector<DataDescriptorMatcher> const& matchers,
                      std::vector<size_t> const& index,
                      VariableContext& context)
{
  for (size_t ri = 0, re = index.size(); ri < re; ++ri) {
    auto& matcher = matchers[index[ri]];

    if (matcher.match(reinterpret_cast<char const*>(data), context)) {
      context.commit();
      return ri;
    }
    context.discard();
  }
  return INVALID_INPUT;
}

/// Send the contents of a context as metrics, so that we can examine them in
/// the GUI.
void sendVariableContextMetrics(VariableContext& context, TimesliceSlot slot,
                                monitoring::Monitoring& metrics, std::vector<std::string> const& names)
{
  static const std::string nullstring{"null"};

  context.publish([](ContextElement::Value const& var, std::string const& name, void* context) {
    monitoring::Monitoring* metrics = reinterpret_cast<monitoring::Monitoring*>(context);
    if (auto pval = std::get_if<uint64_t>(&var)) {
      metrics->send(monitoring::Metric{std::to_string(*pval), name, Verbosity::Debug});
    } else if (auto pval = std::get_if<uint32_t>(&var)) {
      metrics->send(monitoring::Metric{std::to_string(*pval), name, Verbosity::Debug});
    } else if (auto pval2 = std::get_if<std::string>(&var)) {
      metrics->send(monitoring::Metric{*pval2, name, Verbosity::Debug});
    } else {
      metrics->send(monitoring::Metric{nullstring, name, Verbosity::Debug});
    }
  },
                  &metrics, slot, names);
}

void DataRelayer::setOldestPossibleInput(TimesliceId proposed, ChannelIndex channel)
{
  auto newOldest = mTimesliceIndex.setOldestPossibleInput(proposed, channel);
  LOGP(debug, "DataRelayer::setOldestPossibleInput {} from channel {}", newOldest.timeslice.value, newOldest.channel.value);
  static bool dontDrop = getenv("DPL_DONT_DROP_OLD_TIMESLICE") && atoi(getenv("DPL_DONT_DROP_OLD_TIMESLICE"));
  if (dontDrop) {
    return;
  }
  for (size_t si = 0; si < mCache.size() / mInputs.size(); ++si) {
    auto& variables = mTimesliceIndex.getVariablesForSlot({si});
    auto timestamp = VariableContextHelpers::getTimeslice(variables);
    auto valid = mTimesliceIndex.validateSlot({si}, newOldest.timeslice);
    if (valid) {
      if (mTimesliceIndex.isValid({si})) {
        LOGP(debug, "Keeping slot {} because data has timestamp {} while oldest possible timestamp is {}", si, timestamp.value, newOldest.timeslice.value);
      }
      continue;
    }
    mPruneOps.push_back(PruneOp{si});
    for (size_t mi = 0; mi < mInputs.size(); ++mi) {
      auto& input = mInputs[mi];
      auto& element = mCache[si * mInputs.size() + mi];
      if (element.size() != 0) {
        if (input.lifetime != Lifetime::Condition && mCompletionPolicy.name != "internal-dpl-injected-dummy-sink") {
          LOGP(error, "Dropping {} Lifetime::{} data in slot {} with timestamp {} < {}.", DataSpecUtils::describe(input), (int)input.lifetime, si, timestamp.value, newOldest.timeslice.value);
        } else {
          LOGP(debug,
               "Silently dropping data {} in pipeline slot {} because it has timeslice {} < {} after receiving data from channel {}."
               "Because Lifetime::Timeframe data not there and not expected (e.g. due to sampling) we drop non sampled, non timeframe data (e.g. Conditions).",
               DataSpecUtils::describe(input), si, timestamp.value, newOldest.timeslice.value,
               mTimesliceIndex.getChannelInfo(channel).channel->GetName());
        }
      }
    }
  }
}

TimesliceIndex::OldestOutputInfo DataRelayer::getOldestPossibleOutput() const
{
  return mTimesliceIndex.getOldestPossibleOutput();
}

void DataRelayer::prunePending(OnDropCallback onDrop)
{
  for (auto& op : mPruneOps) {
    this->pruneCache(op.slot, onDrop);
  }
  mPruneOps.clear();
}

void DataRelayer::pruneCache(TimesliceSlot slot, OnDropCallback onDrop)
{
  // We need to prune the cache from the old stuff, if any. Otherwise we
  // simply store the payload in the cache and we mark relevant bit in the
  // hence the first if.
  auto pruneCache = [&onDrop,
                     &cache = mCache,
                     &cachedStateMetrics = mCachedStateMetrics,
                     numInputTypes = mDistinctRoutesIndex.size(),
                     &index = mTimesliceIndex,
                     &metrics = mMetrics](TimesliceSlot slot) {
    if (onDrop) {
      auto oldestPossibleTimeslice = index.getOldestPossibleOutput();
      // State of the computation
      std::vector<MessageSet> dropped(numInputTypes);
      for (size_t ai = 0, ae = numInputTypes; ai != ae; ++ai) {
        auto cacheId = slot.index * numInputTypes + ai;
        cachedStateMetrics[cacheId] = CacheEntryStatus::RUNNING;
        // TODO: in the original implementation of the cache, there have been only two messages per entry,
        // check if the 2 above corresponds to the number of messages.
        if (cache[cacheId].size() > 0) {
          dropped[ai] = std::move(cache[cacheId]);
        }
      }
      bool anyDropped = std::any_of(dropped.begin(), dropped.end(), [](auto& m) { return m.size(); });
      if (anyDropped) {
        onDrop(slot, dropped, oldestPossibleTimeslice);
      }
    }
    assert(cache.empty() == false);
    assert(index.size() * numInputTypes == cache.size());
    // Prune old stuff from the cache, hopefully deleting it...
    // We set the current slot to the timeslice value, so that old stuff
    // will be ignored.
    assert(numInputTypes * slot.index < cache.size());
    for (size_t ai = slot.index * numInputTypes, ae = ai + numInputTypes; ai != ae; ++ai) {
      cache[ai].clear();
      cachedStateMetrics[ai] = CacheEntryStatus::EMPTY;
    }
  };

  pruneCache(slot);
}

DataRelayer::RelayChoice
  DataRelayer::relay(void const* rawHeader,
                     std::unique_ptr<fair::mq::Message>* messages,
                     size_t nMessages,
                     size_t nPayloads,
                     std::function<void(TimesliceSlot, std::vector<MessageSet>&, TimesliceIndex::OldestOutputInfo)> onDrop)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  DataProcessingHeader const* dph = o2::header::get<DataProcessingHeader*>(rawHeader);
  // STATE HOLDING VARIABLES
  // This is the class level state of the relaying. If we start supporting
  // multithreading this will have to be made thread safe before we can invoke
  // relay concurrently.
  auto const& readonlyCache = mCache;

  // IMPLEMENTATION DETAILS
  //
  // This returns true if a given slot is available for the current number of lanes
  auto isSlotInLane = [currentLane = dph->startTime, maxLanes = mMaxLanes](TimesliceSlot slot) {
    return (slot.index % maxLanes) == (currentLane % maxLanes);
  };
  // This returns the identifier for the given input. We use a separate
  // function because while it's trivial now, the actual matchmaking will
  // become more complicated when we will start supporting ranges.
  auto getInputTimeslice = [&matchers = mInputMatchers,
                            &distinctRoutes = mDistinctRoutesIndex,
                            &rawHeader,
                            &index = mTimesliceIndex](VariableContext& context)
    -> std::tuple<int, TimesliceId> {
    /// FIXME: for the moment we only use the first context and reset
    /// between one invokation and the other.
    auto input = matchToContext(rawHeader, matchers, distinctRoutes, context);

    if (input == INVALID_INPUT) {
      return {
        INVALID_INPUT,
        TimesliceId{TimesliceId::INVALID},
      };
    }
    /// The first argument is always matched against the data start time, so
    /// we can assert it's the same as the dph->startTime
    if (auto pval = std::get_if<uint64_t>(&context.get(0))) {
      TimesliceId timeslice{*pval};
      return {input, timeslice};
    }
    // If we get here it means we need to push something out of the cache.
    return {
      INVALID_INPUT,
      TimesliceId{TimesliceId::INVALID},
    };
  };

  // Actually save the header / payload in the slot
  auto saveInSlot = [&cachedStateMetrics = mCachedStateMetrics,
                     &messages,
                     &nMessages,
                     &nPayloads,
                     &cache = mCache,
                     numInputTypes = mDistinctRoutesIndex.size(),
                     &metrics = mMetrics](TimesliceId timeslice, int input, TimesliceSlot slot) {
    auto cacheIdx = numInputTypes * slot.index + input;
    MessageSet& target = cache[cacheIdx];
    cachedStateMetrics[cacheIdx] = CacheEntryStatus::PENDING;
    // TODO: make sure that multiple parts can only be added within the same call of
    // DataRelayer::relay
    assert(nPayloads > 0);
    for (size_t mi = 0; mi < nMessages; ++mi) {
      assert(mi + nPayloads < nMessages);
      target.add([&messages, &mi](size_t i) -> fair::mq::MessagePtr& { return messages[mi + i]; }, nPayloads + 1);
      mi += nPayloads;
    }
  };

  auto updateStatistics = [&stats = mStats](TimesliceIndex::ActionTaken action) {
    // Update statistics for what happened
    switch (action) {
      case TimesliceIndex::ActionTaken::DropObsolete:
        stats.droppedIncomingMessages++;
        break;
      case TimesliceIndex::ActionTaken::DropInvalid:
        stats.malformedInputs++;
        stats.droppedIncomingMessages++;
        break;
      case TimesliceIndex::ActionTaken::ReplaceUnused:
        stats.relayedMessages++;
        break;
      case TimesliceIndex::ActionTaken::ReplaceObsolete:
        stats.droppedComputations++;
        stats.relayedMessages++;
        break;
      case TimesliceIndex::ActionTaken::Wait:
        break;
    }
  };

  // OUTER LOOP
  //
  // This is the actual outer loop processing input as part of a given
  // timeslice. All the other implementation details are hidden by the lambdas
  auto input = INVALID_INPUT;
  auto timeslice = TimesliceId{TimesliceId::INVALID};
  auto slot = TimesliceSlot{TimesliceSlot::INVALID};
  auto& index = mTimesliceIndex;

  bool needsCleaning = false;
  // First look for matching slots which already have some
  // partial match.
  for (size_t ci = 0; ci < index.size(); ++ci) {
    slot = TimesliceSlot{ci};
    if (!isSlotInLane(slot)) {
      continue;
    }
    if (index.isValid(slot) == false) {
      continue;
    }
    std::tie(input, timeslice) = getInputTimeslice(index.getVariablesForSlot(slot));
    if (input != INVALID_INPUT) {
      break;
    }
  }

  // If we did not find anything, look for slots which
  // are invalid.
  if (input == INVALID_INPUT) {
    for (size_t ci = 0; ci < index.size(); ++ci) {
      slot = TimesliceSlot{ci};
      if (index.isValid(slot) == true) {
        continue;
      }
      if (!isSlotInLane(slot)) {
        continue;
      }
      std::tie(input, timeslice) = getInputTimeslice(index.getVariablesForSlot(slot));
      if (input != INVALID_INPUT) {
        needsCleaning = true;
        break;
      }
    }
  }

  /// If we get a valid result, we can store the message in cache.
  if (input != INVALID_INPUT && TimesliceId::isValid(timeslice) && TimesliceSlot::isValid(slot)) {
    O2_SIGNPOST(O2_PROBE_DATARELAYER, timeslice.value, 0, 0, 0);
    if (needsCleaning) {
      this->pruneCache(slot, onDrop);
      mPruneOps.erase(std::remove_if(mPruneOps.begin(), mPruneOps.end(), [slot](const auto& x) { return x.slot == slot; }), mPruneOps.end());
    }
    saveInSlot(timeslice, input, slot);
    index.publishSlot(slot);
    index.markAsDirty(slot, true);
    mStats.relayedMessages++;
    return WillRelay;
  }

  /// If not, we find which timeslice we really were looking at
  /// and see if we can prune something from the cache.
  VariableContext pristineContext;
  std::tie(input, timeslice) = getInputTimeslice(pristineContext);

  auto DataHeaderInfo = [&rawHeader]() {
    std::string error;
    // extract header from message model
    const auto* dh = o2::header::get<o2::header::DataHeader*>(rawHeader);
    if (dh) {
      error += fmt::format("{}/{}/{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification);
    } else {
      error += "invalid header";
    }
    return error;
  };

  if (input == INVALID_INPUT) {
    LOG(error) << "Could not match incoming data to any input route: " << DataHeaderInfo();
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    for (size_t pi = 0; pi < nMessages; ++pi) {
      messages[pi].reset(nullptr);
    }
    return Invalid;
  }

  if (TimesliceId::isValid(timeslice) == false) {
    LOG(error) << "Could not determine the timeslice for input: " << DataHeaderInfo();
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    for (size_t pi = 0; pi < nMessages; ++pi) {
      messages[pi].reset(nullptr);
    }
    return Invalid;
  }

  TimesliceIndex::ActionTaken action;
  std::tie(action, slot) = index.replaceLRUWith(pristineContext, timeslice);

  updateStatistics(action);

  switch (action) {
    case TimesliceIndex::ActionTaken::Wait:
      return Backpressured;
    case TimesliceIndex::ActionTaken::DropObsolete:
      static std::atomic<size_t> obsoleteCount = 0;
      static std::atomic<size_t> mult = 1;
      if ((obsoleteCount++ % (1 * mult)) == 0) {
        LOGP(warning, "Over {} incoming messages are already obsolete, not relaying.", obsoleteCount);
        if (obsoleteCount > mult * 10) {
          mult = mult * 10;
        }
      }
      return Dropped;
    case TimesliceIndex::ActionTaken::DropInvalid:
      LOG(warning) << "Incoming data is invalid, not relaying.";
      mStats.malformedInputs++;
      mStats.droppedIncomingMessages++;
      for (size_t pi = 0; pi < nMessages; ++pi) {
        messages[pi].reset(nullptr);
      }
      return Invalid;
    case TimesliceIndex::ActionTaken::ReplaceUnused:
    case TimesliceIndex::ActionTaken::ReplaceObsolete:
      // At this point the variables match the new input but the
      // cache still holds the old data, so we prune it.
      this->pruneCache(slot, onDrop);
      mPruneOps.erase(std::remove_if(mPruneOps.begin(), mPruneOps.end(), [slot](const auto& x) { return x.slot == slot; }), mPruneOps.end());
      saveInSlot(timeslice, input, slot);
      index.publishSlot(slot);
      index.markAsDirty(slot, true);
      return WillRelay;
  }
  O2_BUILTIN_UNREACHABLE();
}

void DataRelayer::getReadyToProcess(std::vector<DataRelayer::RecordAction>& completed)
{
  LOGP(debug, "DataRelayer::getReadyToProcess");
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  // THE STATE
  const auto& cache = mCache;
  const auto numInputTypes = mDistinctRoutesIndex.size();
  //
  // THE IMPLEMENTATION DETAILS
  //
  // We use this to bail out early from the check as soon as we find something
  // which we know is not complete.
  auto getPartialRecord = [&cache, &numInputTypes](int li) -> gsl::span<MessageSet const> {
    auto offset = li * numInputTypes;
    assert(cache.size() >= offset + numInputTypes);
    auto const start = cache.data() + offset;
    auto const end = cache.data() + offset + numInputTypes;
    return {start, end};
  };

  // These two are trivial, but in principle the whole loop could be parallelised
  // or vectorised so "completed" could be a thread local variable which needs
  // merging at the end.
  auto updateCompletionResults = [&completed](TimesliceSlot li, uint64_t const* timeslice, CompletionPolicy::CompletionOp op) {
    if (timeslice) {
      LOGP(debug, "Doing action {} for slot {} (timeslice: {})", (int)op, li.index, *timeslice);
      completed.emplace_back(RecordAction{li, {*timeslice}, op});
    } else {
      LOGP(debug, "No timeslice associated with slot ", li.index);
    }
  };

  // THE OUTER LOOP
  //
  // To determine if a line is complete, we iterate on all the arguments
  // and check if they are ready. We do it this way, because in the end
  // the number of inputs is going to be small and having a more complex
  // structure will probably result in a larger footprint in any case.
  // Also notice that ai == inputsNumber only when we reach the end of the
  // iteration, that means we have found all the required bits.
  //
  // Notice that the only time numInputTypes is 0 is when we are a dummy
  // device created as a source for timers / conditions.
  if (numInputTypes == 0) {
    LOGP(debug, "numInputTypes == 0, returning.");
    return;
  }
  size_t cacheLines = cache.size() / numInputTypes;
  assert(cacheLines * numInputTypes == cache.size());
  int countConsume = 0;
  int countConsumeExisting = 0;
  int countProcess = 0;
  int countDiscard = 0;
  int countWait = 0;
  int notDirty = 0;

  for (int li = cacheLines - 1; li >= 0; --li) {
    TimesliceSlot slot{(size_t)li};
    // We only check the cachelines which have been updated by an incoming
    // message.
    if (mTimesliceIndex.isDirty(slot) == false) {
      notDirty++;
      continue;
    }
    auto partial = getPartialRecord(li);
    // TODO: get the data ref from message model
    auto getter = [&partial](size_t idx, size_t part) {
      if (partial[idx].size() > 0 && partial[idx].header(part).get()) {
        auto header = partial[idx].header(part).get();
        auto payload = partial[idx].payload(part).get();
        return DataRef{nullptr,
                       reinterpret_cast<const char*>(header->GetData()),
                       reinterpret_cast<char const*>(payload ? payload->GetData() : nullptr),
                       payload ? payload->GetSize() : 0};
      }
      return DataRef{};
    };
    auto nPartsGetter = [&partial](size_t idx) {
      return partial[idx].size();
    };
    InputSpan span{getter, nPartsGetter, static_cast<size_t>(partial.size())};
    CompletionPolicy::CompletionOp action;
    if (mCompletionPolicy.callback) {
      action = mCompletionPolicy.callback(span);
    } else if (mCompletionPolicy.callbackFull) {
      action = mCompletionPolicy.callbackFull(span, mInputs);
    } else {
      throw std::runtime_error("No completion policy found");
    }
    auto& variables = mTimesliceIndex.getVariablesForSlot(slot);
    auto timeslice = std::get_if<uint64_t>(&variables.get(0));
    switch (action) {
      case CompletionPolicy::CompletionOp::Consume:
        countConsume++;
        updateCompletionResults(slot, timeslice, action);
        mTimesliceIndex.markAsDirty(slot, false);
        break;
      case CompletionPolicy::CompletionOp::ConsumeAndRescan:
        // This is just like Consume, but we also mark all slots as dirty
        countConsume++;
        action = CompletionPolicy::CompletionOp::Consume;
        updateCompletionResults(slot, timeslice, action);
        mTimesliceIndex.rescan();
        break;
      case CompletionPolicy::CompletionOp::ConsumeExisting:
        countConsumeExisting++;
        updateCompletionResults(slot, timeslice, action);
        mTimesliceIndex.markAsDirty(slot, false);
        break;
      case CompletionPolicy::CompletionOp::Process:
        countProcess++;
        updateCompletionResults(slot, timeslice, action);
        mTimesliceIndex.markAsDirty(slot, false);
        break;
      case CompletionPolicy::CompletionOp::Discard:
        countDiscard++;
        updateCompletionResults(slot, timeslice, action);
        mTimesliceIndex.markAsDirty(slot, false);
        break;
      case CompletionPolicy::CompletionOp::Retry:
        countWait++;
        mTimesliceIndex.markAsDirty(slot, true);
        action = CompletionPolicy::CompletionOp::Wait;
        break;
      case CompletionPolicy::CompletionOp::Wait:
        countWait++;
        mTimesliceIndex.markAsDirty(slot, false);
        break;
    }
  }
  mTimesliceIndex.updateOldestPossibleOutput();
  LOGP(debug, "DataRelayer::getReadyToProcess results notDirty:{}, consume:{}, consumeExisting:{}, process:{}, discard:{}, wait:{}",
       notDirty, countConsume, countConsumeExisting, countProcess,
       countDiscard, countWait);
}

void DataRelayer::updateCacheStatus(TimesliceSlot slot, CacheEntryStatus oldStatus, CacheEntryStatus newStatus)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  const auto numInputTypes = mDistinctRoutesIndex.size();

  auto markInputDone = [&cachedStateMetrics = mCachedStateMetrics,
                        &numInputTypes](TimesliceSlot s, size_t arg, CacheEntryStatus oldStatus, CacheEntryStatus newStatus) {
    auto cacheId = s.index * numInputTypes + arg;
    if (cachedStateMetrics[cacheId] == oldStatus) {
      cachedStateMetrics[cacheId] = newStatus;
    }
  };

  for (size_t ai = 0, ae = numInputTypes; ai != ae; ++ai) {
    markInputDone(slot, ai, oldStatus, newStatus);
  }
}

std::vector<o2::framework::MessageSet> DataRelayer::consumeAllInputsForTimeslice(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  const auto numInputTypes = mDistinctRoutesIndex.size();
  // State of the computation
  std::vector<MessageSet> messages(numInputTypes);
  auto& cache = mCache;
  auto& index = mTimesliceIndex;

  // Nothing to see here, this is just to make the outer loop more understandable.
  auto jumpToCacheEntryAssociatedWith = [](TimesliceSlot) {
    return;
  };

  // We move ownership so that the cache can be reused once the computation is
  // finished. We mark the given cache slot invalid, so that it can be reused
  // This means we can still handle old messages if there is still space in the
  // cache where to put them.
  auto moveHeaderPayloadToOutput = [&messages,
                                    &cachedStateMetrics = mCachedStateMetrics,
                                    &cache, &index, &numInputTypes](TimesliceSlot s, size_t arg) {
    auto cacheId = s.index * numInputTypes + arg;
    cachedStateMetrics[cacheId] = CacheEntryStatus::RUNNING;
    // TODO: in the original implementation of the cache, there have been only two messages per entry,
    // check if the 2 above corresponds to the number of messages.
    if (cache[cacheId].size() > 0) {
      messages[arg] = std::move(cache[cacheId]);
    }
    index.markAsInvalid(s);
  };

  // An invalid set of arguments is a set of arguments associated to an invalid
  // timeslice, so I can simply do that. I keep the assertion there because in principle
  // we should have dispatched the timeslice already!
  // FIXME: what happens when we have enough timeslices to hit the invalid one?
  auto invalidateCacheFor = [&numInputTypes, &index, &cache](TimesliceSlot s) {
    for (size_t ai = s.index * numInputTypes, ae = ai + numInputTypes; ai != ae; ++ai) {
      assert(std::accumulate(cache[ai].messages.begin(), cache[ai].messages.end(), true, [](bool result, auto const& element) { return result && element.get() == nullptr; }));
      cache[ai].clear();
    }
    index.markAsInvalid(s);
  };

  // Outer loop here.
  jumpToCacheEntryAssociatedWith(slot);
  for (size_t ai = 0, ae = numInputTypes; ai != ae; ++ai) {
    moveHeaderPayloadToOutput(slot, ai);
  }
  invalidateCacheFor(slot);

  return messages;
}

std::vector<o2::framework::MessageSet> DataRelayer::consumeExistingInputsForTimeslice(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  const auto numInputTypes = mDistinctRoutesIndex.size();
  // State of the computation
  std::vector<MessageSet> messages(numInputTypes);
  auto& cache = mCache;
  auto& index = mTimesliceIndex;
  auto& metrics = mMetrics;

  // Nothing to see here, this is just to make the outer loop more understandable.
  auto jumpToCacheEntryAssociatedWith = [](TimesliceSlot) {
    return;
  };

  // We move ownership so that the cache can be reused once the computation is
  // finished. We mark the given cache slot invalid, so that it can be reused
  // This means we can still handle old messages if there is still space in the
  // cache where to put them.
  auto copyHeaderPayloadToOutput = [&messages,
                                    &cachedStateMetrics = mCachedStateMetrics,
                                    &cache, &index, &numInputTypes, &metrics](TimesliceSlot s, size_t arg) {
    auto cacheId = s.index * numInputTypes + arg;
    cachedStateMetrics[cacheId] = CacheEntryStatus::RUNNING;
    // TODO: in the original implementation of the cache, there have been only two messages per entry,
    // check if the 2 above corresponds to the number of messages.
    for (size_t pi = 0; pi < cache[cacheId].size(); pi++) {
      auto& header = cache[cacheId].header(pi);
      auto&& newHeader = header->GetTransport()->CreateMessage();
      newHeader->Copy(*header);
      messages[arg].add(PartRef{std::move(newHeader), std::move(cache[cacheId].payload(pi))});
    }
  };

  // Outer loop here.
  jumpToCacheEntryAssociatedWith(slot);
  for (size_t ai = 0, ae = numInputTypes; ai != ae; ++ai) {
    copyHeaderPayloadToOutput(slot, ai);
  }

  return std::move(messages);
}

void DataRelayer::clear()
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  for (auto& cache : mCache) {
    cache.clear();
  }
  for (size_t s = 0; s < mTimesliceIndex.size(); ++s) {
    mTimesliceIndex.markAsInvalid(TimesliceSlot{s});
  }
}

size_t
  DataRelayer::getParallelTimeslices() const
{
  return mCache.size() / mDistinctRoutesIndex.size();
}

/// Tune the maximum number of in flight timeslices this can handle.
/// Notice that in case we have time pipelining we need to count
/// the actual number of different types, without taking into account
/// the time pipelining.
void DataRelayer::setPipelineLength(size_t s)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  mTimesliceIndex.resize(s);
  mVariableContextes.resize(s);
  publishMetrics();
}

void DataRelayer::publishMetrics()
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  auto numInputTypes = mDistinctRoutesIndex.size();
  // FIXME: many of the DataRelayer function rely on allocated cache, so its
  // maybe misleading to have the allocation in a function primarily for
  // metrics publishing, do better in setPipelineLength?
  mCache.resize(numInputTypes * mTimesliceIndex.size());
  mMetrics.send({(int)numInputTypes, "data_relayer/h", Verbosity::Debug});
  mMetrics.send({(int)mTimesliceIndex.size(), "data_relayer/w", Verbosity::Debug});
  sMetricsNames.resize(mCache.size());
  mCachedStateMetrics.resize(mCache.size());
  for (size_t i = 0; i < sMetricsNames.size(); ++i) {
    sMetricsNames[i] = std::string("data_relayer/") + std::to_string(i);
  }
  // There is maximum 16 variables available. We keep them row-wise so that
  // that we can take mod 16 of the index to understand which variable we
  // are talking about.
  sVariablesMetricsNames.resize(mVariableContextes.size() * 16);
  mMetrics.send({(int)16, "matcher_variables/w", Verbosity::Debug});
  mMetrics.send({(int)mVariableContextes.size(), "matcher_variables/h", Verbosity::Debug});
  for (size_t i = 0; i < sVariablesMetricsNames.size(); ++i) {
    sVariablesMetricsNames[i] = std::string("matcher_variables/") + std::to_string(i);
    mMetrics.send({std::string("null"), sVariablesMetricsNames[i % 16], Verbosity::Debug});
  }

  for (size_t ci = 0; ci < mCache.size(); ci++) {
    assert(ci < sMetricsNames.size());
    mMetrics.send({0, sMetricsNames[ci], Verbosity::Debug});
  }
  for (size_t ci = 0; ci < mVariableContextes.size() * 16; ci++) {
    assert(ci < sVariablesMetricsNames.size());
    mMetrics.send({std::string("null"), sVariablesMetricsNames[ci], Verbosity::Debug});
  }
}

DataRelayerStats const& DataRelayer::getStats() const
{
  return mStats;
}

uint32_t DataRelayer::getFirstTFOrbitForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return VariableContextHelpers::getFirstTFOrbit(mTimesliceIndex.getVariablesForSlot(slot));
}

uint32_t DataRelayer::getFirstTFCounterForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return VariableContextHelpers::getFirstTFCounter(mTimesliceIndex.getVariablesForSlot(slot));
}

uint32_t DataRelayer::getRunNumberForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return VariableContextHelpers::getRunNumber(mTimesliceIndex.getVariablesForSlot(slot));
}

uint64_t DataRelayer::getCreationTimeForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return VariableContextHelpers::getCreationTime(mTimesliceIndex.getVariablesForSlot(slot));
}

void DataRelayer::sendContextState()
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  for (size_t ci = 0; ci < mTimesliceIndex.size(); ++ci) {
    auto slot = TimesliceSlot{ci};
    sendVariableContextMetrics(mTimesliceIndex.getPublishedVariablesForSlot(slot), slot,
                               mMetrics, sVariablesMetricsNames);
  }
  for (size_t si = 0; si < mCachedStateMetrics.size(); ++si) {
    mMetrics.send({static_cast<int>(mCachedStateMetrics[si]), sMetricsNames[si], Verbosity::Debug});
    // Anything which is done is actually already empty,
    // so after we report it we mark it as such.
    if (mCachedStateMetrics[si] == CacheEntryStatus::DONE) {
      mCachedStateMetrics[si] = CacheEntryStatus::EMPTY;
    }
  }
}

std::vector<std::string> DataRelayer::sMetricsNames;
std::vector<std::string> DataRelayer::sVariablesMetricsNames;
std::vector<std::string> DataRelayer::sQueriesMetricsNames;
} // namespace o2::framework
