// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/RootSerializationSupport.h"
#include "Framework/DataRelayer.h"

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
#include "DataProcessingStatus.h"
#include "DataRelayerHelpers.h"

#include "Headers/DataHeaderHelpers.h"

#include <Monitoring/Monitoring.h>

#include <fmt/format.h>
#include <gsl/span>
#include <numeric>
#include <string>

using namespace o2::framework::data_matcher;
using DataHeader = o2::header::DataHeader;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

namespace o2::framework
{

constexpr int INVALID_INPUT = -1;

// 16 is just some reasonable numer
// The number should really be tuned at runtime for each processor.
constexpr int DEFAULT_PIPELINE_LENGTH = 16;

DataRelayer::DataRelayer(const CompletionPolicy& policy,
                         std::vector<InputRoute> const& routes,
                         monitoring::Monitoring& metrics,
                         TimesliceIndex& index)
  : mTimesliceIndex{index},
    mMetrics{metrics},
    mCompletionPolicy{policy},
    mDistinctRoutesIndex{DataRelayerHelpers::createDistinctRouteIndex(routes)},
    mInputMatchers{DataRelayerHelpers::createInputMatchers(routes)}
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  setPipelineLength(DEFAULT_PIPELINE_LENGTH);

  // The queries are all the same, so we only have width 1
  auto numInputTypes = mDistinctRoutesIndex.size();
  sQueriesMetricsNames.resize(numInputTypes * 1);
  mMetrics.send({(int)numInputTypes, "data_queries/h"});
  mMetrics.send({(int)1, "data_queries/w"});
  for (size_t i = 0; i < numInputTypes; ++i) {
    sQueriesMetricsNames[i] = std::string("data_queries/") + std::to_string(i);
    char buffer[128];
    assert(mDistinctRoutesIndex[i] < routes.size());
    auto& matcher = routes[mDistinctRoutesIndex[i]].matcher;
    DataSpecUtils::describe(buffer, 127, matcher);
    mMetrics.send({std::string{buffer}, sQueriesMetricsNames[i]});
  }
}

TimesliceId DataRelayer::getTimesliceForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return mTimesliceIndex.getTimesliceForSlot(slot);
}

DataRelayer::ActivityStats DataRelayer::processDanglingInputs(std::vector<ExpirationHandler> const& expirationHandlers,
                                                              ServiceRegistry& services, bool createNew)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  ActivityStats activity;
  /// Nothing to do if nothing can expire.
  if (expirationHandlers.empty()) {
    return activity;
  }
  // Create any slot for the time based fields
  std::vector<TimesliceSlot> slotsCreatedByHandlers;
  if (createNew) {
    for (auto& handler : expirationHandlers) {
      slotsCreatedByHandlers.push_back(handler.creator(mTimesliceIndex));
    }
  }
  if (slotsCreatedByHandlers.empty() == false) {
    activity.newSlots++;
  }
  // Outer loop, we process all the records because the fact that the record
  // expires is independent from having received data for it.
  for (size_t ti = 0; ti < mTimesliceIndex.size(); ++ti) {
    TimesliceSlot slot{ti};
    if (mTimesliceIndex.isValid(slot) == false) {
      continue;
    }
    assert(mDistinctRoutesIndex.empty() == false);
    auto timestamp = mTimesliceIndex.getTimesliceForSlot(slot);
    auto& variables = mTimesliceIndex.getVariablesForSlot(slot);
    // We iterate on all the hanlders checking if they need to be expired.
    for (size_t ei = 0; ei < expirationHandlers.size(); ++ei) {
      auto& expirator = expirationHandlers[ei];
      // We check that no data is already there for the given cell
      // it is enough to check the first element
      auto& part = mCache[ti * mDistinctRoutesIndex.size() + expirator.routeIndex.value];
      if (part.size() > 0 && part[0].header != nullptr) {
        continue;
      }
      if (part.size() > 0 && part[0].payload != nullptr) {
        continue;
      }
      // We check that the cell can actually be expired.
      if (!expirator.checker) {
        continue;
      }
      if (slotsCreatedByHandlers[ei] != slot) {
        continue;
      }
      if (expirator.checker(timestamp.value) == false) {
        continue;
      }

      assert(ti * mDistinctRoutesIndex.size() + expirator.routeIndex.value < mCache.size());
      assert(expirator.handler);
      // expired, so we create one entry
      if (part.size() == 0) {
        part.parts.resize(1);
      }
      expirator.handler(services, part[0], timestamp.value, variables);
      activity.expiredSlots++;

      mTimesliceIndex.markAsDirty(slot, true);
      assert(part[0].header != nullptr);
      assert(part[0].payload != nullptr);
    }
  }
  return activity;
}

/// This does the mapping between a route and a InputSpec. The
/// reason why these might diffent is that when you have timepipelining
/// you have one route per timeslice, even if the type is the same.
size_t matchToContext(void* data,
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
  const std::string nullstring{"null"};

  for (size_t i = 0; i < MAX_MATCHING_VARIABLE; i++) {
    auto& var = context.get(i);
    if (auto pval = std::get_if<uint64_t>(&var)) {
      metrics.send(monitoring::Metric{std::to_string(*pval), names[16 * slot.index + i]});
    } else if (auto pval = std::get_if<uint32_t>(&var)) {
      metrics.send(monitoring::Metric{std::to_string(*pval), names[16 * slot.index + i]});
    } else if (auto pval2 = std::get_if<std::string>(&var)) {
      metrics.send(monitoring::Metric{*pval2, names[16 * slot.index + i]});
    } else {
      metrics.send(monitoring::Metric{nullstring, names[16 * slot.index + i]});
    }
  }
}

DataRelayer::RelayChoice
  DataRelayer::relay(std::unique_ptr<FairMQMessage>&& header,
                     std::unique_ptr<FairMQMessage>&& payload)
{
  return relay(std::move(header), &payload, 1);
}

DataRelayer::RelayChoice
  DataRelayer::relay(std::unique_ptr<FairMQMessage>&& firstPart,
                     std::unique_ptr<FairMQMessage>* restOfParts,
                     size_t restOfPartsSize)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  // STATE HOLDING VARIABLES
  // This is the class level state of the relaying. If we start supporting
  // multithreading this will have to be made thread safe before we can invoke
  // relay concurrently.
  auto& index = mTimesliceIndex;

  auto& cache = mCache;
  auto const& readonlyCache = mCache;
  auto& metrics = mMetrics;
  auto numInputTypes = mDistinctRoutesIndex.size();

  // IMPLEMENTATION DETAILS
  //
  // This returns the identifier for the given input. We use a separate
  // function because while it's trivial now, the actual matchmaking will
  // become more complicated when we will start supporting ranges.
  auto getInputTimeslice = [&matchers = mInputMatchers,
                            &distinctRoutes = mDistinctRoutesIndex,
                            &firstPart,
                            &index](VariableContext& context)
    -> std::tuple<int, TimesliceId> {
    /// FIXME: for the moment we only use the first context and reset
    /// between one invokation and the other.
    auto input = matchToContext(firstPart->GetData(), matchers, distinctRoutes, context);

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

  // We need to prune the cache from the old stuff, if any. Otherwise we
  // simply store the payload in the cache and we mark relevant bit in the
  // hence the first if.
  auto pruneCache = [&cache,
                     &cachedStateMetrics = mCachedStateMetrics,
                     &numInputTypes,
                     &index,
                     &metrics](TimesliceSlot slot) {
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

  // Actually save the header / payload in the slot
  auto saveInSlot = [&firstPart,
                     &cachedStateMetrics = mCachedStateMetrics,
                     &restOfParts,
                     &restOfPartsSize,
                     &cache,
                     &numInputTypes,
                     &metrics](TimesliceId timeslice, int input, TimesliceSlot slot) {
    auto cacheIdx = numInputTypes * slot.index + input;
    std::vector<PartRef>& parts = cache[cacheIdx].parts;
    cachedStateMetrics[cacheIdx] = CacheEntryStatus::PENDING;
    // TODO: make sure that multiple parts can only be added within the same call of
    // DataRelayer::relay
    PartRef entry{std::move(firstPart), std::move(restOfParts[0])};
    parts.emplace_back(std::move(entry));
    auto rest = restOfParts + 1;
    for (size_t pi = 0; pi < (restOfPartsSize - 1) / 2; ++pi) {
      PartRef entry{std::move(rest[pi * 2]), std::move(rest[pi * 2 + 1])};
      parts.emplace_back(std::move(entry));
    }
  };

  auto updateStatistics = [& stats = mStats](TimesliceIndex::ActionTaken action) {
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
    }
  };

  // OUTER LOOP
  //
  // This is the actual outer loop processing input as part of a given
  // timeslice. All the other implementation details are hidden by the lambdas
  auto input = INVALID_INPUT;
  auto timeslice = TimesliceId{TimesliceId::INVALID};
  auto slot = TimesliceSlot{TimesliceSlot::INVALID};

  bool needsCleaning = false;
  // First look for matching slots which already have some
  // partial match.
  for (size_t ci = 0; ci < index.size(); ++ci) {
    slot = TimesliceSlot{ci};
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
      pruneCache(slot);
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

  auto DataHeaderInfo = [&firstPart]() {
    std::string error;
    const auto* dh = o2::header::get<o2::header::DataHeader*>(firstPart->GetData());
    if (dh) {
      error += fmt::format("{}/{}/{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification);
    } else {
      error += "invalid header";
    }
    return error;
  };

  if (input == INVALID_INPUT) {
    LOG(ERROR) << "Could not match incoming data to any input route: " << DataHeaderInfo();
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    return Invalid;
  }

  if (TimesliceId::isValid(timeslice) == false) {
    LOG(ERROR) << "Could not determine the timeslice for input: " << DataHeaderInfo();
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    return Invalid;
  }

  TimesliceIndex::ActionTaken action;
  std::tie(action, slot) = index.replaceLRUWith(pristineContext);

  updateStatistics(action);

  if (action == TimesliceIndex::ActionTaken::DropObsolete) {
    static std::atomic<size_t> obsoleteCount = 0;
    static std::atomic<size_t> mult = 1;
    if ((obsoleteCount++ % (1 * mult)) == 0) {
      LOGP(WARNING, "Over {} incoming messages are already obsolete, not relaying.", obsoleteCount);
      if (obsoleteCount > mult * 10) {
        mult = mult * 10;
      }
    }
    return Dropped;
  }

  if (action == TimesliceIndex::ActionTaken::DropInvalid) {
    LOG(WARNING) << "Incoming data is invalid, not relaying.";
    return Invalid;
  }

  // At this point the variables match the new input but the
  // cache still holds the old data, so we prune it.
  pruneCache(slot);
  saveInSlot(timeslice, input, slot);
  index.publishSlot(slot);
  index.markAsDirty(slot, true);

  return WillRelay;
}

void DataRelayer::getReadyToProcess(std::vector<DataRelayer::RecordAction>& completed)
{
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
    return gsl::span<MessageSet const>(start, end);
  };

  // These two are trivial, but in principle the whole loop could be parallelised
  // or vectorised so "completed" could be a thread local variable which needs
  // merging at the end.
  auto updateCompletionResults = [&completed](TimesliceSlot li, CompletionPolicy::CompletionOp op) {
    completed.emplace_back(RecordAction{li, op});
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
    return;
  }
  size_t cacheLines = cache.size() / numInputTypes;
  assert(cacheLines * numInputTypes == cache.size());

  for (size_t li = 0; li < cacheLines; ++li) {
    TimesliceSlot slot{li};
    // We only check the cachelines which have been updated by an incoming
    // message.
    if (mTimesliceIndex.isDirty(slot) == false) {
      continue;
    }
    auto partial = getPartialRecord(li);
    auto getter = [&partial](size_t idx, size_t part) {
      if (partial[idx].size() > 0 && partial[idx].at(part).header && partial[idx].at(part).payload) {
        return DataRef{nullptr,
                       reinterpret_cast<const char*>(partial[idx].at(part).header->GetData()),
                       reinterpret_cast<const char*>(partial[idx].at(part).payload->GetData())};
      }
      return DataRef{};
    };
    auto nPartsGetter = [&partial](size_t idx) {
      return partial[idx].size();
    };
    InputSpan span{getter, nPartsGetter, static_cast<size_t>(partial.size())};
    auto action = mCompletionPolicy.callback(span);
    switch (action) {
      case CompletionPolicy::CompletionOp::Consume:
      case CompletionPolicy::CompletionOp::Process:
      case CompletionPolicy::CompletionOp::Discard:
        updateCompletionResults(slot, action);
        break;
      case CompletionPolicy::CompletionOp::Wait:
        break;
    }
    // Given we have created an action for this cacheline, we need to wait for
    // a new message before we look again into the given cacheline.
    mTimesliceIndex.markAsDirty(slot, false);
  }
}

void DataRelayer::updateCacheStatus(TimesliceSlot slot, CacheEntryStatus oldStatus, CacheEntryStatus newStatus)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  const auto numInputTypes = mDistinctRoutesIndex.size();
  auto& index = mTimesliceIndex;

  auto markInputDone = [&cachedStateMetrics = mCachedStateMetrics,
                        &index, &numInputTypes](TimesliceSlot s, size_t arg, CacheEntryStatus oldStatus, CacheEntryStatus newStatus) {
    auto cacheId = s.index * numInputTypes + arg;
    if (cachedStateMetrics[cacheId] == oldStatus) {
      cachedStateMetrics[cacheId] = newStatus;
    }
  };

  for (size_t ai = 0, ae = numInputTypes; ai != ae; ++ai) {
    markInputDone(slot, ai, oldStatus, newStatus);
  }
}

std::vector<o2::framework::MessageSet> DataRelayer::getInputsForTimeslice(TimesliceSlot slot)
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
  auto moveHeaderPayloadToOutput = [&messages,
                                    &cachedStateMetrics = mCachedStateMetrics,
                                    &cache, &index, &numInputTypes, &metrics](TimesliceSlot s, size_t arg) {
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
  auto invalidateCacheFor = [&numInputTypes, &cachedStateMetrics = mCachedStateMetrics, &index, &cache](TimesliceSlot s) {
    for (size_t ai = s.index * numInputTypes, ae = ai + numInputTypes; ai != ae; ++ai) {
      assert(std::accumulate(cache[ai].begin(), cache[ai].end(), true, [](bool result, auto const& element) { return result && element.header.get() == nullptr && element.payload.get() == nullptr; }));
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
  mCache.resize(numInputTypes * mTimesliceIndex.size());
  mMetrics.send({(int)numInputTypes, "data_relayer/h"});
  mMetrics.send({(int)mTimesliceIndex.size(), "data_relayer/w"});
  sMetricsNames.resize(mCache.size());
  mCachedStateMetrics.resize(mCache.size());
  for (size_t i = 0; i < sMetricsNames.size(); ++i) {
    sMetricsNames[i] = std::string("data_relayer/") + std::to_string(i);
  }
  // There is maximum 16 variables available. We keep them row-wise so that
  // that we can take mod 16 of the index to understand which variable we
  // are talking about.
  sVariablesMetricsNames.resize(mVariableContextes.size() * 16);
  mMetrics.send({(int)16, "matcher_variables/w"});
  mMetrics.send({(int)mVariableContextes.size(), "matcher_variables/h"});
  for (size_t i = 0; i < sVariablesMetricsNames.size(); ++i) {
    sVariablesMetricsNames[i] = std::string("matcher_variables/") + std::to_string(i);
    mMetrics.send({std::string("null"), sVariablesMetricsNames[i % 16]});
  }

  for (size_t ci = 0; ci < mCache.size(); ci++) {
    assert(ci < sMetricsNames.size());
    mMetrics.send({0, sMetricsNames[ci]});
  }
  for (size_t ci = 0; ci < mVariableContextes.size() * 16; ci++) {
    assert(ci < sVariablesMetricsNames.size());
    mMetrics.send({std::string("null"), sVariablesMetricsNames[ci]});
  }
}

DataRelayerStats const& DataRelayer::getStats() const
{
  return mStats;
}

uint32_t DataRelayer::getFirstTFOrbitForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return mTimesliceIndex.getFirstTFOrbitForSlot(slot);
}

uint32_t DataRelayer::getFirstTFCounterForSlot(TimesliceSlot slot)
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);
  return mTimesliceIndex.getFirstTFCounterForSlot(slot);
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
    mMetrics.send({static_cast<int>(mCachedStateMetrics[si]), sMetricsNames[si]});
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
