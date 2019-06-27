// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRelayer.h"

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecord.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/PartRef.h"
#include "Framework/TimesliceIndex.h"
#include "DataProcessingStatus.h"
#include "Framework/Signpost.h"

#include <Monitoring/Monitoring.h>

#include <fairmq/FairMQLogger.h>

#include <gsl/span>

using namespace o2::framework::data_matcher;
using DataHeader = o2::header::DataHeader;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

namespace o2
{
namespace framework
{

namespace
{
std::vector<size_t> createDistinctRouteIndex(std::vector<InputRoute> const& routes)
{
  std::vector<size_t> result;
  for (size_t ri = 0; ri < routes.size(); ++ri) {
    auto& route = routes[ri];
    if (route.timeslice == 0) {
      result.push_back(ri);
    }
  }
  return result;
}

DataDescriptorMatcher fromConcreteMatcher(ConcreteDataMatcher const& matcher)
{
  return DataDescriptorMatcher{
    DataDescriptorMatcher::Op::And,
    StartTimeValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{ matcher.origin.str },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{ matcher.description.str },
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          SubSpecificationTypeValueMatcher{ matcher.subSpec })))
  };
}

/// This converts from InputRoute to the associated DataDescriptorMatcher.
std::vector<DataDescriptorMatcher> createInputMatchers(std::vector<InputRoute> const& routes)
{
  std::vector<DataDescriptorMatcher> result;

  for (auto& route : routes) {
    if (auto pval = std::get_if<ConcreteDataMatcher>(&route.matcher.matcher)) {
      result.emplace_back(fromConcreteMatcher(*pval));
    } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&route.matcher.matcher)) {
      result.push_back(*matcher);
    } else {
      throw std::runtime_error("Unsupported InputSpec type");
    }
  }

  return result;
}
}

constexpr int INVALID_INPUT = -1;

// 16 is just some reasonable numer
// The number should really be tuned at runtime for each processor.
constexpr int DEFAULT_PIPELINE_LENGTH = 16;

// FIXME: do we really need to pass the forwards?
DataRelayer::DataRelayer(const CompletionPolicy& policy,
                         const std::vector<InputRoute>& inputRoutes,
                         const std::vector<ForwardRoute>& forwardRoutes,
                         monitoring::Monitoring& metrics,
                         TimesliceIndex& index)
  : mInputRoutes{ inputRoutes },
    mForwardRoutes{ forwardRoutes },
    mTimesliceIndex{ index },
    mMetrics{ metrics },
    mCompletionPolicy{ policy },
    mDistinctRoutesIndex{ createDistinctRouteIndex(inputRoutes) },
    mInputMatchers{ createInputMatchers(inputRoutes) }
{
  setPipelineLength(DEFAULT_PIPELINE_LENGTH);
  for (size_t ci = 0; ci < mCache.size(); ci++) {
    metrics.send({ 0, sMetricsNames[ci] });
  }
  for (size_t ci = 0; ci < mVariableContextes.size() * 16; ci++) {
    metrics.send({ std::string("null"), sVariablesMetricsNames[ci] });
  }
}

void DataRelayer::processDanglingInputs(std::vector<ExpirationHandler> const& expirationHandlers,
                                        ServiceRegistry& services)
{
  // Create any slot for the time based fields
  std::vector<TimesliceSlot> slotsCreatedByHandlers(expirationHandlers.size());
  for (size_t hi = 0; hi < expirationHandlers.size(); ++hi) {
    slotsCreatedByHandlers[hi] = expirationHandlers[hi].creator(mTimesliceIndex);
  }
  // Expire the records as needed.
  for (size_t ti = 0; ti < mTimesliceIndex.size(); ++ti) {
    TimesliceSlot slot{ ti };
    if (mTimesliceIndex.isValid(slot) == false) {
      continue;
    }
    assert(mDistinctRoutesIndex.empty() == false);
    for (size_t ri = 0; ri < mDistinctRoutesIndex.size(); ++ri) {
      auto& route = mInputRoutes[mDistinctRoutesIndex[ri]];
      auto& expirator = expirationHandlers[mDistinctRoutesIndex[ri]];
      auto timestamp = mTimesliceIndex.getTimesliceForSlot(slot);
      auto& part = mCache[ti * mDistinctRoutesIndex.size() + ri];
      if (part.header != nullptr) {
        continue;
      }
      if (part.payload != nullptr) {
        continue;
      }
      if (!expirator.checker) {
        continue;
      }
      if (slotsCreatedByHandlers[mDistinctRoutesIndex[ri]].index != slot.index) {
        continue;
      }
      if (expirator.checker(timestamp.value) == false) {
        continue;
      }

      assert(ti * mDistinctRoutesIndex.size() + ri < mCache.size());
      assert(expirator.handler);
      expirator.handler(services, part, timestamp.value);
      mTimesliceIndex.markAsDirty(slot, true);
      assert(part.header != nullptr);
      assert(part.payload != nullptr);
    }
  }
}

/// This does the mapping between a route and a InputSpec. The
/// reason why these might diffent is that when you have timepipelining
/// you have one route per timeslice, even if the type is the same.
size_t matchToContext(void* data,
                      std::vector<DataDescriptorMatcher> const& matchers,
                      VariableContext& context)
{
  for (size_t ri = 0, re = matchers.size(); ri < re; ++ri) {
    auto& matcher = matchers[ri];

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
      metrics.send(monitoring::Metric{ std::to_string(*pval), names[16 * slot.index + i] });
    } else if (auto pval2 = std::get_if<std::string>(&var)) {
      metrics.send(monitoring::Metric{ *pval2, names[16 * slot.index + i] });
    } else {
      metrics.send(monitoring::Metric{ nullstring, names[16 * slot.index + i] });
    }
  }
}

DataRelayer::RelayChoice
DataRelayer::relay(std::unique_ptr<FairMQMessage> &&header,
                   std::unique_ptr<FairMQMessage> &&payload) {
  // STATE HOLDING VARIABLES
  // This is the class level state of the relaying. If we start supporting
  // multithreading this will have to be made thread safe before we can invoke
  // relay concurrently.
  auto const& inputRoutes = mInputRoutes;
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
  auto getInputTimeslice = [& matchers = mInputMatchers,
                            &header,
                            &index ](VariableContext & context)
                             ->std::tuple<int, TimesliceId>
  {
    /// FIXME: for the moment we only use the first context and reset
    /// between one invokation and the other.
    auto input = matchToContext(header->GetData(), matchers, context);

    if (input == INVALID_INPUT) {
      return {
        INVALID_INPUT,
        TimesliceId{ TimesliceId::INVALID },
      };
    }
    /// The first argument is always matched against the data start time, so
    /// we can assert it's the same as the dph->startTime
    if (auto pval = std::get_if<uint64_t>(&context.get(0))) {
      TimesliceId timeslice{ *pval };
      return { input, timeslice };
    }
    // If we get here it means we need to push something out of the cache.
    return {
      INVALID_INPUT,
      TimesliceId{ TimesliceId::INVALID },
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
      cache[ai].header.reset(nullptr);
      cache[ai].payload.reset(nullptr);
      cachedStateMetrics[ai] = 0;
    }
  };

  // We need to check if the slot for the current input is already taken for
  // the current timeslice.
  // This should never happen, however given this is dependent on the input
  // we want to protect again malicious / bad upstream source.
  auto hasCacheInputAlreadyFor = [&cache, &index, &numInputTypes](TimesliceSlot slot, int input) {
    PartRef& currentPart = cache[numInputTypes * slot.index + input];
    return (currentPart.payload != nullptr) || (currentPart.header != nullptr);
  };

  // Actually save the header / payload in the slot
  auto saveInSlot = [&header,
                     &cachedStateMetrics = mCachedStateMetrics,
                     &payload,
                     &cache,
                     &numInputTypes,
                     &metrics](TimesliceId timeslice, int input, TimesliceSlot slot) {
    auto cacheIdx = numInputTypes * slot.index + input;
    PartRef& currentPart = cache[cacheIdx];
    cachedStateMetrics[cacheIdx] = 1;
    PartRef ref{std::move(header), std::move(payload)};
    currentPart = std::move(ref);
    assert(header.get() == nullptr && payload.get() == nullptr);
  };

  auto updateStatistics = [& stats = mStats](TimesliceIndex::ActionTaken action)
  {
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
  auto timeslice = TimesliceId{ TimesliceId::INVALID };
  auto slot = TimesliceSlot{ TimesliceSlot::INVALID };

  // First look for matching slots which already have some 
  // partial match.
  for (size_t ci = 0; ci < index.size(); ++ci) {
    slot = TimesliceSlot{ ci };
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
      slot = TimesliceSlot{ ci };
      if (index.isValid(slot) == true) {
        continue;
      }
      std::tie(input, timeslice) = getInputTimeslice(index.getVariablesForSlot(slot));
      if (input != INVALID_INPUT) {
        break;
      }
    }
  }

  /// If we get a valid result, we can store the message in cache.
  if (input != INVALID_INPUT && TimesliceId::isValid(timeslice) && TimesliceSlot::isValid(slot)) {
    O2_SIGNPOST(O2_PROBE_DATARELAYER, timeslice.value, 0, 0, 0);
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

  if (input == INVALID_INPUT) {
    LOG(ERROR) << "Could not match incoming data to any input";
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    return WillNotRelay;
  }

  if (TimesliceId::isValid(timeslice) == false) {
    LOG(ERROR) << "Could not determine the timeslice for input";
    mStats.malformedInputs++;
    mStats.droppedIncomingMessages++;
    return WillNotRelay;
  }

  TimesliceIndex::ActionTaken action;
  std::tie(action, slot) = index.replaceLRUWith(pristineContext);

  updateStatistics(action);

  if (action == TimesliceIndex::ActionTaken::DropObsolete) {
    LOG(WARNING) << "Incoming data is already obsolete, not relaying.";
    return WillNotRelay;
  }

  if (action == TimesliceIndex::ActionTaken::DropInvalid) {
    LOG(WARNING) << "Incoming data is invalid, not relaying.";
    return WillNotRelay;
  }

  // At this point the variables match the new input but the
  // cache still holds the old data, so we prune it.
  pruneCache(slot);
  saveInSlot(timeslice, input, slot);
  index.publishSlot(slot);
  index.markAsDirty(slot, true);

  return WillRelay;
}

std::vector<DataRelayer::RecordAction>
DataRelayer::getReadyToProcess() {
  // THE STATE
  std::vector<RecordAction> completed;
  completed.reserve(16);
  const auto &cache = mCache;
  const auto numInputTypes = mDistinctRoutesIndex.size();
  //
  // THE IMPLEMENTATION DETAILS
  //
  // We use this to bail out early from the check as soon as we find something
  // which we know is not complete.
  auto getPartialRecord = [&cache, &numInputTypes](int li) -> gsl::span<const PartRef> {
    auto offset = li * numInputTypes;
    assert(cache.size() >= offset + numInputTypes);
    auto const start = cache.data() + offset;
    auto const end = cache.data() + offset + numInputTypes;
    return gsl::span<const PartRef>(start, end);
  };

  // These two are trivial, but in principle the whole loop could be parallelised
  // or vectorised so "completed" could be a thread local variable which needs
  // merging at the end.
  auto updateCompletionResults = [&completed](TimesliceSlot li, CompletionPolicy::CompletionOp op) {
    completed.emplace_back(RecordAction{ li, op });
  };

  auto completionResults = [&completed]() -> std::vector<RecordAction> {
    return completed;
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
    return {};
  }
  size_t cacheLines = cache.size() / numInputTypes;
  assert(cacheLines * numInputTypes == cache.size());

  for (size_t li = 0; li < cacheLines; ++li) {
    TimesliceSlot slot{ li };
    // We only check the cachelines which have been updated by an incoming
    // message.
    if (mTimesliceIndex.isDirty(slot) == false) {
      continue;
    }
    auto partial = getPartialRecord(li);
    auto action = mCompletionPolicy.callback(partial);
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
  return completionResults();
}

std::vector<std::unique_ptr<FairMQMessage>>
  DataRelayer::getInputsForTimeslice(TimesliceSlot slot)
{
  const auto numInputTypes = mDistinctRoutesIndex.size();
  // State of the computation
  std::vector<std::unique_ptr<FairMQMessage>> messages;
  messages.reserve(numInputTypes*2);
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
    cachedStateMetrics[cacheId] = 2;
    messages.emplace_back(std::move(cache[cacheId].header));
    messages.emplace_back(std::move(cache[cacheId].payload));
    index.markAsInvalid(s);
  };

  // An invalid set of arguments is a set of arguments associated to an invalid
  // timeslice, so I can simply do that. I keep the assertion there because in principle
  // we should have dispatched the timeslice already!
  // FIXME: what happens when we have enough timeslices to hit the invalid one?
  auto invalidateCacheFor = [&numInputTypes, &index, &cache](TimesliceSlot s) {
    for (size_t ai = s.index * numInputTypes, ae = ai + numInputTypes; ai != ae; ++ai) {
      assert(cache[ai].header.get() == nullptr);
      assert(cache[ai].payload.get() == nullptr);
    }
    index.markAsInvalid(s);
  };

  // Outer loop here.
  jumpToCacheEntryAssociatedWith(slot);
  for (size_t ai = 0, ae = numInputTypes; ai != ae;  ++ai) {
    moveHeaderPayloadToOutput(slot, ai);
  }
  invalidateCacheFor(slot);

  return std::move(messages);
}

size_t
DataRelayer::getParallelTimeslices() const {
  return mCache.size() / mDistinctRoutesIndex.size();
}


/// Tune the maximum number of in flight timeslices this can handle.
/// Notice that in case we have time pipelining we need to count
/// the actual number of different types, without taking into account
/// the time pipelining.
void
DataRelayer::setPipelineLength(size_t s) {
  mTimesliceIndex.resize(s);
  mVariableContextes.resize(s);
  auto numInputTypes = mDistinctRoutesIndex.size();
  mCache.resize(numInputTypes * mTimesliceIndex.size());
  mMetrics.send({ (int)numInputTypes, "data_relayer/h" });
  mMetrics.send({ (int)mTimesliceIndex.size(), "data_relayer/w" });
  sMetricsNames.resize(mCache.size());
  mCachedStateMetrics.resize(mCache.size());
  for (size_t i = 0; i < sMetricsNames.size(); ++i) {
    sMetricsNames[i] = std::string("data_relayer/") + std::to_string(i);
  }
  // There is maximum 16 variables available. We keep them row-wise so that
  // that we can take mod 16 of the index to understand which variable we
  // are talking about.
  sVariablesMetricsNames.resize(mVariableContextes.size() * 16);
  mMetrics.send({ (int)16, "matcher_variables/w" });
  mMetrics.send({ (int)mVariableContextes.size(), "matcher_variables/h" });
  for (size_t i = 0; i < sVariablesMetricsNames.size(); ++i) {
    sVariablesMetricsNames[i] = std::string("matcher_variables/") + std::to_string(i);
    mMetrics.send({ std::string("null"), sVariablesMetricsNames[i % 16] });
  }
  // The queries are all the same, so we only have width 1
  sQueriesMetricsNames.resize(numInputTypes * 1);
  mMetrics.send({ (int)numInputTypes, "data_queries/h" });
  mMetrics.send({ (int)1, "data_queries/w" });
  for (size_t i = 0; i < numInputTypes; ++i) {
    sQueriesMetricsNames[i] = std::string("data_queries/") + std::to_string(i);
    char buffer[128];
    auto& matcher = mInputRoutes[mDistinctRoutesIndex[i]].matcher;
    DataSpecUtils::describe(buffer, 127, matcher);
    mMetrics.send({ std::string{ buffer }, sQueriesMetricsNames[i] });
  }
}

DataRelayerStats const& DataRelayer::getStats() const
{
  return mStats;
}

void DataRelayer::sendContextState()
{
  for (size_t ci = 0; ci < mTimesliceIndex.size(); ++ci) {
    auto slot = TimesliceSlot{ ci };
    sendVariableContextMetrics(mTimesliceIndex.getPublishedVariablesForSlot(slot), slot,
                               mMetrics, sVariablesMetricsNames);
  }
  for (size_t si = 0; si < mCachedStateMetrics.size(); ++si) {
    mMetrics.send({ mCachedStateMetrics[si], sMetricsNames[si] });
  }
}

std::vector<std::string> DataRelayer::sMetricsNames;
std::vector<std::string> DataRelayer::sVariablesMetricsNames;
std::vector<std::string> DataRelayer::sQueriesMetricsNames;
}
}
