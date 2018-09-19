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
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecord.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/PartRef.h"
#include "fairmq/FairMQLogger.h"

#include <Monitoring/Monitoring.h>

#include <gsl/span>

using DataHeader = o2::header::DataHeader;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

constexpr size_t MAX_PARALLEL_TIMESLICES = 256;


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
}

constexpr int64_t INVALID_TIMESLICE = -1;
constexpr int INVALID_INPUT = -1;
constexpr DataRelayer::TimesliceId INVALID_TIMESLICE_ID = {INVALID_TIMESLICE};

// 16 is just some reasonable numer
// The number should really be tuned at runtime for each processor.
constexpr int DEFAULT_PIPELINE_LENGTH = 16;

// FIXME: do we really need to pass the forwards?
DataRelayer::DataRelayer(const CompletionPolicy& policy,
                         const std::vector<InputRoute>& inputRoutes,
                         const std::vector<ForwardRoute>& forwardRoutes,
                         monitoring::Monitoring& metrics)
  : mInputRoutes{ inputRoutes },
    mForwardRoutes{ forwardRoutes },
    mMetrics{ metrics },
    mCompletionPolicy{ policy },
    mDistinctRoutesIndex{ createDistinctRouteIndex(inputRoutes) }
{
  setPipelineLength(DEFAULT_PIPELINE_LENGTH);
  for (size_t ci = 0; ci < mCache.size(); ci++) {
    metrics.send({ 0, std::string("data_relayer/") + std::to_string(ci) });
  }
}

void DataRelayer::processDanglingInputs(std::vector<ExpirationHandler> const& expirationHandlers,
                                        ServiceRegistry& services)
{
  for (size_t ti = 0; ti < mTimeslices.size(); ++ti) {
    // FIXME: for the moment we need to have at least one data input before we
    //        can invoke the dangling inputs helpers.
    //        This is useful for stuff like conditions or histograms, but not
    //        (yet) for timers. For those we would need to have a mechanism
    //        to create valid timeslices.
    if (mTimeslices[ti].value == INVALID_TIMESLICE) {
      continue;
    }
    assert(mDistinctRoutesIndex.empty() == false);
    for (size_t ri = 0; ri < mDistinctRoutesIndex.size(); ++ri) {
      auto& route = mInputRoutes[mDistinctRoutesIndex[ri]];
      auto& expirator = expirationHandlers[mDistinctRoutesIndex[ri]];
      auto timestamp = mTimeslices[ti].value;
      auto& part = mCache[ti * mDistinctRoutesIndex.size() + ri];
      if (part.header == nullptr && part.payload == nullptr && expirator.checker && expirator.checker(mTimeslices[ti].value)) {
        assert(ti * mDistinctRoutesIndex.size() + ri < mCache.size());
        assert(expirator.handler);
        expirator.handler(services, part, timestamp);
        mDirty[ti] = true;
        assert(part.header != nullptr);
        assert(part.payload != nullptr);
      }
    }
  }
}

/// This does the mapping between a route and a InputSpec. The
/// reason why these might diffent is that when you have timepipelining
/// you have one route per timeslice, even if the type is the same.
size_t
assignInputSpecId(void *data, std::vector<InputRoute> const &routes) {
  for (size_t ri = 0, re = routes.size(); ri < re; ++ri) {
    auto &route = routes[ri];
    const DataHeader* h = o2::header::get<DataHeader*>(data);
    if (h == nullptr) {
      return re;
    }

    if (DataSpecUtils::match(route.matcher,
                             h->dataOrigin,
                             h->dataDescription,
                             h->subSpecification)) {
      return ri;
    }
  }
  return routes.size();
}

DataRelayer::RelayChoice
DataRelayer::relay(std::unique_ptr<FairMQMessage> &&header,
                   std::unique_ptr<FairMQMessage> &&payload) {
  // STATE HOLDING VARIABLES
  // This is the class level state of the relaying. If we start supporting
  // multithreading this will have to be made thread safe before we can invoke
  // relay concurrently.
  const auto &inputRoutes = mInputRoutes;
  std::vector<TimesliceId> &timeslices = mTimeslices;
  auto &cache = mCache;
  const auto &readonlyCache = mCache;
  auto& metrics = mMetrics;
  auto& dirty = mDirty;
  auto numInputTypes = mDistinctRoutesIndex.size();

  // IMPLEMENTATION DETAILS
  // 
  // This returns the identifier for the given input. We use a separate
  // function because while it's trivial now, the actual matchmaking will
  // become more complicated when we will start supporting ranges.
  auto getInput = [&inputRoutes,&header] () -> int {
    return assignInputSpecId(header->GetData(), inputRoutes);
  };

  // This will check if the input is valid. We hide the details so that
  // in principle the outer code will work regardless of the actual
  // implementation.
  auto isValidInput = [](int inputIdx) {
    // If this is true, it means the message we got does
    // not match any of the expected inputs.
    return inputIdx != INVALID_INPUT;
  };

  // This will check if the timeslice is valid. We hide the details so that in
  // principle outer code will work regardless of the actual implementation.
  auto isValidTimeslice = [](int64_t timesliceId) -> bool {
    return timesliceId != INVALID_TIMESLICE;
  };

  // The timeslice is embedded in the DataProcessingHeader header of the O2
  // header stack. This is an extension to the DataHeader, because apparently
  // we do have data which comes without a timestamp, although I am personally
  // not sure what that would be.
  auto getTimeslice = [&header, &timeslices]() -> int64_t {
    const DataProcessingHeader* dph = o2::header::get<DataProcessingHeader*>(header->GetData());
    if (dph == nullptr) {
      return -1;
    }
    size_t timesliceId = dph->startTime;
    assert(timeslices.size());
    return timesliceId;
  };

  // Late arrival means that the incoming timeslice is actually older 
  // then the one hold in the current cache.
  auto isInputFromObsolete = [&timeslices](int64_t timeslice) {
    auto &current = timeslices[timeslice % timeslices.size()];
    return current.value > timeslice;
  };

  // A cache line is obsolete if the incoming one has a greater timestamp or 
  // if the slot contains an invalid timeslice.
  auto isCacheEntryObsoleteFor = [&timeslices](int64_t timeslice){
    auto &current = timeslices[timeslice % timeslices.size()];
    return current.value == INVALID_TIMESLICE
           || (current.value < timeslice);
  };

  // We need to prune the cache from the old stuff, if any. Otherwise we
  // simply store the payload in the cache and we mark relevant bit in the
  // hence the first if.
  auto pruneCacheSlotFor = [&cache, &numInputTypes, &timeslices, &metrics](int64_t timeslice) {
    assert(cache.empty() == false);
    assert(timeslices.size() * numInputTypes == cache.size());
    size_t slotIndex = timeslice % timeslices.size();
    // Prune old stuff from the cache, hopefully deleting it...
    // We set the current slot to the timeslice value, so that old stuff
    // will be ignored.
    assert(numInputTypes * slotIndex < cache.size());
    timeslices[slotIndex].value = timeslice;
    for (size_t ai = slotIndex*numInputTypes, ae = ai + numInputTypes; ai != ae ; ++ai) {
      cache[ai].header.reset(nullptr);
      cache[ai].payload.reset(nullptr);
      metrics.send({ 0, std::string("data_relayer/") + std::to_string(ai) });
    }
  };

  // We need to check if the slot for the current input is already taken for
  // the current timeslice.
  // This should never happen, however given this is dependent on the input
  // we want to protect again malicious / bad upstream source.
  auto hasCacheInputAlreadyFor = [&cache, &timeslices, &numInputTypes](int64_t timeslice, int input) {
    size_t slotIndex = timeslice % timeslices.size();
    PartRef &currentPart = cache[numInputTypes*slotIndex + input];
    return (currentPart.payload != nullptr) || (currentPart.header != nullptr);
  };

  // Actually save the header / payload in the slot
  auto saveInSlot = [&header, &payload, &cache, &dirty, &timeslices, &numInputTypes, &metrics](int64_t timeslice, int input) {
    size_t slotIndex = timeslice % timeslices.size();
    dirty[slotIndex] = true;
    auto cacheIdx = numInputTypes * slotIndex + input;
    PartRef& currentPart = cache[cacheIdx];
    metrics.send({ 1, std::string("data_relayer/") + std::to_string(cacheIdx) });
    PartRef ref{std::move(header), std::move(payload)};
    currentPart = std::move(ref);
    timeslices[slotIndex] = {timeslice};
    assert(header.get() == nullptr && payload.get() == nullptr);
  };

  // OUTER LOOP
  // 
  // This is the actual outer loop processing input as part of a given
  // timeslice. All the other implementation details are hidden by the lambdas
  auto input = getInput();

  if (isValidInput(input) == false) {
    LOG(ERROR) << "A malformed message just arrived";
    return WillNotRelay;
  }

  auto timeslice = getTimeslice();
  LOG(DEBUG) << "Received timeslice" << timeslice;
  if (isValidTimeslice(timeslice) == false) {
    LOG(ERROR) << "Could not determine the timeslice for input";
    return WillNotRelay;
  }

  if (isInputFromObsolete(timeslice)) {
    LOG(ERROR) << "An entry for timeslice " << timeslice << " just arrived but too late to be processed";
    return WillNotRelay;
  }

  if (isCacheEntryObsoleteFor(timeslice)) {
    pruneCacheSlotFor(timeslice);
  }
  if (hasCacheInputAlreadyFor(timeslice, input)) {
    LOG(ERROR) << "Got a message with the same header and timeslice twice!!";
    return WillNotRelay;
  }
  saveInSlot(timeslice, input);
  return WillRelay;
}


std::vector<DataRelayer::RecordAction>
DataRelayer::getReadyToProcess() {
  // THE STATE
  std::vector<RecordAction> completed;
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
  auto updateCompletionResults = [&completed](size_t li, CompletionPolicy::CompletionOp op) {
    completed.push_back({li, op});
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
  assert(numInputTypes != 0);
  size_t cacheLines = cache.size() / numInputTypes;
  assert(cacheLines * numInputTypes == cache.size());

  for (size_t li = 0; li < cacheLines; ++li) {
    // We only check the cachelines which have been updated by an incoming 
    // message.
    if (mDirty[li] == false) {
      continue;
    }
    auto partial = getPartialRecord(li);
    auto action = mCompletionPolicy.callback(partial);
    switch (action) {
      case CompletionPolicy::CompletionOp::Consume:
      case CompletionPolicy::CompletionOp::Process:
      case CompletionPolicy::CompletionOp::Discard:
        updateCompletionResults(li, action);
        break;
      case CompletionPolicy::CompletionOp::Wait:
        break;
    }
    // Given we have created an action for this cacheline, we need to wait for
    // a new message before we look again into the given cacheline.
    mDirty[li] = false;
  }
  return completionResults();
}

std::vector<std::unique_ptr<FairMQMessage>>
DataRelayer::getInputsForTimeslice(size_t timeslice) {
  const auto numInputTypes = mDistinctRoutesIndex.size();
  // State of the computation
  std::vector<std::unique_ptr<FairMQMessage>> messages;
  messages.reserve(numInputTypes*2);
  auto &cache = mCache;
  auto &timeslices = mTimeslices;
  auto& metrics = mMetrics;

  // Nothing to see here, this is just to make the outer loop more understandable.
  auto jumpToCacheEntryAssociatedWith = [](size_t) {
    return;
  };

  // We move ownership so that the cache can be reused once the computation is
  // finished. We bump by one the timeslice for the given cache entry, so that
  // in case we get (for whatever reason) an old input, it will be
  // automatically discarded by the relay method.
  auto moveHeaderPayloadToOutput = [&messages, &cache, &timeslices, &numInputTypes, &metrics](size_t ti, size_t arg) {
    auto cacheId = ti * numInputTypes + arg;
    metrics.send({ 2, "data_relayer/" + std::to_string(cacheId) });
    messages.emplace_back(std::move(cache[cacheId].header));
    messages.emplace_back(std::move(cache[cacheId].payload));
    timeslices[ti % timeslices.size()].value += 1;
  };

  // An invalid set of arguments is a set of arguments associated to an invalid
  // timeslice, so I can simply do that. I keep the assertion there because in principle
  // we should have dispatched the timeslice already!
  // FIXME: what happens when we have enough timeslices to hit the invalid one?
  auto invalidateCacheFor = [&numInputTypes, &timeslices, &cache](size_t ti) {
    for (size_t ai = ti*numInputTypes, ae = ai + numInputTypes; ai != ae; ++ai) {
       assert(cache[ai].header.get() == nullptr);
       assert(cache[ai].payload.get() == nullptr);
    }
    timeslices[ti % timeslices.size()] = INVALID_TIMESLICE_ID;
  };

  // Outer loop here.
  jumpToCacheEntryAssociatedWith(timeslice);
  for (size_t ai = 0, ae = numInputTypes; ai != ae;  ++ai) {
    moveHeaderPayloadToOutput(timeslice, ai);
  }
  invalidateCacheFor(timeslice);

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
  mTimeslices.resize(s, INVALID_TIMESLICE_ID);
  mDirty.resize(mTimeslices.size(), false);
  auto numInputTypes = mDistinctRoutesIndex.size();
  assert(numInputTypes);
  mCache.resize(numInputTypes * mTimeslices.size());
  mMetrics.send({ (int)numInputTypes, "data_relayer/h" });
  mMetrics.send({ (int)mTimeslices.size(), "data_relayer/w" });
}

}
}
