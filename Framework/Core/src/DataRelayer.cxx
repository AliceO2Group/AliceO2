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
#include "Framework/MetricsService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecord.h"
#include "fairmq/FairMQLogger.h"

using DataHeader = o2::header::DataHeader;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

constexpr size_t MAX_PARALLEL_TIMESLICES = 256;

namespace o2 {
namespace framework {

constexpr int64_t INVALID_TIMESLICE = -1;
constexpr int INVALID_INPUT = -1;
constexpr DataRelayer::TimesliceId INVALID_TIMESLICE_ID = {INVALID_TIMESLICE};

// 4 is just a magic number, assuming that each timeslice is a timeframe.
// The number should really be tuned at runtime for each processor.
constexpr int DEFAULT_PIPELINE_LENGTH = 4;

// FIXME: do we really need to pass the forwards?
DataRelayer::DataRelayer(const std::vector<InputRoute> &inputs,
                         const std::vector<ForwardRoute> &forwards,
                         MetricsService &metrics)
: mInputs{inputs},
  mForwards{forwards},
  mMetrics{metrics}
{
  setPipelineLength(DEFAULT_PIPELINE_LENGTH);
}

size_t
assignInputSpecId(void *data, std::vector<InputRoute> const &routes) {
  for (size_t ri = 0, re = routes.size(); ri < re; ++ri) {
    auto &route = routes[ri];
    const DataHeader *h = o2::header::get<DataHeader>(data);
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
  const auto &inputs = mInputs;
  std::vector<TimesliceId> &timeslices = mTimeslices;
  auto &cache = mCache;
  const auto &readonlyCache = mCache;

  // IMPLEMENTATION DETAILS
  // 
  // This returns the identifier for the given input. We use a separate
  // function because while it's trivial now, the actual matchmaking will
  // become more complicated when we will start supporting ranges.
  auto getInput = [&inputs,&header] () -> int {
    return assignInputSpecId(header->GetData(), inputs);
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
  auto getTimeslice = [&header,&timeslices]() -> int64_t {
    const DataProcessingHeader *dph = o2::header::get<DataProcessingHeader>(header->GetData());
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
  // completion mask. Notice that late arrivals should simply be ignored,
  // hence the first if.
  auto pruneCacheSlotFor = [&cache,&inputs,&timeslices](int64_t timeslice) {
    size_t slotIndex = timeslice % timeslices.size();
    // Prune old stuff from the cache, hopefully deleting it...
    // We set the current slot to the timeslice value, so that old stuff
    // will be ignored.
    assert(inputs.size() * slotIndex < cache.size());
    timeslices[slotIndex].value = timeslice;
    for (size_t ai = slotIndex*inputs.size(), ae = ai + inputs.size(); ai != ae ; ++ai) {
      cache[ai].header.reset(nullptr);
      cache[ai].payload.reset(nullptr);
    }
  };

  // We need to check if the slot for the current input is already taken for
  // the current timeslice.
  // This should never happen, however given this is dependent on the input
  // we want to protect again malicious / bad upstream source.
  auto hasCacheInputAlreadyFor = [&cache, &timeslices, &inputs](int64_t timeslice, int input) {
    size_t slotIndex = timeslice % timeslices.size();
    PartRef &currentPart = cache[inputs.size()*slotIndex + input];
    return (currentPart.payload != nullptr) || (currentPart.header != nullptr);
  };

  // Actually save the header / payload in the slot
  auto saveInSlot = [&header, &payload, &cache, &timeslices, &inputs](int64_t timeslice, int input) {
    size_t slotIndex = timeslice % timeslices.size();
    PartRef &currentPart = cache[inputs.size()*slotIndex + input];
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

std::vector<int>
DataRelayer::getReadyToProcess() {
  // THE STATE
  std::vector<int> completed;
  const auto &cache = mCache;
  const auto &inputs = mInputs;
  //
  // THE IMPLEMENTATION DETAILS
  //
  // We use this to bail out early from the check as soon as we find something
  // which we know is not complete.
  auto theLineWillBeIncomplete = [&cache, &inputs](int li, int ai) -> bool {
    auto &input = cache[li*inputs.size() + ai];
    if (input.header == nullptr || input.payload == nullptr) {
      return true;
    }
    return false;
  };

  // These two are trivial, but in principle the whole loop could be parallelised
  // or vectorised so "completed" could be a thread local variable which needs
  // merging at the end.
  auto updateCompletionResults = [&completed](size_t li) {
    completed.push_back(li);
  };

  auto completionResults = [&completed]() -> std::vector<int> {
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
  assert(!inputs.empty());
  size_t cacheLines = cache.size() / inputs.size();
  assert(cacheLines * inputs.size() == cache.size());

  for (size_t li = 0; li < cacheLines; ++li) {
    size_t ai;
    for (ai = 0; ai < inputs.size(); ++ai) {
      if (theLineWillBeIncomplete(li, ai)) {
        break;
      }
    }
    if (ai == inputs.size()) {
      updateCompletionResults(li);
    }
  }
  return completionResults();
}

std::vector<std::unique_ptr<FairMQMessage>>
DataRelayer::getInputsForTimeslice(size_t timeslice) {
  // State of the computation
  std::vector<std::unique_ptr<FairMQMessage>> messages;
  messages.reserve(mInputs.size()*2);
  auto &cache = mCache;
  auto &timeslices = mTimeslices;
  const auto &inputs = mInputs;

  // Nothing to see here, this is just to make the outer loop more understandable.
  auto jumpToCacheEntryAssociatedWith = [](size_t) {
    return;
  };

  // We move ownership so that the cache can be reused once the computation is
  // finished. We bump by one the timeslice for the given cache entry, so that
  // in case we get (for whatever reason) an old input, it will be
  // automatically discarded by the relay method.
  auto moveHeaderPayloadToOutput = [&messages, &cache, &timeslices, &inputs](size_t ti, size_t arg) {
    messages.emplace_back(std::move(cache[ti*inputs.size() + arg].header));
    messages.emplace_back(std::move(cache[ti*inputs.size() + arg].payload));
    timeslices[ti % timeslices.size()].value += 1;
  };

  // An invalid set of arguments is a set of arguments associated to an invalid
  // timeslice, so I can simply do that. I keep the assertion there because in principle
  // we should have dispatched the timeslice already!
  // FIXME: what happens when we have enough timeslices to hit the invalid one?
  auto invalidateCacheFor = [&inputs, &timeslices, &cache](size_t ti) {
    for (size_t ai = ti*inputs.size(), ae = ai + inputs.size(); ai != ae; ++ai) {
       assert(cache[ai].header.get() == nullptr);
       assert(cache[ai].payload.get() == nullptr);
    }
    timeslices[ti % timeslices.size()] = INVALID_TIMESLICE_ID;
  };

  // Outer loop here.
  jumpToCacheEntryAssociatedWith(timeslice);
  for (size_t ai = 0, ae = inputs.size(); ai != ae;  ++ai) {
    moveHeaderPayloadToOutput(timeslice, ai);
  }
  invalidateCacheFor(timeslice);

  return std::move(messages);
}

size_t
DataRelayer::getParallelTimeslices() const {
  return mCache.size() / mInputs.size();
}

/// Tune the maximum number of in flight timeslices this can handle.
void
DataRelayer::setPipelineLength(size_t s) {
  mTimeslices.resize(s, INVALID_TIMESLICE_ID);
  mCache.resize(mInputs.size() * mTimeslices.size());
}


}
}
