// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRelayer.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/MetricsService.h"
#include "fairmq/FairMQLogger.h"

using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

// FIXME: do we really need to pass the forwards?
DataRelayer::DataRelayer(const InputsMap &inputs,
                         const ForwardsMap &forwards,
                         MetricsService &metrics)
: mInputs{inputs},
  mForwards{forwards},
  mMetrics{metrics}
{
}

size_t
assignInputSpecId(void *data, DataRelayer::InputsMap &specs) {
  size_t inputIdx = 0;
  for (auto &input : specs) {
    const DataHeader *h = reinterpret_cast<const DataHeader*>(data);

    if (DataSpecUtils::match(input.second,
                             h->dataOrigin,
                             h->dataDescription,
                             h->subSpecification)) {
      return inputIdx;
    }
    inputIdx++;
  }
  return inputIdx;
}

DataRelayer::RelayChoice
DataRelayer::relay(std::unique_ptr<FairMQMessage> &&header,
                   std::unique_ptr<FairMQMessage> &&payload) {
  // Find out which input is this and assign a valid id to it.
  size_t inputIdx = assignInputSpecId(header->GetData(), mInputs);
  // If this is true, it means the message we got does
  // not match any of the expected inputs.
  if (inputIdx == mInputs.size()) {
    return WillNotRelay;
  }

  // Calculate the mask for the given index and verify
  size_t inputMask = 1 << inputIdx;

  // FIXME: for the moment we assume that we cannot have overlapping timeframes and 
  //        that same payload for the two different timeframes will be processed
  //        in order. This might actually not be the case and we would need some
  //        way of marking to which timeframe each (header,payload) belongs.
  if (mCompletion.empty()) {
    mCompletion.emplace_back(CompletionMask{TimeframeId{0},0});
  }
  assert(!mCompletion.empty());
  // The input is already there for current timeframe. We assume this
  // means we got an input for the next timeframe, therefore we add a new
  // entry.
  if (mCompletion.back().mask & inputMask) {
    LOG(DEBUG) << "Got an entry for the next timeframe";
    TimeframeId nextTimeframe = TimeframeId{mCompletion.back().timeframeId.value + 1};
    mCompletion.push_back(CompletionMask{nextTimeframe, 0});
  }

  TimeframeId inputTimeframeId = sInvalidTimeframeId;
  // We associate the payload to the first empty spot
  for (auto &completion : mCompletion) {
    if ((completion.mask & inputMask) == 0) {
      completion.mask |= inputMask;
      inputTimeframeId = completion.timeframeId;
      LOG(DEBUG) << "Completion mask for timeframe " << completion.timeframeId.value
                 << " is " << completion.mask;
      break;
    }
  }
  assert(inputTimeframeId.value != sInvalidTimeframeId.value);
  mCache.push_back(std::move(PartRef{inputTimeframeId,
                                     inputIdx,
                                     std::move(header),
                                     std::move(payload)}));

  LOG(DEBUG) << "Adding one part to the cache. Cache size is " << mCache.size();
  return WillRelay;
}

DataRelayer::DataReadyInfo
DataRelayer::getReadyToProcess() {
  size_t allInputsMask = (1 << mInputs.size()) - 1;
  std::vector<TimeframeId> readyInputs;

  decltype(mCompletion) newCompletion;
  newCompletion.reserve(mCompletion.size());

  for (auto &completion : mCompletion) {
    if (completion.mask == allInputsMask) {
      LOG(DEBUG) << "Input from timeframe " << completion.timeframeId.value
                 << " is complete.";
      readyInputs.push_back(completion.timeframeId);
    } else {
      newCompletion.push_back(completion);
    }
  }
  // We remove the completed timeframes from the list of what needs to be
  // completed.
  mCompletion.swap(newCompletion);

  // We sort so that outputs are ordered correctly
  std::sort(mCache.begin(), mCache.end());

  // We create a vector with all the parts which can be processed
  DataReadyInfo result;

  size_t rti = 0;
  size_t ci = 0;
  decltype(mCache) newCache;

  while (true) {
    // We do not have any more ready inputs to process
    if (rti >= readyInputs.size()) {
      LOG(DEBUG) << "All " << readyInputs.size() << " entries processed";
      break;
    }

    // We do not have any more cache items to process
    // (can this happen????)
    if (ci >= mCache.size()) {
      LOG(DEBUG) << "No more entries in cache to process";
      break;
    }

    assert(rti < readyInputs.size());
    assert(ci < mCache.size());
    TimeframeId &ready = readyInputs[rti];
    PartRef &part = mCache[ci];
    // There are three possibilities:
    // - We are now processing parts which are older
    //   than the current ready timeframe we are processing
    //   move to the next ready timeframe.
    // - The part in question belongs to the current timeframe
    //   id. Will append it to the list of those to be processed.
    // - If we are here it means that the current timeframe is
    //   larger than the one of the part we are looking at,
    //   we therefore need to keep it for later.
    if (ready.value < part.timeframeId.value) {
      LOG(DEBUG) << "Timeframe " << part.timeframeId.value
                 << " is not ready. Skipping";
      ++rti;
    } else if (ready.value == part.timeframeId.value) {
      LOG(DEBUG) << "Timeframe " << part.timeframeId.value
                 << " is ready. Adding it to results";
      result.readyInputs.push_back(std::move(part));
      ++ci;
    } else {
      LOG(DEBUG) << "Timeframe " << part.timeframeId.value
                 << " is not ready. Putting it back in cache";
      newCache.push_back(std::move(part));
      ++ci;
    }
  }

  // Add the remaining bits to the new cache
  while(ci < mCache.size()) {
    newCache.push_back(std::move(mCache[ci]));
    ci++;
  }

  // Cleanup the parts which have been scheduled for processing
  mCache.swap(newCache);
  assert(result.readyInputs.size() % mInputs.size() == 0);
  return std::move(result);
}

size_t
DataRelayer::getCacheSize() const {
  return mCache.size();
}

}
}
