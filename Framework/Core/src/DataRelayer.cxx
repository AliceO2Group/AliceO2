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

using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

DataRelayer::DataRelayer(const InputsMap &inputs, const ForwardsMap &forwards)
: mInputs{inputs},
  mForwards{forwards}
{
}

DataRelayer::RelayChoice
DataRelayer::relay(std::unique_ptr<FairMQMessage> &&header, std::unique_ptr<FairMQMessage> &&payload) {
  size_t inputIdx = 0;
  // Find out which input is this and assign a valid id to it.
  for (auto &input : mInputs) {
    const DataHeader *h = reinterpret_cast<const DataHeader*>(header->GetData());

    if (DataSpecUtils::match(input.second, h->dataOrigin, h->dataDescription, h->subSpecification)) {
      break;
    }
    inputIdx++;
  }

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
  if (mCompletion.back().mask & inputMask) {
    TimeframeId nextTimeframe = TimeframeId{mCompletion.back().timeframeId.value + 1};
    mCompletion.push_back(CompletionMask{nextTimeframe, 0});
  }

  TimeframeId inputTimeframeId = sInvalidTimeframeId;
  // We associate the payload to the first empty spot
  for (auto &completion : mCompletion) {
    if ((completion.mask & inputMask) == 0) {
      completion.mask |= inputMask;
      inputTimeframeId = completion.timeframeId;
      break;
    }
  }
  assert(inputTimeframeId.value != sInvalidTimeframeId.value);
  std::cout << "Adding one part to the cache";
  mCache.push_back(std::move(PartRef{inputTimeframeId,
                                     inputIdx,
                                     std::move(header),
                                     std::move(payload)}));

  return WillRelay;
}

DataRelayer::DataReadyInfo
DataRelayer::getReadyToProcess() {
  size_t allInputsMask = (1 << mInputs.size()) - 1;
  std::vector<TimeframeId> readyInputs;

  for (auto &completion : mCompletion) {
    if (completion.mask == allInputsMask) {
      readyInputs.push_back(completion.timeframeId);
    }
  }

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
      break;
    }
    // We do not have any more cache items to process
    // (can this happen????)
    if (ci >= mCache.size()) {
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
      ++rti;
    }
    else if (ready.value == part.timeframeId.value) {
      result.readyInputs.push_back(std::move(part));
      ++ci;
    }
    else {
      newCache.push_back(std::move(part));
      ++ci;
    }
  }

  // Cleanup the parts which have been scheduled for move
  mCache.swap(newCache);
  return std::move(result);
}

}
}
