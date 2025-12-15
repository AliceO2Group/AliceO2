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

#include "Framework/ComputingQuotaEvaluator.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceState.h"
#include "Framework/Signpost.h"
#include <Monitoring/Monitoring.h>

#include <vector>
#include <uv.h>
#include <cassert>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

O2_DECLARE_DYNAMIC_LOG(quota);

namespace o2::framework
{

ComputingQuotaEvaluator::ComputingQuotaEvaluator(ServiceRegistryRef ref)
  : mRef(ref)
{
  auto& state = mRef.get<DeviceState>();
  // The first offer is valid, but does not contain any resource
  // so this will only work with some device which does not require
  // any CPU. Notice this will have troubles if a given DPL process
  // runs for more than a year.
  mOffers[0] = ComputingQuotaOffer{
    .cpu = 0,
    .memory = 0,
    .sharedMemory = 0,
    .timeslices = 0,
    .runtime = -1,
    .score = OfferScore::Unneeded,
    .valid = true};
  mInfos[0] = {
    uv_now(state.loop),
    0,
    0};

  // Creating a timer to check for expired offers
  mTimer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
  uv_timer_init(state.loop, mTimer);
}

struct QuotaEvaluatorStats {
  std::vector<int> invalidOffers;
  std::vector<int> otherUser;
  std::vector<int> unexpiring;
  std::vector<int> selectedOffers;
  std::vector<int> expired;
};

bool ComputingQuotaEvaluator::selectOffer(int task, ComputingQuotaRequest const& selector, uint64_t now)
{
  O2_SIGNPOST_ID_GENERATE(qid, quota);

  auto selectOffer = [&offers = this->mOffers, &infos = this->mInfos, task](int ref, uint64_t now) {
    auto& selected = offers[ref];
    auto& info = infos[ref];
    selected.user = task;
    if (info.firstUsed == 0) {
      info.firstUsed = now;
    }
    info.lastUsed = now;
  };

  ComputingQuotaOffer accumulated;
  static QuotaEvaluatorStats stats;

  stats.invalidOffers.clear();
  stats.otherUser.clear();
  stats.unexpiring.clear();
  stats.selectedOffers.clear();
  stats.expired.clear();

  auto summarizeWhatHappended = [ref = mRef](bool enough, std::vector<int> const& result, ComputingQuotaOffer const& totalOffer, QuotaEvaluatorStats& stats) -> bool {
    auto& dpStats = ref.get<DataProcessingStats>();
    if (result.size() == 1 && result[0] == 0) {
      //      LOG(LOGLEVEL) << "No particular resource was requested, so we schedule task anyways";
      return enough;
    }
    O2_SIGNPOST_ID_GENERATE(sid, quota);
    if (enough) {
      O2_SIGNPOST_START(quota, sid, "summary", "%zu offers were selected for a total of: cpu %d, memory %lli, shared memory %lli",
                        result.size(), totalOffer.cpu, totalOffer.memory, totalOffer.sharedMemory);
      for (auto& offer : result) {
        // We pretend each offer id is a pointer, to have a unique id.
        O2_SIGNPOST_ID_FROM_POINTER(oid, quota, (void*)(int64_t)(offer * 8));
        O2_SIGNPOST_START(quota, oid, "offers", "Offer %d has been selected.", offer);
      }
      dpStats.updateStats({static_cast<short>(ProcessingStatsId::RESOURCES_SATISFACTORY), DataProcessingStats::Op::Add, 1});
    } else {
      O2_SIGNPOST_START(quota, sid, "summary", "Not enough resources to select offers.");
      dpStats.updateStats({static_cast<short>(ProcessingStatsId::RESOURCES_MISSING), DataProcessingStats::Op::Add, 1});
      if (result.size()) {
        dpStats.updateStats({static_cast<short>(ProcessingStatsId::RESOURCES_INSUFFICIENT), DataProcessingStats::Op::Add, 1});
      }
    }
    if (stats.invalidOffers.size()) {
      O2_SIGNPOST_EVENT_EMIT(quota, sid, "summary", "The following offers were invalid: %s", fmt::format("{}", fmt::join(stats.invalidOffers, ", ")).c_str());
    }
    if (stats.otherUser.size()) {
      O2_SIGNPOST_EVENT_EMIT(quota, sid, "summary", "The following offers were owned by other users: %s", fmt::format("{}", fmt::join(stats.otherUser, ", ")).c_str());
    }
    if (stats.expired.size()) {
      O2_SIGNPOST_EVENT_EMIT(quota, sid, "summary", "The following offers are expired: %s", fmt::format("{}", fmt::join(stats.expired, ", ")).c_str());
    }
    if (stats.unexpiring.size() > 1) {
      O2_SIGNPOST_EVENT_EMIT(quota, sid, "summary", "The following offers will never expire: %s", fmt::format("{}", fmt::join(stats.unexpiring, ", ")).c_str());
    }
    O2_SIGNPOST_END(quota, sid, "summary", "Done selecting offers.");

    return enough;
  };

  bool enough = false;
  int64_t minValidity = 0;

  for (int i = 0; i != mOffers.size(); ++i) {
    auto& offer = mOffers[i];
    auto& info = mInfos[i];
    if (enough) {
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "We have enough offers. We can continue for computation.");
      break;
    }
    // Ignore:
    // - Invalid offers
    // - Offers which belong to another task
    // - Expired offers
    if (offer.valid == false) {
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d is not valid. Skipping", i);
      stats.invalidOffers.push_back(i);
      continue;
    }
    if (offer.user != -1 && offer.user != task) {
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d already offered to some other user", i);
      stats.otherUser.push_back(i);
      continue;
    }
    if (offer.runtime < 0) {
      stats.unexpiring.push_back(i);
    } else if (offer.runtime + info.received < now) {
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d expired since %llu milliseconds and holds %llu MB and %llu timeslices",
                             i, now - offer.runtime - info.received, offer.sharedMemory / 1000000, offer.timeslices);
      mExpiredOffers.push_back(ComputingQuotaOfferRef{i});
      stats.expired.push_back(i);
      continue;
    } else {
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d still valid for %llu milliseconds, providing %llu MB and %llu timeslices",
                             i, offer.runtime + info.received - now, offer.sharedMemory / 1000000, offer.timeslices);
      if (minValidity == 0) {
        minValidity = offer.runtime + info.received - now;
      }
      minValidity = std::min(minValidity, (int64_t)(offer.runtime + info.received - now));
    }
    /// We then check if the offer is suitable
    assert(offer.sharedMemory >= 0);
    auto tmp = accumulated;
    tmp.cpu += offer.cpu;
    tmp.memory += offer.memory;
    tmp.sharedMemory += offer.sharedMemory;
    tmp.timeslices += offer.timeslices;
    offer.score = selector(offer, accumulated);
    switch (offer.score) {
      case OfferScore::Unneeded:
        O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d considered not needed. Skipping", i);
        continue;
      case OfferScore::Unsuitable:
        O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d considered Unsuitable. Skipping", i);
        continue;
      case OfferScore::More:
        selectOffer(i, now);
        accumulated = tmp;
        stats.selectedOffers.push_back(i);
        O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Offer %d selected but not enough. %llu MB, %d cores and %llu timeslices are not enough.",
                               i, tmp.sharedMemory / 1000000, tmp.cpu, tmp.timeslices);
        continue;
      case OfferScore::Enough:
        selectOffer(i, now);
        accumulated = tmp;
        stats.selectedOffers.push_back(i);
        enough = true;
        O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Selected %zu offers providing %llu MB, %d cores and %llu timeslices are deemed enough.",
                               stats.selectedOffers.size(), tmp.sharedMemory / 1000000, tmp.cpu, tmp.timeslices);
        break;
    };
  }

  if (minValidity != 0) {
    O2_SIGNPOST_EVENT_EMIT(quota, qid, "select", "Next offer to expire in %llu milliseconds", minValidity);
    uv_timer_start(mTimer, [](uv_timer_t* handle) {
      O2_SIGNPOST_ID_GENERATE(tid, quota);
      O2_SIGNPOST_EVENT_EMIT(quota, tid, "select", "Offer should be expired by now, checking again."); }, minValidity + 100, 0);
  }
  // If we get here it means we never got enough offers, so we return false.
  return summarizeWhatHappended(enough, stats.selectedOffers, accumulated, stats);
}

void ComputingQuotaEvaluator::consume(int id, ComputingQuotaConsumer& consumer, std::function<void(ComputingQuotaOffer const& accumulatedConsumed, ComputingQuotaStats& reportConsumedOffer)>& reportConsumedOffer)
{
  // This will report how much of the offers has to be considered consumed.
  // Notice that actual memory usage might be larger, because we can over
  // allocate.
  consumer(id, mOffers, mStats, reportConsumedOffer);
}

void ComputingQuotaEvaluator::dispose(int taskId)
{
  for (int oi = 0; oi < mOffers.size(); ++oi) {
    auto& offer = mOffers[oi];
    if (offer.user != taskId) {
      continue;
    }
    offer.user = -1;
    // Disposing the offer so that the resource can be recyled.
    /// An offer with index 0 is always there.
    /// All the others are reset.
    if (oi == 0) {
      return;
    }
    if (offer.valid == false) {
      continue;
    }
    if (offer.sharedMemory <= 0) {
      O2_SIGNPOST_ID_FROM_POINTER(oid, quota, (void*)(int64_t)(oi * 8));
      O2_SIGNPOST_END(quota, oid, "offers", "Offer %d back to not needed.", oi);
      offer.valid = false;
      offer.score = OfferScore::Unneeded;
    }
  }
}

/// Move offers from the pending list to the actual available offers
void ComputingQuotaEvaluator::updateOffers(std::vector<ComputingQuotaOffer>& pending, uint64_t now)
{
  O2_SIGNPOST_ID_GENERATE(oid, quota);
  O2_SIGNPOST_START(quota, oid, "updateOffers", "Starting to process %zu received offers", pending.size());
  int lastValid = -1;
  for (size_t oi = 0; oi < mOffers.size(); oi++) {
    auto& storeOffer = mOffers[oi];
    auto& info = mInfos[oi];
    if (pending.empty()) {
      O2_SIGNPOST_END(quota, oid, "updateOffers", "No more pending offers to process");
      return;
    }
    if (storeOffer.valid == true) {
      O2_SIGNPOST_EVENT_EMIT(quota, oid, "updateOffers", "Skipping update of offer %zu because it's still valid", oi);
      // In general we want to fill an invalid offer. If we do not find any
      // we add to the last valid offer we found.
      lastValid = oi;
      continue;
    }
    info.received = now;
    auto& offer = pending.back();
    O2_SIGNPOST_EVENT_EMIT(quota, oid, "updateOffers", "Updating of offer %zu at %llu. Cpu: %d, Shared Memory %lli, Timeslices: %lli",
                           oi, now, offer.cpu, offer.sharedMemory, offer.timeslices);
    storeOffer = offer;
    storeOffer.valid = true;
    pending.pop_back();
  }
  if (lastValid == -1) {
    O2_SIGNPOST_END_WITH_ERROR(quota, oid, "updateOffers", "ComputingQuotaOffer losts. This should never happen.");
    return;
  }
  auto& lastValidOffer = mOffers[lastValid];
  for (auto& stillPending : pending) {
    lastValidOffer.cpu += stillPending.cpu;
    lastValidOffer.memory += stillPending.memory;
    lastValidOffer.sharedMemory += stillPending.sharedMemory;
    lastValidOffer.timeslices += stillPending.timeslices;
    lastValidOffer.runtime = std::max(lastValidOffer.runtime, stillPending.runtime);
  }
  pending.clear();
  auto& updatedOffer = mOffers[lastValid];
  O2_SIGNPOST_END(quota, oid, "updateOffers", "Remaining offers cohalesced to %d. New values: Cpu%d, Shared Memory %lli, Timeslices %lli",
                  lastValid, updatedOffer.cpu, updatedOffer.sharedMemory, updatedOffer.timeslices);
}

void ComputingQuotaEvaluator::handleExpired(std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const& stats)> expirator)
{
  static int nothingToDoCount = mExpiredOffers.size();
  O2_SIGNPOST_ID_GENERATE(qid, quota);
  if (mExpiredOffers.size()) {
    O2_SIGNPOST_EVENT_EMIT(quota, qid, "handleExpired", "Handling %zu expired offers", mExpiredOffers.size());
    nothingToDoCount = 0;
  } else {
    if (nothingToDoCount == 0) {
      nothingToDoCount++;
      O2_SIGNPOST_EVENT_EMIT(quota, qid, "handleExpired", "No expired offers");
    }
  }
  /// Whenever an offer is expired, we give back the resources
  /// to the driver.
  for (auto& ref : mExpiredOffers) {
    auto& offer = mOffers[ref.index];
    O2_SIGNPOST_ID_FROM_POINTER(oid, quota, (void*)(int64_t)(ref.index * 8));
    if (offer.sharedMemory < 0 && offer.timeslices < 0) {
      O2_SIGNPOST_END(quota, oid, "handleExpired", "Offer %d does not have any more resources. Marking it as invalid.", ref.index);
      offer.valid = false;
      offer.score = OfferScore::Unneeded;
      continue;
    }
    // FIXME: offers should go through the driver client, not the monitoring
    // api.
    O2_SIGNPOST_END(quota, oid, "handleExpired", "Offer %d expired. Giving back %llu MB, %d cores and %llu timeslices",
                    ref.index, offer.sharedMemory / 1000000, offer.cpu, offer.timeslices);
    mStats.totalExpiredBytes += std::max<int64_t>(offer.sharedMemory, 0);
    mStats.totalExpiredTimeslices += std::max<int64_t>(offer.timeslices, 0);
    mStats.totalExpiredOffers++;
    expirator(offer, mStats);
    // driverClient.tell("expired shmem {}", offer.sharedMemory);
    // driverClient.tell("expired cpu {}", offer.cpu);
    offer.sharedMemory = -1;
    offer.timeslices = -1;
    offer.valid = false;
    offer.score = OfferScore::Unneeded;
  }
  mExpiredOffers.clear();
}

} // namespace o2::framework
