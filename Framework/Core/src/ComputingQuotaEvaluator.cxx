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
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceState.h"
#include "Framework/DriverClient.h"
#include "Framework/Monitoring.h"
#include "Framework/Logger.h"
#include <Monitoring/Monitoring.h>

#include <vector>
#include <uv.h>
#include <cassert>

namespace o2::framework
{

ComputingQuotaEvaluator::ComputingQuotaEvaluator(ServiceRegistry& registry)
  : mRegistry{registry},
    mLoop{registry.get<DeviceState>().loop}
{
  // The first offer is valid, but does not contain any resource
  // so this will only work with some device which does not require
  // any CPU. Notice this will have troubles if a given DPL process
  // runs for more than a year.
  mOffers[0] = {
    0,
    0,
    0,
    -1,
    -1,
    OfferScore::Unneeded,
    true};
  mInfos[0] = {
    uv_now(mLoop),
    0,
    0};
}

struct QuotaEvaluatorStats {
  std::vector<int> invalidOffers;
  std::vector<int> otherUser;
  std::vector<int> unexpiring;
  std::vector<int> selectedOffers;
  std::vector<int> expired;
};

bool ComputingQuotaEvaluator::selectOffer(int task, ComputingQuotaRequest const& selector)
{
  auto selectOffer = [&loop = mLoop, &offers = this->mOffers, &infos = this->mInfos, task](int ref) {
    auto& selected = offers[ref];
    auto& info = infos[ref];
    selected.user = task;
    if (info.firstUsed == 0) {
      info.firstUsed = uv_now(loop);
    }
    info.lastUsed = uv_now(loop);
  };

  ComputingQuotaOffer accumulated;
  static QuotaEvaluatorStats stats;

  stats.invalidOffers.clear();
  stats.otherUser.clear();
  stats.unexpiring.clear();
  stats.selectedOffers.clear();
  stats.expired.clear();

  auto summarizeWhatHappended = [](bool enough, std::vector<int> const& result, ComputingQuotaOffer const& totalOffer, QuotaEvaluatorStats& stats) -> bool {
    if (result.size() == 1 && result[0] == 0) {
      //      LOG(INFO) << "No particular resource was requested, so we schedule task anyways";
      return enough;
    }
    if (enough) {
      LOGP(INFO, "{} offers were selected for a total of: cpu {}, memory {}, shared memory {}", result.size(), totalOffer.cpu, totalOffer.memory, totalOffer.sharedMemory);
      LOGP(INFO, "  The following offers were selected for computation: {} ", fmt::join(result, ","));
    } else {
      LOG(INFO) << "No offer was selected";
      if (result.size()) {
        LOGP(INFO, "  The following offers were selected for computation but not enough: {} ", fmt::join(result, ","));
      }
    }
    if (stats.invalidOffers.size()) {
      LOGP(INFO, "  The following offers were invalid: {}", fmt::join(stats.invalidOffers, ", "));
    }
    if (stats.otherUser.size()) {
      LOGP(INFO, "  The following offers were owned by other users: {}", fmt::join(stats.otherUser, ", "));
    }
    if (stats.expired.size()) {
      LOGP(INFO, "  The following offers are expired: {}", fmt::join(stats.expired, ", "));
    }
    if (stats.unexpiring.size() > 1) {
      LOGP(INFO, "  The following offers will never expire: {}", fmt::join(stats.unexpiring, ", "));
    }

    return enough;
  };

  bool enough = false;

  for (int i = 0; i != mOffers.size(); ++i) {
    auto& offer = mOffers[i];
    auto& info = mInfos[i];
    if (enough) {
      break;
    }
    // Ignore:
    // - Invalid offers
    // - Offers which belong to another task
    // - Expired offers
    if (offer.valid == false) {
      stats.invalidOffers.push_back(i);
      continue;
    }
    if (offer.user != -1 && offer.user != task) {
      stats.otherUser.push_back(i);
      continue;
    }
    if (offer.runtime < 0) {
      stats.unexpiring.push_back(i);
    } else if (offer.runtime + info.received < uv_now(mLoop)) {
      LOGP(INFO, "Offer {} expired since {} milliseconds and holds {}MB", i, uv_now(mLoop) - offer.runtime - info.received, offer.sharedMemory / 1000000);
      mExpiredOffers.push_back(ComputingQuotaOfferRef{i});
      stats.expired.push_back(i);
      continue;
    } else {
      LOGP(INFO, "Offer {} still valid for {} milliseconds, providing {}MB", i, offer.runtime + info.received - uv_now(mLoop), offer.sharedMemory / 1000000);
    }
    /// We then check if the offer is suitable
    assert(offer.sharedMemory >= 0);
    auto tmp = accumulated;
    tmp.cpu += offer.cpu;
    tmp.memory += offer.memory;
    tmp.sharedMemory += offer.sharedMemory;
    offer.score = selector(offer, tmp);
    switch (offer.score) {
      case OfferScore::Unneeded:
        continue;
      case OfferScore::Unsuitable:
        continue;
      case OfferScore::More:
        selectOffer(i);
        accumulated = tmp;
        stats.selectedOffers.push_back(i);
        continue;
      case OfferScore::Enough:
        selectOffer(i);
        accumulated = tmp;
        stats.selectedOffers.push_back(i);
        enough = true;
        break;
    };
  }
  // If we get here it means we never got enough offers, so we return false.
  return summarizeWhatHappended(enough, stats.selectedOffers, accumulated, stats);
}

void ComputingQuotaEvaluator::consume(int id, ComputingQuotaConsumer& consumer)
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;
  // This will report how much of the offers has to be considered consumed.
  // Notice that actual memory usage might be larger, because we can over
  // allocate.
  auto reportConsumedOffer = [&totalDisposedMemory = mTotalDisposedSharedMemory, &monitoring = mRegistry.get<Monitoring>()](ComputingQuotaOffer const& accumulatedConsumed) {
    totalDisposedMemory += accumulatedConsumed.sharedMemory;
    monitoring.send(Metric{(uint64_t)totalDisposedMemory, "shm-offer-consumed"}.addTag(Key::Subsystem, Value::DPL));
  };
  consumer(id, mOffers, reportConsumedOffer);
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
      offer.valid = false;
      offer.score = OfferScore::Unneeded;
    }
  }
}

/// Move offers from the pending list to the actual available offers
void ComputingQuotaEvaluator::updateOffers(std::vector<ComputingQuotaOffer>& pending)
{
  for (size_t oi = 0; oi < mOffers.size(); oi++) {
    auto& storeOffer = mOffers[oi];
    auto& info = mInfos[oi];
    if (pending.empty()) {
      return;
    }
    if (storeOffer.valid == true) {
      continue;
    }
    info.received = uv_now(mLoop);
    auto& offer = pending.back();
    storeOffer = offer;
    pending.pop_back();
  }
}

void ComputingQuotaEvaluator::handleExpired()
{
  static int nothingToDoCount = mExpiredOffers.size();
  if (mExpiredOffers.size()) {
    LOGP(INFO, "Handling {} expired offers", mExpiredOffers.size());
    nothingToDoCount = 0;
  } else {
    if (nothingToDoCount == 0) {
      nothingToDoCount++;
      LOGP(INFO, "No expired offers");
    }
  }
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;
  auto& monitoring = mRegistry.get<o2::monitoring::Monitoring>();
  /// Whenever an offer is expired, we give back the resources
  /// to the driver.
  static uint64_t expiredOffers = 0;
  static uint64_t expiredBytes = 0;

  for (auto& ref : mExpiredOffers) {
    auto& offer = mOffers[ref.index];
    if (offer.sharedMemory < 0) {
      LOGP(INFO, "Offer {} does not have any more memory. Marking it as invalid.", ref.index);
      offer.valid = false;
      offer.score = OfferScore::Unneeded;
      continue;
    }
    // FIXME: offers should go through the driver client, not the monitoring
    // api.
    auto& monitoring = mRegistry.get<o2::monitoring::Monitoring>();
    monitoring.send(o2::monitoring::Metric{expiredOffers++, "resource-offer-expired"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    expiredBytes += offer.sharedMemory;
    monitoring.send(o2::monitoring::Metric{(uint64_t)expiredBytes, "arrow-bytes-expired"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    LOGP(INFO, "Offer {} expired. Giving back {}MB and {} cores", ref.index, offer.sharedMemory / 1000000, offer.cpu);
    //driverClient.tell("expired shmem {}", offer.sharedMemory);
    //driverClient.tell("expired cpu {}", offer.cpu);
    offer.sharedMemory = -1;
    offer.valid = false;
    offer.score = OfferScore::Unneeded;
  }
  mExpiredOffers.clear();
}

} // namespace o2::framework
