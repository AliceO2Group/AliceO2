// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  for (int i = 0; i != mOffers.size(); ++i) {
    auto& offer = mOffers[i];
    auto& info = mInfos[i];

    // Ignore:
    // - Invalid offers
    // - Offers which belong to another task
    // - Expired offers
    if (offer.valid == false) {
      continue;
    }
    if (offer.user != -1 && offer.user != task) {
      continue;
    }
    if (offer.runtime > 0 && offer.runtime + info.received < uv_now(mLoop)) {
      mExpiredOffers.push_back(ComputingQuotaOfferRef{i});
      continue;
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
      case OfferScore::Unsuitable:
        continue;
      case OfferScore::More:
        selectOffer(i);
        break;
      case OfferScore::Enough:
        selectOffer(i);
        return true;
    };
  }
  // If we get here it means we never got enough offers, so we return false.
  return false;
}

void ComputingQuotaEvaluator::consume(int id, ComputingQuotaConsumer& consumer)
{
  consumer(id, mOffers);
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
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;
  auto& monitoring = mRegistry.get<o2::monitoring::Monitoring>();
  /// Whenever an offer is expired, we give back the resources
  /// to the driver.
  for (auto& ref : mExpiredOffers) {
    auto& offer = mOffers[ref.index];
    // FIXME: offers should go through the driver client, not the monitoring
    // api.
    auto& monitoring = mRegistry.get<o2::monitoring::Monitoring>();
    monitoring.send(o2::monitoring::Metric{(uint64_t)offer.sharedMemory, "arrow-bytes-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    //    LOGP(INFO, "Offer expired {} {}", offer.sharedMemory, offer.cpu);
    //    driverClient.tell("expired shmem {}", offer.sharedMemory);
    //    driverClient.tell("expired cpu {}", offer.cpu);
  }
  mExpiredOffers.clear();
}

} // namespace o2::framework
