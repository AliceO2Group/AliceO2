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
#include <vector>
#include <uv.h>

namespace o2::framework
{

ComputingQuotaEvaluator::ComputingQuotaEvaluator(uv_loop_t* loop)
  : mLoop{loop}
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
    uv_now(loop),
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
      continue;
    }
    /// We then check if the offer is suitable
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
    }
  }
}

} // namespace o2::framework
