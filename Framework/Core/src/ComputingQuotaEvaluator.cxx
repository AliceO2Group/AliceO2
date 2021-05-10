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
    false,
    true};
  mInfos[0] = {
    uv_now(loop),
    0,
    0};
}

ComputingQuotaOfferRef ComputingQuotaEvaluator::selectOffer(ComputingQuotaRequest const& evaluator)
{
  int8_t bestOfferScore = -1;
  int bestOfferIndex = -1;

  for (int i = 0; i != mOffers.size(); ++i) {
    auto& offer = mOffers[i];
    auto& info = mInfos[i];

    // Ignore:
    // - Invalid offers
    // - Used offers
    // - Expired offers
    if (offer.valid == false) {
      continue;
    }
    if (offer.used == true) {
      continue;
    }
    if (offer.runtime > 0 && offer.runtime + info.received < uv_now(mLoop)) {
      continue;
    }
    int score = evaluator(offer);
    if (score > bestOfferScore) {
      bestOfferScore = score;
      bestOfferIndex = i;
    }
    if (score == 127) {
      break;
    }
  }

  if (bestOfferScore < 0) {
    return ComputingQuotaOfferRef{-1};
  }
  auto& selected = mOffers[bestOfferIndex];
  auto& info = mInfos[bestOfferIndex];
  selected.used = true;
  if (info.firstUsed == 0) {
    info.firstUsed = uv_now(mLoop);
  }
  info.lastUsed = uv_now(mLoop);
  return ComputingQuotaOfferRef{bestOfferIndex};
}

void ComputingQuotaEvaluator::dispose(ComputingQuotaOfferRef offer)
{
  mOffers[offer.index].used = false;
}

} // namespace o2::framework
