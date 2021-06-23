// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_COMPUTINGQUOTAEVALUATOR_H_
#define O2_COMPUTINGQUOTAEVALUATOR_H_

#include "Framework/ComputingQuotaOffer.h"

#include <cstdint>
#include <functional>
#include <array>
#include <vector>

typedef struct uv_loop_s uv_loop_t;

namespace o2::framework
{
struct ServiceRegistry;

class ComputingQuotaEvaluator
{
 public:
  // Maximum number of offers this evaluator can hold
  static constexpr int MAX_INFLIGHT_OFFERS = 16;
  ComputingQuotaEvaluator(ServiceRegistry&);
  bool selectOffer(int task, ComputingQuotaRequest const& request);

  /// Consume offers for a given taskId
  void consume(int taskId, ComputingQuotaConsumer& consumed);
  /// Dispose offers for a given taskId
  void dispose(int taskId);
  /// Handle all the offers which have timed out giving
  /// them back to the driver.
  void handleExpired();
  void updateOffers(std::vector<ComputingQuotaOffer>& offers);

  /// All the available offerts
  std::array<ComputingQuotaOffer, MAX_INFLIGHT_OFFERS> mOffers;
  /// The offers which expired and need to be given back.
  std::vector<ComputingQuotaOfferRef> mExpiredOffers;
  /// Information about a given computing offer (e.g. when it was started to be used)
  std::array<ComputingQuotaInfo, MAX_INFLIGHT_OFFERS> mInfos;
  ServiceRegistry& mRegistry;
  uv_loop_t* mLoop;
};

} // namespace o2::framework

#endif // O2_COMPUTINGQUOTAEVALUATOR_H_
