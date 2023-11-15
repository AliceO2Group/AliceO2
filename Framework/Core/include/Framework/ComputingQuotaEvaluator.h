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

#ifndef O2_COMPUTINGQUOTAEVALUATOR_H_
#define O2_COMPUTINGQUOTAEVALUATOR_H_

#include "Framework/ComputingQuotaOffer.h"
#include "Framework/ServiceRegistryRef.h"

#include <cstdint>
#include <functional>
#include <array>
#include <vector>
#include <cstddef>

using uv_loop_t = struct uv_loop_s;
using uv_timer_t = struct uv_timer_s;

namespace o2::framework
{
struct ServiceRegistry;

class ComputingQuotaEvaluator
{
 public:
  // Maximum number of offers this evaluator can hold
  static constexpr int MAX_INFLIGHT_OFFERS = 16;
  ComputingQuotaEvaluator(ServiceRegistryRef ref);
  /// @a task the task which needs some quota
  /// @a request the resource request the @a task needs
  /// @a now the time (e.g. uv_now) when invoked.
  bool selectOffer(int task, ComputingQuotaRequest const& request, uint64_t now);
  /// Consume offers for a given taskId
  /// @a reportConsumedOffer callback which reports back that an offer has been consumed.
  void consume(int taskId,
               ComputingQuotaConsumer& consumed,
               std::function<void(ComputingQuotaOffer const& accumulatedConsumed, ComputingQuotaStats&)>& reportConsumedOffer);
  /// Dispose offers for a given taskId
  void dispose(int taskId);
  /// Handle all the offers which have timed out giving
  /// them back to the driver.
  /// @a expirator callback with expired offers
  void handleExpired(std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const&)> reportExpired);
  /// @a now the time (e.g. uv_now) when invoked.
  void updateOffers(std::vector<ComputingQuotaOffer>& offers, uint64_t now);

  /// All the available offerts
  std::array<ComputingQuotaOffer, MAX_INFLIGHT_OFFERS> mOffers;
  /// The offers which expired and need to be given back.
  std::vector<ComputingQuotaOfferRef> mExpiredOffers;
  /// Information about a given computing offer (e.g. when it was started to be used)
  std::array<ComputingQuotaInfo, MAX_INFLIGHT_OFFERS> mInfos;
  ComputingQuotaStats mStats;
  ServiceRegistryRef mRef;
  uv_timer_t* mTimer;
};

} // namespace o2::framework

#endif // O2_COMPUTINGQUOTAEVALUATOR_H_
