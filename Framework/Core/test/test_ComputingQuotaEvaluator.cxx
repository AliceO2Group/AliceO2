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

#include <catch_amalgamated.hpp>
#include "Framework/ComputingQuotaEvaluator.h"
#include "Framework/DeviceState.h"
#include "Framework/ResourcePolicyHelpers.h"
#include "Framework/Logger.h"
#include "Framework/TimingHelpers.h"
#include "Framework/DataProcessingStats.h"
#include "uv.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using namespace o2::framework;

TEST_CASE("TestComputingQuotaEvaluator")
{
  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> reportConsumedOffer = [](ComputingQuotaOffer const& accumulatedConsumed, ComputingQuotaStats& stats) {
    stats.totalConsumedBytes += accumulatedConsumed.sharedMemory;
  };

  ComputingQuotaConsumer dispose2MB = [bs = 2000000](int taskId,
                                                     std::array<ComputingQuotaOffer, 16>& offers,
                                                     ComputingQuotaStats& stats,
                                                     std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> accountDisposed) {
    ComputingQuotaOffer disposed;
    disposed.sharedMemory = 0;
    int64_t bytesSent = bs;
    for (size_t oi = 0; oi < offers.size(); oi++) {
      auto& offer = offers[oi];
      if (offer.user != taskId) {
        continue;
      }
      int64_t toRemove = std::min((int64_t)bytesSent, offer.sharedMemory);
      offer.sharedMemory -= toRemove;
      bytesSent -= toRemove;
      disposed.sharedMemory += toRemove;
      if (bytesSent <= 0) {
        break;
      }
    }
    return accountDisposed(disposed, stats);
  };

  ComputingQuotaConsumer dispose10MB = [bs = 10000000](int taskId,
                                                       std::array<ComputingQuotaOffer, 16>& offers,
                                                       ComputingQuotaStats& stats,
                                                       std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> accountDisposed) {
    ComputingQuotaOffer disposed;
    disposed.sharedMemory = 0;
    int64_t bytesSent = bs;
    for (size_t oi = 0; oi < offers.size(); oi++) {
      auto& offer = offers[oi];
      if (offer.user != taskId) {
        continue;
      }
      int64_t toRemove = std::min((int64_t)bytesSent, offer.sharedMemory);
      offer.sharedMemory -= toRemove;
      bytesSent -= toRemove;
      disposed.sharedMemory += toRemove;
      if (bytesSent <= 0) {
        break;
      }
    }
    return accountDisposed(disposed, stats);
  };

  DataProcessingStats stats(TimingHelpers::defaultRealtimeBaseConfigurator(0, uv_default_loop()),
                            TimingHelpers::defaultCPUTimeConfigurator(uv_default_loop()));

  ServiceRegistry registry;
  ServiceRegistryRef ref(registry);
  auto state = std::make_unique<DeviceState>();
  state->loop = uv_default_loop();
  using MetricSpec = DataProcessingStats::MetricSpec;
  using Kind = DataProcessingStats::Kind;
  using Scope = DataProcessingStats::Scope;
  std::vector<o2::framework::DataProcessingStats::MetricSpec> metrics{
    MetricSpec{.name = "resources-missing",
               .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_MISSING),
               .kind = Kind::UInt64,
               .scope = Scope::DPL,
               .minPublishInterval = 1000,
               .maxRefreshLatency = 1000,
               .sendInitialValue = true},
    MetricSpec{.name = "resources-insufficient",
               .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_INSUFFICIENT),
               .kind = Kind::UInt64,
               .scope = Scope::DPL,
               .minPublishInterval = 1000,
               .maxRefreshLatency = 1000,
               .sendInitialValue = true},
    MetricSpec{.name = "resources-satisfactory",
               .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_SATISFACTORY),
               .kind = Kind::UInt64,
               .scope = Scope::DPL,
               .minPublishInterval = 1000,
               .maxRefreshLatency = 1000,
               .sendInitialValue = true},
  };
  for (auto& metric : metrics) {
    stats.registerMetric(metric);
  }
  ref.registerService(ServiceRegistryHelpers::handleForService<DeviceState>(state.get()));
  ref.registerService(ServiceRegistryHelpers::handleForService<DataProcessingStats>(&stats));

  ComputingQuotaEvaluator evaluator{ref};
  std::vector<ComputingQuotaOffer> offers{{.sharedMemory = 1000000}};
  evaluator.updateOffers(offers, 1);
  REQUIRE(evaluator.mOffers[1].sharedMemory == 1000000);
  std::vector<ComputingQuotaOffer> offers2{{.sharedMemory = 1000000}};
  evaluator.updateOffers(offers2, 2);
  REQUIRE(evaluator.mOffers[2].sharedMemory == 1000000);
  std::vector<ComputingQuotaOffer> offers3{{.sharedMemory = 2000000}, {.sharedMemory = 3000000}};
  evaluator.updateOffers(offers3, 3);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 3000000);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 2000000);
  auto policy = ResourcePolicyHelpers::sharedMemoryBoundTask("internal-dpl-aod-reader.*", 2000000);
  bool selected = evaluator.selectOffer(1, policy.request, 3);
  REQUIRE(selected);
  REQUIRE(evaluator.mOffers[0].user == -1);
  REQUIRE(evaluator.mOffers[1].user == -1);
  REQUIRE(evaluator.mOffers[2].user == -1);
  REQUIRE(evaluator.mOffers[3].user == 1);
  REQUIRE(evaluator.mOffers[4].user == -1);

  evaluator.consume(1, dispose2MB, reportConsumedOffer);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 1000000);
  REQUIRE(evaluator.mOffers[3].user == 1);

  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const&)> reportExpiredOffer = [](ComputingQuotaOffer const& offer, ComputingQuotaStats const& stats) {
  };

  REQUIRE(evaluator.mOffers[2].sharedMemory == 1000000);
  evaluator.handleExpired(reportExpiredOffer);
  REQUIRE(evaluator.mOffers[2].sharedMemory == -1);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 1000000);
  evaluator.dispose(1);
  REQUIRE(evaluator.mOffers[3].user == -1);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 1000000);
  REQUIRE(evaluator.mStats.totalExpiredBytes == 2000000);
  REQUIRE(evaluator.mStats.totalConsumedBytes == 2000000);

  selected = evaluator.selectOffer(1, policy.request, 3);
  REQUIRE(selected);

  REQUIRE(evaluator.mOffers[0].user == -1);
  REQUIRE(evaluator.mOffers[1].user == -1);
  REQUIRE(evaluator.mOffers[2].user == -1);
  REQUIRE(evaluator.mOffers[3].user == 1);
  REQUIRE(evaluator.mOffers[3].valid == true);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 1000000);
  REQUIRE(evaluator.mOffers[4].user == 1);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 2000000);

  evaluator.consume(1, dispose2MB, reportConsumedOffer);
  REQUIRE(evaluator.mOffers[2].sharedMemory == -1);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 1000000);
  evaluator.handleExpired(reportExpiredOffer);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 1000000);
  evaluator.dispose(1);
  REQUIRE(evaluator.mOffers[3].user == -1);
  REQUIRE(evaluator.mOffers[4].user == -1);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 1000000);
  REQUIRE(evaluator.mStats.totalExpiredBytes == 2000000);
  REQUIRE(evaluator.mStats.totalConsumedBytes == 4000000);

  std::vector<ComputingQuotaOffer> offers4{{.sharedMemory = 1000000, .runtime = 100}};
  evaluator.updateOffers(offers4, 2);
  REQUIRE(evaluator.mOffers[1].sharedMemory == 1000000);
  REQUIRE(evaluator.mOffers[2].sharedMemory == -1);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[4].sharedMemory == 1000000);

  selected = evaluator.selectOffer(1, policy.request, 10);
  evaluator.handleExpired(reportExpiredOffer);
  selected = evaluator.selectOffer(1, policy.request, 11);
  evaluator.handleExpired(reportExpiredOffer);
  std::vector<ComputingQuotaOffer> offers5{{.sharedMemory = 1000000, .runtime = 100}};
  evaluator.updateOffers(offers5, 4);
  selected = evaluator.selectOffer(1, policy.request, 13);
  evaluator.consume(1, dispose2MB, reportConsumedOffer);
  evaluator.handleExpired(reportExpiredOffer);
  evaluator.dispose(1);
  REQUIRE(evaluator.mStats.totalExpiredBytes == 3000000);
  REQUIRE(evaluator.mStats.totalConsumedBytes == 6000000);

  REQUIRE(evaluator.mOffers[1].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[2].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[3].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[4].sharedMemory == -1);

  std::vector<ComputingQuotaOffer> offers6{{.sharedMemory = 2000000, .runtime = 100}, {.sharedMemory = 1000000, .runtime = 100}};
  evaluator.updateOffers(offers6, 19);
  REQUIRE(evaluator.mOffers[1].sharedMemory == 1000000);
  REQUIRE(evaluator.mOffers[2].sharedMemory == 2000000);
  /// Check if we request 2MB and consume 10 works
  selected = evaluator.selectOffer(1, policy.request, 20);
  evaluator.consume(1, dispose10MB, reportConsumedOffer);
  evaluator.handleExpired(reportExpiredOffer);
  evaluator.dispose(1);
  REQUIRE(evaluator.mOffers[1].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[2].sharedMemory == 0);
  REQUIRE(evaluator.mOffers[1].user == -1);
  REQUIRE(evaluator.mOffers[2].user == -1);
  REQUIRE(evaluator.mOffers[1].valid == false);
  REQUIRE(evaluator.mOffers[2].valid == false);
}

#pragma GGC diagnostic pop
