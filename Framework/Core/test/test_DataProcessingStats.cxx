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

#include "Framework/DataProcessingStats.h"
#include "Framework/TimingHelpers.h"
#include "Framework/DeviceState.h"
#include "Framework/RuntimeError.h"
#include <catch_amalgamated.hpp>
#include <uv.h>

using namespace o2::framework;

enum TestMetricsId {
  DummyMetric = 0,
  DummyMetric2 = 1,
  Missing = 2,
  ZeroSize = 3,
};

using namespace o2::framework;

TEST_CASE("DataProcessingStats")
{
  DataProcessingStats stats(TimingHelpers::defaultRealtimeBaseConfigurator(0, uv_default_loop()),
                            TimingHelpers::defaultCPUTimeConfigurator(uv_default_loop()));

  o2::framework::clean_all_runtime_errors();
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric});
  /// Registering twice should throw.
  REQUIRE_THROWS(stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric2}));
  /// Registering with a different name should throw.
  REQUIRE_THROWS(stats.registerMetric({.name = "dummy_metric2", .metricId = DummyMetric}));
  /// Registering with a different name should throw.
  REQUIRE_THROWS(stats.registerMetric({.name = "", .metricId = ZeroSize}));

  stats.registerMetric({.name = "dummy_metric2", .metricId = DummyMetric2});
  REQUIRE(stats.metricsNames[DummyMetric] == "dummy_metric");
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 1});
  REQUIRE_THROWS(stats.updateStats({Missing, DataProcessingStats::Op::Add, 1}));
  REQUIRE(stats.nextCmd.load() == 1);
  REQUIRE(stats.updatedMetricsLapse.load() == 1);
  REQUIRE(stats.pushedMetricsLapse == 0);
  REQUIRE(stats.publishedMetricsLapse == 0);
  // Queue was not yet processed here
  REQUIRE(stats.metrics[DummyMetric] == 0);
  REQUIRE(stats.updated[DummyMetric] == false);
  REQUIRE(stats.metrics[DummyMetric2] == false);
  stats.processCommandQueue();
  REQUIRE(stats.updatedMetricsLapse.load() == 1);
  REQUIRE(stats.pushedMetricsLapse == 1);
  REQUIRE(stats.publishedMetricsLapse == 0);
  REQUIRE(stats.nextCmd.load() == 0);
  REQUIRE(stats.metrics[DummyMetric] == 1);
  REQUIRE(stats.updated[DummyMetric] == true);
  REQUIRE(stats.metrics[DummyMetric2] == 0);
  REQUIRE(stats.updated[DummyMetric2] == false);
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 1});
  REQUIRE(stats.nextCmd.load() == 1);
  // Queue was not yet processed here
  REQUIRE(stats.metrics[DummyMetric] == 1);
  // This is true because we have not flushed it yet
  REQUIRE(stats.updated[DummyMetric] == true);
  REQUIRE(stats.metrics[DummyMetric2] == false);
  stats.processCommandQueue();
  REQUIRE(stats.metrics[DummyMetric] == 2);
  REQUIRE(stats.updated[DummyMetric] == true);
  REQUIRE(stats.metrics[DummyMetric2] == 0);
  REQUIRE(stats.updated[DummyMetric2] == false);

  std::vector<std::string> updated;
  auto simpleFlush = [&updated](DataProcessingStats::MetricSpec const& spec, int64_t timestamp, int64_t value) {
    updated.emplace_back(spec.name);
  };

  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(stats.updatedMetricsLapse.load() == 2);
  REQUIRE(stats.pushedMetricsLapse == 2);
  REQUIRE(stats.publishedMetricsLapse == 1);
  REQUIRE(updated.size() == 1);
  REQUIRE(updated[0] == "dummy_metric");
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(stats.updated[DummyMetric] == false);
  REQUIRE(stats.updated[DummyMetric2] == false);
  REQUIRE(updated.size() == 1);
  REQUIRE(updated[0] == "dummy_metric");
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Sub, 1});
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 2});
  REQUIRE(stats.nextCmd.load() == 2);
  stats.processCommandQueue();
  REQUIRE(stats.updatedMetricsLapse.load() == 4);
  REQUIRE(stats.pushedMetricsLapse == 4);
  REQUIRE(stats.publishedMetricsLapse == 1);
  REQUIRE(stats.nextCmd.load() == 0);
  REQUIRE(stats.updated[DummyMetric] == true);
  REQUIRE(stats.updated[DummyMetric2] == false);
  REQUIRE(stats.metrics[DummyMetric] == 3);
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 2});
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 1});
  REQUIRE(stats.updatedMetricsLapse.load() == 6);
  REQUIRE(stats.pushedMetricsLapse == 4);
  REQUIRE(stats.publishedMetricsLapse == 1);
  stats.processCommandQueue();
  REQUIRE(stats.updatedMetricsLapse.load() == 6);
  REQUIRE(stats.pushedMetricsLapse == 6);
  REQUIRE(stats.publishedMetricsLapse == 1);
  REQUIRE(stats.updated[DummyMetric] == true);
  REQUIRE(stats.metrics[DummyMetric] == 1);
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(stats.updated[DummyMetric] == false);
  REQUIRE(stats.metrics[DummyMetric] == 1);

  // Setting the same value does not change the updated flag
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 1});
  stats.processCommandQueue();
  REQUIRE(stats.updatedMetricsLapse.load() == 7);
  REQUIRE(stats.pushedMetricsLapse == 6);
  REQUIRE(stats.publishedMetricsLapse == 2);
  REQUIRE(stats.metrics[DummyMetric] == 1);
  REQUIRE(stats.updated[DummyMetric] == false);

  REQUIRE(stats.nextCmd.load() == 0);
  for (size_t i = 0; i < 65; ++i) {
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 1});
  }
  REQUIRE(stats.nextCmd.load() == 1);
  REQUIRE(stats.metrics[DummyMetric] == 65);
  REQUIRE(stats.updated[DummyMetric] == true);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(stats.metrics[DummyMetric] == 66);
  REQUIRE(stats.updated[DummyMetric] == false);

  SECTION("Test all operations")
  {
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 100});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 100);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 1});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 101);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 3});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 104);
    stats.flushChangedMetrics(simpleFlush);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Add, 0});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 104);
    REQUIRE(stats.updated[DummyMetric] == false);

    stats.updateStats({DummyMetric, DataProcessingStats::Op::Sub, 1});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 103);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Sub, 2});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 101);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Min, 102});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 101);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::Min, 10});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 10);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::SetIfPositive, 11});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 11);
    stats.updateStats({DummyMetric, DataProcessingStats::Op::SetIfPositive, -10});
    stats.processCommandQueue();
    REQUIRE(stats.metrics[DummyMetric] == 11);
  }
}

// Here we artificially create a situation where the metrics are not added in
// time order (e.g. two different threads) but we still make sure that the
// metrics are updated correctly.
TEST_CASE("DataProcessingStatsOutOfOrder")
{
  auto realtimeTime = [](int64_t& base, int64_t& offset) {
    base = 10;
    offset = 1000;
  };
  auto cpuTime = [](int64_t base, int64_t offset) -> int64_t {
    static int count = 0;
    int64_t value[] = {0, 1000, 999, 998};
    return base + value[count++] - offset;
  };
  DataProcessingStats stats(realtimeTime, cpuTime);
  // Notice this will consume one value in the cpuTime.
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric});
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 2});
  // In reality this should have a lower timestamp than the previous one
  // so it will be committed before.
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 1});
  stats.processCommandQueue();
  REQUIRE(stats.metrics[DummyMetric] == 2);

  // In realtiy this should have a lower timestamp than the first
  // so we do not set it, even if it happens after a processCommandQueue.
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 3});
  stats.processCommandQueue();
  REQUIRE(stats.metrics[DummyMetric] == 2);
}

/// We verify that the running average is correctly computed.
TEST_CASE("DataProcessingStatsInstantaneousRate")
{
  auto realtimeConfigurator = [](int64_t& base, int64_t& offset) {
    base = 0;
    offset = 0;
  };
  auto cpuTimeConfigurator = [](int64_t base, int64_t offset) -> int64_t {
    static int count = 0;
    int64_t value[] = {0, 1000, 2000, 5000, 10000};
    return base + value[count++] - offset;
  };

  // I want to push deltas since the last update and have the immediate time
  // averaged being stored.
  DataProcessingStats stats(realtimeConfigurator, cpuTimeConfigurator);
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric, .kind = DataProcessingStats::Kind::Rate});
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 0);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 0);
  // Fake to be after 1 second
  stats.updateStats({DummyMetric, DataProcessingStats::Op::InstantaneousRate, 2000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 0);
  // Faked to be after 2 seconds
  stats.updateStats({DummyMetric, DataProcessingStats::Op::InstantaneousRate, 2000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 0);
  // Faked to be after 5 seconds
  stats.updateStats({DummyMetric, DataProcessingStats::Op::InstantaneousRate, 6000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 0);
  REQUIRE(stats.metrics[DummyMetric] == 6000);

  stats.updateStats({DummyMetric, DataProcessingStats::Op::InstantaneousRate, 5000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 0);
  REQUIRE(stats.metrics[DummyMetric] == 5000);
}

/// We verify that the running average is correctly computed.
TEST_CASE("DataProcessingStatsCumulativeRate")
{
  auto realtimeConfigurator = [](int64_t& base, int64_t& offset) {
    base = 1000;
    offset = 0;
  };
  int64_t count = 0;
  auto cpuTimeConfigurator = [&count](int64_t base, int64_t offset) -> int64_t {
    int64_t value[] = {0, 1000, 2000, 5000, 10000};
    return base + value[count++] - offset;
  };

  // I want to push deltas since the last update and have the immediate time
  // averaged being stored.
  DataProcessingStats stats(realtimeConfigurator, cpuTimeConfigurator);
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric, .kind = DataProcessingStats::Kind::Rate});
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 0);
  // Fake to be after 1 second
  stats.updateStats({DummyMetric, DataProcessingStats::Op::CumulativeRate, 2000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 2000);
  // Faked to be after 2 seconds
  stats.updateStats({DummyMetric, DataProcessingStats::Op::CumulativeRate, 2000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 4000);
  // Faked to be after 5 seconds
  stats.updateStats({DummyMetric, DataProcessingStats::Op::CumulativeRate, 6000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 10000);

  stats.updateStats({DummyMetric, DataProcessingStats::Op::CumulativeRate, 1000});
  stats.processCommandQueue();
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 11000);
}

TEST_CASE("DataProcessingStatsPublishing")
{
  auto realtimeTimestamp = [](int64_t& base, int64_t& offset) {
    base = 1000;
    offset = 0;
  };
  int64_t count = 0;
  auto cpuTimeTimestamp = [&count](int64_t base, int64_t offset) -> int64_t {
    int64_t value[] = {0, 1000, 1001, 2001, 2002, 3000, 5000, 10000, 11000, 12000};
    return base + value[count++] - offset;
  };

  // I want to push deltas since the last update and have the immediate time
  // averaged being stored.
  DataProcessingStats stats(realtimeTimestamp, cpuTimeTimestamp);
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric, .minPublishInterval = 5000});
  stats.registerMetric({.name = "dummy_metric2", .metricId = DummyMetric2, .minPublishInterval = 2000});
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 0);

  std::vector<std::string> updated;
  auto simpleFlush = [&updated](o2::framework::DataProcessingStats::MetricSpec const& spec, int64_t timestamp, int64_t value) {
    updated.emplace_back(spec.name);
  };

  // Fake to be after 1 second
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 2000});
  stats.updateStats({DummyMetric2, DataProcessingStats::Op::Set, 1000});
  REQUIRE(stats.updateInfos[0].timestamp == 1000);
  REQUIRE(stats.updateInfos[1].timestamp == 2000);
  stats.processCommandQueue();
  REQUIRE(count == 4);

  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 5);
  REQUIRE(updated.empty());
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 6);
  REQUIRE(updated.size() == 1);
  updated.clear();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(updated.size() == 1);
}

TEST_CASE("DataProcessingStatsPublishingRepeated")
{
  auto realtimeTimestamp = [](int64_t& base, int64_t& offset) {
    base = 1000;
    offset = 0;
  };
  int64_t count = 0;
  static int64_t timestamps[] = {0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 19000, 20000};
  auto cpuTimeTimestamp = [&count](int64_t base, int64_t offset) -> int64_t {
    return base + timestamps[count++] - offset;
  };

  // I want to push deltas since the last update and have the immediate time
  // averaged being stored.
  DataProcessingStats stats(realtimeTimestamp, cpuTimeTimestamp);
  stats.registerMetric({.name = "dummy_metric", .metricId = DummyMetric, .minPublishInterval = 3000, .maxRefreshLatency = 9000});
  REQUIRE(stats.updateInfos[DummyMetric].timestamp == 1000);
  REQUIRE(stats.updateInfos[DummyMetric].lastPublished == 1000);
  REQUIRE(stats.metrics[DummyMetric] == 0);

  std::vector<std::string> updated;
  auto simpleFlush = [&updated](o2::framework::DataProcessingStats::MetricSpec const& spec, int64_t timestamp, int64_t value) {
    updated.emplace_back(spec.name);
  };

  // Fake to be after 1 second
  stats.updateStats({DummyMetric, DataProcessingStats::Op::Set, 1000});
  REQUIRE(stats.updateInfos[0].timestamp == 1000);
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 3);
  REQUIRE(updated.empty());
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 4);
  CHECK(updated.size() == 1);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 5);
  CHECK(updated.size() == 1);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 6);
  CHECK(updated.size() == 1);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(count == 7);
  CHECK(updated.size() == 1);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  CHECK(updated.size() == 1);
  REQUIRE(count == 8);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  CHECK(updated.size() == 1);
  REQUIRE(count == 9);
  stats.processCommandQueue();
  stats.flushChangedMetrics(simpleFlush);
  REQUIRE(updated.size() == 2);
  REQUIRE(count == 10);
}
