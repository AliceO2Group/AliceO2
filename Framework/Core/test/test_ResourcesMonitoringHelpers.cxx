// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "../src/ResourcesMonitoringHelper.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"

#include <catch_amalgamated.hpp>
#include <regex>
#include <sstream>
#include <iostream>

TEST_CASE("StreamMetrics")
{
  using namespace o2::framework;
  std::vector<DeviceSpec> specs{
    DeviceSpec{
      .name = "someDevice",
      .id = "someDevice",
      .inputChannels = {},
      .outputChannels = {},
      .arguments = {},
      .options = {},
      .services = {},
      .algorithm = AlgorithmSpec{},
      .inputs = {},
      .outputs = {},
      .forwards = {},
      .rank = 0,
      .nSlots = 0,
      .inputTimesliceId = 0,
      .maxInputTimeslices = 0,
      .completionPolicy = CompletionPolicy{},
      .dispatchPolicy = DispatchPolicy{},
      .callbacksPolicy = CallbacksPolicy{},
      .sendingPolicy = SendingPolicy{},
      .resourcePolicy = ResourcePolicy{},
      .resource = {},
      .resourceMonitoringInterval = 10,
      .labels = {},
      .metadata = {}},
    DeviceSpec{
      .name = "anotherDevice",
      .id = "anotherDevice",
      .inputChannels = {},
      .outputChannels = {},
      .arguments = {},
      .options = {},
      .services = {},
      .algorithm = AlgorithmSpec{},
      .inputs = {},
      .outputs = {},
      .forwards = {},
      .rank = 0,
      .nSlots = 0,
      .inputTimesliceId = 0,
      .maxInputTimeslices = 0,
      .completionPolicy = CompletionPolicy{},
      .dispatchPolicy = DispatchPolicy{},
      .callbacksPolicy = CallbacksPolicy{},
      .sendingPolicy = SendingPolicy{},
      .resourcePolicy = ResourcePolicy{},
      .resource = {},
      .resourceMonitoringInterval = 10,
      .labels = {},
      .metadata = {}},

  };

  // This is the device metrics
  std::vector<DeviceMetricsInfo> metrics;
  metrics.resize(2);
  {
    DeviceMetricsInfo& info = metrics[0];
    auto bkey = DeviceMetricsHelper::createNumericMetric<int>(info, "bkey");
    REQUIRE(info.metricLabels.size() == 1);
    REQUIRE(info.metricPrefixes.size() == 1);
    auto akey = DeviceMetricsHelper::createNumericMetric<float>(info, "akey");
    REQUIRE(info.metricLabels.size() == 2);
    REQUIRE(info.metricPrefixes.size() == 2);
    auto ckey = DeviceMetricsHelper::createNumericMetric<uint64_t>(info, "ckey");
    REQUIRE(info.metricLabels.size() == 3);
    REQUIRE(info.metricPrefixes.size() == 3);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("akey", info) == 1);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("bkey", info) == 0);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("ckey", info) == 2);
    REQUIRE(info.changed.size() == 3);
    REQUIRE(info.changed.at(0) == false);
    size_t t = 1000;
    ckey(info, 0, t++);
    ckey(info, 1, t++);
    ckey(info, 2, t++);
    ckey(info, 3, t++);
    ckey(info, 4, t++);
    ckey(info, 5, t++);
  }
  // Metrics for the second device
  {
    DeviceMetricsInfo& info = metrics[1];
    auto bkey = DeviceMetricsHelper::createNumericMetric<int>(info, "bkey");
    REQUIRE(info.metricLabels.size() == 1);
    REQUIRE(info.metricPrefixes.size() == 1);
    auto akey = DeviceMetricsHelper::createNumericMetric<float>(info, "akey");
    REQUIRE(info.metricLabels.size() == 2);
    REQUIRE(info.metricPrefixes.size() == 2);
    auto ckey = DeviceMetricsHelper::createNumericMetric<uint64_t>(info, "ckey");
    REQUIRE(info.metricLabels.size() == 3);
    REQUIRE(info.metricPrefixes.size() == 3);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("akey", info) == 1);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("bkey", info) == 0);
    REQUIRE(DeviceMetricsHelper::metricIdxByName("ckey", info) == 2);
    REQUIRE(info.changed.size() == 3);
    REQUIRE(info.changed.at(0) == false);
    size_t t = 1000;
    bkey(info, 0, t++);
    bkey(info, 1, t++);
    bkey(info, 2, t++);
    bkey(info, 3, t++);
    bkey(info, 4, t++);
    bkey(info, 5, t++);
  }

  // This is the driver metrics
  DeviceMetricsInfo driverMetrics;
  auto dbkey = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "bkey");
  REQUIRE(driverMetrics.metricLabels.size() == 1);
  REQUIRE(driverMetrics.metricPrefixes.size() == 1);
  auto dakey = DeviceMetricsHelper::createNumericMetric<float>(driverMetrics, "akey");
  REQUIRE(driverMetrics.metricLabels.size() == 2);
  REQUIRE(driverMetrics.metricPrefixes.size() == 2);
  auto dckey = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "ckey");
  REQUIRE(driverMetrics.metricLabels.size() == 3);
  REQUIRE(driverMetrics.metricPrefixes.size() == 3);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("akey", driverMetrics) == 1);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("bkey", driverMetrics) == 0);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("ckey", driverMetrics) == 2);
  REQUIRE(driverMetrics.changed.size() == 3);
  REQUIRE(driverMetrics.changed.at(0) == false);
  size_t t = 2000;
  dbkey(driverMetrics, 0, t++);
  dbkey(driverMetrics, 1, t++);
  dbkey(driverMetrics, 2, t++);
  dbkey(driverMetrics, 3, t++);
  dbkey(driverMetrics, 4, t++);
  dbkey(driverMetrics, 5, t++);

  dbkey(driverMetrics, 0, t++);
  dbkey(driverMetrics, 1, t++);

  dckey(driverMetrics, 0, t++);

  std::stringstream streamer;
  std::vector<std::regex> performanceMetrics{
    std::regex("bkey"),
    std::regex("ckey"),
  };

  ResourcesMonitoringHelper::dumpMetricsToJSON(metrics,
                                               driverMetrics, specs, performanceMetrics,
                                               streamer);
  std::string streamed = streamer.str();
  std::string expected = R"JSON({
    "someDevice": {
        "ckey": [
            {
                "timestamp": "1000",
                "value": "0"
            },
            {
                "timestamp": "1001",
                "value": "1"
            },
            {
                "timestamp": "1002",
                "value": "2"
            },
            {
                "timestamp": "1003",
                "value": "3"
            },
            {
                "timestamp": "1004",
                "value": "4"
            },
            {
                "timestamp": "1005",
                "value": "5"
            }
        ]
    },
    "anotherDevice": {
        "bkey": [
            {
                "timestamp": "1000",
                "value": "0"
            },
            {
                "timestamp": "1001",
                "value": "1"
            },
            {
                "timestamp": "1002",
                "value": "2"
            },
            {
                "timestamp": "1003",
                "value": "3"
            },
            {
                "timestamp": "1004",
                "value": "4"
            },
            {
                "timestamp": "1005",
                "value": "5"
            }
        ]
    },
    "driver": {
        "bkey": [
            {
                "timestamp": "2000",
                "value": "0"
            },
            {
                "timestamp": "2001",
                "value": "1"
            },
            {
                "timestamp": "2002",
                "value": "2"
            },
            {
                "timestamp": "2003",
                "value": "3"
            },
            {
                "timestamp": "2004",
                "value": "4"
            },
            {
                "timestamp": "2005",
                "value": "5"
            },
            {
                "timestamp": "2006",
                "value": "0"
            },
            {
                "timestamp": "2007",
                "value": "1"
            }
        ],
        "ckey": [
            {
                "timestamp": "2008",
                "value": "0"
            }
        ]
    }
}
)JSON";
  REQUIRE(std::regex_replace(streamed, std::regex(R"(\s+)"), "") == std::regex_replace(expected, std::regex(R"(\s+)"), ""));
}
