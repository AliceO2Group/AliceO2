// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DeviceMetricsInfo
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DeviceMetricsInfo.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <regex>
#include <string_view>

BOOST_AUTO_TEST_CASE(TestDeviceMetricsInfo)
{
  using namespace o2::framework;
  std::string metricString;
  ParsedMetricMatch match;
  bool result;
  DeviceMetricsInfo info;

  // Parse a simple metric
  metricString = "foo[METRIC] bkey,0 12 1789372894 hostname=test.cern.chbar";
  std::string_view metric{metricString.data() + 3, metricString.size() - 6};
  BOOST_REQUIRE_EQUAL(metric, std::string("[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch"));
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_REQUIRE_EQUAL(result, true);
  BOOST_CHECK(strncmp(match.beginKey, "bkey", 4) == 0);
  BOOST_CHECK_EQUAL(match.timestamp, 1789372894);
  BOOST_REQUIRE_EQUAL(match.type, MetricType::Int);
  BOOST_CHECK_EQUAL(match.intValue, 12);
  // Add the first metric to the store
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 1);
  BOOST_CHECK(strncmp(info.metricLabelsIdx[0].label, "bkey", 4) == 0);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx[0].index, 0);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 1);
  BOOST_CHECK_EQUAL(info.floatMetrics.size(), 0);
  BOOST_CHECK_EQUAL(info.timestamps.size(), 1);
  BOOST_CHECK_EQUAL(info.metrics.size(), 1);
  BOOST_CHECK_EQUAL(info.metrics[0].type, MetricType::Int);
  BOOST_CHECK_EQUAL(info.metrics[0].storeIdx, 0);
  BOOST_CHECK_EQUAL(info.metrics[0].pos, 1);

  BOOST_CHECK_EQUAL(info.timestamps[0][0], 1789372894);
  BOOST_CHECK_EQUAL(info.intMetrics[0][0], 12);
  BOOST_CHECK_EQUAL(info.intMetrics[0][1], 0);

  // Parse a second metric with the same key
  metric = "[METRIC] bkey,0 13 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(match.intValue, 13);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 1);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 1);
  BOOST_CHECK_EQUAL(info.intMetrics[0][0], 12);
  BOOST_CHECK_EQUAL(info.intMetrics[0][1], 13);
  BOOST_CHECK_EQUAL(info.intMetrics[0][2], 0);
  BOOST_CHECK_EQUAL(info.metrics[0].pos, 2);

  // Parse a third metric with a different key
  metric = "[METRIC] akey,0 14 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 2);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 2);
  BOOST_CHECK_EQUAL(info.intMetrics[0][0], 12);
  BOOST_CHECK_EQUAL(info.intMetrics[0][1], 13);
  BOOST_CHECK_EQUAL(info.intMetrics[0][2], 0);
  BOOST_CHECK_EQUAL(info.intMetrics[1][0], 14);
  BOOST_CHECK_EQUAL(info.metrics.size(), 2);
  BOOST_CHECK_EQUAL(info.metrics[1].type, MetricType::Int);
  BOOST_CHECK_EQUAL(info.metrics[1].storeIdx, 1);
  BOOST_CHECK_EQUAL(info.metrics[1].pos, 1);

  // Parse a fourth metric, now a float one
  metric = "[METRIC] key3,2 16.0 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 2);
  BOOST_CHECK_EQUAL(info.floatMetrics.size(), 1);
  BOOST_CHECK_EQUAL(info.metrics.size(), 3);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][0], 16.0);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][1], 0);
  BOOST_CHECK_EQUAL(info.metrics[2].type, MetricType::Float);
  BOOST_CHECK_EQUAL(info.metrics[2].storeIdx, 0);
  BOOST_CHECK_EQUAL(info.metrics[2].pos, 1);

  // Parse a fifth metric, same float one
  metric = "[METRIC] key3,2 17.0 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 2);
  BOOST_CHECK_EQUAL(info.floatMetrics.size(), 1);
  BOOST_CHECK_EQUAL(info.metrics.size(), 3);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][0], 16.0);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][1], 17.0);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][2], 0);
  BOOST_CHECK_EQUAL(info.metrics[2].type, MetricType::Float);
  BOOST_CHECK_EQUAL(info.metrics[2].storeIdx, 0);
  BOOST_CHECK_EQUAL(info.metrics[2].pos, 2);

  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("akey", info), 1);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("bkey", info), 0);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("key3", info), 2);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("foo", info), 3);

  // Parse a string metric
  metric = "[METRIC] key4,1 some_string 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 4);
  BOOST_CHECK_EQUAL(info.metrics.size(), 4);
  BOOST_CHECK_EQUAL(info.stringMetrics.size(), 1);
  BOOST_CHECK_EQUAL(info.metrics[3].type, MetricType::String);
  BOOST_CHECK_EQUAL(info.metrics[3].storeIdx, 0);
  BOOST_CHECK_EQUAL(info.metrics[3].pos, 1);

  // Parse a string metric with a file description in it
  memset(&match, 0, sizeof(match));
  metric = "[METRIC] alien-file-name,1 alien:///alice/data/2015/LHC15o/000244918/pass5_lowIR/PWGZZ/Run3_Conversion/96_20201013-1346_child_1/0028/AO2D.root:/,631838549,ALICE::CERN::EOS 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabelsIdx.size(), 5);
  BOOST_CHECK_EQUAL(info.metrics.size(), 5);
  BOOST_CHECK_EQUAL(info.stringMetrics.size(), 2);
  BOOST_CHECK_EQUAL(info.metrics[4].type, MetricType::String);
  BOOST_CHECK_EQUAL(info.metrics[4].storeIdx, 1);
  BOOST_CHECK_EQUAL(info.metrics[4].pos, 1);
  BOOST_CHECK_EQUAL(std::string(info.stringMetrics[1][0].data), std::string("alien:///alice/data/2015/LHC15o/000244918/pass5_lowIR/PWGZZ/Run3_Conversion/96_20201013-1346_child_1/0028/AO2D.root:/,631838549,ALICE::CERN::EOS"));
}

BOOST_AUTO_TEST_CASE(TestDeviceMetricsInfo2)
{
  using namespace o2::framework;
  DeviceMetricsInfo info;
  auto bkey = DeviceMetricsHelper::createNumericMetric<int>(info, "bkey");
  auto akey = DeviceMetricsHelper::createNumericMetric<float>(info, "akey");
  auto ckey = DeviceMetricsHelper::createNumericMetric<uint64_t>(info, "ckey");
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("akey", info), 1);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("bkey", info), 0);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("ckey", info), 2);
  BOOST_CHECK_EQUAL(info.changed.at(0), true);
  size_t t = 1000;
  bkey(info, 0, t++);
  bkey(info, 1, t++);
  bkey(info, 2, t++);
  bkey(info, 3, t++);
  bkey(info, 4, t++);
  bkey(info, 5, t++);
  BOOST_CHECK_EQUAL(info.changed[0], true);
  BOOST_CHECK_EQUAL(info.intMetrics[0][0], 0);
  BOOST_CHECK_EQUAL(info.intMetrics[0][1], 1);
  BOOST_CHECK_EQUAL(info.intMetrics[0][2], 2);
  BOOST_CHECK_EQUAL(info.intMetrics[0][3], 3);
  BOOST_CHECK_EQUAL(info.intMetrics[0][4], 4);
  BOOST_CHECK_EQUAL(info.intMetrics[0][5], 5);
  BOOST_CHECK_EQUAL(info.changed[1], true);
  bkey(info, 5., t++);
  akey(info, 1., t++);
  akey(info, 2., t++);
  akey(info, 3., t++);
  akey(info, 4., t++);
  BOOST_CHECK_EQUAL(info.intMetrics[0][6], 5);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][0], 1.);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][1], 2.);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][2], 3.);
  BOOST_CHECK_EQUAL(info.floatMetrics[0][3], 4.);
  BOOST_CHECK_EQUAL(info.timestamps.size(), 3);
  BOOST_CHECK_EQUAL(info.timestamps[0][0], 1000);
  BOOST_CHECK_EQUAL(info.timestamps[0][1], 1001);
  BOOST_CHECK_EQUAL(info.timestamps[0][2], 1002);
  BOOST_CHECK_EQUAL(info.timestamps[0][3], 1003);
  BOOST_CHECK_EQUAL(info.timestamps[0][4], 1004);
  BOOST_CHECK_EQUAL(info.timestamps[0][5], 1005);
  BOOST_CHECK_EQUAL(info.timestamps[1][0], 1007);
  BOOST_CHECK_EQUAL(info.timestamps[1][1], 1008);
  BOOST_CHECK_EQUAL(info.timestamps[1][2], 1009);
  BOOST_CHECK_EQUAL(info.timestamps[1][3], 1010);
  BOOST_CHECK_EQUAL(info.changed.size(), 3);
  for (int i = 0; i < 1026; ++i) {
    ckey(info, i, t++);
  }
  BOOST_CHECK_EQUAL(info.uint64Metrics[0][0], 1024);
  BOOST_CHECK_EQUAL(info.uint64Metrics[0][1], 1025);
  BOOST_CHECK_EQUAL(info.uint64Metrics[0][2], 2);
}
