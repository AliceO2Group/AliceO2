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


BOOST_AUTO_TEST_CASE(TestDeviceMetricsInfo) {
  using namespace o2::framework;
  std::string metric;
  std::smatch match;
  bool result;
  DeviceMetricsInfo info;

  // Parse a simple metric
  metric = "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch";
  result = parseMetric(metric, match);
  BOOST_CHECK(result == true);
  BOOST_CHECK(match[2] == "0");
  BOOST_CHECK(match[1].str() == "bkey");
  BOOST_CHECK(match[4].str() == "1789372894");
  BOOST_CHECK(match[3].str() == "12");
  // Add the first metric to the store
  result = processMetric(match, info);
  BOOST_CHECK(result == true);
  BOOST_CHECK(info.metricLabelsIdx.size() == 1);
  BOOST_CHECK(info.metricLabelsIdx[0].first == "bkey");
  BOOST_CHECK(info.metricLabelsIdx[0].second == 0);
  BOOST_CHECK(info.intMetrics.size() == 1);
  BOOST_CHECK(info.floatMetrics.size() == 0);
  BOOST_CHECK(info.timestamps.size() == 1);
  BOOST_CHECK(info.metrics.size() == 1);
  BOOST_CHECK(info.metrics[0].type == MetricType::Int);
  BOOST_CHECK(info.metrics[0].storeIdx == 0);
  BOOST_CHECK(info.metrics[0].pos == 1);

  BOOST_CHECK(info.timestamps[0][0] == 1789372894);
  BOOST_CHECK(info.intMetrics[0][0] == 12);
  BOOST_CHECK(info.intMetrics[0][1] == 0);

  // Parse a second metric with the same key
  metric = "[METRIC] bkey,0 13 1789372894 hostname=test.cern.ch";
  result = parseMetric(metric, match);
  BOOST_CHECK(result == true);
  BOOST_CHECK(match[3].str() == "13");
  result = processMetric(match, info);
  BOOST_CHECK(result == true);
  BOOST_CHECK(info.metricLabelsIdx.size() == 1);
  BOOST_CHECK(info.intMetrics.size() == 1);
  BOOST_CHECK(info.intMetrics[0][0] == 12);
  BOOST_CHECK(info.intMetrics[0][1] == 13);
  BOOST_CHECK(info.intMetrics[0][2] == 0);
  BOOST_CHECK(info.metrics[0].pos == 2);

  // Parse a third metric with a different key
  metric = "[METRIC] akey,0 14 1789372894 hostname=test.cern.ch";
  result = parseMetric(metric, match);
  BOOST_CHECK(result == true);
  result = processMetric(match, info);
  BOOST_CHECK(result == true);
  BOOST_CHECK(info.metricLabelsIdx.size() == 2);
  BOOST_CHECK(info.intMetrics.size() == 2);
  BOOST_CHECK(info.intMetrics[0][0] == 12);
  BOOST_CHECK(info.intMetrics[0][1] == 13);
  BOOST_CHECK(info.intMetrics[0][2] == 0);
  BOOST_CHECK(info.intMetrics[1][0] == 14);
  BOOST_CHECK(info.metrics.size() == 2);
  BOOST_CHECK(info.metrics[1].type == MetricType::Int);
  BOOST_CHECK(info.metrics[1].storeIdx == 1);
  BOOST_CHECK(info.metrics[1].pos == 1);

  // Parse a fourth metric, now a float one
  metric = "[METRIC] key3,2 16.0 1789372894 hostname=test.cern.ch";
  result = parseMetric(metric, match);
  BOOST_CHECK(result == true);
  result = processMetric(match, info);
  BOOST_CHECK(result == true);
  BOOST_CHECK(info.metricLabelsIdx.size() == 3);
  BOOST_CHECK(info.intMetrics.size() == 2);
  BOOST_CHECK(info.floatMetrics.size() == 1);
  BOOST_CHECK(info.metrics.size() == 3);
  BOOST_CHECK(info.floatMetrics[0][0] == 16.0);
  BOOST_CHECK(info.floatMetrics[0][1] == 0);
  BOOST_CHECK(info.metrics[2].type == MetricType::Float);
  BOOST_CHECK(info.metrics[2].storeIdx == 0);
  BOOST_CHECK(info.metrics[2].pos == 1);

  // Parse a fifth metric, same float one
  metric = "[METRIC] key3,2 17.0 1789372895 hostname=test.cern.ch";
  result = parseMetric(metric, match);
  BOOST_CHECK(result == true);
  result = processMetric(match, info);
  BOOST_CHECK(result == true);
  BOOST_CHECK(info.metricLabelsIdx.size() == 3);
  BOOST_CHECK(info.intMetrics.size() == 2);
  BOOST_CHECK(info.floatMetrics.size() == 1);
  BOOST_CHECK(info.metrics.size() == 3);
  BOOST_CHECK(info.floatMetrics[0][0] == 16.0);
  BOOST_CHECK(info.floatMetrics[0][1] == 17.0);
  BOOST_CHECK(info.floatMetrics[0][2] == 0);
  BOOST_CHECK(info.metrics[2].type == MetricType::Float);
  BOOST_CHECK(info.metrics[2].storeIdx == 0);
  BOOST_CHECK(info.metrics[2].pos == 2);

  BOOST_CHECK(metricIdxByName("akey", info) == 1);
  BOOST_CHECK(metricIdxByName("bkey", info) == 0);
  BOOST_CHECK(metricIdxByName("key3", info) == 2);
}
