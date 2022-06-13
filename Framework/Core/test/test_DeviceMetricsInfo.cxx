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
#define BOOST_TEST_MODULE Test Framework DeviceMetricsInfo
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
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
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 1);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 1);
  BOOST_CHECK(strncmp(info.metricLabels[0].label, "bkey", 4) == 0);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 0);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 1);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 0);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 0);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 1);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 0);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 2);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 2);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 0);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 3);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 3);
  BOOST_REQUIRE_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_REQUIRE_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 0);
  BOOST_REQUIRE_EQUAL(info.metricLabelsAlphabeticallySortedIdx[2].index, 2);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 2);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[2].index, 2);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 2);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[2].index, 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[3].index, 3);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 3);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 2);
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
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 5);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 5);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 4);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[2].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[3].index, 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[4].index, 3);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 3);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 2);
  BOOST_CHECK_EQUAL(info.metrics.size(), 5);
  BOOST_CHECK_EQUAL(info.stringMetrics.size(), 2);
  BOOST_CHECK_EQUAL(info.metrics[4].type, MetricType::String);
  BOOST_CHECK_EQUAL(info.metrics[4].storeIdx, 1);
  BOOST_CHECK_EQUAL(info.metrics[4].pos, 1);
  BOOST_CHECK_EQUAL(std::string(info.stringMetrics[1][0].data), std::string("alien:///alice/data/2015/LHC15o/000244918/pass5_lowIR/PWGZZ/Run3_Conversion/96_20201013-1346_child_1/0028/AO2D.root:/,631838549,ALICE::CERN::EOS"));

  // Parse a vector metric
  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/w,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 6);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[3].prefix, "array"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[3].index, 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 6);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[1].index, 4);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[2].index, 5);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[3].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[4].index, 2);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[5].index, 3);
  BOOST_CHECK_EQUAL(info.metrics.size(), 6);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 3);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/h,0 3 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 7);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx.size(), 7);
  BOOST_CHECK_EQUAL(info.metrics.size(), 7);
  BOOST_CHECK_EQUAL(info.intMetrics.size(), 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/0,0 0 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 8);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/1,0 1 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 9);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/2,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 10);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);

  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[3].prefix, "array"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[3].index, 2);

  BOOST_CHECK_EQUAL(info.metricPrefixes[0].begin, 7);
  BOOST_CHECK_EQUAL(info.metricPrefixes[0].end, 8);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[7].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[7].index].size, 4);
  BOOST_CHECK_EQUAL(std::string_view(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[7].index].label, 4), "bkey");
  BOOST_CHECK_EQUAL(info.metricPrefixes[1].begin, 0);
  BOOST_CHECK_EQUAL(info.metricPrefixes[1].end, 2);
  BOOST_CHECK_EQUAL(info.metricPrefixes[2].begin, 8);
  BOOST_CHECK_EQUAL(info.metricPrefixes[2].end, 10);
  BOOST_CHECK_EQUAL(info.metricPrefixes[3].begin, 2);
  BOOST_CHECK_EQUAL(info.metricPrefixes[3].end, 7);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/2,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  BOOST_CHECK_EQUAL(result, true);
  result = DeviceMetricsHelper::processMetric(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 10);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);

  auto array3 = DeviceMetricsHelper::createNumericMetric<int>(info, "array/3");

  BOOST_CHECK_EQUAL(info.metricLabels.size(), 11);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 4);

  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[0].prefix, "b"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[1].prefix, "a"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[2].prefix, "k"), 0);
  BOOST_CHECK_EQUAL(strcmp(info.metricPrefixes[3].prefix, "array"), 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx.size(), 4);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[0].index, 1);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[1].index, 3);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[2].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabelsPrefixesSortedIdx[3].index, 2);

  BOOST_CHECK_EQUAL(info.metricPrefixes[0].begin, 8);
  BOOST_CHECK_EQUAL(info.metricPrefixes[0].end, 9);
  BOOST_CHECK_EQUAL(info.metricLabelsAlphabeticallySortedIdx[8].index, 0);
  BOOST_CHECK_EQUAL(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[8].index].size, 4);
  BOOST_CHECK_EQUAL(std::string_view(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[8].index].label, 4), "bkey");
  BOOST_CHECK_EQUAL(info.metricPrefixes[1].begin, 0);
  BOOST_CHECK_EQUAL(info.metricPrefixes[1].end, 2);
  BOOST_CHECK_EQUAL(info.metricPrefixes[2].begin, 9);
  BOOST_CHECK_EQUAL(info.metricPrefixes[2].end, 11);
  BOOST_CHECK_EQUAL(info.metricPrefixes[3].begin, 2);
  BOOST_CHECK_EQUAL(info.metricPrefixes[3].end, 8);
}

BOOST_AUTO_TEST_CASE(TestDeviceMetricsInfo2)
{
  using namespace o2::framework;
  DeviceMetricsInfo info;
  auto bkey = DeviceMetricsHelper::createNumericMetric<int>(info, "bkey");
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 1);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 1);
  auto akey = DeviceMetricsHelper::createNumericMetric<float>(info, "akey");
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 2);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 2);
  auto ckey = DeviceMetricsHelper::createNumericMetric<uint64_t>(info, "ckey");
  BOOST_CHECK_EQUAL(info.metricLabels.size(), 3);
  BOOST_CHECK_EQUAL(info.metricPrefixes.size(), 3);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("akey", info), 1);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("bkey", info), 0);
  BOOST_CHECK_EQUAL(DeviceMetricsHelper::metricIdxByName("ckey", info), 2);
  BOOST_REQUIRE_EQUAL(info.changed.size(), 3);
  BOOST_CHECK_EQUAL(info.changed.at(0), false);
  size_t t = 1000;
  bkey(info, 0, t++);
  bkey(info, 1, t++);
  bkey(info, 2, t++);
  bkey(info, 3, t++);
  bkey(info, 4, t++);
  bkey(info, 5, t++);
  BOOST_CHECK_EQUAL(info.metrics[0].filledMetrics, 6);
  BOOST_CHECK_EQUAL(info.metrics[1].filledMetrics, 0);
  BOOST_CHECK_EQUAL(info.metrics[2].filledMetrics, 0);
  BOOST_CHECK_EQUAL(info.changed[0], true);
  BOOST_CHECK_EQUAL(info.intMetrics[0][0], 0);
  BOOST_CHECK_EQUAL(info.intMetrics[0][1], 1);
  BOOST_CHECK_EQUAL(info.intMetrics[0][2], 2);
  BOOST_CHECK_EQUAL(info.intMetrics[0][3], 3);
  BOOST_CHECK_EQUAL(info.intMetrics[0][4], 4);
  BOOST_CHECK_EQUAL(info.intMetrics[0][5], 5);
  BOOST_CHECK_EQUAL(info.changed[1], false);
  info.changed[0] = 0;
  bkey(info, 5., t++);
  BOOST_CHECK_EQUAL(info.changed[0], true);
  BOOST_CHECK_EQUAL(info.changed[1], false);
  akey(info, 1., t++);
  akey(info, 2., t++);
  akey(info, 3., t++);
  akey(info, 4., t++);
  BOOST_CHECK_EQUAL(info.changed[0], true);
  BOOST_CHECK_EQUAL(info.changed[1], true);
  BOOST_CHECK_EQUAL(info.changed[2], false);
  BOOST_CHECK_EQUAL(info.metrics[0].filledMetrics, 7);
  BOOST_CHECK_EQUAL(info.metrics[1].filledMetrics, 4);
  BOOST_CHECK_EQUAL(info.metrics[2].filledMetrics, 0);
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
  BOOST_CHECK_EQUAL(info.metrics[0].filledMetrics, 7);
  BOOST_CHECK_EQUAL(info.metrics[1].filledMetrics, 4);
  BOOST_CHECK_EQUAL(info.metrics[2].filledMetrics, 1026);
}

BOOST_AUTO_TEST_CASE(TestHelpers)
{
  using namespace o2::framework;
  DeviceMetricsInfo info;
  auto metric1 = DeviceMetricsHelper::bookMetricInfo(info, "bkey", MetricType::Int);
  auto metric2 = DeviceMetricsHelper::bookMetricInfo(info, "bkey", MetricType::Int);
  auto metric3 = DeviceMetricsHelper::bookMetricInfo(info, "akey", MetricType::Int);
  BOOST_CHECK_EQUAL(metric1, 0);
  BOOST_CHECK_EQUAL(metric2, 0);
  BOOST_CHECK_EQUAL(metric3, 1);
}
