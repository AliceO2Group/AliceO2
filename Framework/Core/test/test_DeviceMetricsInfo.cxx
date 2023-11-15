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

#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include <catch_amalgamated.hpp>
#include <catch_amalgamated.hpp>
#include <regex>
#include <string_view>

TEST_CASE("TestIndexedMetrics")
{
  using namespace o2::framework;
  std::string metricString;
  ParsedMetricMatch match;
  bool result;
  DeviceMetricsInfo info;
  metricString = "[METRIC] array/1-10,0 12 1789372894 hostname=test.cern.chbar";
  result = DeviceMetricsHelper::parseMetric(metricString, match);
  REQUIRE(result);
  REQUIRE(strncmp(match.beginKey, "array", 4) == 0);
  REQUIRE(match.firstIndex == 1);
  REQUIRE(match.lastIndex == 10);
  REQUIRE(match.timestamp == 1789372894);
  REQUIRE(match.type == MetricType::Int);
  REQUIRE(match.intValue == 12);
}

TEST_CASE("TestEnums")
{
  using namespace o2::framework;
  {
    std::string metricString;
    ParsedMetricMatch match;
    bool result;
    DeviceMetricsInfo info;
    metricString = "[METRIC] data_relayer/1,0 1 1789372894 hostname=test.cern.chbar";
    result = DeviceMetricsHelper::parseMetric(metricString, match);
    REQUIRE(result);
    REQUIRE(strncmp(match.beginKey, "data_relayer/1", strlen("data_relayer/1")) == 0);
    REQUIRE(match.endKey - match.beginKey == strlen("data_relayer/1"));
    REQUIRE(match.timestamp == 1789372894);
    REQUIRE(match.type == MetricType::Enum);
    REQUIRE(match.intValue == 1);
    REQUIRE(match.uint64Value == 1);
  }

  {
    std::string metricString;
    ParsedMetricMatch match;
    bool result;
    DeviceMetricsInfo info;
    metricString = "[METRIC] data_relayer/h,0 3 1662377068602 hostname=stillwater.dyndns.cern.ch,dataprocessor_id=D,dataprocessor_name=D,dpl_instance=0";
    result = DeviceMetricsHelper::parseMetric(metricString, match);
    REQUIRE(result);
    REQUIRE(strncmp(match.beginKey, "data_relayer/h", strlen("data_relayer/h")) == 0);
    REQUIRE(match.endKey - match.beginKey == strlen("data_relayer/h"));
    REQUIRE(match.timestamp == 1662377068602);
    REQUIRE(match.type == MetricType::Int);
    REQUIRE(match.intValue == 3);
    REQUIRE(match.uint64Value == 3);
  }

  {
    std::string metricString;
    ParsedMetricMatch match;
    bool result;
    DeviceMetricsInfo info;
    metricString = "[14:00:44][INFO] metric-feedback[0]: in: 0 (0 MB) out: 0 (0";
    result = DeviceMetricsHelper::parseMetric(metricString, match);
    REQUIRE(result == false);
  }
}

TEST_CASE("TestDeviceMetricsInfo")
{
  using namespace o2::framework;
  std::string metricString;
  ParsedMetricMatch match;
  bool result;
  DeviceMetricsInfo info;

  // Parse a simple metric
  metricString = "foo[METRIC] bkey,0 12 1789372894 hostname=test.cern.chbar";
  std::string_view metric{metricString.data() + 3, metricString.size() - 6};
  REQUIRE(metric == std::string_view("[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch"));
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  REQUIRE(strncmp(match.beginKey, "bkey", 4) == 0);
  REQUIRE(match.timestamp == 1789372894);
  REQUIRE(match.type == MetricType::Int);
  REQUIRE(match.intValue == 12);
  // Add the first metric to the store
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 1);
  REQUIRE(info.metricLabels.size() == 1);
  REQUIRE(strncmp(info.metricLabels[0].label, "bkey", 4) == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 0);
  REQUIRE(info.metricPrefixes.size() == 1);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 0);
  REQUIRE(info.intMetrics.size() == 1);
  REQUIRE(info.floatMetrics.size() == 0);
  REQUIRE(info.intTimestamps.size() == 1);
  REQUIRE(info.metrics.size() == 1);
  REQUIRE(info.metrics[0].type == MetricType::Int);
  REQUIRE(info.metrics[0].storeIdx == 0);
  REQUIRE(info.metrics[0].pos == 1);

  REQUIRE(info.intTimestamps[0][0] == 1789372894);
  REQUIRE(info.intMetrics[0][0] == 12);
  REQUIRE(info.intMetrics[0][1] == 0);

  // Parse a second metric with the same key
  metric = "[METRIC] bkey,0 13 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  REQUIRE(match.intValue == 13);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 0);
  REQUIRE(info.metricPrefixes.size() == 1);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 0);
  REQUIRE(info.intMetrics.size() == 1);
  REQUIRE(info.intMetrics[0][0] == 12);
  REQUIRE(info.intMetrics[0][1] == 13);
  REQUIRE(info.intMetrics[0][2] == 0);
  REQUIRE(info.metrics[0].pos == 2);

  // Parse a third metric with a different key
  metric = "[METRIC] akey,0 14 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 0);
  REQUIRE(info.metricPrefixes.size() == 2);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 2);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 0);
  REQUIRE(info.intMetrics.size() == 2);
  REQUIRE(info.intMetrics[0][0] == 12);
  REQUIRE(info.intMetrics[0][1] == 13);
  REQUIRE(info.intMetrics[0][2] == 0);
  REQUIRE(info.intMetrics[1][0] == 14);
  REQUIRE(info.metrics.size() == 2);
  REQUIRE(info.metrics[1].type == MetricType::Int);
  REQUIRE(info.metrics[1].storeIdx == 1);
  REQUIRE(info.metrics[1].pos == 1);

  // Parse a fourth metric, now a float one
  metric = "[METRIC] key3,2 16.0 1789372894 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 3);
  REQUIRE(info.metricPrefixes.size() == 3);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 3);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[2].index == 2);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 2);
  REQUIRE(info.intMetrics.size() == 2);
  REQUIRE(info.floatMetrics.size() == 1);
  REQUIRE(info.metrics.size() == 3);
  REQUIRE(info.floatMetrics[0][0] == 16.0);
  REQUIRE(info.floatMetrics[0][1] == 0);
  REQUIRE(info.metrics[2].type == MetricType::Float);
  REQUIRE(info.metrics[2].storeIdx == 0);
  REQUIRE(info.metrics[2].pos == 1);

  // Parse a fifth metric, same float one
  metric = "[METRIC] key3,2 17.0 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 3);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 3);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[2].index == 2);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 2);
  REQUIRE(info.intMetrics.size() == 2);
  REQUIRE(info.floatMetrics.size() == 1);
  REQUIRE(info.metrics.size() == 3);
  REQUIRE(info.floatMetrics[0][0] == 16.0);
  REQUIRE(info.floatMetrics[0][1] == 17.0);
  REQUIRE(info.floatMetrics[0][2] == 0);
  REQUIRE(info.metrics[2].type == MetricType::Float);
  REQUIRE(info.metrics[2].storeIdx == 0);
  REQUIRE(info.metrics[2].pos == 2);

  REQUIRE(DeviceMetricsHelper::metricIdxByName("akey", info) == 1);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("bkey", info) == 0);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("key3", info) == 2);
  REQUIRE(DeviceMetricsHelper::metricIdxByName("foo", info) == 3);

  // Parse a string metric
  metric = "[METRIC] key4,1 some_string 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 4);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 4);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[2].index == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[3].index == 3);
  REQUIRE(info.metricPrefixes.size() == 3);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 2);
  REQUIRE(info.metrics.size() == 4);
  REQUIRE(info.stringMetrics.size() == 1);
  REQUIRE(info.metrics[3].type == MetricType::String);
  REQUIRE(info.metrics[3].storeIdx == 0);
  REQUIRE(info.metrics[3].pos == 1);

  // Parse a string metric with a file description in it
  memset(&match, 0, sizeof(match));
  metric = "[METRIC] alien-file-name,1 alien:///alice/data/2015/LHC15o/000244918/pass5_lowIR/PWGZZ/Run3_Conversion/96_20201013-1346_child_1/0028/AO2D.root:/,631838549,ALICE::CERN::EOS 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 5);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 5);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 4);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[2].index == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[3].index == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[4].index == 3);
  REQUIRE(info.metricPrefixes.size() == 3);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 2);
  REQUIRE(info.metrics.size() == 5);
  REQUIRE(info.stringMetrics.size() == 2);
  REQUIRE(info.metrics[4].type == MetricType::String);
  REQUIRE(info.metrics[4].storeIdx == 1);
  REQUIRE(info.metrics[4].pos == 1);
  REQUIRE(std::string(info.stringMetrics[1][0].data) == std::string("alien:///alice/data/2015/LHC15o/000244918/pass5_lowIR/PWGZZ/Run3_Conversion/96_20201013-1346_child_1/0028/AO2D.root:/,631838549,ALICE::CERN::EOS"));

  // Parse a vector metric
  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/w,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 6);
  REQUIRE(info.metricPrefixes.size() == 4);
  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(strcmp(info.metricPrefixes[3].prefix, "array") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 4);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[3].index == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 6);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[1].index == 4);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[2].index == 5);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[3].index == 0);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[4].index == 2);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[5].index == 3);
  REQUIRE(info.metrics.size() == 6);
  REQUIRE(info.intMetrics.size() == 3);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/h,0 3 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 7);
  REQUIRE(info.metricPrefixes.size() == 4);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx.size() == 7);
  REQUIRE(info.metrics.size() == 7);
  REQUIRE(info.intMetrics.size() == 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/0,0 0 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 8);
  REQUIRE(info.metricPrefixes.size() == 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/1,0 1 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 9);
  REQUIRE(info.metricPrefixes.size() == 4);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/2,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 10);
  REQUIRE(info.metricPrefixes.size() == 4);

  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(strcmp(info.metricPrefixes[3].prefix, "array") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 4);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[3].index == 2);

  REQUIRE(info.metricPrefixes[0].begin == 7);
  REQUIRE(info.metricPrefixes[0].end == 8);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[7].index == 0);
  REQUIRE(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[7].index].size == 4);
  REQUIRE(std::string_view(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[7].index].label, 4) == std::string_view("bkey"));
  REQUIRE(info.metricPrefixes[1].begin == 0);
  REQUIRE(info.metricPrefixes[1].end == 2);
  REQUIRE(info.metricPrefixes[2].begin == 8);
  REQUIRE(info.metricPrefixes[2].end == 10);
  REQUIRE(info.metricPrefixes[3].begin == 2);
  REQUIRE(info.metricPrefixes[3].end == 7);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] array/2,0 2 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result == true);
  result = DeviceMetricsHelper::processMetric(match, info);
  REQUIRE(result == true);
  REQUIRE(info.metricLabels.size() == 10);
  REQUIRE(info.metricPrefixes.size() == 4);

  auto array3 = DeviceMetricsHelper::createNumericMetric<int>(info, "array/3");

  REQUIRE(info.metricLabels.size() == 11);
  REQUIRE(info.metricPrefixes.size() == 4);

  REQUIRE(strcmp(info.metricPrefixes[0].prefix, "b") == 0);
  REQUIRE(strcmp(info.metricPrefixes[1].prefix, "a") == 0);
  REQUIRE(strcmp(info.metricPrefixes[2].prefix, "k") == 0);
  REQUIRE(strcmp(info.metricPrefixes[3].prefix, "array") == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx.size() == 4);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[0].index == 1);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[1].index == 3);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[2].index == 0);
  REQUIRE(info.metricLabelsPrefixesSortedIdx[3].index == 2);

  REQUIRE(info.metricPrefixes[0].begin == 8);
  REQUIRE(info.metricPrefixes[0].end == 9);
  REQUIRE(info.metricLabelsAlphabeticallySortedIdx[8].index == 0);
  REQUIRE(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[8].index].size == 4);
  REQUIRE(std::string_view(info.metricLabels[info.metricLabelsAlphabeticallySortedIdx[8].index].label, 4) == std::string_view("bkey"));
  REQUIRE(info.metricPrefixes[1].begin == 0);
  REQUIRE(info.metricPrefixes[1].end == 2);
  REQUIRE(info.metricPrefixes[2].begin == 9);
  REQUIRE(info.metricPrefixes[2].end == 11);
  REQUIRE(info.metricPrefixes[3].begin == 2);
  REQUIRE(info.metricPrefixes[3].end == 8);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] data_relayer/w,0 1 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result);
  REQUIRE(match.type == MetricType::Int);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] data_relayer/h,0 1 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result);
  REQUIRE(match.type == MetricType::Int);

  memset(&match, 0, sizeof(match));
  metric = "[METRIC] data_relayer/1,0 8 1789372895 hostname=test.cern.ch";
  result = DeviceMetricsHelper::parseMetric(metric, match);
  REQUIRE(result);
  REQUIRE(match.type == MetricType::Enum);
}

TEST_CASE("TestDeviceMetricsInfo2")
{
  using namespace o2::framework;
  DeviceMetricsInfo info;
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
  REQUIRE(info.metrics[0].filledMetrics == 6);
  REQUIRE(info.metrics[1].filledMetrics == 0);
  REQUIRE(info.metrics[2].filledMetrics == 0);
  REQUIRE(info.changed[0] == true);
  REQUIRE(info.intMetrics[0][0] == 0);
  REQUIRE(info.intMetrics[0][1] == 1);
  REQUIRE(info.intMetrics[0][2] == 2);
  REQUIRE(info.intMetrics[0][3] == 3);
  REQUIRE(info.intMetrics[0][4] == 4);
  REQUIRE(info.intMetrics[0][5] == 5);
  REQUIRE(info.changed[1] == false);
  info.changed[0] = 0;
  bkey(info, 5., t++);
  REQUIRE(info.changed[0] == true);
  REQUIRE(info.changed[1] == false);
  akey(info, 1., t++);
  akey(info, 2., t++);
  akey(info, 3., t++);
  akey(info, 4., t++);
  REQUIRE(info.changed[0] == true);
  REQUIRE(info.changed[1] == true);
  REQUIRE(info.changed[2] == false);
  REQUIRE(info.metrics[0].filledMetrics == 7);
  REQUIRE(info.metrics[1].filledMetrics == 4);
  REQUIRE(info.metrics[2].filledMetrics == 0);
  REQUIRE(info.intMetrics[0][6] == 5);
  REQUIRE(info.floatMetrics[0][0] == 1.);
  REQUIRE(info.floatMetrics[0][1] == 2.);
  REQUIRE(info.floatMetrics[0][2] == 3.);
  REQUIRE(info.floatMetrics[0][3] == 4.);
  REQUIRE(info.intTimestamps.size() == 1);
  REQUIRE(info.floatTimestamps.size() == 1);
  REQUIRE(info.uint64Timestamps.size() == 1);
  REQUIRE(info.intTimestamps[0][0] == 1000);
  REQUIRE(info.intTimestamps[0][1] == 1001);
  REQUIRE(info.intTimestamps[0][2] == 1002);
  REQUIRE(info.intTimestamps[0][3] == 1003);
  REQUIRE(info.intTimestamps[0][4] == 1004);
  REQUIRE(info.intTimestamps[0][5] == 1005);
  REQUIRE(info.floatTimestamps[0][0] == 1007);
  REQUIRE(info.floatTimestamps[0][1] == 1008);
  REQUIRE(info.floatTimestamps[0][2] == 1009);
  REQUIRE(info.floatTimestamps[0][3] == 1010);
  REQUIRE(info.changed.size() == 3);
  for (int i = 0; i < 1026; ++i) {
    ckey(info, i, t++);
  }
  REQUIRE(info.uint64Metrics[0][0] == 1024);
  REQUIRE(info.uint64Metrics[0][1] == 1025);
  REQUIRE(info.uint64Metrics[0][2] == 2);
  REQUIRE(info.metrics[0].filledMetrics == 7);
  REQUIRE(info.metrics[1].filledMetrics == 4);
  REQUIRE(info.metrics[2].filledMetrics == 1026);
}

TEST_CASE("TestHelpers")
{
  using namespace o2::framework;
  DeviceMetricsInfo info;
  auto metric1 = DeviceMetricsHelper::bookMetricInfo(info, "bkey", MetricType::Int);
  auto metric2 = DeviceMetricsHelper::bookMetricInfo(info, "bkey", MetricType::Int);
  auto metric3 = DeviceMetricsHelper::bookMetricInfo(info, "akey", MetricType::Int);
  REQUIRE(metric1 == 0);
  REQUIRE(metric2 == 0);
  REQUIRE(metric3 == 1);
}
