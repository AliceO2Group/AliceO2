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
#include "Framework/RuntimeError.h"
#include <cassert>
#include <cinttypes>
#include <cstdlib>

#include <algorithm>
#include <regex>
#include <string_view>
#include <tuple>
#include <iostream>

namespace o2::framework
{

std::ostream& operator<<(std::ostream& oss, MetricType const& val)
{
  switch (val) {
    case MetricType::Float:
      oss << "float";
      break;
    case MetricType::String:
      oss << "string";
      break;
    case MetricType::Int:
      oss << "int";
      break;
    case MetricType::Uint64:
      oss << "uint64";
      break;
    case MetricType::Enum:
      oss << "enum";
      break;
    case MetricType::Unknown:
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

void DeviceMetricsInfoHelpers::clearMetrics(std::vector<DeviceMetricsInfo>& infos)
{
  for (auto& info : infos) {
    info.intMetrics.clear();
    info.uint64Metrics.clear();
    info.stringMetrics.clear(); // We do not keep so many strings as metrics as history is less relevant.
    info.floatMetrics.clear();
    info.enumMetrics.clear();
    info.intTimestamps.clear();
    info.uint64Timestamps.clear();
    info.floatTimestamps.clear();
    info.stringTimestamps.clear();
    info.enumTimestamps.clear();
    info.max.clear();
    info.min.clear();
    info.average.clear();
    info.minDomain.clear();
    info.maxDomain.clear();
    info.metricLabels.clear();
    info.metricPrefixes.clear();
    info.metricLabelsAlphabeticallySortedIdx.clear();
    info.metricLabelsPrefixesSortedIdx.clear();
    info.metrics.clear();
    info.changed.clear();
  }
}

size_t DeviceMetricsInfoHelpers::metricsStorageSize(std::span<DeviceMetricsInfo const> infos)
{
  // Count the size of the metrics storage
  size_t totalSize = 0;
  for (auto& info : infos) {
    totalSize += info.intMetrics.size() * sizeof(MetricsStorage<int>);
    totalSize += info.uint64Metrics.size() * sizeof(MetricsStorage<uint64_t>);
    totalSize += info.stringMetrics.size() * sizeof(MetricsStorage<StringMetric>);
    totalSize += info.floatMetrics.size() * sizeof(MetricsStorage<float>);
    totalSize += info.enumMetrics.size() * sizeof(MetricsStorage<int8_t>);
    totalSize += info.intTimestamps.size() * sizeof(TimestampsStorage<int>);
    totalSize += info.uint64Timestamps.size() * sizeof(TimestampsStorage<uint64_t>);
    totalSize += info.floatTimestamps.size() * sizeof(TimestampsStorage<float>);
    totalSize += info.stringTimestamps.size() * sizeof(TimestampsStorage<StringMetric>);
    totalSize += info.enumTimestamps.size() * sizeof(TimestampsStorage<int8_t>);
    totalSize += info.max.size() * sizeof(float);
    totalSize += info.min.size() * sizeof(float);
    totalSize += info.average.size() * sizeof(float);
    totalSize += info.minDomain.size() * sizeof(size_t);
    totalSize += info.maxDomain.size() * sizeof(size_t);
    totalSize += info.metricLabels.size() * sizeof(MetricLabel);
    totalSize += info.metricPrefixes.size() * sizeof(MetricPrefix);
    totalSize += info.metricLabelsAlphabeticallySortedIdx.size() * sizeof(MetricLabelIndex);
    totalSize += info.metricLabelsPrefixesSortedIdx.size() * sizeof(MetricPrefixIndex);
    totalSize += info.metrics.size() * sizeof(MetricInfo);
    totalSize += info.changed.size() * sizeof(bool);
  }

  return totalSize;
}

} // namespace o2::framework
