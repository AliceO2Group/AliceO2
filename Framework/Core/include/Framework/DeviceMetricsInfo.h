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
//
#ifndef O2_FRAMEWORK_DEVICEMETRICSINFO_H_
#define O2_FRAMEWORK_DEVICEMETRICSINFO_H_

#include "Framework/RuntimeError.h"
#include "Framework/Traits.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace o2::framework
{

enum class MetricType {
  Int = 0,
  String = 1,
  Float = 2,
  Uint64 = 3,
  Unknown
};

std::ostream& operator<<(std::ostream& oss, MetricType const& val);

struct MetricInfo {
  enum MetricType type = MetricType::Unknown;
  size_t storeIdx = -1;     // Index in the actual store
  size_t pos = 0;           // Last position in the circular buffer
  size_t filledMetrics = 0; // How many metrics were filled
};

// We keep only fixed lenght strings for metrics, as in the end this is not
// really needed. They should be nevertheless 0 terminated.
struct StringMetric {
  static constexpr ptrdiff_t MAX_SIZE = 512;
  char data[MAX_SIZE];
};

// Also for the keys it does not make much sense to keep more than 255 chars.
// They should be nevertheless 0 terminated.
struct MetricLabel {
  static constexpr size_t MAX_METRIC_LABEL_SIZE = 256 - sizeof(unsigned char); // Maximum size for a metric name.
  unsigned char size = 0;
  char label[MAX_METRIC_LABEL_SIZE];
};

struct MetricPrefix {
  static constexpr size_t MAX_METRIC_PREFIX_SIZE = 256 - sizeof(unsigned char); // Maximum size for a metric name.
  unsigned char size = 0;
  char prefix[MAX_METRIC_PREFIX_SIZE];
  int begin = 0;
  int end = 0;
};

struct MetricLabelIndex {
  size_t index;
};

struct MetricPrefixIndex {
  size_t index;
};

/// Temporary struct to hold a metric after it has been parsed.
struct ParsedMetricMatch {
  char const* beginKey;
  char const* endKey;
  size_t timestamp;
  MetricType type;
  int intValue;
  float floatValue;
  uint64_t uint64Value;
  char const* beginStringValue;
  char const* endStringValue;
};

/// This struct hold information about device metrics when running
/// in standalone mode. It's position in the holding vector is
/// the same as the DeviceSpec in its own vector.
struct DeviceMetricsInfo {
  // We keep the size of each metric to 4096 bytes. No need for more
  // for the debug GUI
  std::vector<std::array<int, 1024>> intMetrics;
  std::vector<std::array<uint64_t, 1024>> uint64Metrics;
  std::vector<std::array<StringMetric, 32>> stringMetrics; // We do not keep so many strings as metrics as history is less relevant.
  std::vector<std::array<float, 1024>> floatMetrics;
  std::vector<std::array<size_t, 1024>> timestamps;
  std::vector<float> max;
  std::vector<float> min;
  std::vector<float> average;
  std::vector<size_t> minDomain;
  std::vector<size_t> maxDomain;
  std::vector<MetricLabel> metricLabels;
  std::vector<MetricPrefix> metricPrefixes;
  std::vector<MetricLabelIndex> metricLabelsAlphabeticallySortedIdx;
  std::vector<MetricPrefixIndex> metricLabelsPrefixesSortedIdx;
  std::vector<MetricInfo> metrics;
  std::vector<bool> changed;
};

struct DeviceMetricsInfoHelpers {
  template <typename T>
  static std::array<T, 1024> const& get(DeviceMetricsInfo const& info, size_t metricIdx)
  {
    if constexpr (std::is_same_v<T, int>) {
      return info.intMetrics[metricIdx];
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return info.uint64Metrics[metricIdx];
    } else if constexpr (std::is_same_v<T, StringMetric>) {
      return info.stringMetrics[metricIdx];
    } else if constexpr (std::is_same_v<T, float>) {
      return info.floatMetrics[metricIdx];
    } else {
      static_assert(always_static_assert_v<T>, "Unsupported type");
    }
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
