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
#include "Framework/CompilerBuiltins.h"
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
  // DPL specific type, used for the GUI. Maximum 8 bits
  // and we keep only the last 8 entries in the history.
  Enum = 16,
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
  // Pointer to the first character of the metric name.
  char const* beginKey;
  // Pointer to the last character of the metric name.
  char const* endKey;
  // If the metric is in the form name/<begin>-<end>, this is /
  // nullptr otherwise. If not nullptr, the actual
  // metric name is given by the range [beginKey, beginRange) applied
  // the key. I.e.:
  // name/<begin>-<end> -> name/<begin>...name/<end>
  // If the metric is in the form name/<begin>-<end>, this is <begin>
  int firstIndex = -1;
  // If the metric is in the form name/<begin>-<end>, this is <end>
  int lastIndex = -1;
  size_t timestamp;
  MetricType type;
  int intValue;
  float floatValue;
  uint64_t uint64Value;
  char const* beginStringValue;
  char const* endStringValue;
};

template <typename T>
inline constexpr size_t metricStorageSize()
{
  if constexpr (std::is_same_v<T, int>) {
    return 1024;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return 1024;
  } else if constexpr (std::is_same_v<T, StringMetric>) {
    return 32;
  } else if constexpr (std::is_same_v<T, float>) {
    return 1024;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return 8;
  } else {
    static_assert(always_static_assert_v<T>, "Unsupported type");
  }
}

static inline constexpr size_t metricStorageSize(enum MetricType type)
{
  switch (type) {
    case MetricType::Int:
      return 1024;
    case MetricType::Uint64:
      return 1024;
    case MetricType::Float:
      return 1024;
    case MetricType::String:
      return 32;
    case MetricType::Enum:
      return 8;
    case MetricType::Unknown:
      return 0;
  }
  O2_BUILTIN_UNREACHABLE();
}

template <typename T>
using MetricsStorage = std::array<T, metricStorageSize<T>()>;

template <typename T>
using TimestampsStorage = std::array<size_t, metricStorageSize<T>()>;

/// This struct hold information about device metrics when running
/// in standalone mode. It's position in the holding vector is
/// the same as the DeviceSpec in its own vector.
struct DeviceMetricsInfo {
  // We keep the size of each metric to 4096 bytes. No need for more
  // for the debug GUI
  std::vector<MetricsStorage<int>> intMetrics;
  std::vector<MetricsStorage<uint64_t>> uint64Metrics;
  std::vector<MetricsStorage<StringMetric>> stringMetrics; // We do not keep so many strings as metrics as history is less relevant.
  std::vector<MetricsStorage<float>> floatMetrics;
  std::vector<MetricsStorage<int8_t>> enumMetrics;
  std::vector<std::array<size_t, metricStorageSize<int>()>> intTimestamps;
  std::vector<std::array<size_t, metricStorageSize<uint64_t>()>> uint64Timestamps;
  std::vector<std::array<size_t, metricStorageSize<float>()>> floatTimestamps;
  std::vector<std::array<size_t, metricStorageSize<StringMetric>()>> stringTimestamps;
  std::vector<std::array<size_t, metricStorageSize<int8_t>()>> enumTimestamps;
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
  template <typename T, size_t I = metricStorageSize<T>()>
  static std::array<T, I> const& get(DeviceMetricsInfo const& info, size_t metricIdx)
  {
    if constexpr (std::is_same_v<T, int>) {
      return info.intMetrics[metricIdx];
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return info.uint64Metrics[metricIdx];
    } else if constexpr (std::is_same_v<T, StringMetric>) {
      return info.stringMetrics[metricIdx];
    } else if constexpr (std::is_same_v<T, float>) {
      return info.floatMetrics[metricIdx];
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return info.enumMetrics[metricIdx];
    } else {
      static_assert(always_static_assert_v<T>, "Unsupported type");
    }
  }

  static void clearMetrics(std::vector<DeviceMetricsInfo>& infos)
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
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
