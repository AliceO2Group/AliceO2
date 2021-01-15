// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
#ifndef O2_FRAMEWORK_DEVICEMETRICSINFO_H_
#define O2_FRAMEWORK_DEVICEMETRICSINFO_H_

#include "Framework/RuntimeError.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
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

// Also for the keys it does not make much sense to keep more than 247 chars.
// They should be nevertheless 0 terminated.
struct MetricLabelIndex {
  static constexpr size_t MAX_METRIC_LABEL_SIZE = 256 - sizeof(size_t) - sizeof(unsigned char); // Maximum size for a metric name.
  size_t index;
  unsigned char size = 0;
  char label[MAX_METRIC_LABEL_SIZE];
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
/// in standalone mode
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
  std::vector<size_t> minDomain;
  std::vector<size_t> maxDomain;
  std::vector<MetricLabelIndex> metricLabelsIdx;
  std::vector<MetricInfo> metrics;
  std::vector<bool> changed;
};

struct DeviceMetricsHelper {
  /// Type of the callback which can be provided to be invoked every time a new
  /// metric is found by the system.
  using NewMetricCallback = std::function<void(std::string const&, MetricInfo const&, int value, size_t metricIndex)>;

  /// Helper function to parse a metric string.
  static bool parseMetric(std::string_view const s, ParsedMetricMatch& results);

  /// Processes a parsed metric and stores in the backend store.
  ///
  /// @matches is the regexp_matches from the metric identifying regex
  /// @info is the DeviceInfo associated to the device posting the metric
  /// @newMetricsCallback is a callback that will be invoked every time a new metric is added to the list.
  static bool processMetric(ParsedMetricMatch& results,
                            DeviceMetricsInfo& info,
                            NewMetricCallback newMetricCallback = nullptr);
  /// @return the index in metrics for the information of given metric
  static size_t metricIdxByName(const std::string& name,
                                const DeviceMetricsInfo& info);

  /// Typesafe way to get the actual store
  template <typename T>
  static auto& getMetricsStore(DeviceMetricsInfo& metrics)
  {
    if constexpr (std::is_same_v<T, int>) {
      return metrics.intMetrics;
    } else if constexpr (std::is_same_v<T, float>) {
      return metrics.floatMetrics;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return metrics.uint64Metrics;
    } else {
      throw runtime_error("Unhandled metric type");
    };
  }

  template <typename T>
  static auto getMetricType()
  {
    if constexpr (std::is_same_v<T, int>) {
      return MetricType::Int;
    } else if constexpr (std::is_same_v<T, float>) {
      return MetricType::Float;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return MetricType::Uint64;
    } else {
      throw runtime_error("Unhandled metric type");
    };
  }

  /// @return helper to insert a given value in the metrics
  template <typename T>
  static std::function<void(DeviceMetricsInfo&, T value, size_t timestamp)>
    createNumericMetric(DeviceMetricsInfo& metrics,
                        char const* name,
                        NewMetricCallback newMetricsCallback = nullptr)
  {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, uint64_t> || std::is_same_v<T, float>, "Unsupported metric type");
    // Create a new metric
    MetricInfo metricInfo;
    metricInfo.pos = 0;
    metricInfo.type = getMetricType<T>();
    metricInfo.filledMetrics = 0;
    metricInfo.storeIdx = getMetricsStore<T>(metrics).size();
    getMetricsStore<T>(metrics).emplace_back(std::array<T, 1024>{});

    // Add the timestamp buffer for it
    metrics.timestamps.emplace_back(std::array<size_t, 1024>{});
    metrics.max.push_back(std::numeric_limits<float>::lowest());
    metrics.min.push_back(std::numeric_limits<float>::max());
    metrics.maxDomain.push_back(std::numeric_limits<size_t>::lowest());
    metrics.minDomain.push_back(std::numeric_limits<size_t>::max());
    metrics.changed.push_back(true);

    // Add the index by name in the correct position
    // this will require moving the tail of the index,
    // but inserting should happen only once for each metric,
    // so who cares.
    // Add the the actual Metric info to the store
    MetricLabelIndex metricLabelIdx;
    strncpy(metricLabelIdx.label, name, MetricLabelIndex::MAX_METRIC_LABEL_SIZE - 1);
    metricLabelIdx.label[MetricLabelIndex::MAX_METRIC_LABEL_SIZE - 1] = '\0';
    metricLabelIdx.index = metrics.metrics.size();
    metricLabelIdx.size = strlen(metricLabelIdx.label);
    metrics.metricLabelsIdx.push_back(metricLabelIdx);

    // Add the the actual Metric info to the store
    auto metricIndex = metrics.metrics.size();
    metrics.metrics.push_back(metricInfo);

    if (newMetricsCallback != nullptr) {
      newMetricsCallback(metricLabelIdx.label, metricInfo, 0, metricIndex);
    }
    return [metricIndex](DeviceMetricsInfo& metrics, T value, size_t timestamp) {
      MetricInfo& metric = metrics.metrics[metricIndex];

      metrics.minDomain[metricIndex] = std::min(metrics.minDomain[metricIndex], timestamp);
      metrics.maxDomain[metricIndex] = std::max(metrics.maxDomain[metricIndex], timestamp);
      metrics.max[metricIndex] = std::max(metrics.max[metricIndex], (float)value);
      metrics.min[metricIndex] = std::min(metrics.min[metricIndex], (float)value);
      metrics.changed.at(metricIndex) = true;
      auto& store = getMetricsStore<T>(metrics);
      size_t pos = metric.pos++ % store[metric.storeIdx].size();
      metrics.timestamps[metricIndex][pos] = timestamp;
      store[metric.storeIdx][pos] = value;
      metric.filledMetrics++;
    };
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
