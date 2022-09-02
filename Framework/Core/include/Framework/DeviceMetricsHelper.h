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
#ifndef O2_FRAMEWORK_DEVICEMETRICSHELPERS_H_
#define O2_FRAMEWORK_DEVICEMETRICSHELPERS_H_

#include "Framework/DeviceMetricsInfo.h"
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
struct DriverInfo;

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
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return metrics.enumMetrics;
    } else {
      throw runtime_error("Unhandled metric type");
    };
  }

  /// Typesafe way to get the actual store
  template <typename T>
  static auto& getTimestampsStore(DeviceMetricsInfo& metrics)
  {
    if constexpr (std::is_same_v<T, int>) {
      return metrics.intTimestamps;
    } else if constexpr (std::is_same_v<T, float>) {
      return metrics.floatTimestamps;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return metrics.uint64Timestamps;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return metrics.enumTimestamps;
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
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return MetricType::Enum;
    } else {
      throw runtime_error("Unhandled metric type");
    };
  }

  template <typename T>
  static auto getNumericMetricCursor(size_t metricIndex)
  {
    return [metricIndex](DeviceMetricsInfo& metrics, T value, size_t timestamp) {
      MetricInfo& metric = metrics.metrics[metricIndex];

      metrics.minDomain[metricIndex] = std::min(metrics.minDomain[metricIndex], timestamp);
      metrics.maxDomain[metricIndex] = std::max(metrics.maxDomain[metricIndex], timestamp);
      metrics.max[metricIndex] = std::max(metrics.max[metricIndex], (float)value);
      metrics.min[metricIndex] = std::min(metrics.min[metricIndex], (float)value);
      metrics.changed.at(metricIndex) = true;
      auto& store = getMetricsStore<T>(metrics);
      auto& timestamps = getTimestampsStore<T>(metrics);
      size_t pos = metric.pos++ % store[metric.storeIdx].size();
      timestamps[metric.storeIdx][pos] = timestamp;
      store[metric.storeIdx][pos] = value;
      metric.filledMetrics++;
    };
  }

  static size_t bookMetricInfo(DeviceMetricsInfo& metrics, char const* name, MetricType type);

  /// @return helper to insert a given value in the metrics
  template <typename T>
  static size_t
    bookNumericMetric(DeviceMetricsInfo& metrics,
                      char const* name,
                      NewMetricCallback newMetricsCallback = nullptr)
  {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, uint64_t> || std::is_same_v<T, float>, "Unsupported metric type");
    size_t metricIndex = bookMetricInfo(metrics, name, getMetricType<T>());
    auto& metricInfo = metrics.metrics[metricIndex];
    if (newMetricsCallback != nullptr) {
      newMetricsCallback(name, metricInfo, 0, metricIndex);
    }
    return metricIndex;
  }

  /// @return helper to insert a given value in the metrics
  template <typename T>
  static std::function<void(DeviceMetricsInfo&, T value, size_t timestamp)>
    createNumericMetric(DeviceMetricsInfo& metrics,
                        char const* name,
                        NewMetricCallback newMetricsCallback = nullptr)
  {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, uint64_t> || std::is_same_v<T, float>, "Unsupported metric type");
    size_t metricIndex = bookNumericMetric<T>(metrics, name, newMetricsCallback);
    return getNumericMetricCursor<T>(metricIndex);
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
