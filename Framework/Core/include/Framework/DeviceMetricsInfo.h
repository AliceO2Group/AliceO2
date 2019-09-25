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
#ifndef FRAMEWORK_DEVICEMETRICSINFO_H
#define FRAMEWORK_DEVICEMETRICSINFO_H

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace o2
{
namespace framework
{

enum class MetricType {
  Int = 0,
  String = 1,
  Float = 2,
  Unknown
};

std::ostream& operator<<(std::ostream& oss, MetricType const& val);

struct MetricInfo {
  enum MetricType type;
  size_t storeIdx; // Index in the actual store
  size_t pos;      // Last position in the circular buffer
};

// We keep only fixed lenght strings for metrics, as in the end this is not
// really needed. They should be nevertheless 0 terminated.
struct StringMetric {
  static constexpr ptrdiff_t MAX_SIZE = 128;
  char data[MAX_SIZE];
};

// Also for the keys it does not make much sense to keep more than 256 chars.
// They should be nevertheless 0 terminated.
struct MetricLabelIndex {
  static constexpr size_t MAX_METRIC_LABEL_SIZE = 256 - sizeof(size_t); // Maximum size for a metric name.
  size_t index;
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
  char const* beginStringValue;
  char const* endStringValue;
};

/// This struct hold information about device metrics when running
/// in standalone mode
struct DeviceMetricsInfo {
  // We keep the size of each metric to 4096 bytes. No need for more
  // for the debug GUI
  std::vector<std::array<int, 1024>> intMetrics;
  std::vector<std::array<StringMetric, 32>> stringMetrics; // We do not keep so many strings as metrics as history is less relevant.
  std::vector<std::array<float, 1024>> floatMetrics;
  std::vector<std::array<size_t, 1024>> timestamps;
  std::vector<float> max;
  std::vector<float> min;
  std::vector<size_t> minDomain;
  std::vector<size_t> maxDomain;
  std::vector<MetricLabelIndex> metricLabelsIdx;
  std::vector<MetricInfo> metrics;
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
  static size_t metricIdxByName(const std::string& name,
                                const DeviceMetricsInfo& info);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DEVICEMETRICSINFO_H
