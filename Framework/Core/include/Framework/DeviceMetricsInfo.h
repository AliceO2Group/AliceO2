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
#include <regex>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

enum class MetricType {
  Int,
  String,
  Float,
  Unknown
};

std::ostream& operator<<(std::ostream& oss, MetricType const& val);

struct MetricInfo {
  enum MetricType type;
  size_t storeIdx; // Index in the actual store
  size_t pos; // Last position in the circular buffer
};

// We keep only fixed lenght strings for metrics, as in the end this is not
// really needed. They should be nevertheless 0 terminated.
struct StringMetric {
  char data[128];
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
  std::vector<std::pair<std::string, size_t>> metricLabelsIdx;
  std::vector<MetricInfo> metrics;
};

struct DeviceMetricsHelper {
  /// Type of the callback which can be provided to be invoked every time a new
  /// metric is found by the system.
  using NewMetricCallback = std::function<void(std::string const&, MetricInfo const&, int value, size_t metricIndex)>;

  /// Helper function to parse a metric string.
  static bool parseMetric(const std::string& s, std::smatch& match);

  /// Processes a parsed metric and stores in the backend store.
  ///
  /// @matches is the regexp_matches from the metric identifying regex
  /// @info is the DeviceInfo associated to the device posting the metric
  /// @newMetricsCallback is a callback that will be invoked every time a new metric is added to the list.
  static bool processMetric(const std::smatch& match,
                            DeviceMetricsInfo& info,
                            NewMetricCallback newMetricCallback = nullptr);
  static size_t metricIdxByName(const std::string& name,
                                const DeviceMetricsInfo& info);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DEVICEMETRICSINFO_H
