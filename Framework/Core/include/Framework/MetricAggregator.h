// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_METRICAGGREGATOR_METRICAGGREGATOR_H
#define O2_FRAMEWORK_METRICAGGREGATOR_METRICAGGREGATOR_H

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceMetricsInfo.h"
#include "Framework/Monitoring.h"

#include <fairmq/ProgOptions.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "Framework/AggregationPolicy.h"

namespace o2
{

namespace framework
{

namespace metricaggregator
{
/// Stores a single numeric measurement and its associated timestamp.
struct MetricSample {
  double value = 0.0;
  std::size_t timestamp = 0;
};

/// Collects and reduces metrics across multiple framework devices.
/// Transmits the aggregated results to an external monitoring backend.
class MetricAggregator
{
 public:
  explicit MetricAggregator();
  ~MetricAggregator() = default;
  /// Initializes the internal aggregation policy from environment variables.
  void setPolicy();
  /// Returns the current policy configuration as a formatted string.
  std::string getPolicy();
  /// Routes metrics to the appropriate processing function based on the policy reduction type.
  void mergeMetrics(const std::vector<DeviceMetricsInfo>& metrics,
                    const DeviceMetricsInfo& driverMetrics,
                    const std::vector<DeviceSpec>& specs);

 private:
  /// Appends a suffix to the metric name based on the applied aggregation type.
  std::string getMetricNameFromPolicy(std::string_view metricName, AggregationMetricType aggregationType);
  /// Flushes metrics directly without aggregation.
  void flushMetricsSimple(const std::vector<DeviceMetricsInfo>& deviceMetrics,
                          const DeviceMetricsInfo& driverMetrics,
                          const std::vector<DeviceSpec>& specs);
  /// Flushes metrics by applying the aggregation policy.
  void flushMetrics(const std::vector<DeviceMetricsInfo>& deviceMetrics,
                    const DeviceMetricsInfo& driverMetrics,
                    const std::vector<DeviceSpec>& specs);
  /// Retrieves the monitoring backend type from environment variables.
  const char* getBackendFromEnv();

  const char* mBackend = nullptr;
  std::unique_ptr<o2::monitoring::Monitoring> mMonitoring;
  /// Stores the previous samples required to compute rates over time.
  std::unordered_map<std::string, std::vector<MetricSample>> mLastSentSamples;
  std::unique_ptr<AggregationPolicy> mPolicy;
};
} // namespace metricaggregator
} // namespace framework
} // namespace o2

#endif // O2_FRAMEWORK_METRICAGGREGATOR_METRICAGGREGATOR_H
