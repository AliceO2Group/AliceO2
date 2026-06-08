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

#ifndef O2_FRAMEWORK_METRICAGGREGATOR_AGGREGATIONPOLICY_H
#define O2_FRAMEWORK_METRICAGGREGATOR_AGGREGATIONPOLICY_H

#include <string>
#include <string_view>
#include <vector>
#include <regex>

namespace o2
{

namespace framework
{

namespace metricaggregator
{
/// Defines the selection strategy for devices.
enum class AggregationSelectionType {
  All,
  Specific
};
/// Defines the reduction strategy for metrics.
enum class AggregationMetricType {
  Sum,
  Average,
  Rate,
  Specific,
  Simple
};

/// Parses environment configurations and evaluates aggregation rules.
class AggregationPolicy
{
 public:
  AggregationPolicy() = default;
  ~AggregationPolicy() = default;
  /// Reads configuration from environment variables and sets internal rules.
  void configureFromEnv();
  /// Returns the configured device selection type.
  AggregationSelectionType getSelection() const noexcept;
  /// Returns the configured global metric reduction type.
  AggregationMetricType getReduction() const noexcept;
  /// Determines the specific reduction type required for a given metric name.
  AggregationMetricType getAggregationTypeForMetric(std::string_view metricName) const;
  /// Evaluates whether the policy allows processing for the provided device name.
  bool selectDevice(std::string_view deviceId) const;

 private:
  /// Maps a regular expression pattern to a specific aggregation type.
  struct MetricRule {
    std::regex metricPattern;
    AggregationMetricType type;
  };

  std::vector<std::string> split(std::string_view input, char delim) const;
  /// Converts a string literal into an AggregationSelectionType enum.
  AggregationSelectionType parseSelectionType(const std::string& str);
  /// Converts a string literal into an AggregationMetricType enum.
  AggregationMetricType parseReductionType(const std::string& str);

  AggregationSelectionType mSelection = AggregationSelectionType::All;
  AggregationMetricType mReduction = AggregationMetricType::Sum;
  std::vector<std::string> mSpecificDevices;
  std::vector<MetricRule> mSpecificMetricRules;
};

} // namespace metricaggregator
} // namespace framework
} // namespace o2

#endif // O2_FRAMEWORK_METRICAGGREGATOR_AGGREGATIONPOLICY_H
