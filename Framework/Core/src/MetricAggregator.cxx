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

#include "Framework/MetricAggregator.h"

#include "Framework/CommonServices.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/Logger.h"
#include "Framework/Monitoring.h"
#include "Framework/Plugins.h"
#include "Framework/ServiceSpec.h"
#include "Framework/TypeIdHelpers.h"

#include <Monitoring/MonitoringFactory.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace o2::framework;
using Metric = o2::monitoring::Metric;
#define MONITORING_QUEUE_SIZE 100

namespace o2::framework::metricaggregator
{
/// Reads the most recent value from the ring buffer based on the write position and the history offset.
template <typename ValueT>
bool getNumericValue(std::size_t storeIdx,
                                   std::size_t writePos,
                                   std::size_t filledMetrics,
                                   std::size_t historyOffset,
                                   std::vector<MetricsStorage<ValueT>> const& valuesStorage,
                                   std::vector<TimestampsStorage<ValueT>> const& timestampsStorage,
                                   MetricSample& out)
{
  if (storeIdx >= valuesStorage.size() || storeIdx >= timestampsStorage.size()) {
    return false;
  }

  auto const& values = valuesStorage[storeIdx];
  auto const& timestamps = timestampsStorage[storeIdx];
  auto const capacity = values.size();
  if (capacity == 0 || capacity <= historyOffset) {
    return false;
  }

  auto const ringIndex = (writePos + capacity - 1 - historyOffset) % capacity;
  out.value = static_cast<double>(values[ringIndex]);
  out.timestamp = timestamps[ringIndex];
  return true;
}

/// Routes the metric extraction request to the correct typed storage array.
bool extractNumericSample(DeviceMetricsInfo const& info,
                         std::size_t metricIndex,
                         std::size_t historyOffset,
                         MetricSample& out)
{
  if (metricIndex >= info.metrics.size()) {
    return false;
  }

  auto const& metric = info.metrics[metricIndex];
  switch (metric.type) {
    case MetricType::Int: {
      return getNumericValue(metric.storeIdx,
                                           metric.pos,
                                           metric.filledMetrics,
                                           historyOffset,
                                           info.intMetrics,
                                           info.intTimestamps,
                                           out);
    }
    case MetricType::Uint64: {
      return getNumericValue(metric.storeIdx,
                                           metric.pos,
                                           metric.filledMetrics,
                                           historyOffset,
                                           info.uint64Metrics,
                                           info.uint64Timestamps,
                                           out);
    }
    case MetricType::Float: {
      return getNumericValue(metric.storeIdx,
                                           metric.pos,
                                           metric.filledMetrics,
                                           historyOffset,
                                           info.floatMetrics,
                                           info.floatTimestamps,
                                           out);
    }
    case MetricType::Enum: {
      return getNumericValue(metric.storeIdx,
                                           metric.pos,
                                           metric.filledMetrics,
                                           historyOffset,
                                           info.enumMetrics,
                                           info.enumTimestamps,
                                           out);
    }
    case MetricType::String:
    case MetricType::Unknown:
      return false;
  }
  return false;
}

bool findMetricIndexByName(DeviceMetricsInfo const& info,
                           std::string_view metricName,
                           std::size_t& metricIndex)
{
  auto const nMetrics = std::min(info.metrics.size(), info.metricLabels.size());
  for (std::size_t i = 0; i < nMetrics; ++i) {
    auto const& label = info.metricLabels[i];
    std::string_view currentName{label.label, label.size};
    if (currentName == metricName) {
      metricIndex = i;
      return true;
    }
  }
  return false;
}

MetricAggregator::MetricAggregator()
{
  mBackend = getBackendFromEnv();
  if (mBackend == nullptr) {
    LOGP(error, "[MetricAggregator] No backend configured, skipping initialization");
    return;
  }

  mMonitoring = o2::monitoring::MonitoringFactory::Get(mBackend);
  if (mMonitoring == nullptr) {
    LOGP(error, "[MetricAggregator] Failed to create monitoring backend for '{}'", mBackend);
    return;
  }
  mMonitoring->enableBuffering(MONITORING_QUEUE_SIZE);
  setPolicy();
  LOGP(info, "[MetricAggregator] Initialized with policy '{}'", getPolicy());
}

void MetricAggregator::setPolicy()
{
  mPolicy = std::make_unique<AggregationPolicy>();
  mPolicy->configureFromEnv();
}

std::string MetricAggregator::getPolicy()
{
  std::stringstream ss;
  std::string selectionStr;
  switch (mPolicy->getSelection()) {
    case AggregationSelectionType::All:
      selectionStr = "All";
      break;
    case AggregationSelectionType::Specific:
      selectionStr = "Specific";
      break;
  }
  std::string reductionStr;
  switch (mPolicy->getReduction()) {
    case AggregationMetricType::Sum:
      reductionStr = "Sum";
      break;
    case AggregationMetricType::Average:
      reductionStr = "Average";
      break;
    case AggregationMetricType::Rate:
      reductionStr = "Rate";
      break;
    case AggregationMetricType::Simple:
      reductionStr = "Simple";
      break;
    case AggregationMetricType::Specific:
      reductionStr = "Specific";
      break;
  }
  ss << selectionStr << ":" << reductionStr;
  return ss.str();
}

void MetricAggregator::mergeMetrics(const std::vector<DeviceMetricsInfo>& metrics,
                                    const DeviceMetricsInfo& driverMetrics,
                                    const std::vector<DeviceSpec>& specs)
{
  if (mPolicy->getReduction() == AggregationMetricType::Simple) {
      flushMetricsSimple(metrics, driverMetrics, specs);
    } else {
      flushMetrics(metrics, driverMetrics, specs);
    }
}

std::string MetricAggregator::getMetricNameFromPolicy(std::string_view metricName, AggregationMetricType aggregationType)
{
  std::string metricNameStr{metricName};
  if (aggregationType == AggregationMetricType::Sum) {
    metricNameStr += "_sum";
  } else if (aggregationType == AggregationMetricType::Rate) {
    metricNameStr += "_R";
  } else if (aggregationType == AggregationMetricType::Average) {
    metricNameStr += "_avg";
  }
  return metricNameStr;
}

void MetricAggregator::flushMetricsSimple(const std::vector<DeviceMetricsInfo>& deviceMetricsInfo,
                                    const DeviceMetricsInfo& driverMetrics,
                                    const std::vector<DeviceSpec>& specs)
{
  auto const nDevices = std::min(deviceMetricsInfo.size(), specs.size());
  if (nDevices == 0) {
    return;
  }

  for (std::size_t di = 0; di < nDevices; ++di) {
    auto const& deviceId = specs[di].id;

    if (mPolicy && !mPolicy->selectDevice(deviceId)) {
      continue;
    }

    auto const& deviceMetrics = deviceMetricsInfo[di];
    auto const nMetrics = std::min({deviceMetrics.metrics.size(),
                                    deviceMetrics.metricLabels.size(),
                                    deviceMetrics.changed.size()});

    auto monitoring = o2::monitoring::MonitoringFactory::Get(mBackend);
    if (monitoring == nullptr) {
      LOGP(error, "[MetricAggregator] Failed to create monitoring backend for '{}'", mBackend);
      return;
    }
    monitoring->enableBuffering(MONITORING_QUEUE_SIZE);
    monitoring->addGlobalTag("pipeline_id", std::to_string(specs[di].inputTimesliceId));
    monitoring->addGlobalTag("dataprocessor_name", specs[di].name);
    // FIXME: dpl_instance missing

    for (std::size_t mi = 0; mi < nMetrics; ++mi) {
      MetricSample sample;
      if (!extractNumericSample(deviceMetrics, mi, 0, sample)) {
        continue;
      }

      auto const metricName = std::string(deviceMetrics.metricLabels[mi].label, deviceMetrics.metricLabels[mi].size);
      auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(std::chrono::milliseconds(sample.timestamp));
      auto metric = o2::monitoring::Metric{metricName, o2::monitoring::Metric::DefaultVerbosity, tp};
      metric.addValue(sample.value, "value");
      monitoring->send(std::move(metric));
    }
    monitoring->flushBuffer();
  }
}

void MetricAggregator::flushMetrics(const std::vector<DeviceMetricsInfo>& deviceMetrics,
                                    const DeviceMetricsInfo& driverMetrics,
                                    const std::vector<DeviceSpec>& specs)
{
  auto const nDevices = std::min(deviceMetrics.size(), specs.size());
  if (nDevices == 0) {
    mMonitoring->flushBuffer();
    return;
  }

  // Collect all unique metric names across devices to determine which metrics to aggregate.
  std::unordered_set<std::string_view> allMetricNames;
  for (const auto& deviceMetricsInfo : deviceMetrics) {
    auto const nMetrics = std::min(deviceMetricsInfo.metrics.size(), deviceMetricsInfo.metricLabels.size());
    for (std::size_t i = 0; i < nMetrics; ++i) {
      allMetricNames.insert(std::string_view(deviceMetricsInfo.metricLabels[i].label, deviceMetricsInfo.metricLabels[i].size));
    }
  }

  for (const auto& metricName : allMetricNames) {
    try {
      auto const metricAggregationType = mPolicy->getAggregationTypeForMetric(metricName);
      if (metricAggregationType == AggregationMetricType::Simple) {
        continue;
      }

      // Gather the latest metric values from each valid device
      std::vector<MetricSample> deviceSamples;
      deviceSamples.reserve(nDevices);
      auto metricTimestamp = std::numeric_limits<std::size_t>::max();
      for (std::size_t di = 0; di < nDevices; ++di) {
        auto const& deviceId = specs[di].id;
        if (mPolicy && !mPolicy->selectDevice(deviceId)) {
          continue;
        }
        auto const& deviceMetricsInfo = deviceMetrics[di];
        std::size_t deviceMetricIndex = 0;
        if (!findMetricIndexByName(deviceMetricsInfo, metricName, deviceMetricIndex)) {
          continue;
        }

        MetricSample latest;
        if (!extractNumericSample(deviceMetricsInfo, deviceMetricIndex, 0, latest) || latest.timestamp == 0) {
          continue;
        }
        deviceSamples.push_back(latest);
        metricTimestamp = std::min(metricTimestamp, latest.timestamp);
      }

      if (deviceSamples.empty()) {
        continue;
      }

      const auto handlers = std::unordered_map<AggregationMetricType,
        std::function<double(const std::vector<MetricSample>&)>>{
        {AggregationMetricType::Sum, [](const auto& windows) {
          double sum = 0.0;
          for (const auto& window : windows) {
            sum += window.value;
          }
          return sum;
        }},
        {AggregationMetricType::Average, [](const auto& windows) {
          double sum = 0.0;
          std::size_t count = 0;
          for (const auto& window : windows) {
            sum += window.value;
            ++count;
          }
          return windows.empty() ? 0.0 : sum / count;
        }},
        {AggregationMetricType::Rate, [this, &metricName](const auto& windows) {
          double sumRate = 0.0;
          auto const& previousSamples = mLastSentSamples[std::string{metricName}];
          if (mLastSentSamples.empty() || previousSamples.size() != windows.size()) {
            return 0.0;
          }
          for (const auto& window : windows) {
            double deltaValue = window.value - previousSamples[&window - &windows[0]].value;
            double deltaTimeSec = (window.timestamp - previousSamples[&window - &windows[0]].timestamp) / 1000.0;
            if (deltaTimeSec <= 0) {
              continue;
            }
            sumRate += deltaValue / deltaTimeSec;
          }
          return sumRate;
        }}
      };

      double reducedValue = 0.0;
      auto handlerIt = handlers.find(metricAggregationType);
      if (handlerIt == handlers.end()) {
        LOGP(error, "[MetricAggregator] No handler found for aggregation type '{}'", static_cast<int>(metricAggregationType));
        continue;
      }
      reducedValue = handlerIt->second(deviceSamples);
      if (reducedValue < 0) {
        continue;
      }

      auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(std::chrono::milliseconds(metricTimestamp));
      std::string metricNameWithPolicy = getMetricNameFromPolicy(metricName, metricAggregationType);
      auto metric = Metric{metricNameWithPolicy, Metric::DefaultVerbosity, tp};
      metric.addValue(reducedValue, "value");
      mMonitoring->send(std::move(metric));
      mLastSentSamples[std::string{metricName}] = deviceSamples;
    } catch (const std::exception& e) {
      LOGP(error, "[MetricAggregator] Error determining aggregation type: {}",e.what());
      continue;
    }
  }
  mMonitoring->flushBuffer();
}

const char* MetricAggregator::getBackendFromEnv()
{
  const char* envBackend = std::getenv("APMON_CONFIG");
  if (envBackend == nullptr) {
    LOGP(error, "[MetricAggregator] APMON_CONFIG environment variable is not set");
    return nullptr;
  }

  static std::string result = std::string("apmon://") + envBackend;
  return result.c_str();
}

} // namespace o2::framework::metricaggregator
