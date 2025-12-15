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
#include "ResourcesMonitoringHelper.h"
#include "Framework/DeviceMetricsInfo.h"
#include <rapidjson/prettywriter.h>
#include <rapidjson/ostreamwrapper.h>

#include <ostream>
#include <string>
#include <string_view>
#include <algorithm>
#include <cassert>
#include <regex>

using namespace o2::framework;

template <typename T>
void fillNodeWithValue(rapidjson::Writer<rapidjson::OStreamWrapper>& w,
                       size_t filledMetrics,
                       MetricsStorage<T> const& metricsStorage,
                       TimestampsStorage<T> const& timestampsStorage)
{
  unsigned int loopRange = std::min(filledMetrics, metricsStorage.size());

  w.StartArray();
  for (unsigned int idx = 0; idx < loopRange; ++idx) {
    w.StartObject();
    w.Key("timestamp");
    std::string s = std::to_string(timestampsStorage[idx]);
    w.String(s.c_str(), s.size());
    w.Key("value");
    if constexpr (std::is_arithmetic_v<T>) {
      w.String(std::to_string(metricsStorage[idx]).c_str());
    } else {
      w.String(metricsStorage[idx].data);
    }
    w.EndObject();
  }
  w.EndArray();
}

bool ResourcesMonitoringHelper::dumpMetricsToJSON(const std::vector<DeviceMetricsInfo>& metrics,
                                                  const DeviceMetricsInfo& driverMetrics,
                                                  const std::vector<DeviceSpec>& specs,
                                                  std::vector<std::regex> const& performanceMetricsRegex,
                                                  std::ostream& out) noexcept
{
  assert(metrics.size() == specs.size());

  if (metrics.empty()) {
    return false;
  }

  rapidjson::OStreamWrapper osw(out);
  rapidjson::PrettyWriter<rapidjson::OStreamWrapper> w(osw);

  // Top level obejct for all the metrics
  w.StartObject();

  for (unsigned int idx = 0; idx < metrics.size(); ++idx) {
    w.Key(specs[idx].id.c_str());
    const auto& deviceMetrics = metrics[idx];

    w.StartObject();
    for (size_t mi = 0; mi < deviceMetrics.metricLabels.size(); mi++) {
      std::string_view metricLabel{deviceMetrics.metricLabels[mi].label, deviceMetrics.metricLabels[mi].size};

      auto same = [metricLabel](std::regex const& matcher) -> bool {
        return std::regex_match(metricLabel.begin(), metricLabel.end(), matcher);
      };
      // check if we are interested
      if (std::find_if(std::begin(performanceMetricsRegex), std::end(performanceMetricsRegex), same) == performanceMetricsRegex.end()) {
        continue;
      }
      auto storeIdx = deviceMetrics.metrics[mi].storeIdx;

      size_t filledMetrics = deviceMetrics.metrics[mi].filledMetrics;
      if (deviceMetrics.metrics[mi].filledMetrics == 0) {
        continue;
      }
      w.Key(metricLabel.data(), metricLabel.size());
      switch (deviceMetrics.metrics[mi].type) {
        case MetricType::Int:
          fillNodeWithValue(w, filledMetrics, deviceMetrics.intMetrics[storeIdx],
                            deviceMetrics.intTimestamps[storeIdx]);
          break;

        case MetricType::Float:
          fillNodeWithValue(w, filledMetrics, deviceMetrics.floatMetrics[storeIdx],
                            deviceMetrics.floatTimestamps[storeIdx]);
          break;

        case MetricType::String:
          fillNodeWithValue(w, filledMetrics, deviceMetrics.stringMetrics[storeIdx],
                            deviceMetrics.stringTimestamps[storeIdx]);
          break;

        case MetricType::Uint64:
          fillNodeWithValue(w, filledMetrics, deviceMetrics.uint64Metrics[storeIdx],
                            deviceMetrics.uint64Timestamps[storeIdx]);
          break;

        default:
          continue;
      }
    }

    w.EndObject();
  }

  w.Key("driver");
  w.StartObject();
  for (size_t mi = 0; mi < driverMetrics.metricLabels.size(); mi++) {
    std::string_view const metricLabel{driverMetrics.metricLabels[mi].label, driverMetrics.metricLabels[mi].size};
    auto same = [metricLabel](std::regex const& matcher) -> bool {
      return std::regex_match(metricLabel.begin(), metricLabel.end(), matcher);
    };

    // check if we are interested
    if (std::find_if(std::begin(performanceMetricsRegex), std::end(performanceMetricsRegex), same) == performanceMetricsRegex.end()) {
      continue;
    }

    auto storeIdx = driverMetrics.metrics[mi].storeIdx;
    // and if data is there
    size_t filledMetrics = driverMetrics.metrics[mi].filledMetrics;
    if (filledMetrics == 0) {
      continue;
    }

    w.Key(metricLabel.data(), metricLabel.size());
    switch (driverMetrics.metrics[mi].type) {
      case MetricType::Int:
        fillNodeWithValue(w, filledMetrics, driverMetrics.intMetrics[storeIdx],
                          driverMetrics.intTimestamps[storeIdx]);
        break;

      case MetricType::Float:
        fillNodeWithValue(w, filledMetrics, driverMetrics.floatMetrics[storeIdx],
                          driverMetrics.floatTimestamps[storeIdx]);
        break;

      case MetricType::String:
        fillNodeWithValue(w, filledMetrics, driverMetrics.stringMetrics[storeIdx],
                          driverMetrics.stringTimestamps[storeIdx]);
        break;

      case MetricType::Uint64:
        fillNodeWithValue(w, filledMetrics, driverMetrics.uint64Metrics[storeIdx],
                          driverMetrics.uint64Timestamps[storeIdx]);
        break;

      default:
        continue;
    }
  }
  w.EndObject();
  w.EndObject();

  return true;
}
