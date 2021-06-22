// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ResourcesMonitoringHelper.h"
#include "Framework/DeviceMetricsInfo.h"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <string_view>
#include <algorithm>
#include <cassert>
#include <regex>

using namespace o2::framework;

template <typename T>
inline static T retriveValue(T val)
{
  return val;
}

inline static std::string retriveValue(const std::reference_wrapper<const StringMetric> val)
{
  return std::string(val.get().data);
}

template <typename T>
boost::property_tree::ptree fillNodeWithValue(const DeviceMetricsInfo& deviceMetrics,
                                              const T& metricsStorage, size_t labelIndex, size_t storageIndex)
{

  unsigned int loopRange = std::min(deviceMetrics.metrics[labelIndex].filledMetrics, metricsStorage[storageIndex].size());
  boost::property_tree::ptree metricNode;

  for (unsigned int idx = 0; idx < loopRange; ++idx) {
    boost::property_tree::ptree values;
    values.add("timestamp", deviceMetrics.timestamps[labelIndex][idx]);
    if constexpr (std::is_arithmetic_v<T>) {
      values.add("value", std::to_string(retriveValue(std::cref(metricsStorage[storageIndex][idx]))));
    } else {
      values.add("value", retriveValue(std::cref(metricsStorage[storageIndex][idx])));
    }
    metricNode.push_back(std::make_pair("", values));
  }
  return metricNode;
}

bool ResourcesMonitoringHelper::dumpMetricsToJSON(const std::vector<DeviceMetricsInfo>& metrics,
                                                  const DeviceMetricsInfo& driverMetrics,
                                                  const std::vector<DeviceSpec>& specs,
                                                  std::vector<std::string> const& performanceMetrics) noexcept
{

  assert(metrics.size() == specs.size());

  if (metrics.empty()) {
    return false;
  }

  boost::property_tree::ptree root;
  for (unsigned int idx = 0; idx < metrics.size(); ++idx) {

    const auto& deviceMetrics = metrics[idx];
    boost::property_tree::ptree deviceRoot;

    for (size_t mi = 0; mi < deviceMetrics.metricLabels.size(); mi++) {
      std::string metricLabel = deviceMetrics.metricLabels[mi].label;

      auto same = [metricLabel](std::string const& matcher) -> bool {
        std::regex r{matcher};
        return std::regex_match(metricLabel, r);
      };
      //check if we are interested
      if (std::find_if(std::begin(performanceMetrics), std::end(performanceMetrics), same) == performanceMetrics.end()) {
        continue;
      }

      //if so

      boost::property_tree::ptree metricNode;

      switch (deviceMetrics.metrics[mi].type) {
        case MetricType::Int:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.intMetrics, mi, deviceMetrics.metrics[mi].storeIdx);
          break;

        case MetricType::Float:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.floatMetrics, mi, deviceMetrics.metrics[mi].storeIdx);
          break;

        case MetricType::String:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.stringMetrics, mi, deviceMetrics.metrics[mi].storeIdx);
          break;

        case MetricType::Uint64:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.uint64Metrics, mi, deviceMetrics.metrics[mi].storeIdx);
          break;

        default:
          continue;
      }
      deviceRoot.add_child(metricLabel, metricNode);
    }

    root.add_child(specs[idx].id, deviceRoot);
  }

  boost::property_tree::ptree driverRoot;
  for (size_t mi = 0; mi < driverMetrics.metricLabels.size(); mi++) {
    const char* metricLabel = driverMetrics.metricLabels[mi].label;

    //check if we are interested
    if (std::find(std::begin(performanceMetrics), std::end(performanceMetrics), metricLabel) == std::end(performanceMetrics)) {
      continue;
    }

    //if so

    boost::property_tree::ptree metricNode;

    switch (driverMetrics.metrics[mi].type) {
      case MetricType::Int:
        metricNode = fillNodeWithValue(driverMetrics, driverMetrics.intMetrics,
                                       mi, driverMetrics.metrics[mi].storeIdx);
        break;

      case MetricType::Float:
        metricNode = fillNodeWithValue(driverMetrics, driverMetrics.floatMetrics,
                                       mi, driverMetrics.metrics[mi].storeIdx);
        break;

      case MetricType::String:
        metricNode = fillNodeWithValue(driverMetrics, driverMetrics.stringMetrics,
                                       mi, driverMetrics.metrics[mi].storeIdx);
        break;

      case MetricType::Uint64:
        metricNode = fillNodeWithValue(driverMetrics, driverMetrics.uint64Metrics,
                                       mi, driverMetrics.metrics[mi].storeIdx);
        break;

      default:
        continue;
    }
    driverRoot.add_child(metricLabel, metricNode);
  }

  root.add_child("driver", driverRoot);

  std::ofstream file("performanceMetrics.json", std::ios::out);
  if (file.is_open()) {
    boost::property_tree::json_parser::write_json(file, root);
  } else {
    return false;
  }

  file.close();

  return true;
}
