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
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <string_view>
#include <algorithm>
#include <cassert>

using namespace o2::framework;

bool ResourcesMonitoringHelper::dumpMetricsToJSON(const std::vector<DeviceMetricsInfo>& metrics,
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

    for (const auto& metricLabel : deviceMetrics.metricLabelsIdx) {

      //check if we are interested
      if (std::find(std::begin(performanceMetrics), std::end(performanceMetrics), metricLabel.label) == std::end(performanceMetrics)) {
        continue;
      }

      //if so

      boost::property_tree::ptree metricNode;

      switch (deviceMetrics.metrics[metricLabel.index].type) {
        case MetricType::Int:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.intMetrics,
                                         metricLabel.index, deviceMetrics.metrics[metricLabel.index].storeIdx);
          break;

        case MetricType::Float:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.floatMetrics,
                                         metricLabel.index, deviceMetrics.metrics[metricLabel.index].storeIdx);
          break;

        case MetricType::String:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.stringMetrics,
                                         metricLabel.index, deviceMetrics.metrics[metricLabel.index].storeIdx);
          break;

        case MetricType::Uint64:
          metricNode = fillNodeWithValue(deviceMetrics, deviceMetrics.uint64Metrics,
                                         metricLabel.index, deviceMetrics.metrics[metricLabel.index].storeIdx);
          break;

        default:
          continue;
      }
      deviceRoot.add_child(metricLabel.label, metricNode);
    }

    root.add_child(specs[idx].name, deviceRoot);
  }

  std::ofstream file("performanceMetrics.json", std::ios::out);
  if (file.is_open()) {
    boost::property_tree::json_parser::write_json(file, root);
  } else {
    return false;
  }

  file.close();

  return true;
}
