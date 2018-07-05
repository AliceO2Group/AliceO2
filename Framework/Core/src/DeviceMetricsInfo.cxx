// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DeviceMetricsInfo.h"
#include <cassert>
#include <cinttypes>
#include <cstdlib>

#include <algorithm>
#include <regex>
#include <tuple>
#include <iostream>

namespace o2
{
namespace framework
{

// Parses a metric in the form
//
// [METRIC] <name>,<type> <value> <timestamp> [<tags>]
bool DeviceMetricsHelper::parseMetric(const std::string& s, std::smatch& match)
{
  const static std::regex metricsRE(R"regex(\[METRIC\] ([a-zA-Z0-9/_-]+),(0|1|2|4) ([0-9.]+) ([0-9]+))regex", std::regex::optimize);
  return std::regex_search(s, match, metricsRE);
}

bool DeviceMetricsHelper::processMetric(const std::smatch& match,
                                        DeviceMetricsInfo& info,
                                        DeviceMetricsHelper::NewMetricCallback newMetricsCallback)
{
  auto type = match[2];
  auto name = match[1];
  char *ep = nullptr;
  auto timestamp = strtol(match[4].str().c_str(), &ep, 10);
  if (ep == nullptr || *ep != '\0') {
    return false;
  }
  auto stringValue = match[3];
  size_t metricIndex = -1;

  auto metricType = MetricType::Unknown;
  if (type.str() == "0") {
    metricType = MetricType::Int;
  } else if (type.str() == "2") {
    metricType = MetricType::Float;
  }

  int intValue = 0;
  float floatValue = 0;
  switch (metricType) {
    case MetricType::Int:
      intValue = strtol(stringValue.str().c_str(), &ep, 10);
      if (!ep || *ep != '\0') {
        return false;
      }
      break;
    case MetricType::Float:
      floatValue = strtof(stringValue.str().c_str(), &ep);
      if (!ep || *ep != '\0') {
        return false;
      }
      break;
    default:
      return false;
      break;
  };

  // Find the metric based on the label. Create it if not found.
  using IndexElement = std::pair<std::string, size_t>;
  auto cmpFn = [](const IndexElement &a, const IndexElement &b) -> bool {
    return std::tie(a.first, a.second) < std::tie(b.first, b.second);
  };
  IndexElement metricLabelIdx = std::make_pair(name.str(), 0);
  auto mi = std::lower_bound(info.metricLabelsIdx.begin(),
                             info.metricLabelsIdx.end(),
                             metricLabelIdx,
                             cmpFn);

  // We could not find the metric, lets insert a new one.
  if (mi == info.metricLabelsIdx.end()
      || mi->first != metricLabelIdx.first) {
    // Create a new metric
    MetricInfo metricInfo;
    metricInfo.pos = 0;
    metricInfo.type = metricType;
    // Add a new empty buffer for it of the correct kind
    switch(metricType) {
      case MetricType::Int:
        metricInfo.storeIdx = info.intMetrics.size();
        info.intMetrics.emplace_back(std::array<int, 1024>{});
        break;
      case MetricType::Float:
        metricInfo.storeIdx = info.floatMetrics.size();
        info.floatMetrics.emplace_back(std::array<float, 1024>{});
        break;
      default:
        return false;
    };
    // Add the timestamp buffer for it
    info.timestamps.emplace_back(std::array<size_t, 1024>{});
    info.max.push_back(std::numeric_limits<float>::lowest());
    info.min.push_back(std::numeric_limits<float>::max());

    // Add the index by name in the correct position
    // this will require moving the tail of the index,
    // but inserting should happen only once for each metric,
    // so who cares.
    metricLabelIdx.second = info.metrics.size();
    info.metricLabelsIdx.insert(mi, metricLabelIdx);
    // Add the the actual Metric info to the store
    metricIndex = info.metrics.size();
    assert(metricInfo.storeIdx != -1);
    assert(metricLabelIdx.first.empty() == false);
    if (newMetricsCallback != nullptr) {
      newMetricsCallback(metricLabelIdx.first, metricInfo, intValue);
    }
    info.metrics.push_back(metricInfo);
  } else {
    metricIndex = mi->second;
  }
  assert(metricIndex != -1);
  // We are now guaranteed our metric is present at metricIndex.
  MetricInfo &metricInfo = info.metrics[metricIndex];

  auto mod = info.timestamps[metricIndex].size();

  switch(metricInfo.type) {
    case MetricType::Int: {
      intValue = strtol(stringValue.str().c_str(), &ep, 10);
      if (!ep || *ep != '\0') {
        return false;
      }
      info.intMetrics[metricInfo.storeIdx][metricInfo.pos] = intValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], (float)intValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], (float)intValue);
    } break;
    case MetricType::Float: {
      floatValue = strtof(stringValue.str().c_str(), &ep);
      if (!ep || *ep != '\0') {
        return false;
      }
      info.floatMetrics[metricInfo.storeIdx][metricInfo.pos] = floatValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], floatValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], floatValue);
    } break;
    default:
      return false;
      break;
  };

  // Save the timestamp for the current metric we do it here
  // so that we do not update timestamps for broken metrics
  info.timestamps[metricIndex][metricInfo.pos] = timestamp;
  // Update the position where to write the next metric
  metricInfo.pos = (metricInfo.pos + 1) % mod;
  return true;
}

/// @return the index in metrics for the information of given metric
size_t
DeviceMetricsHelper::metricIdxByName(const std::string& name, const DeviceMetricsInfo& info)
{
  size_t i = 0;
  while (i < info.metricLabelsIdx.size()) {
    auto &metricName = info.metricLabelsIdx[i];
    if (metricName.first == name) {
      return metricName.second;
    }
    ++i;
  }
  return i;
}

std::ostream& operator<<(std::ostream& oss, MetricType const& val)
{
  switch (val) {
    case MetricType::Float:
      oss << "float";
      break;
    case MetricType::Int:
      oss << "float";
      break;
    case MetricType::Unknown:
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

} // namespace framework
} // namespace o2
