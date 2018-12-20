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
// [METRIC] <name>,<type> <value> <timestamp> [<tag>,<tag>]
bool DeviceMetricsHelper::parseMetric(const std::string& s, std::vector<std::pair<char const*, char const*>>& match)
{
  match.clear();
  /// Must start with "[METRIC] "
  ///                  012345678
  constexpr size_t PREFIXSIZE = 9;
  if (s.size() < PREFIXSIZE) {
    return false;
  }
  if (memcmp(s.data(), "[METRIC] ", 9) != 0) {
    return false;
  }
  char const* comma;           // first comma
  char const* spaces[256];     // list of spaces
  char const** space = spaces; // first element to fill

  comma = (char const*)memchr((void*)s.data(), ',', s.size());
  if (comma == nullptr) {
    return false;
  }
  // Find all spaces
  char const* nextSpace = s.data();
  while (space - spaces < 256) {
    *space = strchr(nextSpace, ' ');
    if (*space == nullptr) {
      break;
    }
    nextSpace = *space + 1;
    space++;
  }
  // First space should always come before comma
  if (spaces[0] > comma) {
    return false;
  }
  match.emplace_back(std::make_pair(spaces[0] + 1, comma));
  // type is alway 1 char after the comma, followed by space
  if ((spaces[1] - comma) != 2) {
    return false;
  }
  match.emplace_back(std::make_pair(comma + 1, spaces[1]));
  // We need at least 4 spaces
  if (space - spaces < 4) {
    return false;
  }
  // Value is between the second and the last but one space
  match.emplace_back(std::make_pair(spaces[1] + 1, *(space - 2)));
  // Timestamp is between the last but one and the last space
  match.emplace_back(std::make_pair(*(space - 2) + 1, *(space - 1)));
  return true;
}

bool DeviceMetricsHelper::processMetric(std::vector<std::pair<char const*, char const*>> const& match,
                                        DeviceMetricsInfo& info,
                                        DeviceMetricsHelper::NewMetricCallback newMetricsCallback)
{
  // get the type
  char *ep = nullptr;
  auto type = strtol(match[1].first, &ep, 10);
  if (ep == nullptr || *ep != ' ') {
    return false;
  }
  auto metricType = static_cast<MetricType>(type);
  // get the timestamp
  auto timestamp = strtol(match[3].first, &ep, 10);
  if (ep == nullptr || *ep != ' ') {
    return false;
  }
  std::string const name(match[0].first, match[0].second);
  size_t metricIndex = -1;

  int intValue = 0;
  StringMetric stringValue;
  float floatValue = 0;
  switch (metricType) {
    case MetricType::Int:
      intValue = strtol(match[2].first, &ep, 10);
      if (!ep || *ep != ' ') {
        return false;
      }
      break;
    case MetricType::String:
      strncpy(stringValue.data, match[2].first, match[2].second - match[2].first);
      stringValue.data[std::min(match[2].second - match[2].first, StringMetric::MAX_SIZE - 1)] = '\0';
      break;
    case MetricType::Float:
      floatValue = strtof(match[2].first, &ep);
      if (!ep || *ep != ' ') {
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
  IndexElement metricLabelIdx = std::make_pair(name, 0);
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
      case MetricType::String:
        metricInfo.storeIdx = info.stringMetrics.size();
        info.stringMetrics.emplace_back(std::array<StringMetric, 32>{});
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
    info.maxDomain.push_back(std::numeric_limits<size_t>::lowest());
    info.minDomain.push_back(std::numeric_limits<size_t>::max());

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
      newMetricsCallback(metricLabelIdx.first, metricInfo, intValue, metricIndex);
    }
    info.metrics.push_back(metricInfo);
  } else {
    metricIndex = mi->second;
  }
  assert(metricIndex != -1);
  // We are now guaranteed our metric is present at metricIndex.
  MetricInfo &metricInfo = info.metrics[metricIndex];

  //  auto mod = info.timestamps[metricIndex].size();
  info.minDomain[metricIndex] = std::min(info.minDomain[metricIndex], (size_t)timestamp);
  info.maxDomain[metricIndex] = std::max(info.maxDomain[metricIndex], (size_t)timestamp);

  switch(metricInfo.type) {
    case MetricType::Int: {
      info.intMetrics[metricInfo.storeIdx][metricInfo.pos] = intValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], (float)intValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], (float)intValue);
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = timestamp;
      // Update the position where to write the next metric
      metricInfo.pos = (metricInfo.pos + 1) % info.intMetrics[metricInfo.storeIdx].size();
    } break;
    case MetricType::String: {
      info.stringMetrics[metricInfo.storeIdx][metricInfo.pos] = stringValue;
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = timestamp;
      metricInfo.pos = (metricInfo.pos + 1) % info.stringMetrics[metricInfo.storeIdx].size();
    } break;
    case MetricType::Float: {
      info.floatMetrics[metricInfo.storeIdx][metricInfo.pos] = floatValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], floatValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], floatValue);
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = timestamp;
      metricInfo.pos = (metricInfo.pos + 1) % info.floatMetrics[metricInfo.storeIdx].size();
    } break;
    default:
      return false;
      break;
  };

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
    case MetricType::String:
      oss << "string";
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
