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
#include <string_view>
#include <tuple>
#include <iostream>

namespace o2
{
namespace framework
{

// Parses a metric in the form
//
// [METRIC] <name>,<type> <value> <timestamp> [<tag>,<tag>]
bool DeviceMetricsHelper::parseMetric(std::string_view const s, ParsedMetricMatch& match)
{
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
  match.beginKey = spaces[0] + 1;
  match.endKey = comma;
  // type is alway 1 char after the comma, followed by space
  if ((spaces[1] - comma) != 2) {
    return false;
  }
  char* ep = nullptr;
  match.type = static_cast<MetricType>(strtol(comma + 1, &ep, 10));
  if (ep != spaces[1]) {
    return false;
  }
  // We need at least 4 spaces
  if (space - spaces < 4) {
    return false;
  }
  // Value is between the second and the last but one space
  switch (match.type) {
    case MetricType::Int:
      match.intValue = strtol(spaces[1] + 1, &ep, 10);
      if (ep != *(space - 2)) {
        return false;
      }
      break;
    case MetricType::Float:
      match.floatValue = strtof(spaces[1] + 1, &ep);
      if (ep != *(space - 2)) {
        return false;
      }
      break;
    case MetricType::String:
      match.beginStringValue = spaces[1] + 1;
      match.endStringValue = *(space - 2);
      break;
    default:
      return false;
  }
  // Timestamp is between the last but one and the last space
  match.timestamp = strtol(*(space - 2) + 1, &ep, 10);
  if (ep != *(space - 1)) {
    return false;
  }
  return true;
}

bool DeviceMetricsHelper::processMetric(ParsedMetricMatch& match,
                                        DeviceMetricsInfo& info,
                                        DeviceMetricsHelper::NewMetricCallback newMetricsCallback)
{
  // get the type
  size_t metricIndex = -1;

  StringMetric stringValue;
  switch (match.type) {
    case MetricType::Float:
    case MetricType::Int:
      break;
    case MetricType::String: {
      auto lastChar = std::min(match.endStringValue - match.beginStringValue, StringMetric::MAX_SIZE - 1);
      memcpy(stringValue.data, match.beginStringValue, lastChar);
      stringValue.data[lastChar] = '\0';
    } break;
    default:
      return false;
      break;
  };

  // Find the metric based on the label. Create it if not found.
  auto cmpFn = [namePtr = match.beginKey,
                nameSize = match.endKey - match.beginKey](MetricLabelIndex const& a, MetricLabelIndex const& b)
    -> bool {
    return strncmp(a.label, namePtr, nameSize) < 0;
  };
  auto mi = std::lower_bound(info.metricLabelsIdx.begin(),
                             info.metricLabelsIdx.end(),
                             MetricLabelIndex{},
                             cmpFn);

  // We could not find the metric, lets insert a new one.
  if (mi == info.metricLabelsIdx.end() || (strncmp(mi->label, match.beginKey, std::min(match.endKey - match.beginKey, (long)MetricLabelIndex::MAX_METRIC_LABEL_SIZE - 1)) != 0)) {
    // Create a new metric
    MetricInfo metricInfo;
    metricInfo.pos = 0;
    metricInfo.type = match.type;
    // Add a new empty buffer for it of the correct kind
    switch (match.type) {
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
    MetricLabelIndex metricLabelIdx;
    auto lastChar = std::min(match.endKey - match.beginKey, (ptrdiff_t)MetricLabelIndex::MAX_METRIC_LABEL_SIZE - 1);
    memcpy(metricLabelIdx.label, match.beginKey, lastChar);
    metricLabelIdx.label[lastChar] = '\0';
    metricLabelIdx.index = info.metrics.size();
    info.metricLabelsIdx.insert(mi, metricLabelIdx);
    // Add the the actual Metric info to the store
    metricIndex = info.metrics.size();
    assert(metricInfo.storeIdx != -1);
    assert(metricLabelIdx.label[0] != '\0');
    if (newMetricsCallback != nullptr) {
      newMetricsCallback(metricLabelIdx.label, metricInfo, match.intValue, metricIndex);
    }
    info.metrics.push_back(metricInfo);
  } else {
    metricIndex = mi->index;
  }
  assert(metricIndex != -1);
  // We are now guaranteed our metric is present at metricIndex.
  MetricInfo& metricInfo = info.metrics[metricIndex];

  //  auto mod = info.timestamps[metricIndex].size();
  info.minDomain[metricIndex] = std::min(info.minDomain[metricIndex], (size_t)match.timestamp);
  info.maxDomain[metricIndex] = std::max(info.maxDomain[metricIndex], (size_t)match.timestamp);

  switch (metricInfo.type) {
    case MetricType::Int: {
      info.intMetrics[metricInfo.storeIdx][metricInfo.pos] = match.intValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], (float)match.intValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], (float)match.intValue);
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = match.timestamp;
      // Update the position where to write the next metric
      metricInfo.pos = (metricInfo.pos + 1) % info.intMetrics[metricInfo.storeIdx].size();
    } break;
    case MetricType::String: {
      info.stringMetrics[metricInfo.storeIdx][metricInfo.pos] = stringValue;
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = match.timestamp;
      metricInfo.pos = (metricInfo.pos + 1) % info.stringMetrics[metricInfo.storeIdx].size();
    } break;
    case MetricType::Float: {
      info.floatMetrics[metricInfo.storeIdx][metricInfo.pos] = match.floatValue;
      info.max[metricIndex] = std::max(info.max[metricIndex], match.floatValue);
      info.min[metricIndex] = std::min(info.min[metricIndex], match.floatValue);
      // Save the timestamp for the current metric we do it here
      // so that we do not update timestamps for broken metrics
      info.timestamps[metricIndex][metricInfo.pos] = match.timestamp;
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
    auto& metricName = info.metricLabelsIdx[i];
    if (metricName.label == name) {
      return metricName.index;
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
