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

#include "Framework/DeviceMetricsHelper.h"
#include "Framework/DriverInfo.h"
#include "Framework/RuntimeError.h"
#include "Framework/Logger.h"
#include <cassert>
#include <cinttypes>
#include <cstdlib>

#include <algorithm>
#include <regex>
#include <string_view>
#include <tuple>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace o2::framework
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
    *space = (const char*)memchr(nextSpace, ' ', s.size() - (nextSpace - s.data()));
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
  // Some metrics are special
  match.type = static_cast<MetricType>(strtol(comma + 1, &ep, 10));
  if (strncmp(match.beginKey, "data_relayer/", 13) == 0 && match.beginKey[13] != 'w' && match.beginKey[13] != 'h') {
    match.type = MetricType::Enum;
  }
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
      match.floatValue = (float)match.intValue;
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
      match.floatValue = 0;
      break;
    case MetricType::Uint64:
      match.uint64Value = strtoul(spaces[1] + 1, &ep, 10);
      if (ep != *(space - 2)) {
        return false;
      }
      match.floatValue = (float)match.uint64Value;
      break;
    case MetricType::Enum:
      match.intValue = strtoul(spaces[1] + 1, &ep, 10);
      if (ep != *(space - 2)) {
        return false;
      }
      match.floatValue = (float)match.uint64Value;
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

static auto updatePrefix = [](std::string_view prefix, DeviceMetricsInfo& info, bool hasPrefix, std::vector<MetricPrefixIndex>::iterator pi) -> void {
  // Insert the prefix if needed
  if (!hasPrefix) {
    MetricPrefix metricPrefix;
    metricPrefix.size = prefix.size();
    memcpy(metricPrefix.prefix, prefix.data(), prefix.size());
    metricPrefix.prefix[prefix.size()] = '\0';

    MetricPrefixIndex metricPrefixIdx;
    metricPrefixIdx.index = info.metricPrefixes.size();
    info.metricPrefixes.push_back(metricPrefix);

    auto sortedInsert = std::distance(info.metricLabelsPrefixesSortedIdx.begin(), pi);
    info.metricLabelsPrefixesSortedIdx.insert(info.metricLabelsPrefixesSortedIdx.begin() + sortedInsert, metricPrefixIdx);

    auto previousEnd = 0;
    if (sortedInsert != 0) {
      previousEnd = info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[sortedInsert - 1].index].end;
    }
    info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[sortedInsert].index].begin = previousEnd;
    info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[sortedInsert].index].end = previousEnd + 1;
    for (size_t i = sortedInsert + 1; i < info.metricLabelsPrefixesSortedIdx.size(); i++) {
      info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[i].index].begin++;
      info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[i].index].end++;
    }
  } else {
    info.metricPrefixes[pi->index].end++;
    auto insertLocation = std::distance(info.metricLabelsPrefixesSortedIdx.begin(), pi);
    for (size_t i = insertLocation + 1; i < info.metricLabelsPrefixesSortedIdx.size(); i++) {
      info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[i].index].begin++;
      info.metricPrefixes[info.metricLabelsPrefixesSortedIdx[i].index].end++;
    }
  }
};

static constexpr int DPL_MAX_METRICS_PER_DEVICE = 1024*128;

static auto initMetric = [](DeviceMetricsInfo& info) -> void {
  // Add the timestamp buffer for it
  info.max.push_back(std::numeric_limits<float>::lowest());
  info.min.push_back(std::numeric_limits<float>::max());
  info.average.push_back(0);
  info.maxDomain.push_back(std::numeric_limits<size_t>::lowest());
  info.minDomain.push_back(std::numeric_limits<size_t>::max());
  info.changed.push_back(false);
  if (info.metricLabels.size() > DPL_MAX_METRICS_PER_DEVICE) {
    for (size_t i = 0; i < info.metricLabels.size(); i++) {
      std::cout << info.metricLabels[i].label << std::endl;
    }
    throw runtime_error_f("Too many metrics for a given device. Max is DPL_MAX_METRICS_PER_DEVICE=%d.", DPL_MAX_METRICS_PER_DEVICE );
  }
};

auto storeIdx = [](DeviceMetricsInfo& info, MetricType type) -> size_t {
  switch (type) {
    case MetricType::Int:
      return info.intMetrics.size();
    case MetricType::String:
      return info.stringMetrics.size();
    case MetricType::Float:
      return info.floatMetrics.size();
    case MetricType::Uint64:
      return info.uint64Metrics.size();
    case MetricType::Enum:
      return info.enumMetrics.size();
    default:
      return -1;
  }
};

auto createMetricInfo = [](DeviceMetricsInfo& info, MetricType type) -> MetricInfo {
  // Create a new metric
  MetricInfo metricInfo{
    .type = type,
    .storeIdx = storeIdx(info, type),
    .pos = 0,
    .filledMetrics = 0,
  };
  // Add a new empty buffer for it of the correct kind
  switch (type) {
    case MetricType::Int:
      info.intMetrics.emplace_back(MetricsStorage<int>{});
      info.intTimestamps.emplace_back(TimestampsStorage<int>{});
      break;
    case MetricType::String:
      info.stringMetrics.emplace_back(MetricsStorage<StringMetric>{});
      info.stringTimestamps.emplace_back(TimestampsStorage<StringMetric>{});
      break;
    case MetricType::Float:
      info.floatMetrics.emplace_back(MetricsStorage<float>{});
      info.floatTimestamps.emplace_back(TimestampsStorage<float>{});
      break;
    case MetricType::Uint64:
      info.uint64Metrics.emplace_back(MetricsStorage<uint64_t>{});
      info.uint64Timestamps.emplace_back(TimestampsStorage<uint64_t>{});
      break;
    case MetricType::Enum:
      info.enumMetrics.emplace_back(MetricsStorage<int8_t>{});
      info.enumTimestamps.emplace_back(TimestampsStorage<int8_t>{});
      break;
    default:
      throw std::runtime_error("Unknown metric type");
  };
  initMetric(info);
  return metricInfo;
};

size_t DeviceMetricsHelper::bookMetricInfo(DeviceMetricsInfo& info, char const* name, MetricType type)
{
  // Find the prefix for the metric
  auto key = std::string_view(name, strlen(name));

  auto slash = key.find_first_of("/");
  if (slash == std::string_view::npos) {
    slash = 1;
  }
  auto prefix = std::string_view(key.data(), slash);

  // Find the prefix.
  auto cmpPrefixFn = [&prefixes = info.metricPrefixes](MetricPrefixIndex const& a, std::string_view b) -> bool {
    return std::string_view(prefixes[a.index].prefix, prefixes[a.index].size) < b;
  };

  auto pi = std::lower_bound(info.metricLabelsPrefixesSortedIdx.begin(),
                             info.metricLabelsPrefixesSortedIdx.end(),
                             prefix,
                             cmpPrefixFn);
  bool hasPrefix = pi != info.metricLabelsPrefixesSortedIdx.end() && (std::string_view(info.metricPrefixes[pi->index].prefix, info.metricPrefixes[pi->index].size) == prefix);

  auto rb = info.metricLabelsAlphabeticallySortedIdx.begin() + (hasPrefix ? info.metricPrefixes[pi->index].begin : 0);
  auto re = info.metricLabelsAlphabeticallySortedIdx.begin() + (hasPrefix ? info.metricPrefixes[pi->index].end : info.metricLabelsAlphabeticallySortedIdx.size());

  // Find the metric based on the label. Create it if not found.
  auto cmpFn = [&labels = info.metricLabels, offset = hasPrefix ? prefix.size() : 0](MetricLabelIndex const& a, std::string_view b) -> bool {
    return std::string_view(labels[a.index].label + offset, labels[a.index].size - offset) < std::string_view(b.data() + offset, b.size() - offset);
  };
  auto mi = std::lower_bound(rb, re, key, cmpFn);

  auto miIndex = std::distance(info.metricLabelsAlphabeticallySortedIdx.begin(), mi);
  auto miInsertionPoint = info.metricLabelsAlphabeticallySortedIdx.begin() + miIndex;

  // Add the index by name in the correct position
  // this will require moving the tail of the index,
  // but inserting should happen only once for each metric,
  // so who cares.
  // Add the the actual Metric info to the store
  MetricLabel metricLabel;
  strncpy(metricLabel.label, name, MetricLabel::MAX_METRIC_LABEL_SIZE - 1);
  metricLabel.label[MetricLabel::MAX_METRIC_LABEL_SIZE - 1] = '\0';
  metricLabel.size = strlen(metricLabel.label);

  // If it was already there, return the old index.
  if (mi != re && (strncmp(info.metricLabels[mi->index].label, metricLabel.label, std::min((size_t)metricLabel.size, (size_t)MetricLabel::MAX_METRIC_LABEL_SIZE - 1)) == 0)) {
    return mi->index;
  }

  // Add the the actual Metric info to the store
  auto metricIndex = info.metrics.size();

  // Insert the sorted location where it belongs to.
  MetricLabelIndex metricLabelIdx{metricIndex};
  info.metricLabelsAlphabeticallySortedIdx.insert(miInsertionPoint, metricLabelIdx);
  auto metricInfo = createMetricInfo(info, type);

  info.metricLabels.push_back(metricLabel);
  updatePrefix(prefix, info, hasPrefix, pi);
  info.metrics.emplace_back(metricInfo);

  return metricIndex;
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
    case MetricType::Uint64:
    case MetricType::Enum:
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

  // Find the prefix for the metric
  auto key = std::string_view(match.beginKey, match.endKey - match.beginKey);

  auto slash = key.find_first_of("/");
  if (slash == std::string_view::npos) {
    slash = 1;
  }
  auto prefix = std::string_view(key.data(), slash);

  // Find the prefix.
  auto cmpPrefixFn = [&prefixes = info.metricPrefixes](MetricPrefixIndex const& a, std::string_view b) -> bool {
    return std::string_view(prefixes[a.index].prefix, prefixes[a.index].size) < b;
  };

  auto pi = std::lower_bound(info.metricLabelsPrefixesSortedIdx.begin(),
                             info.metricLabelsPrefixesSortedIdx.end(),
                             prefix,
                             cmpPrefixFn);
  bool hasPrefix = pi != info.metricLabelsPrefixesSortedIdx.end() && (std::string_view(info.metricPrefixes[pi->index].prefix, info.metricPrefixes[pi->index].size) == prefix);

  auto rb = info.metricLabelsAlphabeticallySortedIdx.begin() + (hasPrefix ? info.metricPrefixes[pi->index].begin : 0);
  auto re = info.metricLabelsAlphabeticallySortedIdx.begin() + (hasPrefix ? info.metricPrefixes[pi->index].end : info.metricLabelsAlphabeticallySortedIdx.size());

  // Find the metric based on the label. Create it if not found.
  auto cmpFn = [&labels = info.metricLabels, offset = hasPrefix ? prefix.size() : 0](MetricLabelIndex const& a, std::string_view b) -> bool {
    auto result = std::string_view(labels[a.index].label + offset, labels[a.index].size - offset) < std::string_view(b.data() + offset, b.size() - offset);
    return result;
  };
  auto mi = std::lower_bound(rb, re, key, cmpFn);

  auto miIndex = std::distance(info.metricLabelsAlphabeticallySortedIdx.begin(), mi);
  auto miInsertionPoint = info.metricLabelsAlphabeticallySortedIdx.begin() + miIndex;

  // We could not find the metric, lets insert a new one.
  if (mi == re || (strncmp(info.metricLabels[mi->index].label, key.data(), std::min(key.size(), (size_t)MetricLabel::MAX_METRIC_LABEL_SIZE - 1)) != 0)) {
    auto metricInfo = createMetricInfo(info, match.type);

    // Add the index by name in the correct position
    // this will require moving the tail of the index,
    // but inserting should happen only once for each metric,
    // so who cares.
    MetricLabel metricLabel;
    auto lastChar = std::min(match.endKey - match.beginKey, (ptrdiff_t)MetricLabel::MAX_METRIC_LABEL_SIZE - 1);
    memcpy(metricLabel.label, match.beginKey, lastChar);
    metricLabel.label[lastChar] = '\0';
    metricLabel.size = lastChar;
    MetricLabelIndex metricLabelIdx;
    metricLabelIdx.index = info.metrics.size();
    info.metricLabels.push_back(metricLabel);
    info.metricLabelsAlphabeticallySortedIdx.insert(miInsertionPoint, metricLabelIdx);

    updatePrefix(prefix, info, hasPrefix, pi);

    // Add the the actual Metric info to the store
    metricIndex = info.metrics.size();
    assert(metricInfo.storeIdx != -1);
    assert(metricLabel.label[0] != '\0');
    if (newMetricsCallback != nullptr) {
      newMetricsCallback(metricLabel.label, metricInfo, match.intValue, metricIndex);
    }
    info.metrics.push_back(metricInfo);
  } else {
    metricIndex = mi->index;
  }
  assert(metricIndex != -1);
  // We are now guaranteed our metric is present at metricIndex.
  MetricInfo& metricInfo = info.metrics[metricIndex];

  //  auto mod = info.timestamps[metricIndex].size();
  auto sizeOfCollection = 0;
  switch (metricInfo.type) {
    case MetricType::Int: {
      info.intMetrics[metricInfo.storeIdx][metricInfo.pos] = match.intValue;
      sizeOfCollection = info.intMetrics[metricInfo.storeIdx].size();
      info.intTimestamps[metricInfo.storeIdx][metricInfo.pos] = match.timestamp;
    } break;
    case MetricType::String: {
      info.stringMetrics[metricInfo.storeIdx][metricInfo.pos] = stringValue;
      sizeOfCollection = info.stringMetrics[metricInfo.storeIdx].size();
      info.stringTimestamps[metricInfo.storeIdx][metricInfo.pos] = match.timestamp;
    } break;
    case MetricType::Float: {
      info.floatMetrics[metricInfo.storeIdx][metricInfo.pos] = match.floatValue;
      sizeOfCollection = info.floatMetrics[metricInfo.storeIdx].size();
      info.floatTimestamps[metricInfo.storeIdx][metricInfo.pos] = match.timestamp;
    } break;
    case MetricType::Uint64: {
      info.uint64Metrics[metricInfo.storeIdx][metricInfo.pos] = match.uint64Value;
      sizeOfCollection = info.uint64Metrics[metricInfo.storeIdx].size();
      info.uint64Timestamps[metricInfo.storeIdx][metricInfo.pos] = match.timestamp;
    } break;
    case MetricType::Enum: {
      info.enumMetrics[metricInfo.storeIdx][metricInfo.pos] = match.intValue;
      sizeOfCollection = info.enumMetrics[metricInfo.storeIdx].size();
      info.enumTimestamps[metricInfo.storeIdx][metricInfo.pos] = match.timestamp;
    } break;
    default:
      return false;
      break;
  };
  // We do all the updates here, so that not update timestamps for broken metrics
  // Notice how we always fill floatValue with the float equivalent of the metric
  // regardless of it's type.
  info.minDomain[metricIndex] = std::min(info.minDomain[metricIndex], (size_t)match.timestamp);
  info.maxDomain[metricIndex] = std::max(info.maxDomain[metricIndex], (size_t)match.timestamp);
  info.max[metricIndex] = std::max(info.max[metricIndex], match.floatValue);
  info.min[metricIndex] = std::min(info.min[metricIndex], match.floatValue);
  auto onlineAverage = [](float nextValue, float previousAverage, float previousCount) {
    return previousAverage + (nextValue - previousAverage) / (previousCount + 1);
  };
  info.average[metricIndex] = onlineAverage(match.floatValue, info.average[metricIndex], metricInfo.filledMetrics);
  // We point to the next metric
  metricInfo.pos = (metricInfo.pos + 1) % sizeOfCollection;
  ++metricInfo.filledMetrics;
  // Note that we updated a given metric.
  info.changed[metricIndex] = true;
  return true;
}

size_t DeviceMetricsHelper::metricIdxByName(const std::string& name, const DeviceMetricsInfo& info)
{
  size_t i = 0;
  while (i < info.metricLabels.size()) {
    auto& metricName = info.metricLabels[i];
    // We check the size first and then the last character because that's
    // likely to be different for multi-index metrics
    if (metricName.size == name.size() && metricName.label[metricName.size - 1] == name[metricName.size - 1] && memcmp(metricName.label, name.c_str(), metricName.size) == 0) {
      return i;
    }
    ++i;
  }
  return i;
}

} // namespace o2::framework
