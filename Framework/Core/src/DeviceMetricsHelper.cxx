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

// Parser state for the metric string
enum struct ParsingState {
  IN_START,
  IN_PREFIX,
  IN_BEGIN_INDEX,
  IN_END_INDEX,
  IN_TYPE,
  IN_INT_VALUE,
  IN_UINT64_VALUE,
  IN_FLOAT_VALUE,
  IN_STRING_VALUE,
  IN_TIMESTAMP,
  IN_TAGS,
  IN_EXIT,
  IN_ERROR,
};
// Parses a metric in the form
//
// [METRIC] <name>[/<begin>[-<end>]],<type> <value> <timestamp>[ <tag>,<tag>]
//
// FIXME: I should probably also support the following format:
//
// [METRIC] <name>/<index1>:<index2>:<index3>,<type> <value>,<value>,<value> <timestamp>[ <tag>,<tag>]
//
// I.e. allow to specify multiple values for different indices.
bool DeviceMetricsHelper::parseMetric(std::string_view const s, ParsedMetricMatch& match)
{
  ParsingState state = ParsingState::IN_START;
  char const* cur = s.data();
  char const* next = s.data();
  char* err = nullptr;
  // We need to keep track of the last and last but one space
  // to be able to parse the timestamp and tags.
  char const* lastSpace = nullptr;
  char const* previousLastSpace = nullptr;

  while (true) {
    auto previousState = state;
    state = ParsingState::IN_ERROR;
    err = nullptr;
    switch (previousState) {
      case ParsingState::IN_START: {
        if (strncmp(cur, "[METRIC] ", 9) == 0) {
          next = cur + 8;
          state = ParsingState::IN_PREFIX;
          match.beginKey = cur + 9;
        }
      } break;
      case ParsingState::IN_PREFIX: {
        next = strpbrk(cur, ",/\n");
        if (next == nullptr || *next == '\n' || ((cur == next) && (match.endKey - match.beginKey == 0))) {
        } else if (*next == '/') {
          // Notice that in case of indexed metrics
          // we start by considering the key only
          // the prefix, in case there is a range
          // afterwards. If not, we will update the
          // key to include the index.
          match.endKey = next;
          state = ParsingState::IN_BEGIN_INDEX;
        } else if (*next == ',') {
          match.endKey = next;
          state = ParsingState::IN_TYPE;
        }
      } break;
      case ParsingState::IN_BEGIN_INDEX: {
        match.firstIndex = strtol(cur, &err, 10);
        next = err;
        if (*next == '-') {
          state = ParsingState::IN_END_INDEX;
        } else if (*next == ',') {
          // In case there is no range, we
          // maintain backwards compatibility
          // with the old key format.
          match.lastIndex = match.firstIndex + 1;
          match.endKey = next;
          state = ParsingState::IN_TYPE;
        } else {
          // We are still in prefix, we simply
          // it a / in the middle of the metric
          // so we skip it and go back to prefix.
          match.firstIndex = -1;
          match.endKey = next + 1;
          state = ParsingState::IN_PREFIX;
        }
      } break;
      case ParsingState::IN_END_INDEX: {
        match.lastIndex = strtol(cur, &err, 10);
        next = err;
        if (cur == next) {
        } else if (*next == ',') {
          state = ParsingState::IN_TYPE;
        }
      } break;
      case ParsingState::IN_TYPE: {
        match.type = (MetricType)strtol(cur, &err, 10);
        next = err;
        if (cur == next) {
        } else if (*next == ' ') {
          if (match.type == MetricType::Int) {
            state = ParsingState::IN_INT_VALUE;
          } else if (match.type == MetricType::Uint64) {
            state = ParsingState::IN_UINT64_VALUE;
          } else if (match.type == MetricType::Float) {
            state = ParsingState::IN_FLOAT_VALUE;
          } else if (match.type == MetricType::String) {
            state = ParsingState::IN_STRING_VALUE;
            match.beginStringValue = next + 1;
            match.endStringValue = next + 1;
          } else {
            break;
          }
          if (strncmp(match.beginKey, "data_relayer/", 13) == 0 && match.beginKey[13] != 'w' && match.beginKey[13] != 'h') {
            match.type = MetricType::Enum;
          }
        }
      } break;
      case ParsingState::IN_INT_VALUE: {
        match.intValue = strtol(cur, &err, 10);
        next = err;
        if (cur == next) {
        } else if (*next == ' ') {
          match.uint64Value = match.intValue;
          match.floatValue = match.intValue;
          state = ParsingState::IN_TIMESTAMP;
        }
      } break;
      case ParsingState::IN_UINT64_VALUE: {
        match.uint64Value = strtoull(cur, &err, 10);
        next = err;
        if (cur == next) {
        } else if (*next == ' ') {
          match.intValue = match.uint64Value;
          match.floatValue = match.uint64Value;
          state = ParsingState::IN_TIMESTAMP;
        }
      } break;
      case ParsingState::IN_FLOAT_VALUE: {
        match.floatValue = strtof(cur, &err);
        next = err;
        if (cur == next) {
        } else if (*next == ' ') {
          match.uint64Value = match.floatValue;
          match.intValue = match.floatValue;
          state = ParsingState::IN_TIMESTAMP;
        }
      } break;
      case ParsingState::IN_STRING_VALUE: {
        next = (char*)memchr(cur, ' ', s.data() + s.size() - cur);
        if (next == nullptr) {
          auto timestamp = strtoull(cur, &err, 10);
          if (*err != '\n' && *err != '\0') {
            // last word is not a number, backtrack
            // to the previous space and assume it is the
            // timestamp
            match.endStringValue = previousLastSpace;
            next = previousLastSpace;
            state = ParsingState::IN_TIMESTAMP;
          } else {
            // We found a timestamp as last word.
            match.endStringValue = lastSpace;
            match.timestamp = timestamp;
            next = err;
            state = ParsingState::IN_EXIT;
          }
          break;
          // This is the end of the string
        } else if (*next == ' ') {
          previousLastSpace = lastSpace;
          lastSpace = next;
          state = ParsingState::IN_STRING_VALUE;
          break;
        }
      } break;
      case ParsingState::IN_TIMESTAMP: {
        match.timestamp = strtoull(cur, &err, 10);
        next = err;
        if (cur == next) {
        } else if (*next == ' ') {
          state = ParsingState::IN_TAGS;
        } else if (next == s.data() + s.size() || *next == '\n') {
          state = ParsingState::IN_EXIT;
        }
      } break;
      case ParsingState::IN_TAGS: {
        // Tags not handled for now.
        state = ParsingState::IN_EXIT;
      } break;
      case ParsingState::IN_EXIT:
        return true;
      case ParsingState::IN_ERROR: {
        return false;
      }
    }
    cur = next + 1;
  }
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

static constexpr int DPL_MAX_METRICS_PER_DEVICE = 1024 * 128;

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
    throw runtime_error_f("Too many metrics for a given device. Max is DPL_MAX_METRICS_PER_DEVICE=%d.", DPL_MAX_METRICS_PER_DEVICE);
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
