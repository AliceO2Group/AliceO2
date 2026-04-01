// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "StatusWebSocketHandler.h"
#include "DPLWebSocket.h"
#include "DriverServerContext.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceController.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/DeviceStateEnums.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/Signpost.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace o2::framework
{

namespace
{

std::string jsonEscape(std::string_view s)
{
  std::string out;
  out.reserve(s.size() + 4);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

char const* logLevelName(LogParsingHelpers::LogLevel level)
{
  switch (level) {
    case LogParsingHelpers::LogLevel::Debug:
      return "debug";
    case LogParsingHelpers::LogLevel::Info:
      return "info";
    case LogParsingHelpers::LogLevel::Important:
      return "important";
    case LogParsingHelpers::LogLevel::Warning:
      return "warning";
    case LogParsingHelpers::LogLevel::Alarm:
      return "alarm";
    case LogParsingHelpers::LogLevel::Error:
      return "error";
    case LogParsingHelpers::LogLevel::Critical:
      return "critical";
    case LogParsingHelpers::LogLevel::Fatal:
      return "fatal";
    default:
      return "unknown";
  }
}

char const* streamingStateName(StreamingState s)
{
  switch (s) {
    case StreamingState::Streaming:
      return "Streaming";
    case StreamingState::EndOfStreaming:
      return "EndOfStreaming";
    case StreamingState::Idle:
      return "Idle";
    default:
      return "Unknown";
  }
}

void appendMetricValue(std::string& out, DeviceMetricsInfo const& info, size_t mi)
{
  auto const& metric = info.metrics[mi];
  if (metric.pos == 0) {
    out += "null";
    return;
  }
  size_t last = (metric.pos - 1) % metricStorageSize(metric.type);
  switch (metric.type) {
    case MetricType::Int:
      out += std::to_string(info.intMetrics[metric.storeIdx][last]);
      break;
    case MetricType::Float: {
      char buf[32];
      snprintf(buf, sizeof(buf), "%g", static_cast<double>(info.floatMetrics[metric.storeIdx][last]));
      out += buf;
      break;
    }
    case MetricType::Uint64:
      out += std::to_string(info.uint64Metrics[metric.storeIdx][last]);
      break;
    default:
      out += "null";
  }
}

/// Extract the value of a simple string field from a flat JSON object.
/// e.g. extractField(R"({"cmd":"subscribe","device":"prod"})", "device") → "prod"
/// Returns empty string_view if not found.
std::string_view extractStringField(std::string_view json, std::string_view key)
{
  std::string needle;
  needle += '"';
  needle += key;
  needle += "\":";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  pos += needle.size();
  // skip optional whitespace between ':' and '"'
  while (pos < json.size() && json[pos] == ' ') {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != '"') {
    return {};
  }
  ++pos; // skip opening quote
  auto end = json.find('"', pos);
  if (end == std::string_view::npos) {
    return {};
  }
  return json.substr(pos, end - pos);
}

/// Extract the raw value of an array field from a flat JSON object.
/// e.g. extractArrayField(R"({"metrics":["a","b"]})", "metrics") → R"(["a","b"])"
std::string_view extractArrayField(std::string_view json, std::string_view key)
{
  std::string needle;
  needle += '"';
  needle += key;
  needle += "\":";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  pos += needle.size();
  // skip whitespace
  while (pos < json.size() && json[pos] == ' ') {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != '[') {
    return {};
  }
  auto start = pos;
  size_t depth = 0;
  while (pos < json.size()) {
    if (json[pos] == '[') {
      ++depth;
    } else if (json[pos] == ']') {
      --depth;
      if (depth == 0) {
        return json.substr(start, pos - start + 1);
      }
    }
    ++pos;
  }
  return {};
}

/// Iterate over the string elements of a JSON array of strings.
/// Calls @a callback for each unescaped string value.
template <typename F>
void forEachStringInArray(std::string_view arr, F&& callback)
{
  // arr is like ["name1","name2"]
  size_t pos = 0;
  while (pos < arr.size()) {
    auto q = arr.find('"', pos);
    if (q == std::string_view::npos) {
      break;
    }
    auto end = arr.find('"', q + 1);
    if (end == std::string_view::npos) {
      break;
    }
    callback(arr.substr(q + 1, end - q - 1));
    pos = end + 1;
  }
}

} // anonymous namespace

StatusWebSocketHandler::StatusWebSocketHandler(DriverServerContext& context, WSDPLHandler* handler)
  : mContext{context}, mHandler{handler}
{
}

StatusWebSocketHandler::~StatusWebSocketHandler()
{
  auto& handlers = mContext.statusHandlers;
  handlers.erase(std::remove(handlers.begin(), handlers.end(), this), handlers.end());
}

void StatusWebSocketHandler::headers(std::map<std::string, std::string> const&)
{
  sendSnapshot();
}

void StatusWebSocketHandler::frame(char const* data, size_t s)
{
  std::string_view msg{data, s};
  auto cmd = extractStringField(msg, "cmd");
  if (cmd.empty()) {
    return;
  }
  auto deviceName = extractStringField(msg, "device");

  if (cmd == "list_metrics") {
    handleListMetrics(deviceName);
  } else if (cmd == "subscribe") {
    handleSubscribe(deviceName, extractArrayField(msg, "metrics"));
  } else if (cmd == "unsubscribe") {
    handleUnsubscribe(deviceName, extractArrayField(msg, "metrics"));
  } else if (cmd == "subscribe_logs") {
    handleSubscribeLogs(deviceName);
  } else if (cmd == "unsubscribe_logs") {
    handleUnsubscribeLogs(deviceName);
  } else if (cmd == "enable_signpost") {
    handleEnableSignpost(deviceName, extractArrayField(msg, "streams"));
  } else if (cmd == "disable_signpost") {
    handleDisableSignpost(deviceName, extractArrayField(msg, "streams"));
  }
}

void StatusWebSocketHandler::sendText(std::string const& json)
{
  std::vector<uv_buf_t> outputs;
  encode_websocket_frames(outputs, json.data(), json.size(), WebSocketOpCode::Text, 0);
  mHandler->write(outputs);
}

void StatusWebSocketHandler::sendSnapshot()
{
  auto const& specs = *mContext.specs;
  auto const& infos = *mContext.infos;

  // Size subscription tables to current device count; grow lazily as needed.
  mSubscribedMetrics.resize(specs.size());
  mLastLogSeq.resize(infos.size());
  for (size_t di = 0; di < infos.size(); ++di) {
    mLastLogSeq[di] = infos[di].logSeq;
  }

  std::string out;
  out.reserve(512 + specs.size() * 128);
  out += R"({"type":"snapshot","devices":[)";
  for (size_t di = 0; di < specs.size(); ++di) {
    if (di > 0) {
      out += ',';
    }
    auto const& info = infos[di];
    out += R"({"name":")";
    out += jsonEscape(specs[di].name);
    out += R"(","pid":)";
    out += std::to_string(info.pid);
    out += R"(,"active":)";
    out += info.active ? "true" : "false";
    out += R"(,"streamingState":")";
    out += streamingStateName(info.streamingState);
    out += R"(","deviceState":")";
    out += jsonEscape(info.deviceState);
    out += R"("})";
  }
  out += "]}";
  sendText(out);
}

void StatusWebSocketHandler::sendUpdate(size_t deviceIndex)
{
  auto const& specs = *mContext.specs;
  auto const& metrics = *mContext.metrics;

  if (deviceIndex >= specs.size() || deviceIndex >= metrics.size()) {
    return;
  }

  // Lazily grow the subscription table if new devices were added after snapshot.
  if (mSubscribedMetrics.size() <= deviceIndex) {
    mSubscribedMetrics.resize(deviceIndex + 1);
  }

  auto const& subscribed = mSubscribedMetrics[deviceIndex];
  if (subscribed.empty()) {
    return;
  }

  auto const& info = metrics[deviceIndex];
  std::string metricsJson;
  metricsJson += '{';
  bool first = true;
  for (size_t mi = 0; mi < info.metrics.size(); ++mi) {
    if (!info.changed[mi]) {
      continue;
    }
    auto const& metric = info.metrics[mi];
    if (metric.type == MetricType::String ||
        metric.type == MetricType::Enum ||
        metric.type == MetricType::Unknown) {
      continue;
    }
    auto const& label = info.metricLabels[mi];
    std::string_view labelSV{label.label, label.size};
    if (subscribed.find(std::string(labelSV)) == subscribed.end()) {
      continue;
    }
    if (!first) {
      metricsJson += ',';
    }
    first = false;
    metricsJson += '"';
    metricsJson += jsonEscape(labelSV);
    metricsJson += "\":";
    appendMetricValue(metricsJson, info, mi);
  }
  metricsJson += '}';

  if (first) {
    // Nothing subscribed changed in this cycle.
    return;
  }

  std::string out;
  out += R"({"type":"update","device":)";
  out += std::to_string(deviceIndex);
  out += R"(,"name":")";
  out += jsonEscape(specs[deviceIndex].name);
  out += R"(","metrics":)";
  out += metricsJson;
  out += '}';
  sendText(out);
}

void StatusWebSocketHandler::handleListMetrics(std::string_view deviceName)
{
  size_t di = findDeviceIndex(deviceName);
  if (di == SIZE_MAX) {
    return;
  }
  auto const& metrics = *mContext.metrics;

  std::string out;
  out += R"({"type":"metrics_list","device":")";
  out += jsonEscape(deviceName);
  out += R"(","metrics":[)";
  bool first = true;
  if (di < metrics.size()) {
    auto const& info = metrics[di];
    for (size_t mi = 0; mi < info.metrics.size(); ++mi) {
      auto const& metric = info.metrics[mi];
      if (metric.type == MetricType::String ||
          metric.type == MetricType::Enum ||
          metric.type == MetricType::Unknown) {
        continue;
      }
      if (!first) {
        out += ',';
      }
      first = false;
      auto const& label = info.metricLabels[mi];
      out += '"';
      out += jsonEscape({label.label, label.size});
      out += '"';
    }
  }
  out += "]}";
  sendText(out);
}

void StatusWebSocketHandler::handleSubscribe(std::string_view deviceName, std::string_view metricsArr)
{
  size_t di = findDeviceIndex(deviceName);
  if (di == SIZE_MAX || metricsArr.empty()) {
    return;
  }
  if (mSubscribedMetrics.size() <= di) {
    mSubscribedMetrics.resize(di + 1);
  }
  forEachStringInArray(metricsArr, [&](std::string_view name) {
    mSubscribedMetrics[di].emplace(name);
  });
}

void StatusWebSocketHandler::handleUnsubscribe(std::string_view deviceName, std::string_view metricsArr)
{
  size_t di = findDeviceIndex(deviceName);
  if (di == SIZE_MAX || metricsArr.empty() || di >= mSubscribedMetrics.size()) {
    return;
  }
  forEachStringInArray(metricsArr, [&](std::string_view name) {
    mSubscribedMetrics[di].erase(std::string(name));
  });
}

size_t StatusWebSocketHandler::findDeviceIndex(std::string_view name) const
{
  auto const& specs = *mContext.specs;
  for (size_t di = 0; di < specs.size(); ++di) {
    if (specs[di].name == name) {
      return di;
    }
  }
  return SIZE_MAX;
}

void StatusWebSocketHandler::handleEnableSignpost(std::string_view deviceName, std::string_view streamsArr)
{
  if (streamsArr.empty()) {
    return;
  }
  if (deviceName.empty()) {
    // Driver process — toggle in-process via o2_walk_logs.
    forEachStringInArray(streamsArr, [](std::string_view streamName) {
      std::string target(streamName);
      o2_walk_logs([](char const* name, void* l, void* context) -> bool {
        auto* log = static_cast<_o2_log_t*>(l);
        if (static_cast<std::string*>(context)->compare(name) == 0) {
          _o2_log_set_stacktrace(log, log->defaultStacktrace);
          return false;
        }
        return true;
      }, &target);
    });
  } else {
    size_t di = findDeviceIndex(deviceName);
    if (di == SIZE_MAX || di >= mContext.controls->size() || !(*mContext.controls)[di].controller) {
      return;
    }
    auto* controller = (*mContext.controls)[di].controller;
    forEachStringInArray(streamsArr, [controller](std::string_view name) {
      std::string cmd = "/signpost:enable ";
      cmd += name;
      controller->write(cmd.c_str(), cmd.size());
    });
  }
}

void StatusWebSocketHandler::handleDisableSignpost(std::string_view deviceName, std::string_view streamsArr)
{
  if (streamsArr.empty()) {
    return;
  }
  if (deviceName.empty()) {
    forEachStringInArray(streamsArr, [](std::string_view streamName) {
      std::string target(streamName);
      o2_walk_logs([](char const* name, void* l, void* context) -> bool {
        auto* log = static_cast<_o2_log_t*>(l);
        if (static_cast<std::string*>(context)->compare(name) == 0) {
          _o2_log_set_stacktrace(log, 0);
          return false;
        }
        return true;
      }, &target);
    });
  } else {
    size_t di = findDeviceIndex(deviceName);
    if (di == SIZE_MAX || di >= mContext.controls->size() || !(*mContext.controls)[di].controller) {
      return;
    }
    auto* controller = (*mContext.controls)[di].controller;
    forEachStringInArray(streamsArr, [controller](std::string_view name) {
      std::string cmd = "/signpost:disable ";
      cmd += name;
      controller->write(cmd.c_str(), cmd.size());
    });
  }
}

void StatusWebSocketHandler::handleSubscribeLogs(std::string_view deviceName)
{
  size_t di = findDeviceIndex(deviceName);
  if (di == SIZE_MAX) {
    return;
  }
  if (mLastLogSeq.size() <= di) {
    mLastLogSeq.resize(di + 1, 0);
  }
  // Start the cursor at the current log position so we only push future lines.
  mLastLogSeq[di] = (*mContext.infos)[di].logSeq;
  mLogSubscriptions.insert(di);
}

void StatusWebSocketHandler::handleUnsubscribeLogs(std::string_view deviceName)
{
  size_t di = findDeviceIndex(deviceName);
  if (di == SIZE_MAX) {
    return;
  }
  mLogSubscriptions.erase(di);
}

void StatusWebSocketHandler::sendNewLogs(size_t deviceIndex)
{
  if (mLogSubscriptions.find(deviceIndex) == mLogSubscriptions.end()) {
    return;
  }
  auto const& infos = *mContext.infos;
  auto const& specs = *mContext.specs;
  if (deviceIndex >= infos.size() || deviceIndex >= specs.size()) {
    return;
  }
  if (mLastLogSeq.size() <= deviceIndex) {
    mLastLogSeq.resize(deviceIndex + 1, 0);
  }

  auto const& info = infos[deviceIndex];
  size_t newLines = info.logSeq - mLastLogSeq[deviceIndex];
  if (newLines == 0) {
    return;
  }
  // Cap to buffer size to avoid re-reading overwritten entries.
  if (newLines > info.history.size()) {
    newLines = info.history.size();
  }

  size_t histSize = info.history.size();
  // The oldest unread entry sits at (historyPos - newLines + histSize) % histSize.
  size_t startPos = (info.historyPos + histSize - newLines) % histSize;

  std::string_view devName = specs[deviceIndex].name;
  for (size_t i = 0; i < newLines; ++i) {
    size_t pos = (startPos + i) % histSize;
    std::string out;
    out += R"({"type":"log","device":")";
    out += jsonEscape(devName);
    out += R"(","level":")";
    out += logLevelName(info.historyLevel[pos]);
    out += R"(","line":")";
    out += jsonEscape(info.history[pos]);
    out += R"("})";
    sendText(out);
  }
  mLastLogSeq[deviceIndex] = info.logSeq;
}

} // namespace o2::framework
