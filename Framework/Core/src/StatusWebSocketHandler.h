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
#ifndef O2_FRAMEWORK_STATUSWEBSOCKETHANDLER_H_
#define O2_FRAMEWORK_STATUSWEBSOCKETHANDLER_H_

#include "HTTPParser.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include <cstddef>

namespace o2::framework
{
struct DriverServerContext;
struct WSDPLHandler;

/// WebSocket handler for the /status endpoint.
///
/// Protocol (client → driver):
///   {"cmd":"list_metrics","device":"<name>"}
///     → driver replies with {"type":"metrics_list","device":"<name>","metrics":[...]}
///
///   {"cmd":"subscribe","device":"<name>","metrics":["m1","m2",...]}
///     → driver starts including those metrics in subsequent update frames
///
///   {"cmd":"unsubscribe","device":"<name>","metrics":["m1","m2",...]}
///     → driver stops sending those metrics
///
///   {"cmd":"subscribe_logs","device":"<name>"}
///     → driver starts pushing new log lines for the device
///
///   {"cmd":"unsubscribe_logs","device":"<name>"}
///     → driver stops pushing log lines for the device
///
///   {"cmd":"enable_signpost","device":"<name>","streams":["device","completion",...]}
///     → enable the named signpost log streams for a device (or the driver if device=="")
///     → known streams: "device","completion","monitoring_service","data_processor_context","stream_context"
///
///   {"cmd":"disable_signpost","device":"<name>","streams":["device","completion",...]}
///     → disable the named signpost log streams for a device
///
///   {"cmd":"list_signposts"}
///     → driver replies with {"type":"signposts_list","streams":["device","completion",...]}
///     → lists the known stream names
///
/// Protocol (driver → client):
///   {"type":"snapshot","devices":[{"name","pid","active","streamingState","deviceState"},...]}
///     → sent once on connect; contains no metrics or logs
///
///   {"type":"update","device":<index>,"name":"<name>","metrics":{<subscribed & changed>}}
///     → sent after each metrics cycle for devices with subscribed metrics that changed
///
///   {"type":"metrics_list","device":"<name>","metrics":["m1","m2",...]}
///     → reply to list_metrics command
///
///   {"type":"log","device":"<name>","level":"<level>","line":"<text>"}
///     → pushed for each new log line from a subscribed device
struct StatusWebSocketHandler : public WebSocketHandler {
  StatusWebSocketHandler(DriverServerContext& context, WSDPLHandler* handler);
  ~StatusWebSocketHandler() override;

  /// Sends the minimal snapshot on handshake completion.
  void headers(std::map<std::string, std::string> const& headers) override;
  /// Handles incoming commands from the MCP client.
  void frame(char const* data, size_t s) override;
  void beginChunk() override {}
  void endChunk() override {}
  void beginFragmentation() override {}
  void endFragmentation() override {}
  void control(char const* frame, size_t s) override {}

  /// Send a minimal JSON snapshot (device list + basic state, no metrics/logs).
  void sendSnapshot();
  /// Push an update for device at @a deviceIndex.
  /// Only metrics that are both changed[] and subscribed are included.
  /// No-op if nothing subscribed or nothing changed for this device.
  void sendUpdate(size_t deviceIndex);
  /// Push any log lines for @a deviceIndex that arrived since the last call.
  /// No-op if the device is not subscribed for logs.
  void sendNewLogs(size_t deviceIndex);

 private:
  void sendText(std::string const& json);
  void handleListMetrics(std::string_view deviceName);
  void handleSubscribe(std::string_view deviceName, std::string_view metricsJson);
  void handleUnsubscribe(std::string_view deviceName, std::string_view metricsJson);
  void handleSubscribeLogs(std::string_view deviceName);
  void handleUnsubscribeLogs(std::string_view deviceName);
  void handleStartDevices();
  void handleEnableSignpost(std::string_view deviceName, std::string_view streamsArr);
  void handleDisableSignpost(std::string_view deviceName, std::string_view streamsArr);
  size_t findDeviceIndex(std::string_view name) const;

  DriverServerContext& mContext;
  WSDPLHandler* mHandler;
  /// Per-device set of subscribed metric label strings.
  /// Sized to specs->size() on sendSnapshot(); grows if new devices appear.
  std::vector<std::unordered_set<std::string>> mSubscribedMetrics;
  /// Per-device log cursor: value of DeviceInfo::logSeq when we last sent logs.
  std::vector<size_t> mLastLogSeq;
  /// Set of device indices whose logs are being streamed.
  std::unordered_set<size_t> mLogSubscriptions;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_STATUSWEBSOCKETHANDLER_H_
