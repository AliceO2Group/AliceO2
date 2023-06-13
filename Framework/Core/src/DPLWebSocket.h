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
#ifndef O2_FRAMEWORK_DPLWEBSOCKET_H_
#define O2_FRAMEWORK_DPLWEBSOCKET_H_

#include "Framework/ServiceRegistryRef.h"
#include <uv.h>
#include "HTTPParser.h"
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <atomic>

class uv_stream_s;

namespace o2::framework
{

struct DeviceSpec;
struct DriverServerContext;
struct DriverClientContext;

struct WSError {
  int code;
  std::string message;
};

struct WSDPLHandler : public HTTPParser {
  /// A http parser suitable to be used by DPL as a server
  /// @a stream is the stream from which the data is read,
  /// @a context to use to register the Handler to the appropriate
  ///  DeviceInfo.
  /// @a handler is the websocket handler to react on the
  /// various frames
  WSDPLHandler(uv_stream_t* stream, DriverServerContext* context);
  virtual ~WSDPLHandler() = default;
  void method(std::string_view const& s) override;
  void target(std::string_view const& s) override;
  void header(std::string_view const& k, std::string_view const& v) override;
  void endHeaders() override;
  /// Actual handling of WS frames happens inside here.
  void body(char* data, size_t s) override;
  /// Helper to write a message to the associated client
  void write(char const*, size_t);

  /// Helper to write n buffers containing websockets frames to a server
  void write(std::vector<uv_buf_t>& outputs);

  /// Helper to return an error
  void error(int code, char const* message);

  std::unique_ptr<WebSocketHandler> mHandler;
  bool mHandshaken = false;
  uv_stream_t* mStream = nullptr;
  std::map<std::string, std::string> mHeaders;
  DriverServerContext* mServerContext;
};

struct WSDPLClient : public HTTPParser {
  WSDPLClient();
  /// @a stream where the communication happens and @a spec of the device connecting
  /// to the driver.
  /// @a spec the DeviceSpec associated with this client
  /// @a handshake a callback to invoke whenever we have a successful handshake
  void connect(ServiceRegistryRef ref,
               uv_stream_t* stream,
               std::function<void()> handshake,
               std::unique_ptr<WebSocketHandler> handler);

  void replyVersion(std::string_view const& s) override;
  void replyCode(std::string_view const& s) override;
  void header(std::string_view const& k, std::string_view const& v) override;
  void endHeaders() override;
  /// Actual handling of WS frames happens inside here.
  void body(char* data, size_t s) override;
  /// Helper to write a message to the server
  void write(char const*, size_t);

  /// Helper to write n buffers containing websockets frames to a server
  void write(std::vector<uv_buf_t>& outputs);

  /// Dump headers
  void dumpHeaders();
  void sendHandshake();
  bool isHandshaken() { return mHandshaken; }

  std::string mNonce;
  std::atomic<bool> mHandshaken = false;
  std::function<void()> mHandshake;
  std::unique_ptr<DriverClientContext> mContext;
  std::unique_ptr<WebSocketHandler> mHandler;
  uv_stream_t* mStream = nullptr;
  std::map<std::string, std::string> mHeaders;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DPL_WEBSOCKET_H_
