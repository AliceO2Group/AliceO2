// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DPLWEBSOCKET_H_
#define O2_FRAMEWORK_DPLWEBSOCKET_H_

#include <uv.h>
#include "HTTPParser.h"
#include <memory>
#include <string>
#include <map>

class uv_stream_s;

namespace o2::framework
{

struct DeviceSpec;

struct WSError {
  int code;
  std::string message;
};

struct WSDPLHandler : public HTTPParser {
  /// A http parser suitable to be used by DPL as a server
  /// @a stream is the stream from which the data is read,
  /// @a handler is the websocket handler to react on the
  /// various frames
  WSDPLHandler(uv_stream_t* stream, std::unique_ptr<WebSocketHandler> handler);
  void method(std::string_view const& s) override;
  void target(std::string_view const& s) override;
  void header(std::string_view const& k, std::string_view const& v) override;
  void endHeaders() override;
  /// Actual handling of WS frames happens inside here.
  void body(char* data, size_t s) override;

  /// Helper to return an error
  void error(int code, char const* message);

  std::unique_ptr<WebSocketHandler> mHandler;
  bool mHandshaken = false;
  uv_stream_t* mStream = nullptr;
  std::map<std::string, std::string> mHeaders;
};

struct WSDPLClient : public HTTPParser {
  /// @a stream where the communication happens and @a spec of the device connecting
  /// to the driver.
  WSDPLClient(uv_stream_t* stream, DeviceSpec const& spec);
  void replyVersion(std::string_view const& s) override;
  void replyCode(std::string_view const& s) override;
  void header(std::string_view const& k, std::string_view const& v) override;
  void endHeaders() override;
  /// Helper to write a message to the server
  void write(char const*, size_t);

  /// Helper to write n buffers containing websockets frames to a server
  void write(std::vector<uv_buf_t>& outputs);

  /// Dump headers
  void dumpHeaders();
  void sendHandshake();
  bool isConnected() { return mHandshaken; }

  std::string mNonce;
  DeviceSpec const& mSpec;
  bool mHandshaken = false;
  uv_stream_t* mStream = nullptr;
  std::map<std::string, std::string> mHeaders;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DPL_WEBSOCKET_H_
