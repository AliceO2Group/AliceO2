// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_HTTPPARSER_H_
#define O2_FRAMEWORK_HTTPPARSER_H_

#include <fmt/format.h>
#include <uv.h>
#include <string>
#include <vector>
#include <map>

namespace o2::framework
{

struct __attribute__((__packed__)) WebSocketFrameTiny {
  unsigned char fin : 1;
  unsigned char rsv1 : 1;
  unsigned char rsv2 : 1;
  unsigned char rsv3 : 1;
  unsigned char opcode : 4;
  unsigned char mask : 1;
  unsigned char len : 7;
};

struct __attribute__((__packed__)) WebSocketFrameShort {
  unsigned char fin : 1;
  unsigned char rsv1 : 1;
  unsigned char rsv2 : 1;
  unsigned char rsv3 : 1;
  unsigned char opcode : 4;
  unsigned char mask : 1;
  unsigned char len : 7;
  uint16_t len16;
};

struct __attribute__((__packed__)) WebSocketFrameHuge {
  unsigned char fin : 1;
  unsigned char rsv1 : 1;
  unsigned char rsv2 : 1;
  unsigned char rsv3 : 1;
  unsigned char opcode : 4;
  unsigned char mask : 1;
  unsigned char len : 7;
  uint64_t len64;
};

enum struct WebSocketFrameKind {
  Tiny,
  Short,
  Huge
};

enum struct WebSocketOpCode : uint8_t {
  Continuation = 0,
  Text = 1,
  Binary = 2,
  Close = 8,
  Ping = 9,
  Pong = 10
};

/// Encodes the request handshake for a given path / protocol / version.
/// @a path is the path of the websocket endpoint
/// @a protocol is the protocol required for the websocket connection
/// @a version is the protocol version
/// @a nonce is a unique randomly selected 16-byte value that has been base64-encoded.
/// @a headers with extra headers to be added to the request
std::string encode_websocket_handshake_request(const char* path, const char* protocol, int version, char const* nonce,
                                               std::vector<std::pair<std::string, std::string>> headers = {});

/// Encodes the server reply for a given websocket connection
/// @a nonce the nonce of the request.
std::string encode_websocket_handshake_reply(char const* nonce);

/// Encodes the buffer @a src which is @a size long to a number of buffers suitable to be sent via libuv.
/// If @a binary is provided the binary bit is set.
/// If @a mask is non zero, payload will be xored with the mask, as required by the WebSockets RFC
void encode_websocket_frames(std::vector<uv_buf_t>& outputs, char const* src, size_t size, WebSocketOpCode opcode, uint32_t mask);

/// An handler for a websocket message stream.
struct WebSocketHandler {
  /// Invoked when all the headers are received.
  virtual void headers(std::map<std::string, std::string> const& headers){};
  /// FIXME: not implemented
  virtual void beginFragmentation(){};
  /// Invoked when a frame it's parsed. Notice you do not own the data and you must
  /// not free the memory.
  virtual void frame(char const* frame, size_t s) {}
  /// Invoked before processing the next round of input
  virtual void beginChunk() {}
  /// Invoked whenever we have no more input to process
  virtual void endChunk() {}
  /// FIXME: not implemented
  virtual void endFragmentation() {}
  /// FIXME: not implemented
  virtual void control(char const* frame, size_t s) {}
};

/// Decoder for websocket data. For now we assume that the frame was not split. However multiple
/// frames might be present.
void decode_websocket(char* src, size_t size, WebSocketHandler& handler);

enum struct HTTPState {
  IN_START,
  IN_START_REPLY,
  BEGIN_REPLY_VERSION,
  END_REPLY_VERSION,
  BEGIN_REPLY_CODE,
  END_REPLY_CODE,
  BEGIN_REPLY_MESSAGE,
  END_REPLY_MESSAGE,
  BEGIN_METHOD,
  END_METHOD,
  BEGIN_TARGET,
  END_TARGET,
  BEGIN_VERSION,
  END_VERSION,
  BEGIN_HEADERS,
  BEGIN_HEADER,
  BEGIN_HEADER_KEY,
  END_HEADER_KEY,
  BEGIN_HEADER_VALUE,
  END_HEADER_VALUE,
  END_HEADER,
  END_HEADERS,
  BEGIN_BODY,
  IN_DONE,
  IN_ERROR,
  IN_SKIP_CHARS,         /// skip any "delimiters" char.
  IN_CAPTURE_DELIMITERS, /// capture until any or the "delimiters" characters
  IN_CAPTURE_SEPARATOR,  /// capture until a specific "separator"
  IN_SEPARATOR,          /// skip a specific "separator"
  IN_CHUNKED
};

struct HTTPParser {
  std::string remaining;
  std::string error;
  std::vector<HTTPState> states;
  virtual void method(std::string_view const& s){};
  virtual void target(std::string_view const& s){};
  virtual void version(std::string_view const& s){};
  virtual void header(std::string_view const& k, std::string_view const& v){};
  virtual void endHeaders(){};
  /// Invoked whenever we are parsing data.
  /// In order to allow for xoring (as required by the websocket standard)
  /// in place, we pass it as a mutable pointer.
  virtual void body(char* data, size_t s){};
  virtual void replyVersion(std::string_view const& s){};
  virtual void replyCode(std::string_view const& s){};
  virtual void replyMessage(std::string_view const& s){};
};

struct HTTPParserHelpers {
  /// Helper to calculate the reply to a nonce
  static std::string calculateAccept(const char* nonce);
};

void parse_http_request(char* start, size_t size, HTTPParser* parser);

std::pair<std::string, unsigned short> parse_websocket_url(const char* s);
} // namespace o2::framework
#endif
