// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE Test Framework HTTPParser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "../src/HTTPParser.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

class DPLParser : public HTTPParser
{
 public:
  std::string mMethod;
  std::string mPath;
  std::string mVersion;
  std::string mBody;
  std::map<std::string, std::string> mHeaders;
  void method(std::string_view const& m) override
  {
    mMethod = m;
  }
  void target(std::string_view const& p) override
  {
    mPath = p;
  }

  void version(std::string_view const& v) override
  {
    mVersion = v;
  }
  void header(std::string_view const& k, std::string_view const& v) override
  {
    mHeaders[std::string(k)] = v;
  }
  void body(char* buf, size_t s) override
  {
    mBody = buf;
  }
};

class DPLClientParser : public HTTPParser
{
 public:
  std::string mReplyVersion;
  std::string mReplyCode;
  std::string mReplyMessage;
  std::string mBody;
  std::map<std::string, std::string> mHeaders;

  void replyMessage(std::string_view const& s) override
  {
    mReplyMessage = s;
  }
  void replyCode(std::string_view const& s) override
  {
    mReplyCode = s;
  }
  void replyVersion(std::string_view const& s) override
  {
    mReplyVersion = s;
  }

  void header(std::string_view const& k, std::string_view const& v) override
  {
    mHeaders[std::string(k)] = v;
  }
  void body(char* buf, size_t s) override
  {
    mBody = buf;
  }
};

class TestWSHandler : public WebSocketHandler
{
 public:
  std::vector<char const*> mFrame;
  std::vector<size_t> mSize;
  void frame(const char* f, size_t s) final
  {
    mFrame.push_back(f);
    mSize.push_back(s);
  }
};

BOOST_AUTO_TEST_CASE(HTTPParser1)
{
  {
    char* request = strdup(
      "GET / HTTP/1.1\r\n"
      "x-dpl-pid: 124679842\r\n\r\nCONTROL QUIT");
    DPLParser parser;
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mMethod), std::string("GET"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mPath), std::string("/"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 1);
  }
  {
    char* request = strdup(
      "GET / HTTP/1.1\r\n"
      "x-dpl-pid: 124679842\r\n"
      "Somethingelse: cjnjsdnjks\r\n\r\nCONTROL QUIT");
    DPLParser parser;
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mMethod), std::string("GET"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mPath), std::string("/"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 2);
    BOOST_REQUIRE_EQUAL(parser.mHeaders["x-dpl-pid"], "124679842");
    BOOST_REQUIRE_EQUAL(parser.mHeaders["Somethingelse"], "cjnjsdnjks");
    BOOST_REQUIRE_EQUAL(parser.mBody, "CONTROL QUIT");
  }
  {
    // handle continuations...
    char* request = strdup(
      "GET / HTTP/1.1\r\n"
      "x-dpl-pid: 124679842\r\n"
      "Somethingelse: cjnjsdnjks\r\n\r\nCONTROL QUIT");
    char* request2 = strdup("FOO BAR");
    DPLParser parser;
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mMethod), std::string("GET"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mPath), std::string("/"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 2);
    BOOST_REQUIRE_EQUAL(parser.mHeaders["x-dpl-pid"], "124679842");
    BOOST_REQUIRE_EQUAL(parser.mHeaders["Somethingelse"], "cjnjsdnjks");
    BOOST_REQUIRE_EQUAL(parser.mBody, "CONTROL QUIT");
    parse_http_request(request2, strlen(request2), &parser);
    BOOST_REQUIRE_EQUAL(parser.mBody, "FOO BAR");
  }

  {
    // WebSocket example
    char* request = strdup(
      "GET /chat HTTP/1.1\r\n"
      "Host: server.example.com\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==\r\n"
      "Sec-WebSocket-Protocol: chat, superchat\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "Origin: http://example.com\r\n\r\n");

    DPLParser parser;
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mMethod), std::string("GET"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mPath), std::string("/chat"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 7);
    BOOST_REQUIRE_EQUAL(parser.mHeaders["Sec-WebSocket-Protocol"], "chat, superchat");
  }
  {
    // WebSocket example
    char* request = strdup(
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");

    DPLClientParser parser;
    parser.states.push_back(HTTPState::IN_START_REPLY);
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyCode), std::string("101"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyMessage), std::string("Switching Protocols"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 3);
    BOOST_REQUIRE_EQUAL(parser.mHeaders["Sec-WebSocket-Accept"], "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    BOOST_REQUIRE_EQUAL(parser.mBody, "");
  }
  {
    // WebSocket frame encoding / decoding
    char* buffer = strdup("hello websockets!");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);
    TestWSHandler handler;
    BOOST_REQUIRE_EQUAL(encoded[0].len, strlen(buffer) + 1 + 2); // 1 for the 0, 2 for the header
    decode_websocket(encoded[0].base, encoded[0].len, handler);
    BOOST_REQUIRE_EQUAL(handler.mSize[0], strlen(buffer) + 1);
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
  }
  {
    // WebSocket multiple frame encoding / decoding
    char* buffer = strdup("hello websockets!");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);
    char const* buffer2 = "and again.";
    encode_websocket_frames(encoded, buffer2, strlen(buffer2) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 2);
    char* multiBuffer = (char*)malloc(encoded[0].len + encoded[1].len + 4);
    memcpy(multiBuffer, encoded[0].base, encoded[0].len);
    memcpy(multiBuffer + encoded[0].len, encoded[1].base, encoded[1].len);

    TestWSHandler handler;
    decode_websocket(multiBuffer, encoded[0].len + encoded[1].len, handler);
    BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 2);
    BOOST_REQUIRE_EQUAL(handler.mSize.size(), 2);
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[1], handler.mSize[1] - 1), std::string(buffer2));
  }
  {
    std::string checkRequest =
      "GET /chat HTTP/1.1\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
      "Sec-WebSocket-Protocol: myprotocol\r\n"
      "Sec-WebSocket-Version: 13\r\n\r\n";
    std::string checkReply =
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";
    int someSeed = 123;
    std::string result = encode_websocket_handshake_request("/chat", "myprotocol", 13, "dGhlIHNhbXBsZSBub25jZQ==");
    BOOST_REQUIRE_EQUAL(result, checkRequest);

    std::string reply = encode_websocket_handshake_reply("dGhlIHNhbXBsZSBub25jZQ==");
    BOOST_CHECK_EQUAL(reply, checkReply);
  }
}

BOOST_AUTO_TEST_CASE(URLParser)
{
  {
    auto [ip, port] = o2::framework::parse_websocket_url("ws://");
    BOOST_CHECK_EQUAL(ip, "127.0.0.1");
    BOOST_CHECK_EQUAL(port, 8080);
  }
  {
    auto [ip, port] = o2::framework::parse_websocket_url("ws://127.0.0.1:8080");
    BOOST_CHECK_EQUAL(ip, "127.0.0.1");
    BOOST_CHECK_EQUAL(port, 8080);
  }
  {
    auto [ip, port] = o2::framework::parse_websocket_url("ws://0.0.0.0:8080");
    BOOST_CHECK_EQUAL(ip, "0.0.0.0");
    BOOST_CHECK_EQUAL(port, 8080);
  }
  {
    auto [ip, port] = o2::framework::parse_websocket_url("ws://0.0.0.0:8081");
    BOOST_CHECK_EQUAL(ip, "0.0.0.0");
    BOOST_CHECK_EQUAL(port, 8081);
  }
}
