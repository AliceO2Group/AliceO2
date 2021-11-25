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
    mFrame.push_back(strdup(f));
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
      "Access-Control-Allow-Origin: \"*\"\r\n"
      "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");

    DPLClientParser parser;
    parser.states.push_back(HTTPState::IN_START_REPLY);
    parse_http_request(request, strlen(request), &parser);
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyCode), std::string("101"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyMessage), std::string("Switching Protocols"));
    BOOST_REQUIRE_EQUAL(std::string(parser.mReplyVersion), std::string("HTTP/1.1"));
    BOOST_REQUIRE_EQUAL(parser.mHeaders.size(), 4);
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
    char const* prototype = "and again.";
    char* buffer2 = (char*)malloc(0x20000);
    // fill the buffer with the prototype
    size_t mod = strlen(prototype);
    for (size_t i = 0; i < 0x20000; i++) {
      buffer2[i] = prototype[i % mod];
    }
    buffer2[0x20000 - 1] = '\0';
    encode_websocket_frames(encoded, buffer2, 0x20000, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 2);
    BOOST_REQUIRE_EQUAL(encoded[1].len, 0x20000 + 10);
    char* multiBuffer = (char*)malloc(encoded[0].len + encoded[1].len);
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
    // Decode a frame which is split in two.
    char* buffer = strdup("hello websockets!1");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);

    TestWSHandler handler;
    decode_websocket(encoded[0].base, encoded[0].len / 2, handler);
    decode_websocket(encoded[0].base + encoded[0].len / 2, encoded[0].len - encoded[0].len / 2, handler);
    BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 1);
    BOOST_REQUIRE_EQUAL(handler.mSize.size(), 1);
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
  }
  {
    // Decode a long frame which is split in two.
    char* buffer = strdup("string with more than 127 characters: cdsklcmalkmc cdmslkc adslkccmkadsc adslkmc dsa ckdls cdksclknds lkndnc anslkc klsad ckl lksad clkas ccdascnkjancjnjkascsa cdascds clsad nclksad ncklsd clkadns lkc sadnlk cklsa cnaksld csad");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);

    TestWSHandler handler;
    decode_websocket(encoded[0].base, encoded[0].len / 2, handler);
    decode_websocket(encoded[0].base + encoded[0].len / 2, encoded[0].len - encoded[0].len / 2, handler);
    BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 1);
    BOOST_REQUIRE_EQUAL(handler.mSize.size(), 1);
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
  }
  {
    // WebSocket multiple frame encoding / decoding, long frames
    char* buffer = strdup("dwqnocewnclkanklcdanslkcndklsnclkdsnckldsnclk  cnldcl dsklc dslk cljdnsck sdlakcn askc sdkla cnsd c sdcn dsklncn dklsc nsdkl cklds clkds ckls dklc shello websockets!");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);
    char const* buffer2 = "xsanjkcnsadjknc dsjc nsdnc dlscndsck dsc ds clds cds vnlsfl nklnjk nj nju n nio nkmnklfmdkl mkld mkl mkl mkl mlk m lkm klfdnkln jkafdnk nk mkldfm lkdamlkdmlkdmlk m klml km lkm kl.";
    encode_websocket_frames(encoded, buffer2, strlen(buffer2) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);

    TestWSHandler handler;
    decode_websocket(encoded[0].base, encoded[0].len, handler);
    BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 2);
    BOOST_REQUIRE_EQUAL(handler.mSize.size(), 2);
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
    BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[1], handler.mSize[1] - 1), std::string(buffer2));
  }
  {
    // Decode a long frame which is split in two, after the first byte.
    char* buffer = strdup("string with more than 127 characters: cdsklcmalkmc cdmslkc adslkccmkadsc adslkmc dsa ckdls cdksclknds lkndnc anslkc klsad ckl lksad clkas ccdascnkjancjnjkascsa cdascds clsad nclksad ncklsd clkadns lkc sadnlk cklsa cnaksld csad");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);

    for (size_t i = 1; i < strlen(buffer); ++i) {
      char buffer1[1024];
      char buffer2[1024];
      memset(buffer1, 0xfa, 1024);
      memset(buffer2, 0xfb, 1024);
      memcpy(buffer1, encoded[0].base, i);
      memcpy(buffer2, encoded[0].base + i, encoded[0].len - i);
      TestWSHandler handler;
      decode_websocket(buffer1, i, handler);
      decode_websocket(buffer2, encoded[0].len - i, handler);
      BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 1);
      BOOST_REQUIRE_EQUAL(handler.mSize.size(), 1);
      BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
    }
  }
  {
    // Decode a long frame which is split in two, after the first byte.
    char* buffer = strdup("string with more than 127 characters: cdsklcmalkmc cdmslkc adslkccmkadsc adslkmc dsa ckdls cdksclknds lkndnc anslkc klsad ckl lksad clkas ccdascnkjancjnjkascsa cdascds clsad nclksad ncklsd clkadns lkc sadnlk cklsa cnaksld csad");
    std::vector<uv_buf_t> encoded;
    encode_websocket_frames(encoded, buffer, strlen(buffer) + 1, WebSocketOpCode::Binary, 0);
    BOOST_REQUIRE_EQUAL(encoded.size(), 1);

    for (size_t i = 0; i < strlen(buffer) - 1; ++i) {
      for (size_t j = i + 1; j < strlen(buffer); ++j) {
        char buffer1[1024];
        char buffer2[1024];
        char buffer3[1024];
        memset(buffer1, 0xfa, 1024);
        memset(buffer2, 0xfb, 1024);
        memset(buffer3, 0xfc, 1024);
        memcpy(buffer1, encoded[0].base, i);
        memcpy(buffer2, encoded[0].base + i, (j - i));
        memcpy(buffer3, encoded[0].base + j, encoded[0].len - j);
        TestWSHandler handler;
        decode_websocket(buffer1, i, handler);
        BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 0);
        decode_websocket(buffer2, (j - i), handler);
        BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 0);
        decode_websocket(buffer3, encoded[0].len - j, handler);
        BOOST_REQUIRE_EQUAL(handler.mFrame.size(), 1);
        BOOST_REQUIRE_EQUAL(handler.mSize.size(), 1);
        BOOST_REQUIRE_EQUAL(std::string(handler.mFrame[0], handler.mSize[0] - 1), std::string(buffer));
      }
    }
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
      "Access-Control-Allow-Origin: \"*\"\r\n"
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
