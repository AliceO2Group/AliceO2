// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HTTPParser.h"
#include "Framework/RuntimeError.h"
#include <string_view>
#include "SHA1.h"
#include "Base64.h"
#include <regex>

using namespace o2::framework::internal;
namespace o2::framework
{

namespace
{
/// WebSocket RFC requires XOR masking from the client
void memmask(char* dst, char const* src, size_t size, uint32_t mask)
{
  if (mask) {
    char* m = (char*)&mask;
    for (size_t len = 0; len < size; ++len) {
      *dst++ = *src++ ^ m[len % 4];
    }
  } else {
    memcpy(dst, src, size);
  }
}

void memunmask(char* data, size_t size, uint32_t mask)
{
  char* m = (char*)&mask;
  for (size_t len = 0; len < size; ++len) {
    *data++ ^= m[len % 4];
  }
}
} // namespace

void encode_websocket_frames(std::vector<uv_buf_t>& outputs, char const* src, size_t size, WebSocketOpCode opcode, uint32_t mask)
{
  void* finalHeader;
  size_t headerSize;
  char* buffer = nullptr;
  char* startPayload = nullptr;
  std::vector<uv_buf_t> result;

  if (size < 126) {
    headerSize = sizeof(WebSocketFrameTiny);
    buffer = (char*)malloc(headerSize + size);
    WebSocketFrameTiny* header = (WebSocketFrameTiny*)buffer;
    memset(buffer, 0, headerSize);
    header->len = size;
  } else if (size < 1 << 16) {
    headerSize = sizeof(WebSocketFrameShort);
    buffer = (char*)malloc(headerSize + size);
    WebSocketFrameShort* header = (WebSocketFrameShort*)buffer;
    memset(buffer, 0, headerSize);
    header->len = 126;
    header->len16 = size;
  } else {
    headerSize = sizeof(WebSocketFrameHuge);
    buffer = (char*)malloc(headerSize + size);
    WebSocketFrameHuge* header;
    memset(buffer, 0, headerSize);
    header->len = 127;
    header->len64 = size;
  }
  size_t fullHeaderSize = (mask ? 4 : 0) + headerSize;
  startPayload = buffer + fullHeaderSize;
  WebSocketFrameTiny* header = (WebSocketFrameTiny*)buffer;
  header->fin = 1;
  header->opcode = (unsigned char)opcode; // binary or text for now
  // Mask is right before payload.
  if (mask) {
    *((uint32_t*)(startPayload - 4)) = mask;
  }
  header->mask = mask ? 1 : 0;
  memmask(startPayload, src, size, mask);
  outputs.push_back(uv_buf_init(buffer, size + fullHeaderSize));
}

void decode_websocket(char* start, size_t size, WebSocketHandler& handler)
{
  char* cur = start;
  handler.beginChunk();
  while (cur - start < size) {
    WebSocketFrameTiny* header = (WebSocketFrameTiny*)cur;
    size_t payloadSize = 0;
    size_t headerSize = 0;
    if (header->len < 126) {
      payloadSize = header->len;
      headerSize = 2 + (header->mask ? 4 : 0);
    } else if (header->len == 126) {
      WebSocketFrameShort* headerSmall = (WebSocketFrameShort*)cur;
      payloadSize = headerSmall->len16;
      headerSize = 2 + 2 + (header->mask ? 4 : 0);
    } else if (header->len == 127) {
      WebSocketFrameHuge* headerSmall = (WebSocketFrameHuge*)cur;
      payloadSize = headerSmall->len64;
      headerSize = 2 + 8 + (header->mask ? 4 : 0);
    }
    if (header->mask) {
      int32_t mask = *(int32_t*)(cur + headerSize - 4);
      memunmask(cur, payloadSize, mask);
    }
    handler.frame(cur + headerSize, payloadSize);
    cur += headerSize + payloadSize;
  }
  handler.endChunk();
}

std::string encode_websocket_handshake_request(const char* endpoint, const char* protocol, int version, char const* nonce,
                                               std::vector<std::pair<std::string, std::string>> headers)
{
  char const* res =
    "GET {} HTTP/1.1\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    "Sec-WebSocket-Key: {}\r\n"
    "Sec-WebSocket-Protocol: {}\r\n"
    "Sec-WebSocket-Version: {}\r\n"
    "{}\r\n";
  std::string encodedHeaders;
  for (auto [k, v] : headers) {
    encodedHeaders += std::string(fmt::format("{}: {}\r\n", k, v));
  }
  return fmt::format(res, endpoint, nonce, protocol, version, encodedHeaders);
}

std::string HTTPParserHelpers::calculateAccept(const char* nonce)
{
  std::string reply = std::string(nonce) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  char sha[20];
  SHA1(sha, reply.data(), reply.size());
  char base[64];
  base64_encode(base, 64, (unsigned char*)sha, 20);
  return fmt::format("{}", base);
}

std::string encode_websocket_handshake_reply(char const* nonce)
{
  char const* res =
    "HTTP/1.1 101 Switching Protocols\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    "Sec-WebSocket-Accept: {}\r\n\r\n";
  return fmt::format(res, HTTPParserHelpers::calculateAccept(nonce));
}

void parse_http_request(char* start, size_t size, HTTPParser* parser)
{
  enum HTTPState state = HTTPState::IN_START;
  // Too short, let's try again...
  if (size < 2) {
    parser->remaining += std::string_view(start, size);
  }
  char* cur = start;
  char* next = cur;
  std::string_view lastToken(cur, 0);
  std::string_view lastKey;
  std::string_view lastValue;
  std::string lastError;
  if (parser->states.empty()) {
    parser->states.push_back(HTTPState::IN_START);
  }
  char const* delimiters = nullptr;
  char const* skippable = nullptr;
  char const* separator = nullptr;
  char const* spaces = "\t \v";
  char const* colon = ":";
  char const* newline = "\r\n";
  bool done = false;

  while (!done) {
    HTTPState state = parser->states.back();
    parser->states.pop_back();
    switch (state) {
      case HTTPState::IN_START:
        parser->states.push_back(HTTPState::BEGIN_METHOD);
        break;
      case HTTPState::IN_START_REPLY:
        parser->states.push_back(HTTPState::BEGIN_REPLY_VERSION);
        break;
      case HTTPState::BEGIN_REPLY_VERSION:
        parser->states.push_back(HTTPState::END_REPLY_VERSION);
        parser->states.push_back(HTTPState::IN_CAPTURE_DELIMITERS);
        delimiters = spaces;
        break;
      case HTTPState::END_REPLY_VERSION:
        parser->replyVersion(lastToken);
        parser->states.push_back(HTTPState::BEGIN_REPLY_CODE);
        parser->states.push_back(HTTPState::IN_SKIP_CHARS);
        skippable = spaces;
        break;
      case HTTPState::BEGIN_REPLY_CODE:
        parser->states.push_back(HTTPState::END_REPLY_CODE);
        parser->states.push_back(HTTPState::IN_CAPTURE_DELIMITERS);
        delimiters = spaces;
        break;
      case HTTPState::END_REPLY_CODE:
        parser->replyCode(lastToken);
        parser->states.push_back(HTTPState::BEGIN_REPLY_MESSAGE);
        parser->states.push_back(HTTPState::IN_SKIP_CHARS);
        skippable = spaces;
        break;
      case HTTPState::BEGIN_REPLY_MESSAGE:
        parser->states.push_back(HTTPState::END_REPLY_MESSAGE);
        parser->states.push_back(HTTPState::IN_CAPTURE_SEPARATOR);
        separator = newline;
        break;
      case HTTPState::END_REPLY_MESSAGE:
        parser->replyMessage(lastToken);
        parser->states.push_back(HTTPState::BEGIN_HEADERS);
        break;
      case HTTPState::BEGIN_METHOD:
        parser->states.push_back(HTTPState::END_METHOD);
        parser->states.push_back(HTTPState::IN_CAPTURE_DELIMITERS);
        delimiters = spaces;
        break;
      case HTTPState::END_METHOD:
        parser->method(lastToken);
        parser->states.push_back(HTTPState::BEGIN_TARGET);
        parser->states.push_back(HTTPState::IN_SKIP_CHARS);
        skippable = spaces;
        break;
      case HTTPState::BEGIN_TARGET:
        parser->states.push_back(HTTPState::END_TARGET);
        parser->states.push_back(HTTPState::IN_CAPTURE_DELIMITERS);
        delimiters = spaces;
        break;
      case HTTPState::END_TARGET:
        parser->target(lastToken);
        parser->states.push_back(HTTPState::BEGIN_VERSION);
        parser->states.push_back(HTTPState::IN_SKIP_CHARS);
        skippable = spaces;
        break;
      case HTTPState::BEGIN_VERSION:
        parser->states.push_back(HTTPState::END_VERSION);
        parser->states.push_back(HTTPState::IN_CAPTURE_SEPARATOR);
        separator = newline;
        break;
      case HTTPState::END_VERSION:
        parser->version(lastToken);
        parser->states.push_back(HTTPState::BEGIN_HEADERS);
        break;
      case HTTPState::BEGIN_HEADERS:
        parser->states.push_back(HTTPState::BEGIN_HEADER);
        break;
      case HTTPState::BEGIN_HEADER:
        parser->states.push_back(HTTPState::BEGIN_HEADER_KEY);
        break;
      case HTTPState::BEGIN_HEADER_KEY:
        parser->states.push_back(HTTPState::END_HEADER_KEY);
        parser->states.push_back(HTTPState::IN_CAPTURE_SEPARATOR);
        separator = colon;
        break;
      case HTTPState::END_HEADER_KEY:
        lastKey = lastToken;
        parser->states.push_back(HTTPState::BEGIN_HEADER_VALUE);
        parser->states.push_back(HTTPState::IN_SKIP_CHARS);
        skippable = spaces;
        break;
      case HTTPState::BEGIN_HEADER_VALUE:
        parser->states.push_back(HTTPState::END_HEADER_VALUE);
        parser->states.push_back(HTTPState::IN_CAPTURE_SEPARATOR);
        separator = newline;
        break;
      case HTTPState::END_HEADER_VALUE:
        lastValue = lastToken;
        parser->states.push_back(HTTPState::END_HEADER);
        break;
      case HTTPState::END_HEADER:
        if (strncmp("\r\n", next, 2) == 0) {
          parser->header(lastKey, lastValue);
          parser->states.push_back(HTTPState::END_HEADERS);
          next += 2;
          cur = next;
        } else {
          parser->header(lastKey, lastValue);
          parser->states.push_back(HTTPState::BEGIN_HEADER);
          cur = next;
        }
        break;
      case HTTPState::END_HEADERS:
        parser->endHeaders();
        parser->states.push_back(HTTPState::BEGIN_BODY);
        break;
      case HTTPState::BEGIN_BODY: {
        size_t bodySize = size - (cur - start);
        parser->body(cur, bodySize);
        next = cur + bodySize;
        cur = next;
        parser->states.push_back(HTTPState::BEGIN_BODY);
        parser->states.push_back(HTTPState::IN_DONE);
      } break;
      case HTTPState::IN_SKIP_CHARS:
        while (true) {
          if (next - start == size) {
            parser->remaining += std::string_view(cur, next - cur);
          }
          if (strchr(skippable, *next)) {
            next++;
            continue;
          }
          cur = next;
          break;
        }
        break;
      case HTTPState::IN_SEPARATOR:
        if (memcmp(separator, cur, strlen(separator)) != 0) {
          parser->states.push_back(HTTPState::IN_ERROR);
          break;
        }
        next += strlen(separator);
        cur = next;
        break;
      case HTTPState::IN_CAPTURE_DELIMITERS:
        while (true) {
          if (next - start == size) {
            parser->remaining += std::string_view(cur, next - cur);
          }
          if (strchr(delimiters, *next) == nullptr) {
            next++;
            continue;
          }
          lastToken = std::string_view(cur, next - cur);
          cur = next;
          break;
        }
        break;
      case HTTPState::IN_CAPTURE_SEPARATOR:
        while (true) {
          if (next + strlen(separator) - start == size) {
            parser->remaining += std::string_view(cur, next - cur);
          }
          if (memcmp(separator, next, strlen(separator)) != 0) {
            next++;
            continue;
          }
          lastToken = std::string_view(cur, next - cur);
          next += strlen(separator);
          cur = next;
          break;
        }
        break;
      case HTTPState::IN_DONE:
        // The only case in which there can be a pending state when IN_DONE, is if
        // we plan to resume processing.
        if (parser->states.size() == 1 && parser->states.back() == HTTPState::BEGIN_BODY) {
          done = true;
        } else if (parser->states.empty()) {
          done = true;
        } else {
          parser->states.push_back(HTTPState::IN_ERROR);
        }
        break;
      case HTTPState::IN_ERROR:
        parser->error = lastError;
        parser->states.clear();
        done = true;
        break;
      default:
        parser->states.push_back(HTTPState::IN_ERROR);
        break;
    }
  }
}

std::pair<std::string, unsigned short> parse_websocket_url(char const* url)
{
  std::string s = url;
  if (s == "ws://") {
    s = "ws://127.0.0.1:8080";
  }
  const std::regex urlMatcher("^ws://([0-9-_.]+)[:]([0-9]+)$");
  std::smatch parts;
  if (!std::regex_match(s, parts, urlMatcher)) {
    throw runtime_error_f(
      "Unable to parse driver client url: %s.\n"
      "Format should be ws://[<driver ip>:<port>] e.g. ws://127.0.0.1:8080 or just ws://");
  }
  std::string ip = std::string{parts[1]};
  auto portS = std::string(parts[2]);
  unsigned short port = std::stoul(portS);
  return {ip, port};
}
} // namespace o2::framework
