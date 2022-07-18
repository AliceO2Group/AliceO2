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
#include "ChannelSpecHelpers.h"
#include "Framework/RuntimeError.h"
#include <fmt/format.h>
#include <ostream>
#include <cassert>
#include <cctype>
#include <regex>
#include <unistd.h>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace
{
std::string getTmpFolder()
{
  std::string tmppath = fs::temp_directory_path().native();
  while (tmppath.back() == '/') {
    tmppath.pop_back();
  }
  return tmppath;
}
} // namespace

namespace o2::framework
{

void OutputChannelSpecConfigParser::beginChannel()
{
  specs.push_back(OutputChannelSpec{});
}

void OutputChannelSpecConfigParser::endChannel()
{
}

bool isIPAddress(const std::string& address)
{
  std::regex ipv4_regex("^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$");
  if (std::regex_match(address, ipv4_regex)) {
    return true;
  }
  return false;
}

void OutputChannelSpecConfigParser::property(std::string_view key, std::string_view value)
{
  std::string valueStr = value.data();
  if (key == "address") {
    auto parseAddress = [v = std::string(value), &outputChannelSpec = specs.back()]() {
      auto value = v;
      std::string protocol = "tcp";
      std::string hostname = "127.0.0.1";
      std::string port = "9090";
      auto pos = value.find("://");
      if (pos != std::string::npos) {
        protocol = value.substr(0, pos);
        value = value.substr(pos + 3);
      }
      if (protocol == "tcp") {
        pos = value.find(':');
        if (pos != std::string::npos) {
          hostname = value.substr(0, pos);
          value = value.substr(pos + 1);
        } else {
          throw runtime_error_f("Port not found in address '%s'", v.c_str());
        }
        port = value;
        if (isIPAddress(hostname) == false) {
          throw runtime_error_f("Invalid ip address '%s'", hostname.c_str());
        }
        outputChannelSpec.hostname = hostname;
        outputChannelSpec.port = std::stoi(port);
        outputChannelSpec.protocol = ChannelProtocol::Network;
      } else if (protocol == "ipc") {
        outputChannelSpec.hostname = value;
        outputChannelSpec.port = 0;
        outputChannelSpec.protocol = ChannelProtocol::IPC;
      } else {
        throw runtime_error_f("Unknown protocol '%s'", protocol.c_str());
      }
    };
    parseAddress();
  }
  auto& outputChannelSpec = specs.back();
  if (key == "name") {
    outputChannelSpec.name = value;
  } else if (key == "type" && value == "pub") {
    outputChannelSpec.type = ChannelType::Pub;
  } else if (key == "type" && value == "sub") {
    outputChannelSpec.type = ChannelType::Sub;
  } else if (key == "type" && value == "push") {
    outputChannelSpec.type = ChannelType::Push;
  } else if (key == "type" && value == "pull") {
    outputChannelSpec.type = ChannelType::Pull;
  } else if (key == "type" && value == "pair") {
    outputChannelSpec.type = ChannelType::Pair;
  } else if (key == "method" && value == "bind") {
    outputChannelSpec.method = ChannelMethod::Bind;
  } else if (key == "method" && value == "connect") {
    outputChannelSpec.method = ChannelMethod::Connect;
  } else if (key == "rateLogging") {
    outputChannelSpec.rateLogging = std::stoi(valueStr);
  } else if (key == "recvBufSize") {
    outputChannelSpec.recvBufferSize = std::stoi(valueStr);
  } else if (key == "sendBufSize") {
    outputChannelSpec.recvBufferSize = std::stoi(valueStr);
  }
}

void OutputChannelSpecConfigParser::error()
{
  throw runtime_error_f("Error in channel config.");
}

char const* ChannelSpecHelpers::typeAsString(enum ChannelType type)
{
  switch (type) {
    case ChannelType::Pub:
      return "pub";
    case ChannelType::Sub:
      return "sub";
    case ChannelType::Push:
      return "push";
    case ChannelType::Pull:
      return "pull";
    case ChannelType::Pair:
      return "pair";
  }
  throw runtime_error("Unknown ChannelType");
}

char const* ChannelSpecHelpers::methodAsString(enum ChannelMethod method)
{
  switch (method) {
    case ChannelMethod::Bind:
      return "bind";
    case ChannelMethod::Connect:
      return "connect";
  }
  throw runtime_error("Unknown ChannelMethod");
}

namespace
{
std::string composeIPCName(std::string const& prefix, std::string const& hostname, short port)
{
  if (prefix.empty() == false && prefix[0] == '@') {
    return fmt::format("ipc://{}{}_{},transport=shmem", prefix, hostname, port);
  }
  if (prefix.empty() == false && prefix.back() == '/') {
    return fmt::format("ipc://{}{}_{},transport=shmem", prefix, hostname, port);
  }
  return fmt::format("ipc://{}/{}_{},transport=shmem", prefix, hostname, port);
}
} // namespace

std::string ChannelSpecHelpers::channelUrl(OutputChannelSpec const& channel)
{
  switch (channel.protocol) {
    case ChannelProtocol::IPC:
      return composeIPCName(channel.ipcPrefix, channel.hostname, channel.port);
    default:
      return channel.method == ChannelMethod::Bind ? fmt::format("tcp://*:{}", channel.port)
                                                   : fmt::format("tcp://{}:{}", channel.hostname, channel.port);
  }
}

std::string ChannelSpecHelpers::channelUrl(InputChannelSpec const& channel)
{
  switch (channel.protocol) {
    case ChannelProtocol::IPC:
      return composeIPCName(channel.ipcPrefix, channel.hostname, channel.port);
    default:
      return channel.method == ChannelMethod::Bind ? fmt::format("tcp://*:{}", channel.port)
                                                   : fmt::format("tcp://{}:{}", channel.hostname, channel.port);
  }
}

/// Stream operators so that we can use ChannelType with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelType const& type)
{
  s << ChannelSpecHelpers::typeAsString(type);
  return s;
}

/// Stream operators so that we can use ChannelString with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelMethod const& method)
{
  s << ChannelSpecHelpers::methodAsString(method);
  return s;
}

enum struct ChannelConfigParserState {
  BEGIN,
  BEGIN_CHANNEL,
  END_CHANNEL,
  BEGIN_KEY,
  BEGIN_VALUE,
  END_VALUE,
  END,
  ERROR
};

void ChannelSpecHelpers::parseChannelConfig(char const* config, FairMQChannelConfigParser& handler)
{
  ChannelConfigParserState state = ChannelConfigParserState::BEGIN;
  char const* cur = config;
  char const* next = config;
  std::string_view key;
  std::string_view value;
  char const* nameKey = "name";
  char const* lastError = "bad configuation string";

  while (true) {
    switch (state) {
      case ChannelConfigParserState::BEGIN: {
        if (*cur == '\0') {
          lastError = "empty config string";
          state = ChannelConfigParserState::ERROR;
        } else if (!isalpha(*cur)) {
          lastError = "first character is not alphabetic";
          state = ChannelConfigParserState::ERROR;
        } else {
          state = ChannelConfigParserState::BEGIN_CHANNEL;
        }
        break;
      }
      case ChannelConfigParserState::BEGIN_CHANNEL: {
        next = strpbrk(cur, ":=;,");
        if (*next == ';' || *next == ',') {
          lastError = "expected channel name";
          state = ChannelConfigParserState::ERROR;
          break;
        } else if (*next == ':') {
          handler.beginChannel();
          key = std::string_view(nameKey, 4);
          value = std::string_view(cur, next - cur);
          handler.property(key, value);
          state = ChannelConfigParserState::BEGIN_KEY;
          cur = next + 1;
          if (*cur == '\0') {
            state = ChannelConfigParserState::END_CHANNEL;
          } else {
            state = ChannelConfigParserState::BEGIN_KEY;
          }
          break;
        }
        handler.beginChannel();
        state = ChannelConfigParserState::BEGIN_KEY;
        break;
      }
      case ChannelConfigParserState::BEGIN_KEY: {
        next = strchr(cur, '=');
        if (next == nullptr) {
          lastError = "expected '='";
          state = ChannelConfigParserState::ERROR;
        } else {
          key = std::string_view(cur, next - cur);
          state = ChannelConfigParserState::BEGIN_VALUE;
          cur = next + 1;
        }
        break;
      }
      case ChannelConfigParserState::BEGIN_VALUE: {
        next = strpbrk(cur, ";,");
        if (next == nullptr) {
          size_t l = strlen(cur);
          value = std::string_view(cur, l);
          state = ChannelConfigParserState::END_VALUE;
          cur = cur + l;
        } else if (*next == ';') {
          value = std::string_view(cur, next - cur);
          state = ChannelConfigParserState::END_CHANNEL;
          cur = next;
        } else if (*next == ',') {
          value = std::string_view(cur, next - cur);
          state = ChannelConfigParserState::END_VALUE;
          cur = next;
        }
        handler.property(key, value);
        break;
      }
      case ChannelConfigParserState::END_VALUE: {
        if (*cur == '\0') {
          state = ChannelConfigParserState::END_CHANNEL;
        } else if (*cur == ',') {
          state = ChannelConfigParserState::BEGIN_KEY;
          cur++;
        } else if (*cur == ';') {
          state = ChannelConfigParserState::END_CHANNEL;
          cur++;
        }
        break;
      }
      case ChannelConfigParserState::END_CHANNEL: {
        handler.endChannel();
        if (*cur == '\0') {
          state = ChannelConfigParserState::END;
        } else if (*cur == ';') {
          state = ChannelConfigParserState::BEGIN_CHANNEL;
          cur++;
        } else {
          lastError = "expected ';'";
          state = ChannelConfigParserState::ERROR;
        }
        break;
      }
      case ChannelConfigParserState::END: {
        return;
      }
      case ChannelConfigParserState::ERROR: {
        throw runtime_error_f("Unable to parse channel config: %s", lastError);
      }
    }
  }
}

std::string ChannelSpecHelpers::defaultIPCFolder()
{
#ifdef __linux__
  // On linux we can use abstract sockets to avoid the need for a file.
  // This is not available on macOS.
  // Notice also that when running inside a docker container, like
  // when on alien, the abstract socket is not isolated, so we need
  // to add some unique identifier to avoid collisions.
  char const* channelPrefix = getenv("ALIEN_PROC_ID");
  if (channelPrefix) {
    return fmt::format("@dpl_{}_", channelPrefix);
  }
  return "@";
#else
  /// Find out a place where we can write the sockets
  char const* channelPrefix = getenv("TMPDIR");
  if (channelPrefix) {
    return {channelPrefix};
  }
  return access("/tmp", W_OK) == 0 ? "/tmp/" : "./";
#endif
}

} // namespace o2::framework
