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

  while (true) {
    switch (state) {
      case ChannelConfigParserState::BEGIN: {
        if (*cur == '\0') {
          state = ChannelConfigParserState::ERROR;
        } else if (!isalpha(*cur)) {
          state = ChannelConfigParserState::ERROR;
        } else {
          state = ChannelConfigParserState::BEGIN_CHANNEL;
        }
        break;
      }
      case ChannelConfigParserState::BEGIN_CHANNEL: {
        next = strpbrk(cur, ":=;,");
        if (*next == ';' || *next == ',') {
          state = ChannelConfigParserState::ERROR;
          break;
        } else if (*next == ':') {
          handler.beginChannel();
          key = std::string_view(nameKey, 4);
          value = std::string_view(cur, next - cur);
          handler.property(key, value);
          state = ChannelConfigParserState::BEGIN_KEY;
          cur = next + 1;
          break;
        }
        handler.beginChannel();
        state = ChannelConfigParserState::BEGIN_KEY;
        break;
      }
      case ChannelConfigParserState::BEGIN_KEY: {
        next = strchr(cur, '=');
        if (next == nullptr) {
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
          state = ChannelConfigParserState::ERROR;
        }
        break;
      }
      case ChannelConfigParserState::END: {
        return;
      }
      case ChannelConfigParserState::ERROR: {
        throw runtime_error("Unable to parse channel config");
      }
    }
  }
}

} // namespace o2::framework
