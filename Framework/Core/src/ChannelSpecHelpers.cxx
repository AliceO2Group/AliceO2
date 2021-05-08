// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ChannelSpecHelpers.h"
#include "Framework/RuntimeError.h"
#include <fmt/format.h>
#include <ostream>
#include <cassert>
#if 0
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<filesystem>)
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
  if (prefix == "@") {
    return fmt::format("ipc://@{}_{},transport=shmem", hostname, port);
  }
  if (prefix.back() == '/') {
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

} // namespace o2::framework
