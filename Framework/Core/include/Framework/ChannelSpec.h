// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CHANNELSPEC_H
#define FRAMEWORK_CHANNELSPEC_H

#include <string>

namespace o2
{
namespace framework
{

/// These map to zeromq connection
/// methods.
enum struct ChannelMethod {
  Bind,
  Connect
};

/// These map to zeromq types for the channels.
enum struct ChannelType {
  Pub,
  Sub,
  Push,
  Pull,
};

/// The kind of backend to use for the channels
enum struct ChannelProtocol {
  Network,
  IPC
};

/// This describes an input channel. Since they are point to
/// point connections, there is not much to say about them.
/// Notice that this should be considered read only once it
/// has been created.
struct InputChannelSpec {
  std::string name;
  enum ChannelType type;
  enum ChannelMethod method;
  std::string hostname;
  unsigned short port;
  ChannelProtocol protocol = ChannelProtocol::Network;
};

/// This describes an output channel. Output channels are semantically
/// different from input channels, because we use subChannels to
/// distinguish between different consumers. Notice that the number of
/// subchannels is actually determined by the number of time pipelined
/// consumers downstream.
struct OutputChannelSpec {
  std::string name;
  enum ChannelType type;
  enum ChannelMethod method;
  std::string hostname;
  unsigned short port;
  size_t listeners;
  ChannelProtocol protocol = ChannelProtocol::Network;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_CHANNELSPEC_H
