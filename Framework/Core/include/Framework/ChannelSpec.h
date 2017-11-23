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

namespace o2 {
namespace framework {

enum ChannelMethod {
  Bind,
  Connect
};

enum ChannelType {
  Pub,
  Sub
};

/// This describes an input channel. Since they are point to 
/// point connections, there is not much to say about them.
/// Notice that their name is always of the kind in_<output-channel name>
struct InputChannelSpec {
  std::string name;
  enum ChannelType type;
  enum ChannelMethod method;
  unsigned short port;
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
  unsigned short port;
  size_t listeners;
};

}
}

#endif // FRAMEWORK_CHANNELSPEC_H
