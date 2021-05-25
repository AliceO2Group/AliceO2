// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMPUBLISHCHANNELHELPER_H
#define O2_SIMPUBLISHCHANNELHELPER_H

#include <string>
#include <FairMQChannel.h>

namespace o2::simpubsub
{

// create an IPC socket name of the type
// ipc:///tmp/base-PID
// base should be for example "o2sim-worker" or "o2sim-merger"
std::string getPublishAddress(std::string const& base, int pid = getpid())
{
  std::stringstream publishsocketname;
  publishsocketname << "ipc:///tmp/" << base << "-" << pid;
  return publishsocketname.str();
}

// some standard format for pub-sub subscribers
std::string simStatusString(std::string const& origin, std::string const& topic, std::string const& message)
{
  return origin + std::string("[") + topic + std::string("] : ") + message;
}

// helper function to publish a message to an outside subscriber
bool publishMessage(FairMQChannel& channel, std::string const& message)
{
  if (channel.IsValid()) {
    auto text = new std::string(message);
    std::unique_ptr<FairMQMessage> payload(channel.NewMessage(
      const_cast<char*>(text->c_str()),
      text->length(), [](void* data, void* hint) { delete static_cast<std::string*>(hint); }, text));
    if (channel.Send(payload) > 0) {
      return true;
    }
  } else {
    LOG(ERROR) << "CHANNEL NOT VALID";
  }
  return false;
}

// make channel (transport factory needs to be injected)
FairMQChannel createPUBChannel(std::string const& address,
                               std::string const& type = "pub")
{
  auto factory = FairMQTransportFactory::CreateTransportFactory("zeromq");
  static int i = 0;
  std::stringstream str;
  str << "channel" << i++;
  auto externalpublishchannel = FairMQChannel{str.str(), type, factory};
  externalpublishchannel.Init();
  if ((type.compare("pub") == 0) || (type.compare("pull") == 0)) {
    externalpublishchannel.Bind(address);
    LOG(INFO) << "BINDING TO ADDRESS " << address << " type " << type;
  } else {
    externalpublishchannel.Connect(address);
    LOG(INFO) << "CONNECTING TO ADDRESS " << address << " type " << type;
  }
  externalpublishchannel.Validate();
  return externalpublishchannel;
}

} // namespace o2::simpubsub

#endif //O2_SIMPUBLISHCHANNELHELPER_H
