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

#ifndef O2_SIMPUBLISHCHANNELHELPER_H
#define O2_SIMPUBLISHCHANNELHELPER_H

#include <string>
#include <fairmq/Channel.h>
#include <fairmq/Message.h>
#include <fairmq/TransportFactory.h>

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
bool publishMessage(fair::mq::Channel& channel, std::string const& message)
{
  if (channel.IsValid()) {
    auto text = new std::string(message);
    std::unique_ptr<fair::mq::Message> payload(channel.NewMessage(
      const_cast<char*>(text->c_str()),
      text->length(), [](void* data, void* hint) { delete static_cast<std::string*>(hint); }, text));
    if (channel.Send(payload) > 0) {
      return true;
    }
  } else {
    LOG(error) << "CHANNEL NOT VALID";
  }
  return false;
}

// make channel (transport factory needs to be injected)
fair::mq::Channel createPUBChannel(std::string const& address,
                                   std::string const& type = "pub")
{
  auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  static int i = 0;
  std::stringstream str;
  str << "channel" << i++;
  auto externalpublishchannel = fair::mq::Channel{str.str(), type, factory};
  externalpublishchannel.Init();
  if ((type.compare("pub") == 0) || (type.compare("pull") == 0)) {
    externalpublishchannel.Bind(address);
    LOG(info) << "BINDING TO ADDRESS " << address << " type " << type;
  } else {
    externalpublishchannel.Connect(address);
    LOG(info) << "CONNECTING TO ADDRESS " << address << " type " << type;
  }
  externalpublishchannel.Validate();
  return externalpublishchannel;
}

} // namespace o2::simpubsub

#endif //O2_SIMPUBLISHCHANNELHELPER_H
