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

#include "Framework/FairMQDeviceProxy.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/Channel.h>
#include <fairmq/FairMQMessage.h>

namespace o2::framework
{

ChannelIndex FairMQDeviceProxy::getChannelIndex(RouteIndex index) const
{
  assert(mRoutes.size());
  assert(index.value < mRoutes.size());
  assert(mRoutes[index.value].channel.value != -1);
  assert(mChannels.size());
  assert(mRoutes[index.value].channel.value < mChannels.size());
  return mRoutes[index.value].channel;
}

fair::mq::Channel* FairMQDeviceProxy::getChannel(ChannelIndex index) const
{
  assert(mChannels.size());
  assert(index.value < mChannels.size());
  return mChannels[index.value];
}

FairMQTransportFactory* FairMQDeviceProxy::getTransport(RouteIndex index) const
{
  auto transport = getChannel(getChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createMessage(RouteIndex routeIndex) const
{
  return getTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createMessage(RouteIndex routeIndex, const size_t size) const
{
  return getTransport(routeIndex)->CreateMessage(size, fair::mq::Alignment{64});
}

void FairMQDeviceProxy::bindRoutes(std::vector<OutputRoute> const& outputs, fair::mq::Device& device)
{
  mRoutes.reserve(outputs.size());
  size_t ri = 0;
  std::unordered_map<std::string, ChannelIndex> channelNameToChannel;
  for (auto& route : outputs) {
    // If the channel is not yet registered, register it.
    // If the channel is already registered, use the existing index.
    auto channelPos = channelNameToChannel.find(route.channel);
    ChannelIndex channelIndex;

    if (channelPos == channelNameToChannel.end()) {
      channelIndex = ChannelIndex{(int)mChannels.size()};
      mChannels.push_back(&device.fChannels.at(route.channel).at(0));
      channelNameToChannel[route.channel] = channelIndex;
      LOGP(debug, "Binding channel {} to channel index {}", route.channel, channelIndex.value);
    } else {
      LOGP(debug, "Using index {} for channel {}", channelPos->second.value, route.channel);
      channelIndex = channelPos->second;
    }
    LOGP(debug, "Binding route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, route.maxTimeslices, ri, channelIndex.value);
    mRoutes.emplace_back(RouteState{channelIndex, false});
    ri++;
  }
  for (auto& route : mRoutes) {
    assert(route.channel.value != -1);
    assert(route.channel.value < mChannels.size());
  }
  LOGP(debug, "Total channels found {}, total routes {}", mChannels.size(), mRoutes.size());
  assert(mRoutes.size() == outputs.size());
}
} // namespace o2::framework
