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
#include "Framework/DataSpecUtils.h"
#include "InputRouteHelpers.h"
#include "Framework/DataProcessingHeader.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/Channel.h>
#include <fairmq/FairMQMessage.h>

namespace o2::framework
{

ChannelIndex FairMQDeviceProxy::getOutputChannelIndex(RouteIndex index) const
{
  assert(mOutputRoutes.size());
  assert(index.value < mOutputRoutes.size());
  assert(mOutputRoutes[index.value].channel.value != -1);
  assert(mOutputChannels.size());
  assert(mOutputRoutes[index.value].channel.value < mOutputChannels.size());
  return mOutputRoutes[index.value].channel;
}

ChannelIndex FairMQDeviceProxy::getInputChannelIndex(RouteIndex index) const
{
  assert(mInputRoutes.size());
  assert(index.value < mInputRoutes.size());
  assert(mInputRoutes[index.value].channel.value != -1);
  assert(mInputChannels.size());
  assert(mInputRoutes[index.value].channel.value < mInputChannels.size());
  return mInputRoutes[index.value].channel;
}

fair::mq::Channel* FairMQDeviceProxy::getOutputChannel(ChannelIndex index) const
{
  assert(mOutputChannels.size());
  assert(index.value < mOutputChannels.size());
  return mOutputChannels[index.value];
}

fair::mq::Channel* FairMQDeviceProxy::getInputChannel(ChannelIndex index) const
{
  assert(mInputChannels.size());
  assert(index.value < mInputChannels.size());
  return mInputChannels[index.value];
}

ChannelIndex FairMQDeviceProxy::getOutputChannelIndex(OutputSpec const& query, size_t timeslice) const
{
  assert(mOutputRoutes.size() == mOutputs.size());
  for (size_t ri = 0; ri < mOutputs.size(); ++ri) {
    auto& route = mOutputs[ri];

    LOG(debug) << "matching: " << DataSpecUtils::describe(query) << " to route " << DataSpecUtils::describe(route.matcher);
    if (DataSpecUtils::match(route.matcher, query) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
      return mOutputRoutes[ri].channel;
    }
  }
  return ChannelIndex{ChannelIndex::INVALID};
};

ChannelIndex FairMQDeviceProxy::getOutputChannelIndexByName(std::string const& name) const
{
  for (int i = 0; i < mOutputChannels.size(); i++) {
    if (mOutputChannelNames[i] == name) {
      return {i};
    }
  }
  return {ChannelIndex::INVALID};
}

ChannelIndex FairMQDeviceProxy::getInputChannelIndexByName(std::string const& name) const
{
  for (int i = 0; i < mInputChannelNames.size(); i++) {
    if (mInputChannelNames[i] == name) {
      return {i};
    }
  }
  return {ChannelIndex::INVALID};
}

FairMQTransportFactory* FairMQDeviceProxy::getOutputTransport(RouteIndex index) const
{
  auto transport = getOutputChannel(getOutputChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

FairMQTransportFactory* FairMQDeviceProxy::getInputTransport(RouteIndex index) const
{
  auto transport = getInputChannel(getInputChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createOutputMessage(RouteIndex routeIndex) const
{
  return getOutputTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createOutputMessage(RouteIndex routeIndex, const size_t size) const
{
  return getOutputTransport(routeIndex)->CreateMessage(size, fair::mq::Alignment{64});
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createInputMessage(RouteIndex routeIndex) const
{
  return getInputTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createInputMessage(RouteIndex routeIndex, const size_t size) const
{
  return getInputTransport(routeIndex)->CreateMessage(size, fair::mq::Alignment{64});
}

void FairMQDeviceProxy::bind(std::vector<OutputRoute> const& outputs, std::vector<InputRoute> const& inputs, fair::mq::Device& device)
{
  mOutputs.clear();
  mOutputRoutes.clear();
  mOutputChannels.clear();
  mOutputChannelNames.clear();
  mInputs.clear();
  mInputRoutes.clear();
  mInputChannels.clear();
  mInputChannelNames.clear();
  {
    mOutputs = outputs;
    mOutputRoutes.reserve(outputs.size());
    size_t ri = 0;
    std::unordered_map<std::string, ChannelIndex> channelNameToChannel;
    for (auto& route : outputs) {
      // If the channel is not yet registered, register it.
      // If the channel is already registered, use the existing index.
      auto channelPos = channelNameToChannel.find(route.channel);
      ChannelIndex channelIndex;

      if (channelPos == channelNameToChannel.end()) {
        channelIndex = ChannelIndex{(int)mOutputChannels.size()};
        mOutputChannels.push_back(&device.fChannels.at(route.channel).at(0));
        mOutputChannelNames.push_back(route.channel);
        channelNameToChannel[route.channel] = channelIndex;
        LOGP(debug, "Binding channel {} to channel index {}", route.channel, channelIndex.value);
      } else {
        LOGP(debug, "Using index {} for channel {}", channelPos->second.value, route.channel);
        channelIndex = channelPos->second;
      }
      LOGP(debug, "Binding route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, route.maxTimeslices, ri, channelIndex.value);
      mOutputRoutes.emplace_back(RouteState{channelIndex, false});
      ri++;
    }
    for (auto& route : mOutputRoutes) {
      assert(route.channel.value != -1);
      assert(route.channel.value < mOutputChannels.size());
    }
    LOGP(debug, "Total channels found {}, total routes {}", mOutputChannels.size(), mOutputRoutes.size());
    assert(mOutputRoutes.size() == outputs.size());
  }

  {
    auto maxLanes = InputRouteHelpers::maxLanes(inputs);
    mInputs = inputs;
    mInputRoutes.reserve(inputs.size());
    size_t ri = 0;
    std::unordered_map<std::string, ChannelIndex> channelNameToChannel;
    for (auto& route : inputs) {
      // If the channel is not yet registered, register it.
      // If the channel is already registered, use the existing index.
      auto channelPos = channelNameToChannel.find(route.sourceChannel);
      ChannelIndex channelIndex;

      if (channelPos == channelNameToChannel.end()) {
        channelIndex = ChannelIndex{(int)mInputChannels.size()};
        mInputChannels.push_back(&device.fChannels.at(route.sourceChannel).at(0));
        mInputChannelNames.push_back(route.sourceChannel);
        channelNameToChannel[route.sourceChannel] = channelIndex;
        LOGP(debug, "Binding channel {} to channel index {}", route.sourceChannel, channelIndex.value);
      } else {
        LOGP(debug, "Using index {} for channel {}", channelPos->second.value, route.sourceChannel);
        channelIndex = channelPos->second;
      }
      LOGP(debug, "Binding route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, maxLanes, ri, channelIndex.value);
      mInputRoutes.emplace_back(RouteState{channelIndex, false});
      ri++;
    }
    for (auto& route : mInputRoutes) {
      assert(route.channel.value != -1);
      assert(route.channel.value < mInputChannels.size());
    }
    LOGP(debug, "Total input channels found {}, total routes {}", mInputChannels.size(), mInputRoutes.size());
    assert(mInputRoutes.size() == inputs.size());
  }
  mStateChangeCallback = [&device]() -> bool { return device.NewStatePending(); };
}
} // namespace o2::framework
