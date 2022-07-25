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
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"

#include <fairmq/Channel.h>
#include <fairmq/Device.h>
#include <fairmq/Message.h>
#include <fairmq/TransportFactory.h>

namespace o2::framework
{

ChannelIndex FairMQDeviceProxy::getOutputChannelIndex(RouteIndex index) const
{
  assert(mOutputRoutes.size());
  assert(index.value < mOutputRoutes.size());
  assert(mOutputRoutes[index.value].channel.value != -1);
  assert(mOutputChannelInfos.size());
  assert(mOutputRoutes[index.value].channel.value < mOutputChannelInfos.size());
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

ChannelIndex FairMQDeviceProxy::getForwardChannelIndex(RouteIndex index) const
{
  assert(mForwardRoutes.size());
  assert(index.value < mForwardRoutes.size());
  assert(mForwardRoutes[index.value].channel.value != -1);
  assert(mForwardChannelInfos.size());
  assert(mForwardRoutes[index.value].channel.value < mForwardChannelInfos.size());
  return mForwardRoutes[index.value].channel;
}

fair::mq::Channel* FairMQDeviceProxy::getOutputChannel(ChannelIndex index) const
{
  assert(mOutputChannelInfos.size());
  assert(index.value < mOutputChannelInfos.size());
  return &mOutputChannelInfos[index.value].channel;
}

OutputChannelInfo const& FairMQDeviceProxy::getOutputChannelInfo(ChannelIndex index) const
{
  assert(mOutputChannelInfos.size());
  assert(index.value < mOutputChannelInfos.size());
  return mOutputChannelInfos[index.value];
}

OutputChannelState& FairMQDeviceProxy::getOutputChannelState(ChannelIndex index)
{
  assert(mOutputChannelInfos.size());
  assert(index.value < mOutputChannelInfos.size());
  return mOutputChannelStates[index.value];
}

fair::mq::Channel* FairMQDeviceProxy::getInputChannel(ChannelIndex index) const
{
  assert(mInputChannels.size());
  assert(index.value < mInputChannels.size());
  return mInputChannels[index.value];
}

fair::mq::Channel* FairMQDeviceProxy::getForwardChannel(ChannelIndex index) const
{
  assert(mForwardChannelInfos.size());
  assert(index.value < mForwardChannelInfos.size());
  return &mForwardChannelInfos[index.value].channel;
}

ForwardChannelInfo const& FairMQDeviceProxy::getForwardChannelInfo(ChannelIndex index) const
{
  assert(mForwardChannelInfos.size());
  assert(index.value < mForwardChannelInfos.size());
  return mForwardChannelInfos[index.value];
}

ForwardChannelState& FairMQDeviceProxy::getForwardChannelState(ChannelIndex index)
{
  assert(mForwardChannelInfos.size());
  assert(index.value < mForwardChannelStates.size());
  return mForwardChannelStates[index.value];
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
}

ChannelIndex FairMQDeviceProxy::getForwardChannelIndex(header::DataHeader const& dh, size_t timeslice) const
{
  assert(mForwardRoutes.size() == mForwards.size());
  // Notice we need to match against a data header and not against
  // the InputMatcher, because an input might match something which
  // is then rerouted to two different output routes, depending on the content.
  for (size_t ri = 0; ri < mForwards.size(); ++ri) {
    auto& route = mForwards[ri];

    LOGP(debug, "matching: {} to route {}", dh, DataSpecUtils::describe(route.matcher));
    if (DataSpecUtils::match(route.matcher, dh.dataOrigin, dh.dataDescription, dh.subSpecification) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
      return mForwardRoutes[ri].channel;
    }
  }
  return ChannelIndex{ChannelIndex::INVALID};
}

ChannelIndex FairMQDeviceProxy::getOutputChannelIndexByName(std::string const& name) const
{
  for (int i = 0; i < mOutputChannelInfos.size(); i++) {
    if (mOutputChannelInfos[i].name == name) {
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

ChannelIndex FairMQDeviceProxy::getForwardChannelIndexByName(std::string const& name) const
{
  for (int i = 0; i < mForwardChannelInfos.size(); i++) {
    if (mForwardChannelInfos[i].name == name) {
      return {i};
    }
  }
  return {ChannelIndex::INVALID};
}

fair::mq::TransportFactory* FairMQDeviceProxy::getOutputTransport(RouteIndex index) const
{
  auto transport = getOutputChannel(getOutputChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

fair::mq::TransportFactory* FairMQDeviceProxy::getInputTransport(RouteIndex index) const
{
  auto transport = getInputChannel(getInputChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

fair::mq::TransportFactory* FairMQDeviceProxy::getForwardTransport(RouteIndex index) const
{
  auto transport = getInputChannel(getInputChannelIndex(index))->Transport();
  assert(transport);
  return transport;
}

std::unique_ptr<fair::mq::Message> FairMQDeviceProxy::createOutputMessage(RouteIndex routeIndex) const
{
  return getOutputTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

std::unique_ptr<fair::mq::Message> FairMQDeviceProxy::createOutputMessage(RouteIndex routeIndex, const size_t size) const
{
  return getOutputTransport(routeIndex)->CreateMessage(size, fair::mq::Alignment{64});
}

std::unique_ptr<fair::mq::Message> FairMQDeviceProxy::createInputMessage(RouteIndex routeIndex) const
{
  return getInputTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

std::unique_ptr<fair::mq::Message> FairMQDeviceProxy::createInputMessage(RouteIndex routeIndex, const size_t size) const
{
  return getInputTransport(routeIndex)->CreateMessage(size, fair::mq::Alignment{64});
}

std::unique_ptr<fair::mq::Message> FairMQDeviceProxy::createForwardMessage(RouteIndex routeIndex) const
{
  return getForwardTransport(routeIndex)->CreateMessage(fair::mq::Alignment{64});
}

void FairMQDeviceProxy::bind(std::vector<OutputRoute> const& outputs, std::vector<InputRoute> const& inputs,
                             std::vector<ForwardRoute> const& forwards,
                             fair::mq::Device& device)
{
  mOutputs.clear();
  mOutputRoutes.clear();
  mOutputChannelInfos.clear();
  mOutputChannelStates.clear();
  mInputs.clear();
  mInputRoutes.clear();
  mInputChannels.clear();
  mInputChannelNames.clear();
  mForwards.clear();
  mForwardRoutes.clear();
  mForwardChannelInfos.clear();
  mForwardChannelStates.clear();
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
        channelIndex = ChannelIndex{(int)mOutputChannelInfos.size()};
        ChannelAccountingType dplChannel = (route.channel.rfind("from_", 0) == 0) ? ChannelAccountingType::DPL : ChannelAccountingType::RAW;
        OutputChannelInfo info{
          .name = route.channel,
          .channelType = dplChannel,
          .channel = device.fChannels.at(route.channel).at(0),
        };
        mOutputChannelInfos.push_back(info);
        mOutputChannelStates.push_back({0});
        channelNameToChannel[route.channel] = channelIndex;
        LOGP(detail, "Binding channel {} to channel index {}", route.channel, channelIndex.value);
      } else {
        LOGP(detail, "Using index {} for channel {}", channelPos->second.value, route.channel);
        channelIndex = channelPos->second;
      }
      LOGP(detail, "Binding route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, route.maxTimeslices, ri, channelIndex.value);
      mOutputRoutes.emplace_back(RouteState{channelIndex, false});
      ri++;
    }
    for (auto& route : mOutputRoutes) {
      assert(route.channel.value != -1);
      assert(route.channel.value < mOutputChannelInfos.size());
    }
    LOGP(detail, "Total channels found {}, total routes {}", mOutputChannelInfos.size(), mOutputRoutes.size());
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
        LOGP(detail, "Binding channel {} to channel index {}", route.sourceChannel, channelIndex.value);
      } else {
        LOGP(detail, "Using index {} for channel {}", channelPos->second.value, route.sourceChannel);
        channelIndex = channelPos->second;
      }
      LOGP(detail, "Binding route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, maxLanes, ri, channelIndex.value);
      mInputRoutes.emplace_back(RouteState{channelIndex, false});
      ri++;
    }
    for (auto& route : mInputRoutes) {
      assert(route.channel.value != -1);
      assert(route.channel.value < mInputChannels.size());
    }
    LOGP(detail, "Total input channels found {}, total routes {}", mInputChannels.size(), mInputRoutes.size());
    assert(mInputRoutes.size() == inputs.size());
  }

  {
    mForwards = forwards;
    mForwardRoutes.reserve(forwards.size());
    LOGP(detail, "Forwards.size(): {}", forwards.size());
    size_t ri = 0;
    std::unordered_map<std::string, ChannelIndex> channelNameToChannel;

    for (auto& route : forwards) {
      // If the channel is not yet registered, register it.
      // If the channel is already registered, use the existing index.
      auto channelPos = channelNameToChannel.find(route.channel);
      ChannelIndex channelIndex;

      if (channelPos == channelNameToChannel.end()) {
        channelIndex = ChannelIndex{(int)mForwardChannelInfos.size()};
        auto& channel = device.fChannels.at(route.channel).at(0);
        ChannelAccountingType dplChannel = (route.channel.rfind("from_", 0) == 0) ? ChannelAccountingType::DPL : ChannelAccountingType::RAW;
        mForwardChannelInfos.push_back(ForwardChannelInfo{route.channel, dplChannel, channel});
        mForwardChannelStates.push_back(ForwardChannelState{0});
        channelNameToChannel[route.channel] = channelIndex;
        LOGP(detail, "Binding forward channel {} to channel index {}", route.channel, channelIndex.value);
      } else {
        LOGP(detail, "Using index {} for forward channel {}", channelPos->second.value, route.channel);
        channelIndex = channelPos->second;
      }
      LOGP(detail, "Binding forward route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, route.maxTimeslices, ri, channelIndex.value);
      mForwardRoutes.emplace_back(RouteState{channelIndex, false});
      ri++;
    }
    for (auto& route : mForwardRoutes) {
      assert(route.channel.value != -1);
      assert(route.channel.value < mForwardChannelInfos.size());
    }
    LOGP(detail, "Total forward channels found {}, total routes {}", mForwardChannelInfos.size(), mForwardRoutes.size());
    // List all routes
    for (auto& route : mForwards) {
      LOGP(detail, "Forward route {}@{}%{} to index {} and channelIndex {}", route.matcher, route.timeslice, route.maxTimeslices);
    }
    assert(mForwardRoutes.size() == forwards.size());
  }
  mStateChangeCallback = [&device]() -> bool { return device.NewStatePending(); };
}
} // namespace o2::framework
