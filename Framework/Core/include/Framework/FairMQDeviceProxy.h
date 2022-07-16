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
#ifndef O2_FRAMEWORK_FAIRMQDEVICEPROXY_H_
#define O2_FRAMEWORK_FAIRMQDEVICEPROXY_H_

#include <memory>

#include "Framework/ChannelInfo.h"
#include "Framework/RoutingIndices.h"
#include "Framework/RouteState.h"
#include "Framework/OutputRoute.h"
#include "Framework/InputRoute.h"
#include "Framework/ForwardRoute.h"
#include <fairmq/FwdDecls.h>
#include <vector>

namespace o2::header
{
struct DataHeader;
};

namespace o2::framework
{
/// Helper class to hide fair::mq::Device headers in the DataAllocator header.
/// This is done because fair::mq::Device brings in a bunch of boost.mpl /
/// boost.fusion stuff, slowing down compilation times enourmously.
class FairMQDeviceProxy
{
 public:
  FairMQDeviceProxy() = default;
  FairMQDeviceProxy(FairMQDeviceProxy const&) = delete;
  void bind(std::vector<OutputRoute> const& outputs, std::vector<InputRoute> const& inputs,
            std::vector<ForwardRoute> const& forwards, fair::mq::Device& device);

  /// Retrieve the transport associated to a given route.
  fair::mq::TransportFactory* getOutputTransport(RouteIndex routeIndex) const;
  /// Retrieve the transport associated to a given route.
  fair::mq::TransportFactory* getInputTransport(RouteIndex routeIndex) const;
  /// Retrieve the transport associated to a given route.
  fair::mq::TransportFactory* getForwardTransport(RouteIndex routeIndex) const;
  /// ChannelIndex from a given channel name
  ChannelIndex getOutputChannelIndexByName(std::string const& channelName) const;
  /// ChannelIndex from a given channel name
  ChannelIndex getInputChannelIndexByName(std::string const& channelName) const;
  /// ChannelIndex from a given channel name
  ChannelIndex getForwardChannelIndexByName(std::string const& channelName) const;
  /// Retrieve the channel index from a given OutputSpec and the associated timeslice
  ChannelIndex getOutputChannelIndex(OutputSpec const& spec, size_t timeslice) const;
  /// Retrieve the channel index from a given OutputSpec and the associated timeslice
  ChannelIndex getForwardChannelIndex(header::DataHeader const& header, size_t timeslice) const;
  /// ChannelIndex from a RouteIndex
  ChannelIndex getOutputChannelIndex(RouteIndex routeIndex) const;
  ChannelIndex getInputChannelIndex(RouteIndex routeIndex) const;
  ChannelIndex getForwardChannelIndex(RouteIndex routeIndex) const;
  /// Retrieve the channel associated to a given output route.
  fair::mq::Channel* getInputChannel(ChannelIndex channelIndex) const;
  fair::mq::Channel* getOutputChannel(ChannelIndex channelIndex) const;
  fair::mq::Channel* getForwardChannel(ChannelIndex channelIndex) const;

  /// Retrieve information associated to a given forward by ChannelIndex
  ForwardChannelInfo const& getForwardChannelInfo(ChannelIndex channelIndex) const;

  std::unique_ptr<fair::mq::Message> createOutputMessage(RouteIndex routeIndex) const;
  std::unique_ptr<fair::mq::Message> createOutputMessage(RouteIndex routeIndex, const size_t size) const;

  std::unique_ptr<fair::mq::Message> createInputMessage(RouteIndex routeIndex) const;
  std::unique_ptr<fair::mq::Message> createInputMessage(RouteIndex routeIndex, const size_t size) const;

  std::unique_ptr<fair::mq::Message> createForwardMessage(RouteIndex routeIndex) const;

  size_t getNumOutputChannels() const { return mOutputChannels.size(); }
  size_t getNumInputChannels() const { return mInputChannels.size(); }
  size_t getNumForwardChannels() const { return mForwardChannelInfos.size(); }
  size_t getNumForwards() const { return mForwards.size(); }

  bool newStateRequested() const { return mStateChangeCallback(); }

 private:
  std::vector<OutputRoute> mOutputs;
  std::vector<RouteState> mOutputRoutes;
  std::vector<fair::mq::Channel*> mOutputChannels;
  std::vector<std::string> mOutputChannelNames;

  std::vector<InputRoute> mInputs;
  std::vector<RouteState> mInputRoutes;
  std::vector<fair::mq::Channel*> mInputChannels;
  std::vector<std::string> mInputChannelNames;

  std::vector<ForwardRoute> mForwards;
  std::vector<RouteState> mForwardRoutes;
  std::vector<ForwardChannelInfo> mForwardChannelInfos;

  std::function<bool()> mStateChangeCallback;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_FAIRMQDEVICEPROXY_H_
