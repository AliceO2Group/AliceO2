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

#include "Framework/RoutingIndices.h"
#include "Framework/RouteState.h"
#include "Framework/OutputRoute.h"
#include <fairmq/FwdDecls.h>
#include <vector>

namespace o2::framework
{
/// Helper class to hide FairMQDevice headers in the DataAllocator header.
/// This is done because FairMQDevice brings in a bunch of boost.mpl /
/// boost.fusion stuff, slowing down compilation times enourmously.
class FairMQDeviceProxy
{
 public:
  FairMQDeviceProxy() = default;
  FairMQDeviceProxy(FairMQDeviceProxy const&) = delete;
  void bindRoutes(std::vector<OutputRoute> const& routes, FairMQDevice& device);

  /// Retrieve the transport associated to a given route.
  fair::mq::TransportFactory* getTransport(RouteIndex routeIndex) const;
  /// ChannelIndex from a RouteIndex
  ChannelIndex getChannelIndex(RouteIndex routeIndex) const;
  /// Retrieve the channel associated to a given route.
  fair::mq::Channel* getChannel(ChannelIndex channelIndex) const;

  std::unique_ptr<FairMQMessage> createMessage(RouteIndex routeIndex) const;
  std::unique_ptr<FairMQMessage> createMessage(RouteIndex routeIndex, const size_t size) const;
  size_t getNumChannels() const { return mChannels.size(); }

 private:
  std::vector<RouteState> mRoutes;
  std::vector<fair::mq::Channel*> mChannels;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_FAIRMQDEVICEPROXY_H_
