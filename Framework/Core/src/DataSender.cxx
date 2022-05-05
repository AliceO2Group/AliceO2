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

#include "Framework/DataSender.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/OutputRoute.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Monitoring.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/LifetimeHelpers.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/DomainInfoHeader.h"

#include <fairmq/Device.h>

#include <fmt/ostream.h>

using namespace o2::monitoring;

namespace o2::framework
{

namespace
{
std::vector<size_t>
  createDistinctOutputRouteIndex(std::vector<OutputRoute> const& routes)
{
  std::vector<size_t> result;
  for (size_t ri = 0; ri < routes.size(); ++ri) {
    auto& route = routes[ri];
    if (route.timeslice == 0) {
      result.push_back(ri);
    }
  }
  return result;
}
} // namespace

DataSender::DataSender(ServiceRegistry& registry,
                       SendingPolicy const& policy)
  : mProxy{registry.get<FairMQDeviceProxy>()},
    mRegistry{registry},
    mSpec{registry.get<DeviceSpec const>()},
    mDistinctRoutesIndex{createDistinctOutputRouteIndex(mSpec.outputs)},
    mPolicy{policy}
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  auto numInputTypes = mDistinctRoutesIndex.size();
  mQueriesMetricsNames.resize(numInputTypes * 1);
  auto& monitoring = mRegistry.get<Monitoring>();
  monitoring.send({(int)numInputTypes, "output_matchers/h", Verbosity::Debug});
  monitoring.send({(int)1, "output_matchers/w", Verbosity::Debug});
  auto& routes = mSpec.outputs;
  for (size_t i = 0; i < numInputTypes; ++i) {
    mQueriesMetricsNames[i] = fmt::format("output_matchers/{}", i);
    char buffer[128];
    assert(mDistinctRoutesIndex[i] < routes.size());
    mOutputs.push_back(routes[mDistinctRoutesIndex[i]].matcher);
    DataSpecUtils::describe(buffer, 127, mOutputs.back());
    monitoring.send({fmt::format("{} ({})", buffer, mOutputs.back().lifetime), mQueriesMetricsNames[i], Verbosity::Debug});
  }
}

std::unique_ptr<FairMQMessage> DataSender::create(RouteIndex routeIndex)
{
  return mProxy.getOutputTransport(routeIndex)->CreateMessage();
}

void DataSender::send(FairMQParts& parts, ChannelIndex channelIndex)
{
  mPolicy.send(mProxy, parts, channelIndex);

  /// We also always propagate the information about what is the oldest possible
  /// timeslice that can be sent from this device.
  /// FIXME: do it at a different level?
  /// FIXME: throttling this information?
  /// FIXME: do it only if it changes?
  TimesliceIndex& index = mRegistry.get<TimesliceIndex>();
  index.updateOldestPossibleOutput();

  auto oldest = index.getOldestPossibleOutput();
  if (oldest.timeslice.value == -1) {
    return;
  }
  auto* channel = mProxy.getOutputChannel(channelIndex);

  FairMQParts oldestParts;
  FairMQMessagePtr payload(channel->Transport()->CreateMessage());
  DomainInfoHeader dih;
  dih.oldestPossibleTimeslice = oldest.timeslice.value;
  auto channelAlloc = o2::pmr::getTransportAllocator(channel->Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  oldestParts.AddPart(std::move(header));
  oldestParts.AddPart(std::move(payload));
  LOGP(debug, "Notifying {} {} about oldest possible timeslice being {} from {} #{}",
       channelIndex.value,
       channel->GetName(),
       oldest.timeslice.value,
       oldest.channel.value == -1 ? "slot" : " input channel",
       oldest.channel.value == -1 ? oldest.slot.index : oldest.channel.value);
  channel->Send(oldestParts);
}

} // namespace o2::framework
