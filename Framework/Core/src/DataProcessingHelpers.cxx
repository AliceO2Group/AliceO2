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
#include "Framework/DataProcessingHelpers.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/DomainInfoHeader.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelInfo.h"
#include "MemoryResources/MemoryResources.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "Framework/Logger.h"
#include "Framework/SendingPolicy.h"
#include "Framework/RawDeviceService.h"

#include <fairmq/Device.h>
#include <fairmq/Channel.h>

namespace o2::framework
{
void DataProcessingHelpers::sendEndOfStream(ServiceRegistryRef const& ref, OutputChannelSpec const& channel)
{
  fair::mq::Device* device = ref.get<RawDeviceService>().device();
  fair::mq::Parts parts;
  fair::mq::MessagePtr payload(device->NewMessage());
  SourceInfoHeader sih;
  sih.state = InputChannelState::Completed;
  auto channelAlloc = o2::pmr::getTransportAllocator(device->GetChannel(channel.name, 0).Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, sih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));
  device->Send(parts, channel.name, 0);
  LOGP(info, "Sending end-of-stream message to channel {}", channel.name);
}

void doSendOldestPossibleTimeframe(ServiceRegistryRef ref, fair::mq::TransportFactory* transport, ChannelIndex index, SendingPolicy::SendingCallback const& callback, size_t timeslice)
{
  fair::mq::Parts parts;
  fair::mq::MessagePtr payload(transport->CreateMessage());
  o2::framework::DomainInfoHeader dih;
  dih.oldestPossibleTimeslice = timeslice;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport);
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));

  callback(parts, index, ref);
}

bool DataProcessingHelpers::sendOldestPossibleTimeframe(ServiceRegistryRef const& ref, ForwardChannelInfo const& info, ForwardChannelState& state, size_t timeslice)
{
  if (state.oldestForChannel.value >= timeslice) {
    return false;
  }
  doSendOldestPossibleTimeframe(ref, info.channel.Transport(), info.index, info.policy->forward, timeslice);
  state.oldestForChannel = {timeslice};
  return true;
}

bool DataProcessingHelpers::sendOldestPossibleTimeframe(ServiceRegistryRef const& ref, OutputChannelInfo const& info, OutputChannelState& state, size_t timeslice)
{
  if (state.oldestForChannel.value >= timeslice) {
    return false;
  }
  doSendOldestPossibleTimeframe(ref, info.channel.Transport(), info.index, info.policy->send, timeslice);
  state.oldestForChannel = {timeslice};
  return true;
}

void DataProcessingHelpers::broadcastOldestPossibleTimeslice(ServiceRegistryRef const& ref, size_t timeslice)
{
  auto& proxy = ref.get<FairMQDeviceProxy>();
  for (int ci = 0; ci < proxy.getNumOutputChannels(); ++ci) {
    auto& info = proxy.getOutputChannelInfo({ci});
    auto& state = proxy.getOutputChannelState({ci});
    sendOldestPossibleTimeframe(ref, info, state, timeslice);
  }
}

} // namespace o2::framework
