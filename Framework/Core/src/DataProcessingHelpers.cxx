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

#include <fairmq/Device.h>
#include <fairmq/Channel.h>

namespace o2::framework
{
void DataProcessingHelpers::sendEndOfStream(FairMQDevice& device, OutputChannelSpec const& channel)
{
  FairMQParts parts;
  FairMQMessagePtr payload(device.NewMessage());
  SourceInfoHeader sih;
  sih.state = InputChannelState::Completed;
  auto channelAlloc = o2::pmr::getTransportAllocator(device.GetChannel(channel.name, 0).Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, sih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));
  device.Send(parts, channel.name, 0);
  LOGP(info, "Sending end-of-stream message to channel {}", channel.name);
}

void DataProcessingHelpers::sendOldestPossibleTimeframe(fair::mq::Channel& channel, size_t timeslice)
{
  FairMQParts parts;
  FairMQMessagePtr payload(channel.Transport()->CreateMessage());
  o2::framework::DomainInfoHeader dih;
  dih.oldestPossibleTimeslice = timeslice;
  auto channelAlloc = o2::pmr::getTransportAllocator(channel.Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));

  auto timeout = 1000;
  auto res = channel.Send(parts, timeout);
  if (res == (size_t)fair::mq::TransferCode::timeout) {
    LOGP(warning, "Timed out sending oldest possible timeslice after {}s. Downstream backpressure detected on {}.", timeout / 1000, channel.GetName());
    channel.Send(parts);
    LOGP(info, "Downstream backpressure on {} recovered.", channel.GetName());
  } else if (res == (size_t)fair::mq::TransferCode::error) {
    LOGP(fatal, "Error while sending on channel {}", channel.GetName());
  }
}

void DataProcessingHelpers::broadcastOldestPossibleTimeslice(FairMQDeviceProxy& proxy, size_t timeslice)
{
  for (int ci = 0; ci < proxy.getNumOutputChannels(); ++ci) {
    auto* channel = proxy.getOutputChannel({ci});
    assert(channel);
    sendOldestPossibleTimeframe(*channel, timeslice);
  }
}

} // namespace o2::framework
