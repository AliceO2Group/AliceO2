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

#include "Framework/SendingPolicy.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/STFHeader.h"
#include "DeviceSpecHelpers.h"
#include <fairmq/Device.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
namespace o2::framework
{

std::vector<SendingPolicy> SendingPolicy::createDefaultPolicies()
{
  return {SendingPolicy{
            .name = "dispatcher",
            .matcher = [](DeviceSpec const& spec, ConfigContext const&) { return spec.name == "Dispatcher" || DeviceSpecHelpers::hasLabel(spec, "Dispatcher"); },
            .send = [](FairMQDeviceProxy& proxy, FairMQParts& parts, ChannelIndex channelIndex) {
              auto *channel = proxy.getOutputChannel(channelIndex);
              channel->Send(parts, -1); }},
          SendingPolicy{
            .name = "default",
            .matcher = [](DeviceSpec const&, ConfigContext const&) { return true; },
            .send = [](FairMQDeviceProxy& proxy, FairMQParts& parts, ChannelIndex channelIndex) { 
              auto *channel = proxy.getOutputChannel(channelIndex);
              channel->Send(parts); }}};
}
} // namespace o2::framework
