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
#include "Framework/DataRelayer.h"
#include "Headers/DataHeaderHelpers.h"
#include "Framework/Logger.h"
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
            .matcher = [](DataProcessorSpec const& source, DataProcessorSpec const& dest, ConfigContext const&) {
                if (source.name == "Dispatcher") {
                  return true;
                }
                // Check if any of the labels has "Dispatcher" as prefix
                for (auto const& label : source.labels) {
                  if (label.value.find("Dispatcher") == 0) {
                    return true;
                  }
                }
                // Check if any of the destination's labels is "expendable" or "non-critical"
                for (auto const& label : dest.labels) {
                  if (label.value == "expendable" || label.value == "non-critical") {
                    return true;
                  }
                }
                return false; },
            .send = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              OutputChannelInfo const& info = proxy.getOutputChannelInfo(channelIndex);
              OutputChannelState& state = proxy.getOutputChannelState(channelIndex);
              // Default timeout is 10ms.
              // We count the number of consecutively dropped messages.
              // If we have more than 10, we switch to a completely
              // non-blocking approach.
              int64_t timeout = 10;
              if (state.droppedMessages == 10 + 1) {
                LOG(warning) << "Failed to send 10 messages with 10ms timeout in a row, switching to completely non-blocking mode";
              }
              if (state.droppedMessages > 10) {
                timeout = 0;
              }
              size_t result = info.channel.Send(parts, timeout);
              if (result > 0) {
                state.droppedMessages = 0;
              } else if (state.droppedMessages < std::numeric_limits<decltype(state.droppedMessages)>::max()) {
                state.droppedMessages++;
              } }},
          SendingPolicy{
            .name = "profiling",
            .matcher = [](DataProcessorSpec const&, DataProcessorSpec const&, ConfigContext const&) { return getenv("DPL_DEBUG_MESSAGE_SIZE"); },
            .send = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getOutputChannel(channelIndex);
              auto timeout = 1000;
              int count = 0;
              auto& relayer = registry.get<DataRelayer>();
              for (auto& part : parts) {
                auto* dh = o2::header::get<o2::header::DataHeader*>(part->GetData());
                if (dh == nullptr) {
                  // This is a payload.
                  continue;
                }
                LOGP(info, "Sent {}/{}/{} for a total of {} bytes", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dh->payloadSize);
                count+= dh->payloadSize;
                auto* dph = o2::header::get<o2::framework::DataProcessingHeader*>(part->GetData());
                if (dph == nullptr) {
                  // This is a payload.
                  continue;
                }
                auto oldestPossibleOutput = relayer.getOldestPossibleOutput();
                if ((size_t)dph->startTime < oldestPossibleOutput.timeslice.value) {
                  LOGP(error, "Sent startTime {} while oldestPossibleOutput is {}. This should not be possible.", dph->startTime, oldestPossibleOutput.timeslice.value);
                }
              }
              LOGP(info, "Sent {} parts for a total of {} bytes", parts.Size(), count);
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on {}.", timeout/1000, channel->GetName());
                channel->Send(parts);
                LOGP(info, "Downstream backpressure on {} recovered.", channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(fatal, "Error while sending on channel {}", channel->GetName());
              } }},
          SendingPolicy{
            .name = "expendable",
            .matcher = [](DataProcessorSpec const& source, DataProcessorSpec const& dest, ConfigContext const&) {
              auto has_label = [](DataProcessorLabel const& label) {
                return label.value == "expendable";
              };
              return std::find_if(dest.labels.begin(), dest.labels.end(), has_label) != dest.labels.end(); },
            .send = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getOutputChannel(channelIndex);
              auto timeout = 1000;
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on expendable channel {}.", timeout/1000, channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(info, "Error while sending on channel {}", channel->GetName());
              } }},
          SendingPolicy{
            .name = "default",
            .matcher = [](DataProcessorSpec const&, DataProcessorSpec const&, ConfigContext const&) { return true; },
            .send = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getOutputChannel(channelIndex);
              auto timeout = 1000;
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on {}.", timeout/1000, channel->GetName());
                channel->Send(parts);
                LOGP(info, "Downstream backpressure on {} recovered.", channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(fatal, "Error while sending on channel {}", channel->GetName());
              } }}};
}

ForwardingPolicy ForwardingPolicy::createDefaultForwardingPolicy()
{
  return ForwardingPolicy{
    .name = "default",
    .matcher = [](DataProcessorSpec const&, DataProcessorSpec const&, ConfigContext const&) { return true; },
    .forward = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getForwardChannel(channelIndex);
              auto timeout = 1000;
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on {}.", timeout/1000, channel->GetName());
                channel->Send(parts);
                LOGP(info, "Downstream backpressure on {} recovered.", channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(fatal, "Error while sending on channel {}", channel->GetName());
              } }};
}

std::vector<ForwardingPolicy> ForwardingPolicy::createDefaultPolicies()
{
  return {ForwardingPolicy{
            .name = "profiling",
            .matcher = [](DataProcessorSpec const&, DataProcessorSpec const&, ConfigContext const&) { return getenv("DPL_DEBUG_MESSAGE_SIZE"); },
            .forward = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getForwardChannel(channelIndex);
              auto timeout = 1000;
              int count = 0;
              auto& relayer = registry.get<DataRelayer>();
              for (auto& part : parts) {
                auto* dh = o2::header::get<o2::header::DataHeader*>(part->GetData());
                if (dh == nullptr) {
                  // This is a payload.
                  continue;
                }
                LOGP(info, "Sent {}/{}/{} for a total of {} bytes", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dh->payloadSize);
                count+= dh->payloadSize;
                auto* dph = o2::header::get<o2::framework::DataProcessingHeader*>(part->GetData());
                if (dph == nullptr) {
                  // This is a payload.
                  continue;
                }
                auto oldestPossibleOutput = relayer.getOldestPossibleOutput();
                if ((size_t)dph->startTime < oldestPossibleOutput.timeslice.value) {
                  LOGP(error, "Sent startTime {} while oldestPossibleOutput is {}. This should not be possible.", dph->startTime, oldestPossibleOutput.timeslice.value);
                }
              }
              LOGP(info, "Sent {} parts for a total of {} bytes", parts.Size(), count);
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on {}.", timeout/1000, channel->GetName());
                channel->Send(parts);
                LOGP(info, "Downstream backpressure on {} recovered.", channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(fatal, "Error while sending on channel {}", channel->GetName());
              } }},
          ForwardingPolicy{
            .name = "expendable",
            .matcher = [](DataProcessorSpec const& source, DataProcessorSpec const& dest, ConfigContext const&) {
              auto has_label = [](DataProcessorLabel const& label) {
                return label.value == "expendable";
              };
              return std::find_if(dest.labels.begin(), dest.labels.end(), has_label) != dest.labels.end(); },
            .forward = [](fair::mq::Parts& parts, ChannelIndex channelIndex, ServiceRegistryRef registry) {
              auto &proxy = registry.get<FairMQDeviceProxy>();
              auto *channel = proxy.getForwardChannel(channelIndex);
              auto timeout = 1000;
              auto res = channel->Send(parts, timeout);
              if (res == (size_t)fair::mq::TransferCode::timeout) {
                LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on expendable channel {}.", timeout/1000, channel->GetName());
              } else if (res == (size_t) fair::mq::TransferCode::error) {
                LOGP(info, "Error while sending on channel {}", channel->GetName());
              } }},
          createDefaultForwardingPolicy()};
}
} // namespace o2::framework
