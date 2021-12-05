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

#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ConfigContext.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

ChannelConfigurationPolicy defaultDispatcherPolicy(ConfigContext const& configContext)
{
  auto& options = configContext.options();
  FairMQChannelConfigSpec spec{
    .rateLogging = options.get<int>("fairmq-rate-logging"),
    .recvBufferSize = options.isDefault("fairmq-recv-buffer-size") ? 32 : options.get<int>("fairmq-recv-buffer-size"),
    .sendBufferSize = options.isDefault("fairmq-send-buffer-size") ? 32 : options.get<int>("fairmq-send-buffer-size"),
    .ipcPrefix = options.get<std::string>("fairmq-ipc-prefix"),
  };

  ChannelConfigurationPolicy policy{
    .match = [](std::string const&, std::string const& consumerId) -> bool { return consumerId == "Dispatcher"; },
    .modifyInput = [spec](InputChannelSpec& channel) -> void {
      channel.method = ChannelMethod::Bind;
      channel.type = ChannelType::Pull;
      channel.rateLogging = spec.rateLogging;
      channel.recvBufferSize = spec.recvBufferSize;
      channel.sendBufferSize = spec.sendBufferSize;
      channel.ipcPrefix = spec.ipcPrefix; },
    .modifyOutput = [spec](OutputChannelSpec& channel) -> void {
      channel.method = ChannelMethod::Connect;
      channel.type = ChannelType::Push;
      channel.rateLogging = spec.rateLogging;
      channel.recvBufferSize = spec.recvBufferSize;
      channel.sendBufferSize = spec.sendBufferSize;
      channel.ipcPrefix = spec.ipcPrefix; }};

  return policy;
}

std::vector<ChannelConfigurationPolicy> ChannelConfigurationPolicy::createDefaultPolicies(ConfigContext const& configContext)
{
  ChannelConfigurationPolicy defaultPolicy;
  FairMQChannelConfigSpec spec;
  spec.rateLogging = configContext.options().get<int>("fairmq-rate-logging");
  spec.recvBufferSize = configContext.options().get<int>("fairmq-recv-buffer-size");
  spec.sendBufferSize = configContext.options().get<int>("fairmq-send-buffer-size");
  spec.ipcPrefix = configContext.options().get<std::string>("fairmq-ipc-prefix");

  defaultPolicy.match = ChannelConfigurationPolicyHelpers::matchAny;
  defaultPolicy.modifyInput = ChannelConfigurationPolicyHelpers::pullInput(spec);
  defaultPolicy.modifyOutput = ChannelConfigurationPolicyHelpers::pushOutput(spec);

  return {defaultDispatcherPolicy(configContext), defaultPolicy};
}

} // namespace o2::framework
