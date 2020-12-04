// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ChannelConfigurationPolicyHelpers.h"
#include <functional>
#include <string>
#include "Framework/ChannelSpec.h"

namespace o2::framework
{

ChannelConfigurationPolicyHelpers::PolicyMatcher ChannelConfigurationPolicyHelpers::matchAny =
  [](std::string const&, std::string const&) { return true; };

ChannelConfigurationPolicyHelpers::PolicyMatcher ChannelConfigurationPolicyHelpers::matchByProducerName(
  const char* name)
{
  std::string nameString = name;

  return [nameString](std::string const& producerId, std::string const&) -> bool { return producerId == nameString; };
}

ChannelConfigurationPolicyHelpers::PolicyMatcher ChannelConfigurationPolicyHelpers::matchByConsumerName(
  const char* name)
{
  std::string nameString = name;

  return [nameString](std::string const&, std::string const& consumerId) -> bool { return consumerId == nameString; };
}

ChannelConfigurationPolicyHelpers::InputChannelModifier ChannelConfigurationPolicyHelpers::subscribeInput(FairMQChannelConfigSpec const& spec)
{
  return [spec](InputChannelSpec& channel) {
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Sub;
    channel.rateLogging = spec.rateLogging;
  };
}

ChannelConfigurationPolicyHelpers::OutputChannelModifier ChannelConfigurationPolicyHelpers::publishOutput(FairMQChannelConfigSpec const& spec)
{
  return [spec](OutputChannelSpec& channel) {
    channel.method = ChannelMethod::Bind;
    channel.type = ChannelType::Pub;
    channel.rateLogging = spec.rateLogging;
  };
}

ChannelConfigurationPolicyHelpers::InputChannelModifier ChannelConfigurationPolicyHelpers::pullInput(FairMQChannelConfigSpec const& spec)
{
  return [spec](InputChannelSpec& channel) {
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Pull;
    channel.rateLogging = spec.rateLogging;
  };
}

ChannelConfigurationPolicyHelpers::OutputChannelModifier ChannelConfigurationPolicyHelpers::pushOutput(FairMQChannelConfigSpec const& spec)
{
  return [spec](OutputChannelSpec& channel) {
    channel.method = ChannelMethod::Bind;
    channel.type = ChannelType::Push;
    channel.rateLogging = spec.rateLogging;
  };
}

ChannelConfigurationPolicyHelpers::InputChannelModifier ChannelConfigurationPolicyHelpers::pairInput(FairMQChannelConfigSpec const& spec)
{
  return [spec](InputChannelSpec& channel) {
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Pair;
    channel.rateLogging = spec.rateLogging;
  };
}

ChannelConfigurationPolicyHelpers::OutputChannelModifier ChannelConfigurationPolicyHelpers::pairOutput(FairMQChannelConfigSpec const& spec)
{
  return [spec](OutputChannelSpec& channel) {
    channel.method = ChannelMethod::Bind;
    channel.type = ChannelType::Pair;
    channel.rateLogging = spec.rateLogging;
  };
}

} // namespace o2::framework
