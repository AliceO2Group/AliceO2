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

namespace o2
{
namespace framework
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

ChannelConfigurationPolicyHelpers::InputChannelModifier ChannelConfigurationPolicyHelpers::subscribeInput =
  [](InputChannelSpec& channel) {
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Sub;
  };

ChannelConfigurationPolicyHelpers::OutputChannelModifier ChannelConfigurationPolicyHelpers::publishOutput =
  [](OutputChannelSpec& channel) {
    channel.method = ChannelMethod::Bind;
    channel.type = ChannelType::Pub;
  };

ChannelConfigurationPolicyHelpers::InputChannelModifier ChannelConfigurationPolicyHelpers::pullInput =
  [](InputChannelSpec& channel) {
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Pull;
  };

ChannelConfigurationPolicyHelpers::OutputChannelModifier ChannelConfigurationPolicyHelpers::pushOutput =
  [](OutputChannelSpec& channel) {
    channel.method = ChannelMethod::Bind;
    channel.type = ChannelType::Push;
  };

} // namespace framework
} // namespace o2
