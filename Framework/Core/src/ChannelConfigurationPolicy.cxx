// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ConfigContext.h"

namespace o2::framework
{

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

  return {defaultPolicy};
}

} // namespace o2::framework
