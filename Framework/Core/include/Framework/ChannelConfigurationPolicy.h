// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CHANNELCONFIGURATIONPOLICY_H
#define FRAMEWORK_CHANNELCONFIGURATIONPOLICY_H

#include <functional>
#include "Framework/ChannelConfigurationPolicyHelpers.h"
#include "Framework/ChannelSpec.h"
#include "Framework/DeviceSpec.h"

namespace o2
{
namespace framework
{

/// A ChannelConfigurationPolicy allows the user
/// to customise connection method and type
/// for a channel created between two devices.
///
///
/// NOTE: is this a device to device decision or an
///       input to output decision? Do we want to allow two
///       devices to exchange different kind of data using
///       diffent channels, with different policies?
struct ChannelConfigurationPolicy {
  using Helpers = ChannelConfigurationPolicyHelpers;

  Helpers::PolicyMatcher match = nullptr;
  Helpers::InputChannelModifier modifyInput = nullptr;
  Helpers::OutputChannelModifier modifyOutput = nullptr;

  static std::vector<ChannelConfigurationPolicy> createDefaultPolicies();
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_CHANNELCONFIGURATIONPOLICY_H
