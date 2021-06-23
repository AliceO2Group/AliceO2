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
#ifndef O2_FRAMEWORK_CHANNELCONFIGURATIONPOLICYHELPERS_H_
#define O2_FRAMEWORK_CHANNELCONFIGURATIONPOLICYHELPERS_H_

#include "Framework/ChannelSpec.h"

#include <functional>

namespace o2::framework
{

struct FairMQChannelConfigSpec {
  int64_t rateLogging;
  int64_t recvBufferSize;
  int64_t sendBufferSize;
  std::string ipcPrefix;
};

/// A set of helpers for common ChannelConfigurationPolicy behaviors
struct ChannelConfigurationPolicyHelpers {
  // TODO: currently we allow matching of the policy only based on
  //       the id of the device. Best would be to be able to use the
  //       DeviceSpec itself to do the matching. This would allow
  //       fancy behaviors like "use shared memory if the device is on the
  //       same node, something else if remote".
  using PolicyMatcher = std::function<bool(std::string const& producer, std::string const& consumer)>;
  using OutputChannelModifier = std::function<void(OutputChannelSpec& spec)>;
  using InputChannelModifier = std::function<void(InputChannelSpec& spec)>;

  /// Catch all policy, used by the last rule.
  static PolicyMatcher matchAny;

  // Example on how to write an helper which allows matching
  // based on some DeviceSpec property.
  ///
  static PolicyMatcher matchByProducerName(const char* name);
  static PolicyMatcher matchByConsumerName(const char* name);

  // Some trivial modifier which can be used by the policy.
  /// Makes the passed input channel connect and subscribe
  static InputChannelModifier subscribeInput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed output channel bind and subscribe
  static OutputChannelModifier publishOutput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed input channel connect and pull
  static InputChannelModifier pullInput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed output channel bind and push
  static OutputChannelModifier pushOutput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed input channel connect and request
  static InputChannelModifier reqInput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed output channel bind and reply
  static OutputChannelModifier replyOutput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed input channel connect and pair
  static InputChannelModifier pairInput(FairMQChannelConfigSpec const& spec);
  /// Makes the passed output channel bind and pair
  static OutputChannelModifier pairOutput(FairMQChannelConfigSpec const& spec);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHANNELCONFIGURATIONPOLICY_H_
