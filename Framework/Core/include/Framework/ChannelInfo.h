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
#ifndef O2_FRAMEWORK_CHANNELINFO_H_
#define O2_FRAMEWORK_CHANNELINFO_H_

#include "Framework/RoutingIndices.h"
#include "Framework/TimesliceSlot.h"
#include <string>
#include <fairmq/Parts.h>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

enum struct InputChannelState {
  /// The channel is actively receiving data
  Running,
  /// The channel was paused
  Paused,
  /// The channel was signaled it will not receive any data
  Completed,
  /// The channel can be used to retrieve data, but it
  /// will not send it on its own.
  Pull
};

enum struct ChannelAccountingType {
  /// A channel which was not explicitly set
  Unknown,
  /// The channel is a normal input channel
  DPL,
  /// A raw FairMQ channel which is not accounted by the framework
  RAWFMQ
};

/// This represent the current state of an input channel.  Its values can be
/// updated by Control or by the by the incoming flow of messages.
struct InputChannelInfo {
  InputChannelState state = InputChannelState::Running;
  uint32_t hasPendingEvents = 0;
  bool readPolled = false;
  fair::mq::Channel* channel = nullptr;
  fair::mq::Parts parts;
  /// Wether we already notified operations are normal.
  /// We start with true given we assume in the beginning
  /// things are ok.
  bool normalOpsNotified = true;
  /// Wether we aready notified about backpressure.
  /// We start with false since we assume there is no
  /// backpressure to start with.
  bool backpressureNotified = false;
  ChannelIndex id = {-1};
  /// Wether its a normal channel or one which
  ChannelAccountingType channelType = ChannelAccountingType::DPL;
  /// Oldest possible timeslice for the given channel
  TimesliceId oldestForChannel;
};

/// Output channel information
struct OutputChannelInfo {
  std::string name = "invalid";
  ChannelAccountingType channelType = ChannelAccountingType::DPL;
  fair::mq::Channel& channel;
};

struct OutputChannelState {
  TimesliceId oldestForChannel = {0};
  // How many times sending on this channel failed
  int64_t droppedMessages = 0;
};

/// Forward channel information
struct ForwardChannelInfo {
  /// The name of the channel
  std::string name = "invalid";
  /// Wether or not it's a DPL internal channel.
  ChannelAccountingType channelType = ChannelAccountingType::DPL;
  fair::mq::Channel& channel;
};

struct ForwardChannelState {
  TimesliceId oldestForChannel = {0};
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHANNELINFO_H_
