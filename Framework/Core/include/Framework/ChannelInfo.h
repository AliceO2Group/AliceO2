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
  /// Oldest possible timeslice for the given channel
  TimesliceId oldestForChannel;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHANNELINFO_H_
