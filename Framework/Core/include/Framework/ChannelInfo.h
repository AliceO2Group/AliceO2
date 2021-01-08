// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CHANNELINFO_H
#define O2_FRAMEWORK_CHANNELINFO_H

#include <string>

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
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHANNELSPEC_H
