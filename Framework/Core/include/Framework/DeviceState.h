// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DEVICESTATE_H_
#define O2_FRAMEWORK_DEVICESTATE_H_

#include "Framework/ChannelInfo.h"
#include <vector>
#include <string>
#include <map>
#include <utility>

typedef struct uv_loop_s uv_loop_t;
typedef struct uv_timer_s uv_timer_t;
typedef struct uv_poll_s uv_poll_t;
typedef struct uv_signal_s uv_signal_t;

namespace o2::framework
{

/// enumeration representing the current state of a given
/// device.
enum struct StreamingState {
  /// Data is being processed
  Streaming,
  /// End of streaming requested, but not notified
  EndOfStreaming,
  /// End of streaming notified
  Idle,
};

/// Running state information of a given device
struct DeviceState {
  std::vector<InputChannelInfo> inputChannelInfos;
  StreamingState streaming = StreamingState::Streaming;
  bool quitRequested = false;
  // The libuv event loop which serves this device.
  uv_loop_t* loop;
  // The list of active timers which notify this device.
  std::vector<uv_timer_t*> activeTimers;
  // The list of pollers for active input channels
  std::vector<uv_poll_t*> activeInputPollers;
  // The list of pollers for active output channels
  std::vector<uv_poll_t*> activeOutputPollers;
  /// The list of active signal handlers
  std::vector<uv_signal_t*> activeSignals;
};

} // namespace o2::framework
#endif
