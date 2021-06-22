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
#include "Framework/ComputingQuotaOffer.h"

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
  /// Motivation for the loop being triggered.
  enum LoopReason : int {
    NO_REASON = 0,          // No tracked reason to wake up
    METRICS_MUST_FLUSH = 1, // Metrics available to flush
    SIGNAL_ARRIVED = 2,     // Signal has arrived
    DATA_SOCKET_POLLED = 4, // Data has arrived
    DATA_INCOMING = 8,      // Data was read
    DATA_OUTGOING = 16,     // Data was written
    WS_COMMUNICATION = 32,  // Communication over WS
    TIMER_EXPIRED = 64,     // Timer expired
    WS_CONNECTED = 128,     // Connection to driver established
    WS_CLOSING = 256,       // Events related to WS shutting down
    WS_READING = 512,       // Events related to WS shutting down
    WS_WRITING = 1024,      // Events related to WS shutting down
    ASYNC_NOTIFICATION = 2048
  };

  std::vector<InputChannelInfo> inputChannelInfos;
  StreamingState streaming = StreamingState::Streaming;
  bool quitRequested = false;

  /// ComputingQuotaOffers which have not yet been
  /// evaluated by the ComputingQuotaEvaluator
  std::vector<ComputingQuotaOffer> pendingOffers;
  /// ComputingQuotaOffers which should be removed
  /// from the queue.
  std::vector<ComputingQuotaConsumer> offerConsumers;

  // The libuv event loop which serves this device.
  uv_loop_t* loop = nullptr;
  // The list of active timers which notify this device.
  std::vector<uv_timer_t*> activeTimers;
  // The list of pollers for active input channels
  std::vector<uv_poll_t*> activeInputPollers;
  // The list of pollers for active output channels
  std::vector<uv_poll_t*> activeOutputPollers;
  /// The list of active signal handlers
  std::vector<uv_signal_t*> activeSignals;
  int loopReason = 0;
};

} // namespace o2::framework
#endif
