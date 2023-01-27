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
typedef struct uv_async_s uv_async_t;

namespace o2::framework
{

/// enumeration representing the current state of a given
/// device.
enum struct StreamingState {
  /// Data is being processed
  Streaming = 0,
  /// End of streaming requested, but not notified
  EndOfStreaming = 1,
  /// End of streaming notified
  Idle = 2,
};

enum struct TransitionHandlingState {
  /// No pending transitions
  NoTransition,
  /// A transition was notified to be requested
  Requested,
  /// A transition needs to be fullfilled ASAP
  Expired
};

/// Running state information of a given device
struct DeviceState {
  /// Motivation for the loop being triggered.
  enum LoopReason : int {
    NO_REASON = 0,                // No tracked reason to wake up
    METRICS_MUST_FLUSH = 1,       // Metrics available to flush
    SIGNAL_ARRIVED = 1 << 1,      // Signal has arrived
    DATA_SOCKET_POLLED = 1 << 2,  // Data has arrived
    DATA_INCOMING = 1 << 3,       // Data was read
    DATA_OUTGOING = 1 << 4,       // Data was written
    WS_COMMUNICATION = 1 << 5,    // Communication over WS
    TIMER_EXPIRED = 1 << 6,       // Timer expired
    WS_CONNECTED = 1 << 7,        // Connection to driver established
    WS_CLOSING = 1 << 8,          // Events related to WS shutting down
    WS_READING = 1 << 9,          // Events related to WS shutting down
    WS_WRITING = 1 << 10,         // Events related to WS shutting down
    ASYNC_NOTIFICATION = 1 << 11, // Some other thread asked the main one to wake up
    OOB_ACTIVITY = 1 << 12,       // Out of band activity
    UNKNOWN = 1 << 13,            // Unknown reason why we are here.
    FIRST_LOOP = 1 << 14,         // First loop to be executed
    NEW_STATE_PENDING = 1 << 15,  // Someone invoked NewStatePending
    PREVIOUSLY_ACTIVE = 1 << 16,  // The previous loop was active
    TRACE_CALLBACKS = 1 << 17,    // Trace callbacks
    TRACE_USERCODE = 1 << 18,     // Trace only usercode
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
  // The list of timers fired in this loop
  std::vector<uv_timer_t*> firedTimers;
  // The list of pollers for active input channels
  std::vector<uv_poll_t*> activeInputPollers;
  // The list of pollers for active output channels
  std::vector<uv_poll_t*> activeOutputPollers;
  /// The list of active signal handlers
  std::vector<uv_signal_t*> activeSignals;
  /// The list for active out-of-bound pollers
  std::vector<uv_poll_t*> activeOutOfBandPollers;

  uv_async_t* awakeMainThread = nullptr;

  // A list of states which we should go to
  std::vector<std::string> nextFairMQState;

  /// Bitmask of LoopReason which caused this iterations.
  int loopReason = 0;
  /// Bitmask of LoopReason to trace
  int tracingFlags = 0;
  /// Stack of the severity, so that we can display only
  /// the bits we are interested in.
  std::vector<int> severityStack;
  TransitionHandlingState transitionHandling = TransitionHandlingState::NoTransition;
};

} // namespace o2::framework
#endif
