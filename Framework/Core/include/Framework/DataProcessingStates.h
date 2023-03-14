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
#ifndef O2_FRAMEWORK_DATAPROCESSINGSTATES_H_
#define O2_FRAMEWORK_DATAPROCESSINGSTATES_H_

#include "DeviceState.h"
#include "Framework/ServiceSpec.h"
#include "Framework/TimingHelpers.h"
#include <atomic>
#include <cstdint>
#include <array>
#include <numeric>
#include <mutex>
#include <utility>

namespace o2::framework
{

struct DataProcessingStatsHelpers {
  /// Return a function which can be used to retrieve the base timestamp and the
  /// associated fast offset for the realtime clock.
  static std::function<void(int64_t& base, int64_t& offset)> defaultRealtimeBaseConfigurator(uint64_t offset, uv_loop_t* loop);
  static std::function<int64_t(int64_t base, int64_t offset)> defaultCPUTimeConfigurator();
};

/// Helper struct to hold state of the data processing while it is running.
/// This is meant to then be used to report the state of the data processing
/// to the driver.
/// This is similar to the DataProcessingStats, however it can only track
/// the fact that a given substate (registered as a metric) has changed. No
/// other operations are supported.
struct DataProcessingStates {
  DataProcessingStates(std::function<void(int64_t& base, int64_t& offset)> getRealtimeBase,
                       std::function<int64_t(int64_t base, int64_t offset)> getTimestamp);

  constexpr static ServiceKind service_kind = ServiceKind::Global;
  constexpr static int STATES_BUFFER_SIZE = 1 << 16;
  constexpr static int MAX_STATES = 256;

  // This is the structure to request the state update
  struct CommandSpec {
    int id = -1;                // Id of the state to update.
    int size = 0;               // Size of the state.
    char const* data = nullptr; // Pointer to the beginning of the state
  };

  // This is the structure to keep track of local updates to the states.
  // Notice how the states are kept in a single buffer in reverse order
  // and that the actual payload is stored after the header.
  // This way we can simply flush the buffer by iterating from the
  // previous insertion point forward, skipping things which have and
  // older timestamp. Besides the generation, we can also keep track of
  struct CommandHeader {
    short id = 0;          // The id of the state to update
    int size = 0;          // The size of the state
    int64_t timestamp = 0; // Timestamp of the update
  };

  // This structure is used to keep track of the last updates
  // for each of the metrics. This can be used to defer the need
  // to flush the buffers to the remote end, so that we do not need to
  // send metrics synchronously but we can do e.g. as a batch update.
  // It also prevents that we send the same metric multiple times, because
  // we keep track of the time of the last update.
  struct UpdateInfo {
    int64_t timestamp = 0;     // When the update actually took place
    int64_t lastPublished = 0; // When the update was last published
    int64_t generation = -1;   // The generation which did the update
  };

  struct StateSpec {
    // Id of the metric. It must match the index in the metrics array.
    // Name of the metric
    std::string name = "";
    int stateId = -1;
    /// The default value for the state
    char defaultValue = 0;
    /// How many milliseconds must have passed since the last publishing
    int64_t minPublishInterval = 0;
    /// After how many milliseconds we should still refresh the metric
    /// -1 means that we never refresh it automatically.
    uint64_t maxRefreshLatency = -1;
    /// Wether or not to consider the metric as updated when we
    /// register it.
    bool sendInitialValue = false;
  };

  struct StateView {
    /// Pointer to the beginning of the state
    int first = 0;
    /// Size of the state
    short size = 0;
    /// Extra capacity for the state
    short capacity = 0;
  };

  void registerState(StateSpec const& spec);
  // Update some stats as specified by the @cmd cmd
  void updateState(CommandSpec state);
  // Flush the states which are pending on the intermediate buffer.
  void processCommandQueue();

  void flushChangedStates(std::function<void(std::string const&, int64_t, std::string_view)> const& callback);
  void repack();

  std::atomic<size_t> statesSize;

  std::array<char, STATES_BUFFER_SIZE> store = {};
  std::array<int64_t, MAX_STATES> statesIndex = {};
  std::vector<char> statesBuffer;
  std::array<StateView, MAX_STATES> statesViews = {};
  std::array<bool, MAX_STATES> updated = {};
  std::array<std::string, MAX_STATES> stateNames = {};
  std::array<UpdateInfo, MAX_STATES> updateInfos;
  std::array<StateSpec, MAX_STATES> stateSpecs;
  // How many commands have been committed to the queue.
  std::atomic<int> insertedStates = 0;
  // The insertion point for the next state. Notice we
  // insert in the buffer backwards, so that on flush we iterate
  // from the last insertion point forward.
  std::atomic<int> nextState = STATES_BUFFER_SIZE;
  // How many commands are currently in flight.
  std::atomic<int> pendingStates = 0;
  int64_t lastFlushedToRemote = 0;
  int64_t lastMetrics = 0;

  // Function to retrieve an aritrary base for the realtime clock.
  std::function<void(int64_t& base, int64_t& offset)> getRealtimeBase;
  // Function to retrieve the timestamp from the value returned by getRealtimeBase.
  std::function<int64_t(int64_t base, int64_t offset)> getTimestamp;
  // The value of the uv_hrtime() at the last update.
  int64_t realTimeBase = 0;
  // The value of the uv_now() at the last update.
  int64_t initialTimeOffset = 0;

  // Invoke to make sure that the updatedMetricsTotal is updated.
  void lapseTelemetry()
  {
    updatedMetricsTotal += updatedMetricsLapse.load();
    pushedMetricsTotal += pushedMetricsLapse;
    publishedMetricsTotal += publishedMetricsLapse;
    updatedMetricsLapse = 0;
    pushedMetricsLapse = 0;
    publishedMetricsLapse = 0;
  }

  // How many times we have invoked the processing of the command queue.
  // Notice that we use this to order the updates, so that we need
  // to update only once per generation, because the items are
  // inserted in the buffer in reverse time order and only
  // the more recent update is interesting for us.
  std::atomic<int64_t> generation = 0;
  // Telemetry for the metric updates and pushes
  std::atomic<int64_t> updatedMetricsLapse = 0;
  int64_t updatedMetricsTotal = 0;
  int64_t pushedMetricsTotal = 0;
  int64_t pushedMetricsLapse = 0;
  int64_t publishedMetricsTotal = 0;
  int64_t publishedMetricsLapse = 0;
  int64_t publishingInvokedTotal = 0;
  int64_t publishingDoneTotal = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAPROCESSINGSTATES_H_
