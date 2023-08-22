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
#ifndef O2_FRAMEWORK_DATAPROCESSINGSTATS_H_
#define O2_FRAMEWORK_DATAPROCESSINGSTATS_H_

#include "DeviceState.h"
#include "Framework/ServiceSpec.h"
#include <atomic>
#include <cstdint>
#include <array>
#include <numeric>
#include <mutex>
#include <utility>

namespace o2::framework
{

enum struct ProcessingStatsId : short {
  ERROR_COUNT,
  EXCEPTION_COUNT,
  PENDING_INPUTS,
  INCOMPLETE_INPUTS,
  TOTAL_INPUTS,
  LAST_ELAPSED_TIME_MS,
  LAST_PROCESSED_SIZE,
  TOTAL_PROCESSED_SIZE,
  TOTAL_SIGUSR1,
  CONSUMED_TIMEFRAMES,
  AVAILABLE_MANAGED_SHM,
  LAST_SLOW_METRIC_SENT_TIMESTAMP,
  LAST_VERY_SLOW_METRIC_SENT_TIMESTAMP,
  LAST_METRIC_FLUSHED_TIMESTAMP,
  BEGIN_ITERATION_TIMESTAMP,
  PERFORMED_COMPUTATIONS,
  LAST_REPORTED_PERFORMED_COMPUTATIONS,
  TOTAL_BYTES_IN,
  TOTAL_BYTES_OUT,
  LAST_MIN_LATENCY,
  LAST_MAX_LATENCY,
  TOTAL_RATE_IN_MB_S,
  TOTAL_RATE_OUT_MB_S,
  PROCESSING_RATE_HZ,
  MALFORMED_INPUTS,
  DROPPED_COMPUTATIONS,
  DROPPED_INCOMING_MESSAGES,
  RELAYED_MESSAGES,
  CPU_USAGE_FRACTION,
  ARROW_BYTES_CREATED,
  ARROW_BYTES_DESTROYED,
  ARROW_MESSAGES_CREATED,
  ARROW_MESSAGES_DESTROYED,
  ARROW_BYTES_EXPIRED,
  RESOURCE_OFFER_EXPIRED,
  SHM_OFFER_BYTES_CONSUMED,
  RESOURCES_MISSING,
  RESOURCES_INSUFFICIENT,
  RESOURCES_SATISFACTORY,
  AVAILABLE_MANAGED_SHM_BASE = 512,
};

/// Helper struct to hold statistics about the data processing happening.
struct DataProcessingStats {
  DataProcessingStats(std::function<void(int64_t& base, int64_t& offset)> getRealtimeBase,
                      std::function<int64_t(int64_t base, int64_t offset)> getTimestamp);

  constexpr static ServiceKind service_kind = ServiceKind::Global;
  constexpr static unsigned short MAX_METRICS = 1 << 15;
  constexpr static short MAX_CMDS = 64;

  enum struct Op : char {
    Nop,               /// No operation
    Set,               /// Set the value to the specified value
    SetIfPositive,     /// Set the value to the specified value if it is positive
    CumulativeRate,    /// Update the rate of the metric given the cumulative value since last time it got published
    InstantaneousRate, /// Update the rate of the metric given the amount since the last time
    Add,               /// Add the value to the current value
    Sub,               /// Subtract the value from the current value
    Max,               /// Set the value to the maximum of the current value and the specified value
    Min                /// Set the value to the minimum of the current value and the specified value
  };

  // Kind of the metric. This is used to know how to interpret the value
  enum struct Kind : char {
    Int,
    UInt64,
    Double,
    Rate, /// A rate metric is sent out as a float and reset to 0 after each update
          /// Use the InstantaneousRate operation to update it. Most likely you also
          /// want that the minPublishInterval is as large as the maxRefreshLatency.
    Unknown,
  };

  // The scope for a given metric. DPL is used for the DPL Monitoring GUI,
  // Online is used for the online monitoring.
  enum struct Scope : char {
    DPL,
    Online
  };

  // This is what the user passes. Notice that there is no
  // need to specify the timestamp, because we calculate it for them
  // using the delta between the last update and the current time.
  struct CommandSpec {
    unsigned short id = 0;
    Op op = Op::Nop;
    int64_t value = 0;
  };

  // This is the structure to keep track of local updates to the stats.
  // Each command will be queued in a buffer and then flushed to the
  // global stats either when the buffer is full (after MAX_CMDS commands)
  // or when the queue is flushed explicitly via the processQueue() method.
  struct Command {
    unsigned short id = 0; // StatsId of the metric to update
    int64_t value = 0;     // Value to update the metric with
    int64_t timestamp = 0; // Timestamp of the update
    Op op = Op::Nop;       // Operation to perform to do the update
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
  };

  struct MetricSpec {
    // Id of the metric. It must match the index in the metrics array.
    // Name of the metric
    std::string name = "";
    // Wether or not the metric is enabled
    bool enabled = true;
    int metricId = -1;
    /// The kind of the metric
    Kind kind = Kind::Int;
    /// The scope of the metric
    Scope scope = Scope::DPL;
    /// The default value for the metric
    int64_t defaultValue = 0;
    /// How many milliseconds must have passed since the last publishing
    int64_t minPublishInterval = 0;
    /// After how many milliseconds we should still refresh the metric
    /// -1 means that we never refresh it automatically.
    uint64_t maxRefreshLatency = -1;
    /// Wether or not to consider the metric as updated when we
    /// register it.
    bool sendInitialValue = false;
  };

  void registerMetric(MetricSpec const& spec);
  // Update some stats as specified by the @cmd cmd
  void updateStats(CommandSpec cmd);

  /// This will process the queue of commands required to update the stats.
  /// It is meant to be called periodically by a single thread.
  void processCommandQueue();

  void flushChangedMetrics(std::function<void(MetricSpec const&, int64_t, int64_t)> const& callback);

  std::atomic<size_t> statesSize;

  std::array<Command, MAX_CMDS> cmds = {};
  std::array<int64_t, MAX_METRICS> metrics = {};
  std::array<bool, MAX_METRICS> updated = {};
  std::array<std::string, MAX_METRICS> metricsNames;
  std::array<UpdateInfo, MAX_METRICS> updateInfos;
  std::array<MetricSpec, MAX_METRICS> metricSpecs;
  std::array<int64_t, MAX_METRICS> lastPublishedMetrics;
  std::vector<int> availableMetrics;
  // How many commands have been committed to the queue.
  std::atomic<int> insertedCmds = 0;
  // The insertion point for the next command.
  std::atomic<int> nextCmd = 0;
  // How many commands are currently in flight.
  std::atomic<int> pendingCmds = 0;
  int64_t lastFlushedToRemote = 0;
  int64_t lastMetrics = 0;
  // This is the mutex to protect the queue of commands.
  std::mutex mMutex;

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

#endif // O2_FRAMEWORK_DATAPROCESSINGSTATS_H_
