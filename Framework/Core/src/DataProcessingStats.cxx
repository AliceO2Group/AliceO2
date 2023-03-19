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

#include "Framework/DataProcessingStats.h"
#include "Framework/RuntimeError.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceState.h"
#include "Framework/Logger.h"
#include <uv.h>
#include <iostream>
#include <atomic>
#include <utility>

namespace o2::framework
{

DataProcessingStats::DataProcessingStats(std::function<void(int64_t& base, int64_t& offset)> getRealtimeBase_,
                                         std::function<int64_t(int64_t base, int64_t offset)> getTimestamp_)
  : getTimestamp(getTimestamp_),
    getRealtimeBase(getRealtimeBase_)
{
  getRealtimeBase(realTimeBase, initialTimeOffset);
}

std::function<void(int64_t&, int64_t&)> DataProcessingStatsHelpers::defaultRealtimeBaseConfigurator(uint64_t startTimeOffset, uv_loop_t* loop)
{
  return [startTimeOffset, loop](int64_t& base, int64_t& offset) {
    uv_update_time(loop);
    base = startTimeOffset + uv_now(loop);
    offset = uv_hrtime();
  };
}

// Implement getTimestampConfigurator based on getRealtimeBaseConfigurator
std::function<int64_t(int64_t, int64_t)> DataProcessingStatsHelpers::defaultCPUTimeConfigurator()
{
  return [](int64_t base, int64_t offset) -> int64_t {
    return base + (uv_hrtime() - offset) / 1000000;
  };
}

void DataProcessingStats::updateStats(CommandSpec cmd)
{
  if (metricSpecs[cmd.id].name.empty()) {
    throw runtime_error_f("MetricID %d was not registered", (int)cmd.id);
  }
  if (cmd.id >= metrics.size()) {
    throw runtime_error_f("MetricID %d is out of range", (int)cmd.id);
  }
  if (cmd.id >= updateInfos.size()) {
    throw runtime_error_f("MetricID %d is out of range", (int)cmd.id);
  }
  pendingCmds++;
  // Add a static mutex to protect the queue
  // Get the next available operation in an atomic way.
  auto idx = nextCmd.fetch_add(1, std::memory_order_relaxed);
  if (idx == cmds.size()) {
    // We abort this command
    pendingCmds--;
    while (pendingCmds.load(std::memory_order_relaxed) > 0) {
      // We need to wait for all the pending commands to be processed.
      // This is needed because we are going to flush the queue.
      // We cannot flush the queue while there are pending commands
      // as we might end up with a command being processed twice.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    processCommandQueue();
    insertedCmds.store(0, std::memory_order_relaxed);
    nextCmd.store(0, std::memory_order_relaxed);
    pendingCmds++;
    idx = nextCmd.fetch_add(1, std::memory_order_relaxed);
  } else if (idx > cmds.size()) {
    while (cmds.size()) {
      // We need to wait for the flushing of the queue
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      idx = nextCmd.load(std::memory_order_relaxed);
    }
    return updateStats(cmd);
  }
  // Save the command.
  assert(idx < cmds.size());
  assert(cmd.id < metrics.size());
  cmds[idx] = Command{cmd.id, cmd.value, getTimestamp(realTimeBase, initialTimeOffset), cmd.op};
  insertedCmds++;
  pendingCmds--;
  // Keep track of the number of commands we have received.
  updatedMetricsLapse++;
}

void DataProcessingStats::processCommandQueue()
{
  std::array<char, MAX_CMDS> order;
  // The range cannot be larger than the number of commands we have.
  auto range = insertedCmds.load(std::memory_order_relaxed);
  if (insertedCmds.load(std::memory_order_relaxed) == 0) {
    return;
  }

  std::iota(order.begin(), order.begin() + range, 0);
  // Shuffle the order in which we will process the commands based
  // on their timestamp. If two commands are inserted at the same
  // time, we expect to process them in the order they were inserted.
  std::stable_sort(order.begin(), order.begin() + range, [this](char a, char b) {
    return cmds[a].timestamp < cmds[b].timestamp;
  });

  // Process the commands in the order we have just computed.
  for (int i = 0; i < range; ++i) {
    auto& cmd = cmds[i];
    assert(cmd.id < updateInfos.size());
    auto& update = updateInfos[cmd.id];
    switch (cmd.op) {
      case Op::Nop:
        break;
      case Op::Set:
        if (cmd.value != metrics[cmd.id] && cmd.timestamp >= update.timestamp) {
          metrics[cmd.id] = cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::Add:
        if (cmd.value) {
          metrics[cmd.id] += cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::Sub:
        if (cmd.value) {
          metrics[cmd.id] -= cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::Max:
        if (cmd.value > metrics[cmd.id]) {
          metrics[cmd.id] = cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::Min:
        if (cmd.value < metrics[cmd.id]) {
          metrics[cmd.id] = cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::SetIfPositive:
        if (cmd.value > 0 && cmd.timestamp >= update.timestamp) {
          metrics[cmd.id] = cmd.value;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
      case Op::InstantaneousRate:
        if (cmd.value > 0) {
          auto delta = cmd.timestamp - update.timestamp;
          if (delta > 0) {
            metrics[cmd.id] = cmd.value / delta;
            updated[cmd.id] = true;
            update.timestamp = cmd.timestamp;
            pushedMetricsLapse++;
          }
        }
        break;
      case Op::CumulativeRate:
        if (cmd.value > 0) {
          auto delta = cmd.timestamp - update.timestamp;
          auto deltaLastPublished = update.timestamp - update.lastPublished;
          // Too close, skip.
          if (delta == 0) {
            break;
          }
          // The sum since last publication is the period of time since last
          // publication multiplied by the rate since last publication.
          auto CAn = metrics[cmd.id];
          auto newV = cmd.value;
          auto totalInt = delta + deltaLastPublished; // This is in milliseconds
          if (totalInt == 0) {
            break;
          }
          auto newRate = ((CAn) + ((newV - CAn) * 1000) / totalInt);
          metrics[cmd.id] = newRate;
          updated[cmd.id] = true;
          update.timestamp = cmd.timestamp;
          pushedMetricsLapse++;
        }
        break;
    }
  }
  // No one should have tried to insert more commands while processing.
  assert(range == insertedCmds.load(std::memory_order_relaxed));
  nextCmd.store(0, std::memory_order_relaxed);
  insertedCmds.store(0, std::memory_order_relaxed);
}

void DataProcessingStats::flushChangedMetrics(std::function<void(std::string const&, int64_t, int64_t)> const& callback)
{
  publishingInvokedTotal++;
  bool publish = false;
  auto currentTimestamp = getTimestamp(realTimeBase, initialTimeOffset);
  for (size_t mi = 0; mi < updated.size(); ++mi) {
    auto& update = updateInfos[mi];
    auto& spec = metricSpecs[mi];
    if (currentTimestamp - update.timestamp > spec.maxRefreshLatency) {
      updated[mi] = true;
      update.timestamp = currentTimestamp;
    }
    if (updated[mi] == false) {
      continue;
    }
    if (currentTimestamp - update.lastPublished < spec.minPublishInterval) {
      continue;
    }
    publish = true;
    callback(spec.name.data(), update.timestamp, metrics[mi]);
    publishedMetricsLapse++;
    update.lastPublished = currentTimestamp;
    updated[mi] = false;
  }
  if (publish) {
    publishingDoneTotal++;
  }
  static int64_t startTime = uv_hrtime();
  int64_t now = uv_hrtime();
  double averageInvocations = (publishingInvokedTotal * 1000000000) / (now - startTime);
  double averagePublishing = (publishedMetricsLapse * 1000000000) / (now - startTime);

  LOGP(debug, "Publishing invoked {} times / s, {} metrics published / s", (int)averageInvocations, (int)averagePublishing);
}

void DataProcessingStats::registerMetric(MetricSpec const& spec)
{
  if (spec.metricId >= metricSpecs.size()) {
    throw runtime_error_f("Metric id %d is out of range. Max is %d", spec.metricId, metricSpecs.size());
  }
  if (metricSpecs[spec.metricId].name.size() != 0 && spec.name != metricSpecs[spec.metricId].name) {
    auto currentName = metricSpecs[spec.metricId].name;
    throw runtime_error_f("Metric %d already registered with name %s", spec.metricId, currentName.data(), spec.name.data());
  }
  auto currentMetric = std::find_if(metricSpecs.begin(), metricSpecs.end(), [&spec](MetricSpec const& s) { return s.name == spec.name && s.metricId != spec.metricId; });
  if (currentMetric != metricSpecs.end()) {
    throw runtime_error_f("Metric %s already registered with id %d. Cannot reregister with %d.", spec.name.data(), currentMetric->metricId, spec.metricId);
  }
  metricSpecs[spec.metricId] = spec;
  metricsNames[spec.metricId] = spec.name;
  metrics[spec.metricId] = spec.defaultValue;
  int64_t currentTime = getTimestamp(realTimeBase, initialTimeOffset);
  updateInfos[spec.metricId] = UpdateInfo{currentTime, currentTime};
  updated[spec.metricId] = spec.sendInitialValue;
}

} // namespace o2::framework
