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

#include "Framework/DataProcessingStates.h"
#include "Framework/RuntimeError.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceState.h"
#include "Framework/Logger.h"
#include <uv.h>
#include <iostream>
#include <atomic>
#include <utility>
#include <string_view>

namespace o2::framework
{

DataProcessingStates::DataProcessingStates(std::function<void(int64_t& base, int64_t& offset)> getRealtimeBase_,
                                           std::function<int64_t(int64_t base, int64_t offset)> getTimestamp_)
  : getTimestamp(getTimestamp_),
    getRealtimeBase(getRealtimeBase_)
{
  getRealtimeBase(realTimeBase, initialTimeOffset);
}

void DataProcessingStates::processCommandQueue()
{
  int position = nextState.load(std::memory_order_relaxed);
  // Process the commands in the order we have just computed.
  while (position < DataProcessingStates::STATES_BUFFER_SIZE) {
    DataProcessingStates::CommandHeader header;
    // Avoid alignment issues.
    memcpy(&header, &store[position], sizeof(DataProcessingStates::CommandHeader));
    int id = header.id;
    int64_t timestamp = header.timestamp;
    int size = header.size;
    assert(id < updateInfos.size());
    auto& update = updateInfos[id];
    // We need to update only once per invoked callback,
    // because the metrics are stored in reverse insertion order.
    if (generation > update.generation) {
      // If we have enough capacity, we reuse the buffer.
      if (statesViews[id].capacity >= size) {
        memcpy(statesBuffer.data() + statesViews[id].first, &store[position + sizeof(DataProcessingStates::CommandHeader)], size);
        statesViews[id].size = size;
        updated[id] = true;
        update.timestamp = timestamp;
        update.generation = generation;
        pushedMetricsLapse++;
      } else if (statesViews[id].capacity < size) {
        // Otherwise we need to reallocate.
        int newCapacity = std::max(size, 64);
        int first = statesBuffer.size();
        statesBuffer.resize(statesBuffer.size() + newCapacity);
        memcpy(statesBuffer.data() + first, &store[position + sizeof(DataProcessingStates::CommandHeader)], size);
        statesViews[id].first = first;
        statesViews[id].size = size;
        statesViews[id].capacity = newCapacity;
        updated[id] = true;
        update.timestamp = timestamp;
        update.generation = generation;
        pushedMetricsLapse++;
      }
    }
    position += sizeof(DataProcessingStates::CommandHeader) + header.size;
  }
  assert(position == DataProcessingStates::STATES_BUFFER_SIZE);
  // We reset the queue. Once again, the queue is filled in reverse order.
  nextState.store(STATES_BUFFER_SIZE, std::memory_order_relaxed);
  insertedStates.store(0, std::memory_order_relaxed);
  generation++;
}

void DataProcessingStates::updateState(CommandSpec cmd)
{
  if (stateSpecs[cmd.id].name.empty()) {
    throw runtime_error_f("StateID %d was not registered", (int)cmd.id);
  }
  if (cmd.id >= stateNames.size()) {
    throw runtime_error_f("StateID %d is out of range", (int)cmd.id);
  }
  if (cmd.id >= updateInfos.size()) {
    throw runtime_error_f("MetricID %d is out of range", (int)cmd.id);
  }
  pendingStates++;
  // Add a static mutex to protect the queue
  // Get the next available operation in an atomic way.
  auto size = sizeof(CommandHeader) + cmd.size;
  int idx = nextState.fetch_sub(size, std::memory_order_relaxed);
  if (idx - size < 0) {
    // We abort this command
    pendingStates--;
    while (pendingStates.load(std::memory_order_relaxed) > 0) {
      // We need to wait for all the pending commands to be processed.
      // This is needed because we are going to flush the queue.
      // We cannot flush the queue while there are pending commands
      // as we might end up with a command being processed twice.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    processCommandQueue();
    insertedStates.store(0, std::memory_order_relaxed);
    nextState.store(STATES_BUFFER_SIZE, std::memory_order_relaxed);
    pendingStates++;
    idx = nextState.fetch_sub(size, std::memory_order_relaxed);
  } else if (idx < 0) {
    while (idx < 0) {
      // We need to wait for the flushing of the queue
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      idx = nextState.load(std::memory_order_relaxed);
    }
    return updateState(cmd);
  }
  // Save the state in the queue
  assert(idx >= 0);
  assert(cmd.id < statesViews.size());
  int64_t timestamp = getTimestamp(realTimeBase, initialTimeOffset);
  // We also write starting from idx - size, because we know this is
  // reserved for us.
  idx -= size;
  CommandHeader header{(short)cmd.id, cmd.size, timestamp};
  memcpy(&store.data()[idx], &header, sizeof(CommandHeader));
  memcpy(&store.data()[idx + sizeof(CommandHeader)], cmd.data, cmd.size);
  insertedStates++;
  pendingStates--;
  // Keep track of the number of commands we have received.
  updatedMetricsLapse++;
}

void DataProcessingStates::flushChangedStates(std::function<void(std::string const&, int64_t, std::string_view)> const& callback)
{
  publishingInvokedTotal++;
  bool publish = false;
  auto currentTimestamp = getTimestamp(realTimeBase, initialTimeOffset);
  for (size_t mi = 0; mi < updated.size(); ++mi) {
    auto& update = updateInfos[mi];
    auto& spec = stateSpecs[mi];
    auto& view = statesViews[mi];
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
    callback(spec.name.data(), update.timestamp, std::string_view(statesBuffer.data() + view.first, view.size));
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

void DataProcessingStates::repack()
{
  // Everytime we publish, we repack the states buffer so that we can minimize the
  // amount of memory me use.
  std::array<int, MAX_STATES> order;
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.begin() + statesViews.size(), [&](int a, int b) {
    return statesViews[a].first < statesViews[b].first;
  });
  int position = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    auto& view = statesViews[order[i]];
    // If we have no size, we do not need to move anything.
    if (view.size == 0) {
      continue;
    }
    // If we are already in the correct place, do nothing.
    if (view.first == position) {
      continue;
    }
    memcpy(statesBuffer.data() + position, statesBuffer.data() + view.first, view.size);
    view.first = position;
    view.capacity = view.size;
    position += view.size;
  }
}

void DataProcessingStates::registerState(StateSpec const& spec)
{
  if (stateSpecs[spec.stateId].name.size() != 0 && spec.name != stateSpecs[spec.stateId].name) {
    auto currentName = stateSpecs[spec.stateId].name;
    throw runtime_error_f("Metric %d already registered with name %s", spec.stateId, currentName.data(), spec.name.data());
  }
  auto currentMetric = std::find_if(stateSpecs.begin(), stateSpecs.end(), [&spec](StateSpec const& s) { return s.name == spec.name && s.stateId != spec.stateId; });
  if (currentMetric != stateSpecs.end()) {
    throw runtime_error_f("Metric %s already registered with id %d. Cannot reregister with %d.", spec.name.data(), currentMetric->stateId, spec.stateId);
  }
  stateSpecs[spec.stateId] = spec;
  stateNames[spec.stateId] = spec.name;
  int64_t currentTime = getTimestamp(realTimeBase, initialTimeOffset);
  updateInfos[spec.stateId] = UpdateInfo{currentTime, currentTime};
  updated[spec.stateId] = spec.sendInitialValue;
}

} // namespace o2::framework
