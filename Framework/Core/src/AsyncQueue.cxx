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

#include "Framework/AsyncQueue.h"
#include <numeric>

namespace o2::framework
{
auto AsyncQueueHelpers::create(AsyncQueue& queue, AsyncTaskSpec spec) -> AsyncTaskId
{
  AsyncTaskId id;
  id.value = queue.prototypes.size();
  queue.prototypes.push_back(spec);
  return id;
}

auto AsyncQueueHelpers::post(AsyncQueue& queue, AsyncTaskId id, std::function<void()> task, int64_t debounce) -> void
{
  AsyncTask taskToPost;
  taskToPost.task = task;
  taskToPost.id = id;
  taskToPost.debounce = debounce;
  queue.tasks.push_back(taskToPost);
}

auto AsyncQueueHelpers::run(AsyncQueue& queue) -> void
{
  std::vector<int> order;
  order.resize(queue.tasks.size());
  std::iota(order.begin(), order.end(), 0);

  // Sort by priority and debounce
  std::sort(order.begin(), order.end(), [&queue](int a, int b) {
    if (queue.tasks[a].id.value == -1 || queue.tasks[b].id.value == -1) {
      return false;
    }
    if (queue.tasks[a].id.value == queue.tasks[b].id.value) {
      return queue.tasks[a].debounce > queue.tasks[b].debounce;
    }
    return queue.prototypes[queue.tasks[a].id.value].score > queue.prototypes[queue.tasks[b].id.value].score;
  });
  // Keep only the tasks with the highest debounce value for a given id
  auto newEnd = std::unique(order.begin(), order.end(), [&queue](int a, int b) {
    return queue.tasks[a].id.value == queue.tasks[b].id.value;
  });
  order.erase(newEnd, order.end());

  for (auto i : order) {
    queue.tasks[i].task();
  }
  queue.tasks.clear();
}

} // namespace o2::framework
