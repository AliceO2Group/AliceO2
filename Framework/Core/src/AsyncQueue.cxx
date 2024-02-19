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
#include "Framework/Logger.h"
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

auto AsyncQueueHelpers::post(AsyncQueue& queue, AsyncTaskId id, std::function<void()> task, TimesliceId timeslice, int64_t debounce) -> void
{
  AsyncTask taskToPost;
  taskToPost.task = task;
  taskToPost.id = id;
  taskToPost.timeslice = timeslice;
  taskToPost.debounce = debounce;
  queue.tasks.push_back(taskToPost);
}

auto AsyncQueueHelpers::run(AsyncQueue& queue, TimesliceId oldestPossible) -> void
{
  if (queue.tasks.empty()) {
    return;
  }
  LOGP(debug, "Attempting at running {} tasks", queue.tasks.size());
  std::vector<int> order;
  order.resize(queue.tasks.size());
  std::iota(order.begin(), order.end(), 0);
  // Decide wether or not they can run as a first thing
  for (auto& task : queue.tasks) {
    if (task.timeslice.value <= oldestPossible.value) {
      task.runnable = true;
    }
  }

  // Sort by runnable, timeslice, then priority and finally debounce
  std::sort(order.begin(), order.end(), [&queue](int a, int b) {
    if (queue.tasks[a].runnable && !queue.tasks[b].runnable) {
      return true;
    }
    if (!queue.tasks[a].runnable && queue.tasks[b].runnable) {
      return false;
    }
    if (queue.tasks[a].timeslice.value == queue.tasks[b].timeslice.value) {
      if (queue.tasks[a].id.value == -1 || queue.tasks[b].id.value == -1) {
        return false;
      }
      if (queue.tasks[a].id.value == queue.tasks[b].id.value) {
        return queue.tasks[a].debounce > queue.tasks[b].debounce;
      }
      return queue.prototypes[queue.tasks[a].id.value].score > queue.prototypes[queue.tasks[b].id.value].score;
    } else {
      return queue.tasks[a].timeslice.value > queue.tasks[b].timeslice.value;
    }
  });
  for (auto i : order) {
    if (queue.tasks[i].runnable) {
      LOGP(debug, "AsyncQueue: Running task {}, timeslice {}, score {}, debounce {}", queue.tasks[i].id.value, queue.tasks[i].timeslice.value, queue.prototypes[queue.tasks[i].id.value].score, queue.tasks[i].debounce);
    } else {
      LOGP(debug, "AsyncQueue: Skipping task {}, timeslice {}, score {}, debounce {}", queue.tasks[i].id.value, queue.tasks[i].timeslice.value, queue.prototypes[queue.tasks[i].id.value].score, queue.tasks[i].debounce);
    }
  }
  // Keep only the tasks with the highest debounce value for a given id
  auto newEnd = std::unique(order.begin(), order.end(), [&queue](int a, int b) {
    return queue.tasks[a].runnable == queue.tasks[b].runnable && queue.tasks[a].id.value == queue.tasks[b].id.value && queue.tasks[a].debounce >= 0 && queue.tasks[b].debounce >= 0;
  });
  order.erase(newEnd, order.end());

  if (order.empty() && queue.tasks.size() > 0) {
    LOGP(debug, "AsyncQueue: not running iteration {} timeslice {} pending {}.", order.size(), queue.iteration, oldestPossible.value, queue.tasks.size());
    return;
  } else if (order.empty()) {
    return;
  }
  LOGP(debug, "AsyncQueue: Running {} tasks in iteration {} timeslice {}", order.size(), queue.iteration, oldestPossible.value);
  bool obsolete = true;

  for (auto i : order) {
    if (queue.tasks[i].runnable) {
      // If a task is runable, we can run the task and remove it from the queue
      LOGP(debug, "Running task {} ({})", queue.prototypes[queue.tasks[i].id.value].name, i);
      queue.tasks[i].task();
      LOGP(debug, "Done running {}", i);
    }
  }
  // Remove all runnable tasks regardless  they actually
  // ran or they were skipped due to debouncing.
  queue.tasks.erase(std::remove_if(queue.tasks.begin(), queue.tasks.end(), [&queue](AsyncTask const& task) {
                      return task.runnable;
                    }),
                    queue.tasks.end());
}

auto AsyncQueueHelpers::reset(AsyncQueue& queue) -> void
{
  queue.tasks.clear();
  queue.iteration = 0;
}

} // namespace o2::framework
