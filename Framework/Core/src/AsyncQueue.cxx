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
#include "Framework/Signpost.h"
#include <numeric>

O2_DECLARE_DYNAMIC_LOG(async_queue);

namespace o2::framework
{
AsyncQueue::AsyncQueue()
{
}

auto AsyncQueueHelpers::create(AsyncQueue& queue, AsyncTaskSpec spec) -> AsyncTaskId
{
  AsyncTaskId id;
  id.value = queue.prototypes.size();
  queue.prototypes.push_back(spec);
  return id;
}

auto AsyncQueueHelpers::post(AsyncQueue& queue, AsyncTask const& task) -> void
{
  queue.tasks.push_back(task);
}

auto AsyncQueueHelpers::run(AsyncQueue& queue, TimesliceId oldestPossible) -> void
{
  if (queue.tasks.empty()) {
    return;
  }
  O2_SIGNPOST_ID_GENERATE(opid, async_queue);
  O2_SIGNPOST_START(async_queue, opid, "run", "Attempting at running %zu tasks with oldestPossible timeframe %zu", queue.tasks.size(), oldestPossible.value);
  std::vector<int> order;
  order.resize(queue.tasks.size());
  std::iota(order.begin(), order.end(), 0);
  // Decide wether or not they can run as a first thing
  for (auto& task : queue.tasks) {
    if (task.timeslice.value <= oldestPossible.value) {
      task.runnable = true;
    }
    O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run",
                           "Task %d (timeslice %zu), score %d, debounce %d is %{public}s when oldestPossible timeframe is %zu",
                           task.id.value, task.timeslice.value, queue.prototypes[task.id.value].score, task.debounce,
                           task.runnable ? "runnable" : "not runnable", oldestPossible.value);
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
      O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run", "Running task %d (%d), (timeslice %zu), score %d, debounce %d", queue.tasks[i].id.value, i, queue.tasks[i].timeslice.value, queue.prototypes[queue.tasks[i].id.value].score, queue.tasks[i].debounce);
    } else {
      O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run", "Skipping task %d (%d) (timeslice %zu), score %d, debounce %d", queue.tasks[i].id.value, i, queue.tasks[i].timeslice.value, queue.prototypes[queue.tasks[i].id.value].score, queue.tasks[i].debounce);
    }
  }
  // Keep only the tasks with the highest debounce value for a given id
  // For this reason I need to keep the callback in the task itself, because
  // two different callbacks with the same id will be coalesced.
  auto newEnd = std::unique(order.begin(), order.end(), [&queue](int a, int b) {
    return queue.tasks[a].runnable == queue.tasks[b].runnable && queue.tasks[a].id.value == queue.tasks[b].id.value && queue.tasks[a].debounce >= 0 && queue.tasks[b].debounce >= 0;
  });
  for (auto ii = newEnd; ii != order.end(); ii++) {
    O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "dropping", "Dropping task %d for timeslice %zu", queue.tasks[*ii].id.value, queue.tasks[*ii].timeslice.value);
  }
  order.erase(newEnd, order.end());

  if (order.empty() && queue.tasks.size() > 0) {
    O2_SIGNPOST_END(async_queue, opid, "run", "Not running iteration %zu pending %zu.",
                    queue.iteration, queue.tasks.size());
    return;
  } else if (order.empty()) {
    O2_SIGNPOST_END(async_queue, opid, "run", "Not running iteration %zu. No tasks.", queue.iteration);
    return;
  }
  O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run", "Running %zu tasks in iteration %zu", order.size(), queue.iteration);

  int runCount = 0;
  for (auto i : order) {
    if (queue.tasks[i].runnable) {
      runCount++;
      // If a task is runable, we can run the task and remove it from the queue
      O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run", "Running task %{public}s (%d) for timeslice %zu",
                             queue.prototypes[queue.tasks[i].id.value].name.c_str(), i,
                             queue.tasks[i].timeslice.value);
      queue.tasks[i].callback(queue.tasks[i], opid.value);
      O2_SIGNPOST_EVENT_EMIT(async_queue, opid, "run", "Done running %d", i);
    }
  }
  // Remove all runnable tasks regardless  they actually
  // ran or they were skipped due to debouncing.
  queue.tasks.erase(std::remove_if(queue.tasks.begin(), queue.tasks.end(), [&queue](AsyncTask const& task) {
                      return task.runnable;
                    }),
                    queue.tasks.end());
  O2_SIGNPOST_END(async_queue, opid, "run", "Done running %d/%zu tasks", runCount, order.size());
}

auto AsyncQueueHelpers::reset(AsyncQueue& queue) -> void
{
  queue.tasks.clear();
  queue.iteration = 0;
}

} // namespace o2::framework
