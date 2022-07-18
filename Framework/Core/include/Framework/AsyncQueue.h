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
#ifndef O2_FRAMEWORK_ASYNCQUUE_H_
#define O2_FRAMEWORK_ASYNCQUUE_H_

#include "Framework/TimesliceSlot.h"
#include <functional>
#include <string>
#include <vector>

namespace o2::framework
{

struct AsyncTaskSpec {
  std::string name;
  // Its priority compared to the other tasks
  int score = 0;
};

/// The position of the TaskSpec in the prototypes
struct AsyncTaskId {
  int value = -1;
};

/// An actuatual task to be executed
struct AsyncTask {
  // The task to be executed
  std::function<void()> task;
  // The associated task spec
  AsyncTaskId id = {-1};
  TimesliceId timeslice = {TimesliceId::INVALID};
  // Only the task with the highest debounce value will be executed
  int debounce = 0;
  bool runnable = false;
};

struct AsyncQueue {
  std::vector<AsyncTaskSpec> prototypes;
  std::vector<AsyncTask> tasks;
  size_t iteration = 0;
};

struct AsyncQueueHelpers {
  static AsyncTaskId create(AsyncQueue& queue, AsyncTaskSpec spec);
  // Schedule a task with @a taskId to be executed whenever the timeslice
  // is past timeslice. If debounce is provided, only execute the task
  static void post(AsyncQueue& queue, AsyncTaskId taskId, std::function<void()> task, TimesliceId timeslice, int64_t debounce = 0);
  /// Run all the tasks which are older than the oldestPossible timeslice
  /// executing them by:
  /// 1. sorting the tasks by timeslice
  /// 2. then priority
  /// 3. only execute the highest (timeslice, debounce) value
  static void run(AsyncQueue& queue, TimesliceId oldestPossibleTimeslice);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_ASYNCQUEUE_H_
