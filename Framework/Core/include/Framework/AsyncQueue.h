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
#include <string>
#include <vector>
#include <atomic>

typedef struct x9_inbox_internal x9_inbox;
typedef struct x9_node_internal x9_node;

namespace o2::framework
{

/// The position of the TaskSpec in the prototypes
/// Up to 127 different kind of tasks are allowed per queue
struct AsyncTaskId {
  int16_t value = -1;
};

/// An actuatual task to be executed
struct AsyncTask {
  // The timeslice after which this callback should be executed.
  TimesliceId timeslice = {TimesliceId::INVALID};
  // Only the task with the highest debounce value will be executed
  // The associated task spec. Notice that the context
  // is stored separately so that we do not need to
  // manage lifetimes of std::functions
  AsyncTaskId id = {-1};
  // Debounce value.
  int8_t debounce = 0;
  bool runnable = false;
  // Some unuser integer
  int32_t unused = 0;
  // The callback to be executed. id can be used as unique
  // id for the signpost in the async_queue stream.
  // Context is provided by the userdata attached to the
  // task the moment we post it.
  // Notice that we do not store the task in the prototype
  // because we do support at the moment coalescing two
  // different callbaks with the same id.
  void (*callback)(AsyncTask& task, size_t id);
  // Some user data e.g. to decode what comes next
  // This can either be used via the .data pointer
  // or by asking a cast to the appropriate type via
  // the user() method.
  void* data[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

  // Helper to return userdata
  template <typename T>
  T& user()
  {
    static_assert(sizeof(T) <= 5 * sizeof(void*), "User object does not fit user data");
    return *(T*)data;
  }

  // Helper to set userdata
  // @return a reference to this task modified. This is meant to be used like:
  //
  // AsyncQueueHelpers::post(queue, AsyncTask{.id = myTask}.user(Context{.contextValue}));
  //
  // Coupled with the other one:
  //
  // task.user<Context>().contextValue;
  //
  // it can be used to mimick capturing lambdas
  template <typename T>
  AsyncTask& user(T&& value)
  {
    static_assert(sizeof(T) <= 5 * sizeof(void*), "User object does not fit user data");
    new (&data[0])(T){value};
    return *this;
  }
};

struct AsyncTaskSpec {
  std::string name;
  // Its priority compared to the other tasks
  int score = 0;
};

struct AsyncQueue {
  std::vector<AsyncTaskSpec> prototypes;
  std::vector<AsyncTask> tasks;
  size_t iteration = 0;

  std::atomic<bool> first = true;

  // Inbox for the message queue used to append
  // tasks to this queue.
  x9_inbox* inbox = nullptr;
  AsyncQueue();
};

struct AsyncQueueHelpers {
  static AsyncTaskId create(AsyncQueue& queue, AsyncTaskSpec spec);
  // Schedule a task with @a taskId to be executed whenever the timeslice
  // is past timeslice. If debounce is provided, only execute the task
  static void post(AsyncQueue& queue, AsyncTask const& task);
  /// Run all the tasks which are older than the oldestPossible timeslice
  /// executing them by:
  /// 1. sorting the tasks by timeslice
  /// 2. then priority
  /// 3. only execute the highest (timeslice, debounce) value
  static void run(AsyncQueue& queue, TimesliceId oldestPossibleTimeslice);

  // Flush tasks which were posted but not yet committed to the queue
  static void flushPending(AsyncQueue& queue);
  /// Reset the queue to its initial state
  static void reset(AsyncQueue& queue);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_ASYNCQUEUE_H_
