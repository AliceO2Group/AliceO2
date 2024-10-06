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
#include <catch_amalgamated.hpp>
#include "Framework/AsyncQueue.h"

struct TaskContext {
  int& count;
};

/// Test debouncing functionality. The same task cannot be executed more than once
/// in a given run.
TEST_CASE("TestDebouncing")
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId = AsyncQueueHelpers::create(queue, {.name = "test", .score = 10});
  // Push two tasks on the queue with the same id
  int count = 0;
  AsyncQueueHelpers::post(queue,
                          AsyncTask{.timeslice = TimesliceId{0}, .id = taskId, .debounce = 10, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 1; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(queue,
                          AsyncTask{.timeslice = TimesliceId{1}, .id = taskId, .debounce = 20, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 2; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  REQUIRE(count == 2);
}

// Test task oridering. The tasks with the highest priority should be executed first.
TEST_CASE("TestPriority")
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  int count = 0;
  AsyncQueueHelpers::post(queue, AsyncTask{.timeslice = TimesliceId{0}, .id = taskId1, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 10; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(queue, AsyncTask{.timeslice = TimesliceId{0}, .id = taskId2, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count /= 10; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  REQUIRE(count == 10);
}

// Make sure we execute tasks only up to the timeslice provided to run.
TEST_CASE("TestOldestTimeslice")
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{1}, .id = taskId1, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 10; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{0}, .id = taskId2, .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 20; }}.user<TaskContext>({.count = count}));
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  REQUIRE(count == 20);
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  REQUIRE(count == 20);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  REQUIRE(count == 30);
}

// Make sure we execute tasks only up to the timeslice provided to run with bouncing enabled.
TEST_CASE("TestOldestTimesliceWithBounce")
{
  using namespace o2::framework;
  AsyncQueue queue;
  // Push two tasks on the queue with the same id
  auto count = 0;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  REQUIRE(taskId1.value != taskId2.value);
  AsyncQueueHelpers::post(queue, AsyncTask{
                                   .timeslice = TimesliceId{2},
                                   .id = taskId1,
                                   .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 10; }}
                                   .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(queue, AsyncTask{
                                   .timeslice = TimesliceId{1},
                                   .id = taskId2,
                                   .debounce = 10,
                                   .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 20; }}
                                   .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(queue, AsyncTask{
                                   .timeslice = TimesliceId{1},
                                   .id = taskId2,
                                   .debounce = 20,
                                   .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 30; }}
                                   .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  REQUIRE(count == 0);
  REQUIRE(queue.tasks.size() == 3);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  REQUIRE(count == 30);
  REQUIRE(queue.tasks.size() == 1);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  REQUIRE(count == 40);
  REQUIRE(queue.tasks.size() == 0);
}

// test bouncing disabled with negative value
TEST_CASE("TestOldestTimesliceWithNegativeBounce")
{
  using namespace o2::framework;
  AsyncQueue queue;
  int count = 0;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{2},
                     .id = taskId1,
                     .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 10; }}
             .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{1},
                     .id = taskId2,
                     .debounce = -10,
                     .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 20; }}
             .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{1},
                     .id = taskId2,
                     .debounce = -20,
                     .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 30; }}
             .user<TaskContext>({.count = count}));
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  REQUIRE(count == 0);
  REQUIRE(queue.tasks.size() == 3);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  REQUIRE(count == 50);
  REQUIRE(queue.tasks.size() == 1);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  REQUIRE(count == 60);
  REQUIRE(queue.tasks.size() == 0);
}

// Make sure we execute tasks only up to the timeslice provided to run.
TEST_CASE("TestOldestTimeslicePerTimeslice")
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  // Push two tasks on the queue with the same id
  int count = 0;
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{1},
                     .id = taskId1,
                     .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 10; }}
             .user<TaskContext>({.count = count}));
  // The size of the queue is only updated once a flush happens
  REQUIRE(queue.tasks.size() == 0);
  AsyncQueueHelpers::flushPending(queue);
  REQUIRE(queue.tasks.size() == 1);
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  REQUIRE(queue.tasks.size() == 1);
  REQUIRE(count == 0);
  AsyncQueueHelpers::post(
    queue, AsyncTask{.timeslice = TimesliceId{2},
                     .id = taskId1,
                     .callback = [](AsyncTask& task, size_t) { task.user<TaskContext>().count += 20; }}
             .user<TaskContext>({.count = count}));
  REQUIRE(queue.tasks.size() == 1);
  AsyncQueueHelpers::flushPending(queue);
  REQUIRE(queue.tasks.size() == 2);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  REQUIRE(queue.tasks.size() == 1);
  REQUIRE(count == 10);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  REQUIRE(queue.tasks.size() == 0);
  REQUIRE(count == 30);
}
