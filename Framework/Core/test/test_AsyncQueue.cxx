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

#define BOOST_TEST_MODULE Test Framework
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/AsyncQueue.h"

/// Test debouncing functionality. The same task cannot be executed more than once
/// in a given run.
BOOST_AUTO_TEST_CASE(TestDebouncing)
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId = AsyncQueueHelpers::create(queue, {.name = "test", .score = 10});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, taskId, [&count]() { count += 1; }, TimesliceId{0}, 10);
  AsyncQueueHelpers::post(
    queue, taskId, [&count]() { count += 2; }, TimesliceId{1}, 20);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  BOOST_CHECK_EQUAL(count, 2);
}

// Test task oridering. The tasks with the highest priority should be executed first.
BOOST_AUTO_TEST_CASE(TestPriority)
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, taskId1, [&count]() { count += 10; }, TimesliceId{0});
  AsyncQueueHelpers::post(
    queue, taskId2, [&count]() { count /= 10; }, TimesliceId{0});
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  BOOST_CHECK_EQUAL(count, 10);
}

// Make sure we execute tasks only up to the timeslice provided to run.
BOOST_AUTO_TEST_CASE(TestOldestTimeslice)
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, taskId1, [&count]() { count += 10; }, TimesliceId{1});
  AsyncQueueHelpers::post(
    queue, taskId2, [&count]() { count += 20; }, TimesliceId{0});
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  BOOST_CHECK_EQUAL(count, 20);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  BOOST_CHECK_EQUAL(count, 20);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  BOOST_CHECK_EQUAL(count, 30);
}

// Make sure we execute tasks only up to the timeslice provided to run.
BOOST_AUTO_TEST_CASE(TestOldestTimesliceWithBounce)
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, taskId1, [&count]() { count += 10; }, TimesliceId{1});
  AsyncQueueHelpers::post(
    queue, taskId2, [&count]() { count += 20; }, TimesliceId{0}, 10);
  AsyncQueueHelpers::post(
    queue, taskId2, [&count]() { count += 30; }, TimesliceId{0}, 20);
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  BOOST_CHECK_EQUAL(count, 0);
  BOOST_CHECK_EQUAL(queue.tasks.size(), 3);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  BOOST_CHECK_EQUAL(count, 30);
  BOOST_CHECK_EQUAL(queue.tasks.size(), 1);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  BOOST_CHECK_EQUAL(count, 40);
  BOOST_CHECK_EQUAL(queue.tasks.size(), 0);
}

// Make sure we execute tasks only up to the timeslice provided to run.
BOOST_AUTO_TEST_CASE(TestOldestTimeslicePerTimeslice)
{
  using namespace o2::framework;
  AsyncQueue queue;
  auto taskId1 = AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  // Push two tasks on the queue with the same id
  auto count = 0;
  AsyncQueueHelpers::post(
    queue, taskId1, [&count]() { count += 10; }, TimesliceId{0});
  BOOST_CHECK_EQUAL(queue.tasks.size(), 1);
  AsyncQueueHelpers::run(queue, TimesliceId{0});
  BOOST_CHECK_EQUAL(queue.tasks.size(), 1);
  BOOST_CHECK_EQUAL(count, 0);
  AsyncQueueHelpers::post(
    queue, taskId1, [&count]() { count += 20; }, TimesliceId{1});
  BOOST_CHECK_EQUAL(queue.tasks.size(), 2);
  AsyncQueueHelpers::run(queue, TimesliceId{1});
  BOOST_CHECK_EQUAL(queue.tasks.size(), 1);
  BOOST_CHECK_EQUAL(count, 10);
  AsyncQueueHelpers::run(queue, TimesliceId{2});
  BOOST_CHECK_EQUAL(queue.tasks.size(), 0);
  BOOST_CHECK_EQUAL(count, 30);
}
