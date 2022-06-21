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
  o2::framework::AsyncQueue queue;
  auto taskId = o2::framework::AsyncQueueHelpers::create(queue, {.name = "test", .score = 10});
  // Push two tasks on the queue with the same id
  auto count = 0;
  o2::framework::AsyncQueueHelpers::post(
    queue, taskId, [&count]() { count += 1; }, 10);
  o2::framework::AsyncQueueHelpers::post(
    queue, taskId, [&count]() { count += 2; }, 20);
  o2::framework::AsyncQueueHelpers::run(queue);
  BOOST_CHECK_EQUAL(count, 2);
}

// Test task oridering. The tasks with the highest priority should be executed first.
BOOST_AUTO_TEST_CASE(TestPriority)
{
  o2::framework::AsyncQueue queue;
  auto taskId1 = o2::framework::AsyncQueueHelpers::create(queue, {.name = "test1", .score = 10});
  auto taskId2 = o2::framework::AsyncQueueHelpers::create(queue, {.name = "test2", .score = 20});
  // Push two tasks on the queue with the same id
  auto count = 0;
  o2::framework::AsyncQueueHelpers::post(queue, taskId1, [&count]() { count += 10; });
  o2::framework::AsyncQueueHelpers::post(queue, taskId2, [&count]() { count /= 10; });
  o2::framework::AsyncQueueHelpers::run(queue);
  BOOST_CHECK_EQUAL(count, 10);
}
