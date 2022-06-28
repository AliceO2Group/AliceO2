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

#define BOOST_TEST_MODULE Test Framework DataRelayer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/TimesliceIndex.h"
#include "Framework/VariableContextHelpers.h"

BOOST_AUTO_TEST_CASE(TestBasics)
{
  using namespace o2::framework;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};
  TimesliceSlot slot;

  BOOST_REQUIRE_EQUAL(index.size(), 0);
  index.resize(10);
  BOOST_REQUIRE_EQUAL(index.size(), 10);
  BOOST_CHECK(index.isValid({0}) == false);
  BOOST_CHECK(index.isDirty({0}) == false);
  index.associate(TimesliceId{10}, TimesliceSlot{0});
  BOOST_CHECK(index.isValid(TimesliceSlot{0}));
  BOOST_CHECK(index.isDirty({0}) == true);
  index.associate(TimesliceId{20}, TimesliceSlot{0});
  BOOST_CHECK(index.isValid(TimesliceSlot{0}));
  BOOST_CHECK_EQUAL(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{0})).value, 20);
  BOOST_CHECK(index.isDirty(TimesliceSlot{0}));
  index.associate(TimesliceId{1}, TimesliceSlot{1});
  BOOST_CHECK_EQUAL(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{0})).value, 20);
  BOOST_CHECK_EQUAL(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{1})).value, 1);
  BOOST_CHECK(index.isValid(TimesliceSlot{2}) == false);
  slot = TimesliceSlot{0};
  BOOST_CHECK(index.isDirty(slot));
  slot = TimesliceSlot{2};
  BOOST_CHECK(index.isValid(slot) == false);
  BOOST_CHECK(index.isDirty(slot) == false);
  index.markAsInvalid(slot);
  BOOST_CHECK(index.isDirty(slot) == false);
  BOOST_CHECK(index.isValid(slot) == false);
}

BOOST_AUTO_TEST_CASE(TestLRUReplacement)
{
  using namespace o2::framework;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};
  index.resize(3);
  data_matcher::VariableContext context;

  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {10});
    BOOST_CHECK_EQUAL(slot.index, 0);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{20}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {20});
    BOOST_CHECK_EQUAL(slot.index, 1);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{30}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {30});
    BOOST_CHECK_EQUAL(slot.index, 2);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{40}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {40});
    BOOST_CHECK_EQUAL(slot.index, TimesliceSlot::INVALID);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::Wait);
  }
  {
    context.put({0, uint64_t{50}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {50});
    BOOST_CHECK_EQUAL(slot.index, TimesliceSlot::INVALID);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::Wait);
  }
  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {10});
    BOOST_CHECK_EQUAL(slot.index, TimesliceSlot::INVALID);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::Wait);
  }
}

/// A test to check the calculations of the oldest possible
/// timeslice in the index.
BOOST_AUTO_TEST_CASE(TestOldestPossibleTimeslice)
{
  using namespace o2::framework;
  std::vector<InputChannelInfo> infos{2};
  TimesliceIndex index{2, infos};

  index.resize(3);

  data_matcher::VariableContext context;
  {
    context.put({0, uint64_t{9}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {9});
    index.markAsDirty(slot, true);
    auto oldest = index.setOldestPossibleInput({9}, {0});
    for (size_t i = 0; i < 3; ++i) {
      bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
      std::cout << "Slot " << i << " valid: " << invalidated << std::endl;
    }
    index.updateOldestPossibleOutput();
    BOOST_CHECK_EQUAL(slot.index, 1);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {10});
    index.markAsDirty(slot, true);
    auto oldest = index.setOldestPossibleInput({10}, {1});
    for (size_t i = 0; i < 3; ++i) {
      bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
    }
    BOOST_CHECK_EQUAL(slot.index, 0);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }

  BOOST_CHECK_EQUAL(index.getOldestPossibleInput().timeslice.value, 9);
  BOOST_CHECK_EQUAL(index.getOldestPossibleOutput().timeslice.value, 0);
  auto oldest = index.setOldestPossibleInput({10}, {0});
  for (size_t i = 0; i < 3; ++i) {
    bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
  }
  index.updateOldestPossibleOutput();
  BOOST_CHECK_EQUAL(index.getOldestPossibleInput().timeslice.value, 10);
  BOOST_CHECK_EQUAL(index.getOldestPossibleOutput().timeslice.value, 9);
  oldest = index.setOldestPossibleInput({11}, {1});
  for (size_t i = 0; i < 3; ++i) {
    bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
  }
  index.updateOldestPossibleOutput();
  BOOST_CHECK_EQUAL(index.getOldestPossibleInput().timeslice.value, 10);
  BOOST_CHECK_EQUAL(index.getOldestPossibleOutput().timeslice.value, 9);
  // We fake the fact that we have processed the slot 0;
  index.markAsDirty({1}, false);
  index.updateOldestPossibleOutput();
  BOOST_CHECK_EQUAL(index.getOldestPossibleOutput().timeslice.value, 9);
  index.markAsInvalid({1});
  index.updateOldestPossibleOutput();
  BOOST_CHECK_EQUAL(index.getOldestPossibleOutput().timeslice.value, 10);
}
