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
#include "Framework/TimesliceIndex.h"
#include "Framework/VariableContextHelpers.h"
#include <iostream>

TEST_CASE("TestBasics")
{
  using namespace o2::framework;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};
  TimesliceSlot slot;

  REQUIRE(index.size() == 0);
  index.resize(10);
  REQUIRE(index.size() == 10);
  REQUIRE(index.isValid({0}) == false);
  REQUIRE(index.isDirty({0}) == false);
  index.associate(TimesliceId{10}, TimesliceSlot{0});
  REQUIRE(index.isValid(TimesliceSlot{0}));
  REQUIRE(index.isDirty({0}) == true);
  index.associate(TimesliceId{20}, TimesliceSlot{0});
  REQUIRE(index.isValid(TimesliceSlot{0}));
  REQUIRE(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{0})).value == 20);
  REQUIRE(index.isDirty(TimesliceSlot{0}));
  index.associate(TimesliceId{1}, TimesliceSlot{1});
  REQUIRE(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{0})).value == 20);
  REQUIRE(VariableContextHelpers::getTimeslice(index.getVariablesForSlot(TimesliceSlot{1})).value == 1);
  REQUIRE(index.isValid(TimesliceSlot{2}) == false);
  slot = TimesliceSlot{0};
  REQUIRE(index.isDirty(slot));
  slot = TimesliceSlot{2};
  REQUIRE(index.isValid(slot) == false);
  REQUIRE(index.isDirty(slot) == false);
  index.markAsInvalid(slot);
  REQUIRE(index.isDirty(slot) == false);
  REQUIRE(index.isValid(slot) == false);
}

TEST_CASE("TestLRUReplacement")
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
    REQUIRE(slot.index == 0);
    REQUIRE(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{20}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {20});
    REQUIRE(slot.index == 1);
    REQUIRE(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{30}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {30});
    REQUIRE(slot.index == 2);
    REQUIRE(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{40}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {40});
    REQUIRE(slot.index == TimesliceSlot::INVALID);
    REQUIRE(action == TimesliceIndex::ActionTaken::Wait);
  }
  {
    context.put({0, uint64_t{50}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {50});
    REQUIRE(slot.index == TimesliceSlot::INVALID);
    REQUIRE(action == TimesliceIndex::ActionTaken::Wait);
  }
  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context, {10});
    REQUIRE(slot.index == TimesliceSlot::INVALID);
    REQUIRE(action == TimesliceIndex::ActionTaken::Wait);
  }
}

/// A test to check the calculations of the oldest possible
/// timeslice in the index.
TEST_CASE("TestOldestPossibleTimeslice")
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
    REQUIRE(slot.index == 1);
    REQUIRE(action == TimesliceIndex::ActionTaken::ReplaceUnused);
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
    REQUIRE(slot.index == 0);
    REQUIRE(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }

  REQUIRE(index.getOldestPossibleInput().timeslice.value == 9);
  REQUIRE(index.getOldestPossibleOutput().timeslice.value == 0);
  auto oldest = index.setOldestPossibleInput({10}, {0});
  for (size_t i = 0; i < 3; ++i) {
    bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
  }
  index.updateOldestPossibleOutput();
  REQUIRE(index.getOldestPossibleInput().timeslice.value == 10);
  REQUIRE(index.getOldestPossibleOutput().timeslice.value == 9);
  oldest = index.setOldestPossibleInput({11}, {1});
  for (size_t i = 0; i < 3; ++i) {
    bool invalidated = index.validateSlot(TimesliceSlot{i}, oldest.timeslice);
  }
  index.updateOldestPossibleOutput();
  REQUIRE(index.getOldestPossibleInput().timeslice.value == 10);
  REQUIRE(index.getOldestPossibleOutput().timeslice.value == 9);
  // We fake the fact that we have processed the slot 0;
  index.markAsDirty({1}, false);
  index.updateOldestPossibleOutput();
  REQUIRE(index.getOldestPossibleOutput().timeslice.value == 9);
  index.markAsInvalid({1});
  index.updateOldestPossibleOutput();
  REQUIRE(index.getOldestPossibleOutput().timeslice.value == 10);
}
