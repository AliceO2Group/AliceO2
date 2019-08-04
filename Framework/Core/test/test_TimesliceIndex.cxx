// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataRelayer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/TimesliceIndex.h"

BOOST_AUTO_TEST_CASE(TestBasics)
{
  using namespace o2::framework;
  TimesliceIndex index;
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
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{0}).value, 20);
  BOOST_CHECK(index.isDirty(TimesliceSlot{0}));
  index.associate(TimesliceId{1}, TimesliceSlot{1});
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{0}).value, 20);
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{1}).value, 1);
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
  TimesliceIndex index;
  index.resize(3);
  data_matcher::VariableContext context;

  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, 0);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{20}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, 1);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{30}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, 2);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceUnused);
  }
  {
    context.put({0, uint64_t{40}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, 0);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceObsolete);
  }
  {
    context.put({0, uint64_t{50}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, 1);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::ReplaceObsolete);
  }
  {
    context.put({0, uint64_t{10}});
    context.commit();
    auto [action, slot] = index.replaceLRUWith(context);
    BOOST_CHECK_EQUAL(slot.index, TimesliceSlot::INVALID);
    BOOST_CHECK(action == TimesliceIndex::ActionTaken::DropObsolete);
  }
}
