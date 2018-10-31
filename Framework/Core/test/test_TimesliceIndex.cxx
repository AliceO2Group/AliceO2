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
  BOOST_CHECK(index.isValid({ 0 }) == false);
  BOOST_CHECK(index.isObsolete({ 0 }) == false);
  BOOST_CHECK(index.isDirty({ 0 }) == false);
  index.bookTimeslice(TimesliceId{ 0 });
  BOOST_CHECK(index.isValid(TimesliceSlot{ 0 }));
  BOOST_CHECK(index.isDirty(TimesliceSlot{ 0 }));
  BOOST_CHECK(index.isObsolete(TimesliceId{ 0 }) == false);
  index.bookTimeslice(TimesliceId{ 10 });
  BOOST_CHECK(index.isValid(TimesliceSlot{ 0 }));
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{ 0 }).value, 10);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 0 }));
  BOOST_CHECK(index.isObsolete(TimesliceId{ 10 }) == false);
  BOOST_CHECK(index.isDirty(TimesliceSlot{ 0 }));
  index.bookTimeslice(TimesliceId{ 1 });
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{ 0 }).value, 10);
  BOOST_CHECK_EQUAL(index.getTimesliceForSlot(TimesliceSlot{ 1 }).value, 1);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 0 }));
  BOOST_CHECK(index.isObsolete(TimesliceId{ 1 }) == false);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 2 }) == false);
  BOOST_CHECK(index.isValid(TimesliceSlot{ 2 }) == false);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 10 }) == false);
  slot = index.getSlotForTimeslice(TimesliceId{ 10 });
  index.markAsObsolete(slot);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 10 }));
  BOOST_CHECK(index.isDirty(slot));
  slot = index.getSlotForTimeslice(TimesliceId{ 2 });
  BOOST_CHECK(index.isObsolete(TimesliceId{ 2 }) == false);
  BOOST_CHECK(index.isValid(slot) == false);
  index.markAsObsolete(slot);
  BOOST_CHECK(index.isObsolete(TimesliceId{ 2 }) == false);
  BOOST_CHECK(index.isValid(slot) == false);
}
