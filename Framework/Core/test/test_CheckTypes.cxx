// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Traits
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/CheckTypes.h"

using namespace o2::framework;

struct Foo {
  bool foo;
};

BOOST_AUTO_TEST_CASE(CallIfUndefined)
{
  bool shouldBeCalled = false;
  bool shouldNotBeCalled = false;
  bool shouldBeCalledOnUndefined = false;

  call_if_defined<struct Foo>([&shouldBeCalled](auto) { shouldBeCalled = true; });
  call_if_defined<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; });
  BOOST_REQUIRE_EQUAL(shouldBeCalled, true);
  BOOST_REQUIRE_EQUAL(shouldNotBeCalled, false);

  shouldBeCalled = false;
  shouldNotBeCalled = false;
  shouldBeCalledOnUndefined = false;

  call_if_defined_full<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; }, []() {});
  BOOST_REQUIRE_EQUAL(shouldNotBeCalled, false);
  BOOST_REQUIRE_EQUAL(shouldBeCalledOnUndefined, false);
  call_if_defined_full<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; }, [&shouldBeCalledOnUndefined]() { shouldBeCalledOnUndefined = true; });
  BOOST_REQUIRE_EQUAL(shouldNotBeCalled, false);
  BOOST_REQUIRE_EQUAL(shouldBeCalledOnUndefined, true);
}
