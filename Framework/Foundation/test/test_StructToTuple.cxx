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
#include "Framework/StructToTuple.h"

struct Foo {
};

// FIXME: this should really struct Bar : Foo, but a c++17 bug
// in GCC 7.3 (solved in 7.4 / 8.x) prevents us from doing so
// for now.
struct Bar {
  int foo = 1;
  int bar = 2;
};

BOOST_AUTO_TEST_CASE(TestStructToTuple)
{
  Foo foo;
  //auto t1 = o2::framework::to_tuple(foo);
  //#BOOST_CHECK_EQUAL(std::get<0>(t1), 1);
  // Expand a struct which inherits from
  // another..
  Bar bar{4, 5};
  BOOST_CHECK_EQUAL(bar.foo, 4);
  BOOST_CHECK_EQUAL(bar.bar, 5);
  auto t2 = o2::framework::to_tuple(bar);
  BOOST_CHECK_EQUAL(std::get<0>(t2), 4);
  BOOST_CHECK_EQUAL(std::get<1>(t2), 5);
  std::get<0>(t2) = 10;
  BOOST_CHECK_EQUAL(std::get<0>(t2), 10);
  BOOST_CHECK_EQUAL(bar.foo, 4);

  auto t3 = o2::framework::to_tuple_refs(bar);
  BOOST_CHECK_EQUAL(std::get<0>(t3), 4);
  BOOST_CHECK_EQUAL(std::get<1>(t3), 5);

  std::get<0>(t3) = 10;
  BOOST_CHECK_EQUAL(std::get<0>(t3), 10);
  BOOST_CHECK_EQUAL(bar.foo, 10);
}
