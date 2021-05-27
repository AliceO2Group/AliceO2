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

/// Largest supported struct
struct FooMax {
  int foo1 = 1;
  int foo2 = 2;
  int foo3 = 3;
  int foo4 = 4;
  int foo5 = 5;
  int foo6 = 6;
  int foo7 = 7;
  int foo8 = 8;
  int foo9 = 9;
  int foo10 = 10;
  int foo11 = 11;
  int foo12 = 12;
  int foo13 = 13;
  int foo14 = 14;
  int foo15 = 15;
  int foo16 = 16;
  int foo17 = 17;
  int foo18 = 18;
  int foo19 = 19;
  int foo20 = 20;
  int foo21 = 21;
  int foo22 = 22;
  int foo23 = 23;
  int foo24 = 24;
  int foo25 = 25;
  int foo26 = 26;
  int foo27 = 27;
  int foo28 = 28;
  int foo29 = 29;
  int foo30 = 30;
  int foo31 = 31;
  int foo32 = 32;
  int foo33 = 33;
  int foo34 = 34;
  int foo35 = 35;
  int foo36 = 36;
  int foo37 = 37;
  int foo38 = 38;
  int foo39 = 39;
};

BOOST_AUTO_TEST_CASE(TestStructToTuple)
{
  FooMax fooMax;

  auto t5 = o2::framework::homogeneous_apply_refs([](auto i) -> bool { return i > 20; }, fooMax);
  BOOST_CHECK_EQUAL(t5[0], false);
  BOOST_CHECK_EQUAL(t5[19], false);
  BOOST_CHECK_EQUAL(t5[20], true);
}
