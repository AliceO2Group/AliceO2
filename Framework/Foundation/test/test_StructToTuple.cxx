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
  int foo01 = 1;
  int foo02 = 2;
  int foo03 = 3;
  int foo04 = 4;
  int foo05 = 5;
  int foo06 = 6;
  int foo07 = 7;
  int foo08 = 8;
  int foo09 = 9;
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
  int foo101 = 1;
  int foo102 = 2;
  int foo103 = 3;
  int foo104 = 4;
  int foo105 = 5;
  int foo106 = 6;
  int foo107 = 7;
  int foo108 = 8;
  int foo109 = 9;
  int foo110 = 10;
  int foo111 = 11;
  int foo112 = 12;
  int foo113 = 13;
  int foo114 = 14;
  int foo115 = 15;
  int foo116 = 16;
  int foo117 = 17;
  int foo118 = 18;
  int foo119 = 19;
  int foo120 = 20;
  int foo121 = 21;
  int foo122 = 22;
  int foo123 = 23;
  int foo124 = 24;
  int foo125 = 25;
  int foo126 = 26;
  int foo127 = 27;
  int foo128 = 28;
  int foo129 = 29;
  int foo130 = 30;
  int foo131 = 31;
  int foo132 = 32;
  int foo133 = 33;
  int foo134 = 34;
  int foo135 = 35;
  int foo136 = 36;
  int foo137 = 37;
  int foo138 = 38;
  int foo139 = 39;
};

struct FooNested {
  int foo;
};

struct Foo2 {
  FooNested foo{
    .foo = 100};
  FooNested foo2{
    .foo = 20};
  int foo3 = 40;
};

TEST_CASE("TestStructToTuple")
{
  FooMax fooMax;

  auto t5 = o2::framework::homogeneous_apply_refs([](auto i) -> bool { return i > 20; }, fooMax);
  REQUIRE(t5[0] == false);
  REQUIRE(t5[19] == false);
  REQUIRE(t5[20] == true);
  Foo2 nestedFoo;
  auto t6 = o2::framework::homogeneous_apply_refs([](auto e) -> bool {
    if constexpr (std::is_same_v<decltype(e), FooNested>) {
      o2::framework::homogeneous_apply_refs([](auto n) -> bool { return n > 20; }, e);
      return true;
    } else {
      return e > 20;
    }
  },
                                                  nestedFoo);
  REQUIRE(t6.size() == 3);
  REQUIRE(t6[0] == true);
}
