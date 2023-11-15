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
#include "Framework/Traits.h"

struct Foo {
  virtual void a() {}
};

struct Bar : public Foo {
  void a() override {}
};

struct FooBar : public Foo {
};

TEST_CASE("TestOverride2")
{
  bool check1 = o2::framework::is_overriding<decltype(&Bar::a), decltype(&Foo::a)>::value;
  REQUIRE(check1 == true);
  bool check2 = o2::framework::is_overriding<decltype(&FooBar::a), decltype(&Foo::a)>::value;
  REQUIRE(check2 == false);
}
