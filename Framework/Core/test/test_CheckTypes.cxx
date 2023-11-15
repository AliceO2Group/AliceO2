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
#include "Framework/CheckTypes.h"

using namespace o2::framework;

struct Foo {
  bool foo;
};

TEST_CASE("CallIfUndefined")
{
  bool shouldBeCalled = false;
  bool shouldNotBeCalled = false;
  bool shouldBeCalledOnUndefined = false;

  call_if_defined<struct Foo>([&shouldBeCalled](auto) { shouldBeCalled = true; });
  call_if_defined<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; });
  REQUIRE(shouldBeCalled == true);
  REQUIRE(shouldNotBeCalled == false);

  shouldBeCalled = false;
  shouldNotBeCalled = false;
  shouldBeCalledOnUndefined = false;

  call_if_defined_full<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; }, []() {});
  REQUIRE(shouldNotBeCalled == false);
  REQUIRE(shouldBeCalledOnUndefined == false);
  call_if_defined_full<struct Bar>([&shouldNotBeCalled](auto) { shouldNotBeCalled = true; }, [&shouldBeCalledOnUndefined]() { shouldBeCalledOnUndefined = true; });
  REQUIRE(shouldNotBeCalled == false);
  REQUIRE(shouldBeCalledOnUndefined == true);
}
