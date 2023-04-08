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

#include "Framework/StringHelpers.h"
#include "Framework/StaticFor.h"

using namespace o2::framework;

template <int someNumber>
void dummyFunc()
{
  INFO("calling function with non-type template argument " << someNumber);
}

TEST_CASE("TestStaticFor")
{
  // check if it is actually static
  static_for<0, 0>([&](auto i) {
    static_assert(std::is_same_v<decltype(i), std::integral_constant<int, 0>>);

    static_assert(std::is_same_v<decltype(i.value), const int>);
    REQUIRE(i.value == 0);
    REQUIRE(i == 0);

    // the following checks will fail
    // static_assert(std::is_same_v<decltype(i), std::integral_constant<int, 1>>);
    // REQUIRE(i.value ==  1);;
    // REQUIRE(i ==  1);;
  });

  // dont start at 0
  static_for<5, 5>([&](auto i) {
    static_assert(std::is_same_v<decltype(i), std::integral_constant<int, 5>>);
  });

  // check if argument can be used as non-type template argument
  static_for<0, 2>([&](auto i) {
    dummyFunc<i>();
    dummyFunc<i.value>();
    constexpr auto index = i.value;
    dummyFunc<index>();
  });

  // use static loop in combination with CONST_STR
  static constexpr std::string_view staticNames[] = {"Bob", "Alice", "Eve"};
  static_for<0, 2>([&](auto i) {
    constexpr int index = i.value;

    // compiler will complain if constexpr is not enforced for index access:
    //CONST_STR(staticNames[index]);    // works
    //CONST_STR(staticNames[i.value]);  // fails

    constexpr auto sayHello = CONST_STR("Hello ") + CONST_STR(staticNames[index]);

    INFO(sayHello.str);
  });
}
