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

TEST_CASE("StringHelpersHash")
{
  std::string s{"test-string"};
  char const* const cs = "test-string";
  REQUIRE(compile_time_hash(s.c_str()) == compile_time_hash("test-string"));
  REQUIRE(compile_time_hash(cs) == compile_time_hash("test-string"));
  REQUIRE(compile_time_hash(s.c_str()) == compile_time_hash(cs));
}

template <typename T>
void printString(const T& constStr)
{
  static_assert(is_const_str<T>::value, "This function can only print compile-time strings!");

  INFO("ConstStr:");
  INFO("str -> " << constStr.str);
  INFO("hash -> " << constStr.hash);
};

TEST_CASE("StringHelpersConstStr")
{
  printString(CONST_STR("this/is/a/histogram"));

  auto myConstStr = CONST_STR("helloWorld");
  printString(myConstStr);
  static_assert(std::is_same_v<decltype(myConstStr), ConstStr<'h', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd'>>);
  static_assert(myConstStr.hash == (uint32_t)942280617);
  REQUIRE(myConstStr.hash == compile_time_hash("helloWorld"));

  if constexpr (is_const_str_v(myConstStr)) {
    INFO("myConstStr is a compile-time string");
  }

  auto myConstStr2 = CONST_STR("hello") + CONST_STR("Universe");
  printString(myConstStr2);
  static_assert(std::is_same_v<decltype(myConstStr2), ConstStr<'h', 'e', 'l', 'l', 'o', 'U', 'n', 'i', 'v', 'e', 'r', 's', 'e'>>);

  enum ParticleSpecies {
    kPion,
    kKaon
  };
  static constexpr std::string_view hist[] = {"ptDist", "etaDist"};
  static constexpr std::string_view particleSuffix[] = {"_pions", "_kaons"};

  printString(CONST_STR(hist[0]) + CONST_STR(particleSuffix[kPion]));
  printString(CONST_STR(hist[0]) + CONST_STR(particleSuffix[kKaon]));
  printString(CONST_STR(hist[1]) + CONST_STR(particleSuffix[kPion]));
  printString(CONST_STR(hist[1]) + CONST_STR(particleSuffix[kKaon]));

  REQUIRE(CONST_STR(hist[0]).hash == CONST_STR("ptDist").hash);
}
