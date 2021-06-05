// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework StringHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/StringHelpers.h"
#include <iostream>

BOOST_AUTO_TEST_CASE(StringHelpersHash)
{
  std::string s{"test-string"};
  char const* const cs = "test-string";
  BOOST_CHECK_EQUAL(compile_time_hash(s.c_str()), compile_time_hash("test-string"));
  BOOST_CHECK_EQUAL(compile_time_hash(cs), compile_time_hash("test-string"));
  BOOST_CHECK_EQUAL(compile_time_hash(s.c_str()), compile_time_hash(cs));
}

template <typename T>
void printString(const T& constStr)
{
  static_assert(is_const_str<T>::value, "This function can only print compile-time strings!");

  std::cout << "ConstStr:" << std::endl;
  std::cout << "str -> " << constStr.str << std::endl;
  std::cout << "hash -> " << constStr.hash << std::endl;
};

BOOST_AUTO_TEST_CASE(StringHelpersConstStr)
{
  printString(CONST_STR("this/is/a/histogram"));

  auto myConstStr = CONST_STR("helloWorld");
  printString(myConstStr);
  static_assert(std::is_same_v<decltype(myConstStr), ConstStr<'h', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd'>>);
  static_assert(myConstStr.hash == (uint32_t)942280617);
  BOOST_CHECK_EQUAL(myConstStr.hash, compile_time_hash("helloWorld"));

  if constexpr (is_const_str_v(myConstStr)) {
    std::cout << "myConstStr is a compile-time string" << std::endl;
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

  BOOST_CHECK_EQUAL(CONST_STR(hist[0]).hash, CONST_STR("ptDist").hash);
}
