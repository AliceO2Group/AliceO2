// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   test_FlattenRestore.cxx
//  @author Matthias Richter
//  @since  2020-04-05
//  @brief  Test program flatten/restore tools

#define BOOST_TEST_MODULE Algorithm FlattenRestore test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "../include/Algorithm/FlattenRestore.h"
#include <vector>
#include <algorithm>

namespace flatten = o2::algorithm::flatten;

namespace o2::test
{
struct DataAccess {
  size_t count = 0;
  char* chars = nullptr;
  int* ints = nullptr;
  float* floats = nullptr;
};
} // namespace o2::test
BOOST_AUTO_TEST_CASE(test_flattenrestore)
{
  o2::test::DataAccess access{static_cast<size_t>(rand() % 32)};
  std::vector<char> chars(access.count);
  std::generate(chars.begin(), chars.end(), []() { return rand() % 256; });
  std::vector<int> ints(access.count);
  std::generate(ints.begin(), ints.end(), []() { return rand() % 256; });
  std::vector<float> floats(access.count);
  std::generate(floats.begin(), floats.end(), []() { return rand() % 256; });
  access.chars = chars.data();
  access.ints = ints.data();
  access.floats = floats.data();

  std::vector<char> raw(flatten::value_size(access.chars, access.ints, access.floats) * access.count);
  char* wrtptr = raw.data();
  auto copied = flatten::copy_to(wrtptr, access.count, access.chars, access.ints, access.floats);
  BOOST_CHECK(copied == raw.size());
  char* checkptr = raw.data();
  BOOST_CHECK(memcmp(checkptr, chars.data(), chars.size() * sizeof(decltype(chars)::value_type)) == 0);
  checkptr += flatten::calc_size(nullptr, chars.size(), chars.data());
  BOOST_CHECK(memcmp(checkptr, ints.data(), ints.size() * sizeof(decltype(ints)::value_type)) == 0);
  checkptr += flatten::calc_size(nullptr, ints.size(), ints.data());
  BOOST_CHECK(memcmp(checkptr, floats.data(), floats.size() * sizeof(decltype(floats)::value_type)) == 0);
  checkptr += flatten::calc_size(nullptr, floats.size(), floats.data());

  o2::test::DataAccess target{access.count, nullptr, nullptr, nullptr};
  char* readptr = raw.data();
  auto readsize = flatten::set_from(readptr, target.count, target.chars, target.ints, target.floats);
  BOOST_CHECK(readsize == copied);
  checkptr = raw.data();
  BOOST_CHECK(reinterpret_cast<decltype(target.chars)>(checkptr) == target.chars);
  checkptr += flatten::calc_size(nullptr, chars.size(), chars.data());
  BOOST_CHECK(reinterpret_cast<decltype(target.ints)>(checkptr) == target.ints);
  checkptr += flatten::calc_size(nullptr, ints.size(), ints.data());
  BOOST_CHECK(reinterpret_cast<decltype(target.floats)>(checkptr) == target.floats);
  checkptr += flatten::calc_size(nullptr, floats.size(), floats.data());
}
