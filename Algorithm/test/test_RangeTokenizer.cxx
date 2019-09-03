// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   test_RangeTokenizer.cxx
//  @author Matthias Richter
//  @since  2018-12-11
//  @brief  Test program for RangeTokenizer

#define BOOST_TEST_MODULE Algorithm RangeTokenizer test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "../include/Algorithm/RangeTokenizer.h"
#include <vector>
#include <map>

using RangeTokenizer = o2::RangeTokenizer;

BOOST_AUTO_TEST_CASE(test_simple_integral)
{
  // the simple case using integral type
  std::vector<int> tokens = RangeTokenizer::tokenize<int>("0-5,10,13-15");
  std::vector<int> expected{0, 1, 2, 3, 4, 5, 10, 13, 14, 15};
  BOOST_CHECK(tokens == expected);
}

BOOST_AUTO_TEST_CASE(test_simple_string)
{
  // simple case using string type
  std::vector<std::string> tokens = RangeTokenizer::tokenize<std::string>("apple,strawberry,tomato");
  BOOST_CHECK(tokens[0] == "apple");
  BOOST_CHECK(tokens[1] == "strawberry");
  BOOST_CHECK(tokens[2] == "tomato");
}

BOOST_AUTO_TEST_CASE(test_mapped_custom)
{
  // process a custom type according to a map
  enum struct Food { Apple,
                     Strawberry,
                     Tomato };

  const std::map<std::string, Food> FoodMap{
    {"apple", Food::Apple},
    {"strawberry", Food::Strawberry},
    {"tomato", Food::Tomato},
  };
  auto tester = [FoodMap](const char* arg) {
    // use a custom mapper function, this evetually throws an exception if the token is not in the map
    return std::move(RangeTokenizer::tokenize<Food>(arg, [FoodMap](auto const& token) { return FoodMap.at(token); }));
  };

  auto tokens = tester("apple,tomato");
  BOOST_CHECK(tokens[0] == Food::Apple);
  BOOST_CHECK(tokens[1] == Food::Tomato);

  BOOST_CHECK_THROW(tester("blueberry"), std::out_of_range);
}
