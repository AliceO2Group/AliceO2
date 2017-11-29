// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework TypeTraits
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/TypeTraits.h"
#include <boost/test/unit_test.hpp>
#include <vector>
#include <list>

using namespace o2::framework;

struct Foo {
  int x;
  int y;
};

// Simple test to do root deserialization.
BOOST_AUTO_TEST_CASE(TestIsSpecialization) {
  std::vector<int> a;
  std::vector<Foo> b;
  std::list<int> c;
  int d;

  bool test1 = is_specialization<decltype(a),std::vector>::value;
  bool test2 = is_specialization<decltype(b),std::vector>::value;
  bool test3 = is_specialization<decltype(b),std::list>::value;
  bool test4 = is_specialization<decltype(c),std::list>::value;
  bool test5 = is_specialization<decltype(c),std::vector>::value;
  bool test6 = is_specialization<decltype(d),std::vector>::value;
  BOOST_REQUIRE_EQUAL(test1, true);
  BOOST_REQUIRE_EQUAL(test2, true);
  BOOST_REQUIRE_EQUAL(test3, false);
  BOOST_REQUIRE_EQUAL(test4, true);
  BOOST_REQUIRE_EQUAL(test5, false);
  BOOST_REQUIRE_EQUAL(test6, false);
}
