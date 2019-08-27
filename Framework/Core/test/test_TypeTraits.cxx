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
#include "Framework/SerializationMethods.h"
#include "TestClasses.h"
#include <boost/test/unit_test.hpp>
#include <vector>
#include <list>
#include <gsl/gsl>

using namespace o2::framework;

struct Foo {
  int x;
  int y;
};

// Simple test to do root deserialization.
BOOST_AUTO_TEST_CASE(TestIsSpecialization)
{
  std::vector<int> a;
  std::vector<Foo> b;
  std::list<int> c;
  int d;

  bool test1 = is_specialization<decltype(a), std::vector>::value;
  bool test2 = is_specialization<decltype(b), std::vector>::value;
  bool test3 = is_specialization<decltype(b), std::list>::value;
  bool test4 = is_specialization<decltype(c), std::list>::value;
  bool test5 = is_specialization<decltype(c), std::vector>::value;
  bool test6 = is_specialization<decltype(d), std::vector>::value;
  BOOST_REQUIRE_EQUAL(test1, true);
  BOOST_REQUIRE_EQUAL(test2, true);
  BOOST_REQUIRE_EQUAL(test3, false);
  BOOST_REQUIRE_EQUAL(test4, true);
  BOOST_REQUIRE_EQUAL(test5, false);
  BOOST_REQUIRE_EQUAL(test6, false);

  ROOTSerialized<decltype(d)> e(d);
  bool test7 = is_specialization<decltype(e), ROOTSerialized>::value;
  BOOST_REQUIRE_EQUAL(test7, true);
}

BOOST_AUTO_TEST_CASE(TestForceNonMessageable)
{
  // a struct explicitly marked to be non-messageable by defining
  // a type alias provided by the framework type traits
  struct ExplicitNonMessageable {
    using non_messageable = o2::framework::MarkAsNonMessageable;
  };

  // a struct using the same name for the type alias, but which should
  // not be forced to be non-messageable
  struct FailedNonMessageable {
    using non_messageable = int;
  };

  ExplicitNonMessageable a;
  Foo b;
  FailedNonMessageable c;

  BOOST_REQUIRE_EQUAL(is_forced_non_messageable<decltype(a)>::value, true);
  BOOST_REQUIRE_EQUAL(is_forced_non_messageable<decltype(b)>::value, false);
  BOOST_REQUIRE_EQUAL(is_forced_non_messageable<decltype(c)>::value, false);
}

BOOST_AUTO_TEST_CASE(TestIsMessageable)
{
  int a;
  Foo b;
  std::vector<int> c;
  o2::test::TriviallyCopyable d;
  o2::test::Polymorphic e;
  gsl::span<o2::test::TriviallyCopyable> spantriv;
  gsl::span<o2::test::Polymorphic> spanpoly;

  BOOST_REQUIRE_EQUAL(is_messageable<decltype(a)>::value, true);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(b)>::value, true);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(c)>::value, false);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(d)>::value, true);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(e)>::value, false);
  BOOST_REQUIRE_EQUAL(is_messageable<ROOTSerialized<decltype(e)>>::value, false);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(spantriv)>::value, false);
  BOOST_REQUIRE_EQUAL(is_messageable<decltype(spanpoly)>::value, false);
}

BOOST_AUTO_TEST_CASE(TestIsStlContainer)
{
  int a;
  o2::test::TriviallyCopyable b;
  std::vector<o2::test::Polymorphic> c;
  std::map<int, o2::test::Polymorphic> d;

  BOOST_REQUIRE_EQUAL(is_container<decltype(a)>::value, false);
  BOOST_REQUIRE_EQUAL(is_container<decltype(b)>::value, false);
  BOOST_REQUIRE_EQUAL(is_container<decltype(c)>::value, true);
  BOOST_REQUIRE_EQUAL(is_container<decltype(d)>::value, true);

  BOOST_REQUIRE_EQUAL(has_messageable_value_type<decltype(b)>::value, false);
  BOOST_REQUIRE_EQUAL(has_messageable_value_type<std::vector<int>>::value, true);
  BOOST_REQUIRE_EQUAL(has_messageable_value_type<decltype(c)>::value, false);
}

BOOST_AUTO_TEST_CASE(TestHasRootStreamer)
{
  o2::test::TriviallyCopyable a;
  o2::test::Polymorphic b;
  std::vector<o2::test::Polymorphic> c;
  int d;
  Foo e;
  std::list<o2::test::TriviallyCopyable> f;
  std::list<int> g;

  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(a)>::value, true);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(b)>::value, true);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(c)>::value, true);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(d)>::value, false);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(e)>::value, false);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(f)>::value, true);
  BOOST_REQUIRE_EQUAL(has_root_dictionary<decltype(g)>::value, false);
}

BOOST_AUTO_TEST_CASE(TestIsSpan)
{
  gsl::span<int> a;
  int b;
  std::vector<char> c;
  BOOST_REQUIRE_EQUAL(is_span<decltype(a)>::value, true);
  BOOST_REQUIRE_EQUAL(is_span<decltype(b)>::value, false);
  BOOST_REQUIRE_EQUAL(is_span<decltype(c)>::value, false);
}
