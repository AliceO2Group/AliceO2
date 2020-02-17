// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ASoAHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ASoAHelpers.h"
#include "Framework/TableBuilder.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace o2::soa;

namespace test
{
DECLARE_SOA_COLUMN(X, x, int32_t, "x");
DECLARE_SOA_COLUMN(Y, y, int32_t, "y");
DECLARE_SOA_COLUMN(Z, z, int32_t, "z");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](int32_t x, int32_t y) { return x + y; });
} // namespace test

BOOST_AUTO_TEST_CASE(AddOneToTuple)
{
  using TupleType2 = typename generateTupleType<int, 2>::type;
  TupleType2 indexTuple2(0, 1);
  bool isEnd = false;

  addOne(indexTuple2, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple2), 0);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple2), 2);
  BOOST_CHECK_EQUAL(isEnd, false);

  addOne(indexTuple2, 1, isEnd);
  addOne(indexTuple2, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple2), 2);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple2), 3);
  BOOST_CHECK_EQUAL(isEnd, true);

  using TupleType3 = typename generateTupleType<int, 3>::type;
  TupleType3 indexTuple3(0, 2, 3);
  isEnd = false;

  addOne(indexTuple3, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple3), 1);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple3), 2);
  BOOST_CHECK_EQUAL(std::get<2>(indexTuple3), 3);
  BOOST_CHECK_EQUAL(isEnd, false);

  addOne(indexTuple3, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple3), 2);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple3), 3);
  BOOST_CHECK_EQUAL(std::get<2>(indexTuple3), 4);
  BOOST_CHECK_EQUAL(isEnd, true);
}

BOOST_AUTO_TEST_CASE(UpdateTuple)
{
  using TupleType2 = typename generateTupleType<int, 2>::type;
  TupleType2 indexTuple2(1, 2);
  bool isEnd = false;

  updateTuple(
    indexTuple2, [](size_t ind) { return ind + 1; }, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple2), 1);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple2), 2);
  BOOST_CHECK_EQUAL(isEnd, false);

  updateTuple(
    indexTuple2, [](size_t ind) { return ind + 2; }, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple2), 2);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple2), 3);
  BOOST_CHECK_EQUAL(isEnd, true);

  using TupleType3 = typename generateTupleType<int, 3>::type;
  TupleType3 indexTuple3(0, 2, 3);
  isEnd = false;

  updateTuple(
    indexTuple3, [](size_t ind) { return ind + 1; }, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple3), 1);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple3), 2);
  BOOST_CHECK_EQUAL(std::get<2>(indexTuple3), 3);
  BOOST_CHECK_EQUAL(isEnd, false);

  updateTuple(
    indexTuple3, [](size_t ind) { return ind + 2; }, 1, isEnd);
  BOOST_CHECK_EQUAL(std::get<0>(indexTuple3), 2);
  BOOST_CHECK_EQUAL(std::get<1>(indexTuple3), 3);
  BOOST_CHECK_EQUAL(std::get<2>(indexTuple3), 4);
  BOOST_CHECK_EQUAL(isEnd, true);
}

//BOOST_AUTO_TEST_CASE(IndicesToIterators)
//{
//  TableBuilder builderA;
//  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
//  rowWriterA(0, 2, 0);
//  rowWriterA(0, 3, 0);
//  rowWriterA(0, 4, 0);
//  auto tableA = builderA.finalize();
//  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 3);
//
//  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
//
//  TestA tests{tableA};
//
//  using IndexTupleType2 = typename generateTupleType<int, 2>::type;
//  using TupleType2 = typename generateTupleType<TestA::iterator, 2>::type;
//  IndexTupleType2 indexTuple2(0, 1);
//  TupleType2 iteratorsTuple2;
//
//  indicesToIterators(indexTuple2, iteratorsTuple2, tests.begin());
//
//  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(iteratorsTuple2)).getIterator().mCurrentPos, nullptr);
//  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(iteratorsTuple2)).getIterator().mCurrentPos), 0);
//  BOOST_CHECK_EQUAL(std::get<0>(iteratorsTuple2).x(), 2);
//  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(iteratorsTuple2)).getIterator().mCurrentPos, nullptr);
//  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(iteratorsTuple2)).getIterator().mCurrentPos), 1);
//  BOOST_CHECK_EQUAL(std::get<1>(iteratorsTuple2).x(), 3);
//
//  using IndexTupleType3 = typename generateTupleType<int, 3>::type;
//  using TupleType3 = typename generateTupleType<TestA::iterator, 3>::type;
//  IndexTupleType3 indexTuple3(0, 2, 1);
//  TupleType3 iteratorsTuple3;
//
//  indicesToIterators(indexTuple3, iteratorsTuple3, tests.begin());
//
//  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(iteratorsTuple3)).getIterator().mCurrentPos, nullptr);
//  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(iteratorsTuple3)).getIterator().mCurrentPos), 0);
//  BOOST_CHECK_EQUAL(std::get<0>(iteratorsTuple3).x(), 2);
//  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(iteratorsTuple3)).getIterator().mCurrentPos, nullptr);
//  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(iteratorsTuple3)).getIterator().mCurrentPos), 2);
//  BOOST_CHECK_EQUAL(std::get<1>(iteratorsTuple3).x(), 4);
//  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(iteratorsTuple3)).getIterator().mCurrentPos, nullptr);
//  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(iteratorsTuple3)).getIterator().mCurrentPos), 1);
//  BOOST_CHECK_EQUAL(std::get<2>(iteratorsTuple3).x(), 3);
//}

BOOST_AUTO_TEST_CASE(TuplesGeneratorConstruction)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  rowWriterA(0, 4, 0);
  rowWriterA(0, 5, 0);
  rowWriterA(0, 6, 0);
  rowWriterA(0, 7, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;

  TestA tests{tableA};

  BOOST_REQUIRE_EQUAL(8, tests.size());

  auto comb2 = TuplesGenerator<TestA, 2>(tests, [](const auto testTuple) { return true; });

  bool isBeginTuplesIterator = std::is_same_v<decltype(comb2.begin()), TuplesGenerator<TestA, 2>::TuplesIterator>;
  BOOST_REQUIRE(isBeginTuplesIterator == true);
  bool isBeginValueTuple = std::is_same_v<decltype(*(comb2.begin())), TuplesGenerator<TestA, 2>::TupleType&>;
  BOOST_REQUIRE(isBeginValueTuple == true);

  auto beginTuple = *(comb2.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginTuple)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginTuple)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginTuple)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginTuple)).getIterator().mCurrentPos), 1);

  BOOST_REQUIRE(comb2.begin() != comb2.end());

  auto endTuple = *(comb2.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endTuple)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endTuple)).getIterator().mCurrentPos), 7);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endTuple)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endTuple)).getIterator().mCurrentPos), 8);
}

BOOST_AUTO_TEST_CASE(Combinations)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  rowWriterA(0, 4, 0);
  rowWriterA(0, 5, 0);
  rowWriterA(0, 6, 0);
  rowWriterA(0, 7, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;

  TestA tests{tableA};

  BOOST_REQUIRE_EQUAL(8, tests.size());
  int n = tests.size();

  auto comb2 = TuplesGenerator<TestA, 2>(tests, [](const auto testTuple) { return true; });

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto comb : comb2) {
    BOOST_CHECK_EQUAL(std::get<0>(comb).x(), i);
    BOOST_CHECK_EQUAL(std::get<1>(comb).x(), j);
    count++;
    j++;
    if (j == n) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 28);

  auto comb2_cond = TuplesGenerator<TestA, 2>(tests, [](const auto testTuple) { return std::get<1>(testTuple).x() == 5; });

  count = 0;
  i = 0;
  j = 5;
  for (auto comb : comb2_cond) {
    BOOST_CHECK_EQUAL(std::get<0>(comb).x(), i);
    BOOST_CHECK_EQUAL(std::get<1>(comb).x(), j);
    count++;
    i++;
  }
  BOOST_CHECK_EQUAL(count, 5);

  auto comb3 = TuplesGenerator<TestA, 3>(tests, [](const auto testTuple) { return true; });

  count = 0;
  i = 0;
  j = 1;
  int k = 2;
  for (auto comb : comb3) {
    BOOST_CHECK_EQUAL(std::get<0>(comb).x(), i);
    BOOST_CHECK_EQUAL(std::get<1>(comb).x(), j);
    BOOST_CHECK_EQUAL(std::get<2>(comb).x(), k);
    count++;
    k++;
    if (k == n) {
      if (j == n - 2) {
        i++;
        j = i;
      }
      j++;
      k = j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 56);

  auto comb3_cond = TuplesGenerator<TestA, 3>(tests, [](const auto testTuple) { return std::get<1>(testTuple).x() == 6; });

  count = 0;
  i = 0;
  j = 6;
  k = 7;
  for (auto comb : comb3_cond) {
    BOOST_CHECK_EQUAL(std::get<0>(comb).x(), i);
    BOOST_CHECK_EQUAL(std::get<1>(comb).x(), j);
    BOOST_CHECK_EQUAL(std::get<2>(comb).x(), k);
    count++;
    k++;
    if (k == n) {
      i++;
      k = j + 1;
    }
  }

  BOOST_CHECK_EQUAL(count, 6);
}
