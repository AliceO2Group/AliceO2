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

BOOST_AUTO_TEST_CASE(IteratorTuple)
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

  TestA::iterator beginIt = tests.begin();
  BOOST_REQUIRE_NE(static_cast<test::X>(beginIt).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(beginIt).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(beginIt.x(), 0);
  BOOST_CHECK_EQUAL(beginIt.mRowIndex, 0);

  auto beginIterators = std::make_tuple(beginIt, beginIt);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginIterators)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginIterators)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<0>(beginIterators).x(), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginIterators)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginIterators)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<1>(beginIterators).x(), 0);

  auto maxIt0 = tests.begin() + 8 - 2 + 1;
  auto maxIt1 = tests.begin() + 8 - 2 + 1 + 1;
  auto maxOffset2 = std::make_tuple(maxIt0, maxIt1);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(maxOffset2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(maxOffset2)).getIterator().mCurrentPos), 7);
  BOOST_CHECK_EQUAL(std::get<0>(maxOffset2).x(), 7);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(maxOffset2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(maxOffset2)).getIterator().mCurrentPos), 8);

  expressions::Filter filter = test::x > 3;
  auto filtered = Filtered<TestA>{{tests.asArrowTable()}, o2::framework::expressions::createSelection(tests.asArrowTable(), filter)};
  std::tuple<Filtered<TestA>, Filtered<TestA>> filteredTuple = std::make_tuple(filtered, filtered);

  auto it1 = std::get<0>(filteredTuple).begin();
  BOOST_REQUIRE_NE(static_cast<test::X>(it1).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(it1).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(it1).getIterator().mCurrentChunk, 0);
  auto it2(it1);
  BOOST_REQUIRE_NE(static_cast<test::X>(it2).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(it2).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(it2).getIterator().mCurrentChunk, 0);
  auto it3 = std::get<1>(filteredTuple).begin();
  BOOST_REQUIRE_NE(static_cast<test::X>(it3).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(it3).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(it3).getIterator().mCurrentChunk, 0);
}

BOOST_AUTO_TEST_CASE(AddOne)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 4);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;

  TestA tests{tableA};

  BOOST_REQUIRE_EQUAL(4, tests.size());

  auto beginIt0 = tests.begin();
  auto beginIt1 = tests.begin() + 1;
  auto maxIt0 = tests.begin() + 4 - 2 + 1;
  auto maxIt1 = tests.begin() + 4 - 2 + 1 + 1;
  auto comb2 = std::make_tuple(beginIt0, beginIt1);
  auto maxOffset2 = std::make_tuple(maxIt0, maxIt1);
  bool isEnd = false;

  // tests.begin() + 2 == max possible position for
  // the first iterator in a combination of 2 out of 4
  addOne(comb2, maxOffset2, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<0>(comb2).x(), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(std::get<1>(comb2).x(), 2);
  BOOST_CHECK_EQUAL(isEnd, false);

  std::get<0>(comb2)++;
  std::get<0>(comb2)++;
  std::get<1>(comb2)++;
  addOne(comb2, maxOffset2, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(std::get<0>(comb2).x(), 3);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(isEnd, true);

  auto beginIt2 = tests.begin() + 2;
  maxIt0 = tests.begin() + 4 - 3 + 1;
  maxIt1 = tests.begin() + 4 - 3 + 1 + 1;
  auto maxIt2 = tests.begin() + 4 - 3 + 2 + 1;
  auto comb3 = std::make_tuple(beginIt0, beginIt1, beginIt2);
  auto maxOffset3 = std::make_tuple(maxIt0, maxIt1, maxIt2);
  isEnd = false;

  // tests.begin() + 1 == max possible position for
  // the first iterator in a combination of 3 out of 4
  addOne(comb3, maxOffset3, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<0>(comb3).x(), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(std::get<1>(comb3).x(), 1);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(std::get<2>(comb3).x(), 3);
  BOOST_CHECK_EQUAL(isEnd, false);

  std::get<0>(comb3)++;
  std::get<1>(comb3)++;
  addOne(comb3, maxOffset3, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(std::get<0>(comb3).x(), 2);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(std::get<1>(comb3).x(), 3);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(isEnd, true);
}

BOOST_AUTO_TEST_CASE(AddOneMultipleChunks)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 4);

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  rowWriterB(0, 4);
  rowWriterB(0, 5);
  rowWriterB(0, 6);
  rowWriterB(0, 7);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 4);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using ConcatTest = Concat<TestA, TestB>;

  ConcatTest tests{tableA, tableB};

  BOOST_REQUIRE_EQUAL(8, tests.size());

  auto beginIt0 = tests.begin() + 2;
  auto beginIt1 = tests.begin() + 3;
  auto maxIt0 = tests.begin() + 8 - 2 + 1;
  auto maxIt1 = tests.begin() + 8 - 2 + 1 + 1;
  auto comb2 = std::make_tuple(beginIt0, beginIt1);
  auto maxOffset2 = std::make_tuple(maxIt0, maxIt1);
  bool isEnd = false;

  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentChunk, 0);

  // tests.begin() + 6 == max possible position for
  // the first iterator in a combination of 2 out of 8
  addOne(comb2, maxOffset2, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(std::get<0>(comb2).x(), 2);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(std::get<1>(comb2).x(), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentChunk, 1);

  BOOST_CHECK_EQUAL(isEnd, false);

  beginIt0 = tests.begin() + 1;
  beginIt1 = tests.begin() + 2;
  auto beginIt2 = tests.begin() + 3;
  maxIt0 = tests.begin() + 8 - 3 + 1;
  maxIt1 = tests.begin() + 8 - 3 + 1 + 1;
  auto maxIt2 = tests.begin() + 8 - 3 + 2 + 1;
  auto comb3 = std::make_tuple(beginIt0, beginIt1, beginIt2);
  auto maxOffset3 = std::make_tuple(maxIt0, maxIt1, maxIt2);
  isEnd = false;

  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentChunk, 0);

  // tests.begin() + 5 == max possible position for
  // the first iterator in a combination of 3 out of 8
  addOne(comb3, maxOffset3, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(std::get<0>(comb3).x(), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(std::get<1>(comb3).x(), 2);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(std::get<2>(comb3).x(), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentChunk, 1);

  BOOST_CHECK_EQUAL(isEnd, false);
}

BOOST_AUTO_TEST_CASE(AddOneDifferentTables)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 4);

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  rowWriterB(0, 4);
  rowWriterB(0, 5);
  rowWriterB(0, 6);
  rowWriterB(0, 7);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 4);

  TableBuilder builderC;
  auto rowWriterC = builderC.persist<int32_t, int32_t, int32_t>({"x", "y", "z"});
  rowWriterC(0, 8, 0, 0);
  rowWriterC(0, 9, 0, 0);
  rowWriterC(0, 10, 0, 0);
  rowWriterC(0, 11, 0, 0);
  auto tableC = builderC.finalize();
  BOOST_REQUIRE_EQUAL(tableC->num_rows(), 4);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using TestC = o2::soa::Table<o2::soa::Index<>, test::X, test::Y, test::Z>;

  TestA testsA{tableA};
  TestB testsB{tableB};
  TestC testsC{tableC};

  BOOST_REQUIRE_EQUAL(4, testsA.size());
  BOOST_REQUIRE_EQUAL(4, testsB.size());
  BOOST_REQUIRE_EQUAL(4, testsC.size());

  auto beginIt0 = testsA.begin();
  auto beginIt1 = testsB.begin();
  auto maxIt0 = testsA.end();
  auto maxIt1 = testsB.end();
  auto comb2 = std::make_tuple(beginIt0, beginIt1);
  auto maxOffset2 = std::make_tuple(maxIt0, maxIt1);
  bool isEnd = false;

  addOne(comb2, maxOffset2, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb2)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<0>(comb2).x(), 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb2)).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(std::get<1>(comb2).x(), 5);

  BOOST_CHECK_EQUAL(isEnd, false);

  auto beginIt2 = testsC.begin();
  auto maxIt2 = testsC.end();
  auto comb3 = std::make_tuple(beginIt0, beginIt1, beginIt2);
  auto maxOffset3 = std::make_tuple(maxIt0, maxIt1, maxIt2);
  isEnd = false;

  // tests.begin() + 5 == max possible position for
  // the first iterator in a combination of 3 out of 8
  addOne(comb3, maxOffset3, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(comb3)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<0>(comb3).x(), 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(comb3)).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(std::get<1>(comb3).x(), 4);

  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(comb3)).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(std::get<2>(comb3).x(), 9);

  BOOST_CHECK_EQUAL(isEnd, false);
}

BOOST_AUTO_TEST_CASE(CombinationsGeneratorConstruction)
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

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  rowWriterB(0, 8);
  rowWriterB(0, 9);
  rowWriterB(0, 10);
  rowWriterB(0, 11);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 4);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using ConcatTest = Concat<TestA, TestB>;

  TestA testsA{tableA};
  TestB testsB{tableB};
  ConcatTest concatTests{tableA, tableB};
  BOOST_REQUIRE_EQUAL(8, testsA.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  CombinationsGenerator<TestA, TestA>::CombinationsIterator combIt(testsA, testsA);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentChunk, 0);

  auto comb2 = combinations(testsA, testsA);

  static_assert(std::is_same_v<decltype(comb2.begin()), CombinationsGenerator<TestA, TestA>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2.begin())), CombinationsGenerator<TestA, TestA>::CombinationType&>, "Wrong combination type");

  auto beginCombination = *(comb2.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginCombination)).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2.begin() != comb2.end());

  auto endCombination = *(comb2.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endCombination)).getIterator().mCurrentPos), 7);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentChunk, 0);

  expressions::Filter filter = test::x > 3;
  auto filtered = Filtered<TestA>{{testsA.asArrowTable()}, o2::framework::expressions::createSelection(testsA.asArrowTable(), filter)};

  CombinationsGenerator<Filtered<TestA>, Filtered<TestA>>::CombinationsIterator combItFiltered(filtered, filtered);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos), 5);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentChunk, 0);

  auto comb2Filter = combinations(filter, testsA, testsA);

  static_assert(std::is_same_v<decltype(comb2Filter.begin()), CombinationsGenerator<Filtered<TestA>, Filtered<TestA>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Filter.begin())), CombinationsGenerator<Filtered<TestA>, Filtered<TestA>>::CombinationType&>, "Wrong combination type");

  auto beginFilterCombination = *(comb2Filter.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginFilterCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginFilterCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginFilterCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginFilterCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginFilterCombination)).getIterator().mCurrentPos), 5);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginFilterCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Filter.begin() != comb2Filter.end());

  auto endFilterCombination = *(comb2Filter.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endFilterCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endFilterCombination)).getIterator().mCurrentPos), 7);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endFilterCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentPos), -1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentChunk, 0);

  auto comb2Concat = combinations(concatTests, concatTests);

  static_assert(std::is_same_v<decltype(comb2Concat.begin()), CombinationsGenerator<ConcatTest, ConcatTest>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Concat.begin())), CombinationsGenerator<ConcatTest, ConcatTest>::CombinationType&>, "Wrong combination type");

  auto beginConcatCombination = *(comb2Concat.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginConcatCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginConcatCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginConcatCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginConcatCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginConcatCombination)).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginConcatCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Concat.begin() != comb2Concat.end());

  // Looks that mCurrentChunk is reset to 0 if an iterator goes too far
  // (the iterators before the end() have correct chunk numbers)
  auto endConcatCombination = *(comb2Concat.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endConcatCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endConcatCombination)).getIterator().mCurrentPos), 11);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endConcatCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentPos), 12);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentChunk, 0);

  auto comb2Diff = combinations(testsA, testsB);

  static_assert(std::is_same_v<decltype(comb2Diff.begin()), CombinationsGenerator<TestA, TestB>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Diff.begin())), CombinationsGenerator<TestA, TestB>::CombinationType&>, "Wrong combination type");

  auto beginDiffCombination = *(comb2Diff.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Diff.begin() != comb2Diff.end());

  // Looks that mCurrentChunk is reset to 0 if an iterator goes too far
  // (the iterators before the end() have correct chunk numbers)
  auto endDiffCombination = *(comb2Diff.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentPos), 7);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentChunk, 0);
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

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  rowWriterB(0, 8);
  rowWriterB(0, 9);
  rowWriterB(0, 10);
  rowWriterB(0, 11);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 4);

  TableBuilder builderC;
  auto rowWriterC = builderC.persist<int32_t, int32_t, int32_t>({"x", "y", "z"});
  rowWriterC(0, 12, 0, 0);
  rowWriterC(0, 13, 0, 0);
  rowWriterC(0, 14, 0, 0);
  rowWriterC(0, 15, 0, 0);
  auto tableC = builderC.finalize();
  BOOST_REQUIRE_EQUAL(tableC->num_rows(), 4);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using TestC = o2::soa::Table<o2::soa::Index<>, test::X, test::Y, test::Z>;
  using ConcatTest = Concat<TestA, TestB>;

  TestA testsA{tableA};
  TestB testsB{tableB};
  TestC testsC{tableC};
  ConcatTest concatTests{tableA, tableB};

  BOOST_REQUIRE_EQUAL(8, testsA.size());
  int n = testsA.size();
  BOOST_REQUIRE_EQUAL(4, testsB.size());
  BOOST_REQUIRE_EQUAL(4, testsC.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto& [t0, t1] : combinations(testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == n) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 28);

  expressions::Filter pairsFilter = test::x > 3;

  count = 0;
  i = 4;
  j = 5;
  for (auto& [t0, t1] : combinations(pairsFilter, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == n) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 6);

  count = 0;
  i = 0;
  j = 1;
  int k = 2;
  for (auto& [t0, t1, t2] : combinations(testsA, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
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

  expressions::Filter triplesFilter = test::x < 4;

  count = 0;
  i = 0;
  j = 1;
  k = 2;
  for (auto& [t0, t1, t2] : combinations(triplesFilter, testsA, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == 4) {
      if (j == 2) {
        i++;
        j = i;
      }
      j++;
      k = j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 4);

  //int nConcat = concatTests.size();

  //count = 0;
  //i = 0;
  //j = 1;
  //for (auto& [t0, t1] : combinations(concatTests, concatTests)) {
  //  BOOST_CHECK_EQUAL(t0.x(), i);
  //  BOOST_CHECK_EQUAL(t1.x(), j);
  //  BOOST_REQUIRE_EQUAL(static_cast<test::X>(t0).getIterator().mCurrentChunk, i < n ? 0 : 1);
  //  BOOST_REQUIRE_EQUAL(static_cast<test::X>(t1).getIterator().mCurrentChunk, j < n ? 0 : 1);
  //  count++;
  //  j++;
  //  if (j == nConcat) {
  //    i++;
  //    j = i + 1;
  //  }
  //}
  //BOOST_CHECK_EQUAL(count, 66);

  //count = 0;
  //i = 0;
  //j = 1;
  //k = 2;
  //for (auto [t0, t1, t2] : combinations(concatTests, concatTests, concatTests)) {
  //  BOOST_CHECK_EQUAL(t0.x(), i);
  //  BOOST_CHECK_EQUAL(t1.x(), j);
  //  BOOST_CHECK_EQUAL(t2.x(), k);
  //  BOOST_REQUIRE_EQUAL(static_cast<test::X>(t0).getIterator().mCurrentChunk, i < n ? 0 : 1);
  //  BOOST_REQUIRE_EQUAL(static_cast<test::X>(t1).getIterator().mCurrentChunk, j < n ? 0 : 1);
  //  BOOST_REQUIRE_EQUAL(static_cast<test::X>(t2).getIterator().mCurrentChunk, k < n ? 0 : 1);
  //  count++;
  //  k++;
  //  if (k == nConcat) {
  //    if (j == nConcat - 2) {
  //      i++;
  //      j = i;
  //    }
  //    j++;
  //    k = j + 1;
  //  }
  //}
  //BOOST_CHECK_EQUAL(count, 220);

  count = 0;
  i = 0;
  j = 1;
  k = 2;
  int l = 3;
  int m = 4;
  for (auto& [t0, t1, t2, t3, t4] : combinations(testsA, testsA, testsA, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    BOOST_CHECK_EQUAL(t3.x(), l);
    BOOST_CHECK_EQUAL(t4.x(), m);
    count++;
    m++;
    if (m == n) {
      if (l == n - 2) {
        if (k == n - 3) {
          if (j == n - 4) {
            i++;
            j = i;
          }
          j++;
          k = j;
        }
        k++;
        l = k;
      }
      l++;
      m = l + 1;
    }
  }

  BOOST_CHECK_EQUAL(count, 56);
}
