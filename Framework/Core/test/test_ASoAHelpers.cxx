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

  CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, TestA, TestA>::CombinationsIterator combIt(CombinationsStrictlyUpperIndexPolicy(testsA, testsA));
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentChunk, 0);

  auto comb2 = combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA));

  static_assert(std::is_same_v<decltype(comb2.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, TestA, TestA>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2.begin())), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, TestA, TestA>::CombinationType&>, "Wrong combination type");

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

  CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, Filtered<TestA>, Filtered<TestA>>::CombinationsIterator combItFiltered(CombinationsStrictlyUpperIndexPolicy(filtered, filtered));
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos), 5);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentChunk, 0);

  auto comb2Filter = combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA), filter, testsA, testsA);

  static_assert(std::is_same_v<decltype(comb2Filter.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, Filtered<TestA>, Filtered<TestA>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Filter.begin())), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, Filtered<TestA>, Filtered<TestA>>::CombinationType&>, "Wrong combination type");

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

  auto comb2Concat = combinations(CombinationsStrictlyUpperIndexPolicy(concatTests, concatTests));

  static_assert(std::is_same_v<decltype(comb2Concat.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, ConcatTest, ConcatTest>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Concat.begin())), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, ConcatTest, ConcatTest>::CombinationType&>, "Wrong combination type");

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

  auto comb2Diff = combinations(CombinationsFullIndexPolicy(testsA, testsB));

  static_assert(std::is_same_v<decltype(comb2Diff.begin()), CombinationsGenerator<CombinationsFullIndexPolicy, TestA, TestB>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Diff.begin())), CombinationsGenerator<CombinationsFullIndexPolicy, TestA, TestB>::CombinationType&>, "Wrong combination type");

  auto beginDiffCombination = *(comb2Diff.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginDiffCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginDiffCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Diff.begin() != comb2Diff.end());

  auto endDiffCombination = *(comb2Diff.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endDiffCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endDiffCombination)).getIterator().mCurrentChunk, 0);

  // More elements required for a combination than number of elements in the table
  auto comb2Bad = combinations(CombinationsStrictlyUpperIndexPolicy(testsB, testsB, testsB, testsB, testsB));

  static_assert(std::is_same_v<decltype(comb2Bad.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, TestB, TestB, TestB, TestB, TestB>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Bad.begin())), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, TestB, TestB, TestB, TestB, TestB>::CombinationType&>, "Wrong combination type");

  auto beginBadCombination = *(comb2Bad.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Bad.begin() == comb2Bad.end());

  auto endBadCombination = *(comb2Bad.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentChunk, 0);
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
  int nA = testsA.size();
  BOOST_REQUIRE_EQUAL(4, testsB.size());
  BOOST_REQUIRE_EQUAL(4, testsC.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto& [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 28);

  expressions::Filter pairsFilter = test::x > 3;

  count = 0;
  i = 4;
  j = 5;
  for (auto& [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA), pairsFilter, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 6);

  count = 0;
  i = 0;
  j = 1;
  int k = 2;
  for (auto& [t0, t1, t2] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA) {
      if (j == nA - 2) {
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
  for (auto& [t0, t1, t2] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA, testsA), triplesFilter, testsA, testsA, testsA)) {
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

  int nConcat = concatTests.size();

  count = 0;
  i = 0;
  j = 1;
  for (auto [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(concatTests, concatTests))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(t0).getIterator().mCurrentChunk, i < nA ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(t1).getIterator().mCurrentChunk, j < nA ? 0 : 1);
    count++;
    j++;
    if (j == nConcat) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 66);

  count = 0;
  i = 0;
  j = 1;
  k = 2;
  for (auto [t0, t1, t2] : combinations(CombinationsStrictlyUpperIndexPolicy(concatTests, concatTests, concatTests))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(t0).getIterator().mCurrentChunk, i < nA ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(t1).getIterator().mCurrentChunk, j < nA ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(t2).getIterator().mCurrentChunk, k < nA ? 0 : 1);
    count++;
    k++;
    if (k == nConcat) {
      if (j == nConcat - 2) {
        i++;
        j = i;
      }
      j++;
      k = j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 220);

  count = 0;
  i = 0;
  j = 1;
  k = 2;
  int l = 3;
  int m = 4;
  for (auto& [t0, t1, t2, t3, t4] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA, testsA, testsA, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    BOOST_CHECK_EQUAL(t3.x(), l);
    BOOST_CHECK_EQUAL(t4.x(), m);
    count++;
    m++;
    if (m == nA) {
      if (l == nA - 2) {
        if (k == nA - 3) {
          if (j == nA - 4) {
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

  int nB = testsB.size();
  count = 0;
  i = 0;
  j = 0 + nA;
  for (auto& [t0, t1] : combinations(CombinationsFullIndexPolicy(testsA, testsB))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA + nB) {
      i++;
      j = 0 + nA;
    }
  }
  BOOST_CHECK_EQUAL(count, 32);

  int nC = testsC.size();
  count = 0;
  i = 0;
  j = 0 + nA;
  k = 0 + nA + nB;
  for (auto& [t0, t1, t2] : combinations(CombinationsFullIndexPolicy(testsA, testsB, testsC))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA + nB + nC) {
      if (j == nA + nB - 1) {
        i++;
        j = 0 + nA;
      } else {
        j++;
      }
      k = 0 + nA + nB;
    }
  }
  BOOST_CHECK_EQUAL(count, 128);

  count = 0;
  i = 0;
  j = 0 + nA;
  for (auto& [t0, t1] : combinations(testsA, testsB)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA + nB) {
      i++;
      j = 0 + nA;
    }
  }
  BOOST_CHECK_EQUAL(count, 32);

  count = 0;
  i = 0;
  j = 1;
  for (auto& [t0, t1] : combinations(testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 28);

  count = 0;
  i = 4;
  j = 5;
  for (auto& [t0, t1] : combinations(pairsFilter, testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 6);
}

BOOST_AUTO_TEST_CASE(BreakingCombinations)
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

  TestA testsA{tableA};

  BOOST_REQUIRE_EQUAL(8, testsA.size());
  int nA = testsA.size();

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto& [t0, t1] : combinations(testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
    if (t0.x() == 4) {
      continue;
      BOOST_REQUIRE_NE(true, true);
    }
    BOOST_REQUIRE_NE(t0.x(), 4);
  }
  BOOST_CHECK_EQUAL(count, 28);

  count = 0;
  i = 0;
  j = 1;
  for (auto& [t0, t1] : combinations(testsA, testsA)) {
    if (t0.x() == 4) {
      break;
      BOOST_REQUIRE_NE(true, true);
    }
    BOOST_REQUIRE(t0.x() < 4);
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == nA) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 22);
}

BOOST_AUTO_TEST_CASE(SmallTableCombinations)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 2);

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  rowWriterB(0, 8);
  rowWriterB(0, 9);
  rowWriterB(0, 10);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 3);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;

  TestA testsA{tableA};
  TestB testsB{tableB};

  BOOST_REQUIRE_EQUAL(2, testsA.size());
  BOOST_REQUIRE_EQUAL(3, testsB.size());
  int nA = testsA.size();
  int nB = testsB.size();

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto& [t0, t1] : combinations(testsA, testsA)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
  }
  BOOST_CHECK_EQUAL(count, 1);

  count = 0;
  i = 8;
  j = 9;
  int k = 10;
  for (auto& [t0, t1, t2] : combinations(testsB, testsB, testsB)) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
  }
  BOOST_CHECK_EQUAL(count, 1);

  count = 0;
  for (auto& [t0, t1, t2] : combinations(testsA, testsA, testsA)) {
    count++;
  }
  BOOST_CHECK_EQUAL(count, 0);
}
