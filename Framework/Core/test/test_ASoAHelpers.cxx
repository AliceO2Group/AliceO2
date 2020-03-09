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

  std::array<TestA::iterator, 2> comb2{tests.begin(), tests.begin() + 1};
  bool isEnd = false;

  // tests.begin() + 2 == max possible position for
  // the first iterator in a combination of 2 out of 4
  addOne(comb2, tests.begin() + 4 - 2 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(comb2[0].x(), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(comb2[1].x(), 2);
  BOOST_CHECK_EQUAL(isEnd, false);

  comb2[0]++;
  comb2[0]++;
  comb2[1]++;
  addOne(comb2, tests.begin() + 4 - 2 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(comb2[0].x(), 3);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(isEnd, true);

  std::array<TestA::iterator, 3> comb3{tests.begin(), tests.begin() + 1, tests.begin() + 2};
  isEnd = false;

  // tests.begin() + 1 == max possible position for
  // the first iterator in a combination of 3 out of 4
  addOne(comb3, tests.begin() + 4 - 3 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos), 0);
  BOOST_CHECK_EQUAL(comb3[0].x(), 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(comb3[1].x(), 1);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(comb3[2].x(), 3);
  BOOST_CHECK_EQUAL(isEnd, false);

  comb3[0]++;
  comb3[1]++;
  addOne(comb3, tests.begin() + 4 - 3 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(comb3[0].x(), 2);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos), 3);
  BOOST_CHECK_EQUAL(comb3[1].x(), 3);
  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos), 4);
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

  std::array<ConcatTest::iterator, 2> comb2{tests.begin() + 2, tests.begin() + 3};
  bool isEnd = false;

  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb2[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb2[1]).getIterator().mCurrentChunk, 0);

  // tests.begin() + 6 == max possible position for
  // the first iterator in a combination of 2 out of 8
  addOne(comb2, tests.begin() + 8 - 2 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[0]).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(comb2[0].x(), 2);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb2[0]).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb2[1]).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(comb2[1].x(), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb2[1]).getIterator().mCurrentChunk, 1);

  BOOST_CHECK_EQUAL(isEnd, false);

  std::array<ConcatTest::iterator, 3> comb3{tests.begin() + 1, tests.begin() + 2, tests.begin() + 3};
  isEnd = false;
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[1]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[2]).getIterator().mCurrentChunk, 0);

  // tests.begin() + 5 == max possible position for
  // the first iterator in a combination of 3 out of 8
  addOne(comb3, tests.begin() + 8 - 3 + 1, isEnd);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[0]).getIterator().mCurrentPos), 1);
  BOOST_CHECK_EQUAL(comb3[0].x(), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[0]).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[1]).getIterator().mCurrentPos), 2);
  BOOST_CHECK_EQUAL(comb3[1].x(), 2);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[1]).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE_NE(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(comb3[2]).getIterator().mCurrentPos), 4);
  BOOST_CHECK_EQUAL(comb3[2].x(), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb3[2]).getIterator().mCurrentChunk, 1);
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

  TestA tests{tableA};
  ConcatTest concatTests{tableA, tableB};

  BOOST_REQUIRE_EQUAL(8, tests.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  auto comb2 = CombinationsGenerator<TestA, 2>(tests, [](const auto testCombination) { return true; });

  bool isBeginCombinationsIterator = std::is_same_v<decltype(comb2.begin()), CombinationsGenerator<TestA, 2>::CombinationsIterator>;
  BOOST_REQUIRE(isBeginCombinationsIterator == true);
  bool isBeginValueCombination = std::is_same_v<decltype(*(comb2.begin())), CombinationsGenerator<TestA, 2>::CombinationType&>;
  BOOST_REQUIRE(isBeginValueCombination == true);

  auto beginCombination = *(comb2.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(beginCombination[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(beginCombination[0]).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(beginCombination[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(beginCombination[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(beginCombination[1]).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(beginCombination[1]).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2.begin() != comb2.end());

  auto endCombination = *(comb2.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(endCombination[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(endCombination[0]).getIterator().mCurrentPos), 7);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(endCombination[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(endCombination[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(endCombination[1]).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(endCombination[1]).getIterator().mCurrentChunk, 0);

  auto comb2Concat = CombinationsGenerator<ConcatTest, 2>(concatTests, [](const auto testCombination) { return true; });

  isBeginCombinationsIterator = std::is_same_v<decltype(comb2Concat.begin()), CombinationsGenerator<ConcatTest, 2>::CombinationsIterator>;
  BOOST_REQUIRE(isBeginCombinationsIterator == true);
  isBeginValueCombination = std::is_same_v<decltype(*(comb2Concat.begin())), CombinationsGenerator<ConcatTest, 2>::CombinationType&>;
  BOOST_REQUIRE(isBeginValueCombination == true);

  auto beginConcatCombination = *(comb2Concat.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(beginConcatCombination[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(beginConcatCombination[0]).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(beginConcatCombination[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(beginConcatCombination[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(beginConcatCombination[1]).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(beginConcatCombination[1]).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Concat.begin() != comb2Concat.end());

  // Looks that mCurrentChunk is reset to 0 if an iterator goes too far
  // (the iterators before the end() have correct chunk numbers)
  auto endConcatCombination = *(comb2Concat.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(endConcatCombination[0]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(endConcatCombination[0]).getIterator().mCurrentPos), 11);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(endConcatCombination[0]).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(endConcatCombination[1]).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(endConcatCombination[1]).getIterator().mCurrentPos), 12);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(endConcatCombination[1]).getIterator().mCurrentChunk, 0);
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

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using ConcatTest = Concat<TestA, TestB>;

  TestA tests{tableA};
  ConcatTest concatTests{tableA, tableB};

  BOOST_REQUIRE_EQUAL(8, tests.size());
  int n = tests.size();
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  auto comb2 = CombinationsGenerator<TestA, 2>(tests, [](const auto testCombination) { return true; });

  int count = 0;
  int i = 0;
  int j = 1;
  for (auto comb : comb2) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    count++;
    j++;
    if (j == n) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 28);

  auto comb2_cond = CombinationsGenerator<TestA, 2>(tests, [](const auto testCombination) { return testCombination[1].x() == 5; });

  count = 0;
  i = 0;
  j = 5;
  for (auto comb : comb2_cond) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    count++;
    i++;
  }
  BOOST_CHECK_EQUAL(count, 5);

  auto comb3 = CombinationsGenerator<TestA, 3>(tests, [](const auto testCombination) { return true; });

  count = 0;
  i = 0;
  j = 1;
  int k = 2;
  for (auto comb : comb3) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    BOOST_CHECK_EQUAL(comb[2].x(), k);
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

  auto comb3_cond = CombinationsGenerator<TestA, 3>(tests, [](const auto testCombination) { return testCombination[1].x() == 6; });

  count = 0;
  i = 0;
  j = 6;
  k = 7;
  for (auto comb : comb3_cond) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    BOOST_CHECK_EQUAL(comb[2].x(), k);
    count++;
    k++;
    if (k == n) {
      i++;
      k = j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 6);

  int nConcat = concatTests.size();

  auto comb2Concat = CombinationsGenerator<ConcatTest, 2>(concatTests, [](const auto testCombination) { return true; });
  count = 0;
  i = 0;
  j = 1;
  for (auto comb : comb2Concat) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb[0]).getIterator().mCurrentChunk, i < n ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb[1]).getIterator().mCurrentChunk, j < n ? 0 : 1);
    count++;
    j++;
    if (j == nConcat) {
      i++;
      j = i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 66);

  auto comb3Concat = CombinationsGenerator<ConcatTest, 3>(concatTests, [](const auto testCombination) { return true; });
  count = 0;
  i = 0;
  j = 1;
  k = 2;
  for (auto comb : comb3Concat) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    BOOST_CHECK_EQUAL(comb[2].x(), k);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb[0]).getIterator().mCurrentChunk, i < n ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb[1]).getIterator().mCurrentChunk, j < n ? 0 : 1);
    BOOST_REQUIRE_EQUAL(static_cast<test::X>(comb[2]).getIterator().mCurrentChunk, k < n ? 0 : 1);
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

  auto comb5 = CombinationsGenerator<TestA, 5>(tests, [](const auto testCombination) { return true; });

  count = 0;
  i = 0;
  j = 1;
  k = 2;
  int l = 3;
  int m = 4;
  for (auto comb : comb5) {
    BOOST_CHECK_EQUAL(comb[0].x(), i);
    BOOST_CHECK_EQUAL(comb[1].x(), j);
    BOOST_CHECK_EQUAL(comb[2].x(), k);
    BOOST_CHECK_EQUAL(comb[3].x(), l);
    BOOST_CHECK_EQUAL(comb[4].x(), m);
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
