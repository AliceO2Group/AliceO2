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
#include "Framework/AnalysisDataModel.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace o2::soa;

namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, int32_t, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, int32_t, "y");
DECLARE_SOA_COLUMN_FULL(Z, z, int32_t, "z");
DECLARE_SOA_COLUMN_FULL(FloatZ, floatZ, float, "floatZ");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](int32_t x, int32_t y) { return x + y; });
} // namespace test

// Calculate hash for an element based on 2 properties and their bins.
int32_t getHash(const std::vector<uint32_t>& yBins, const std::vector<float>& zBins, uint32_t colY, float colZ, bool ignoreOverflows = false)
{
  if (ignoreOverflows) {
    if (colY < yBins[0] || colZ < zBins[0]) {
      return -1;
    }
  }

  for (int i = 0; i < yBins.size(); i++) {
    if (colY < yBins[i]) {
      for (int j = 0; j < zBins.size(); j++) {
        if (colZ < zBins[j]) {
          return i + j * (yBins.size() + 1);
        }
      }
      // overflow for zBins only
      return ignoreOverflows ? -1 : i + zBins.size() * (yBins.size() + 1);
    }
  }

  if (ignoreOverflows) {
    return -1;
  }

  // overflow for yBins only
  for (int j = 0; j < zBins.size(); j++) {
    if (colZ < zBins[j]) {
      return yBins.size() + j * (yBins.size() + 1);
    }
  }

  // overflow for both bins
  return (zBins.size() + 1) * (yBins.size() + 1) - 1;
}

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
  auto rowWriterA = builderA.persist<int32_t, int32_t, float>({"x", "y", "floatZ"});
  rowWriterA(0, 0, 25, -6.0f);
  rowWriterA(0, 1, 18, 0.0f);
  rowWriterA(0, 2, 48, 8.0f);
  rowWriterA(0, 3, 103, 2.0f);
  rowWriterA(0, 4, 28, -6.0f);
  rowWriterA(0, 5, 102, 2.0f);
  rowWriterA(0, 6, 12, 0.0f);
  rowWriterA(0, 7, 24, -7.0f);
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

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y, test::FloatZ>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using ConcatTest = Concat<TestA, TestB>;

  TestA testsA{tableA};
  TestB testsB{tableB};
  ConcatTest concatTests{tableA, tableB};
  BOOST_REQUIRE_EQUAL(8, testsA.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());

  // Grouped data:
  // [3, 5] [0, 4, 7], [1, 6], [2]
  // Assuming bins intervals: [ , )
  std::vector<uint32_t> yBins{0, 5, 10, 20, 30, 40, 50, 101};
  std::vector<float> zBins{-7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f};

  TableBuilder builderAux;
  auto rowWriterAux = builderAux.persist<int32_t, int32_t>({"x", "y"});
  for (auto it = testsA.begin(); it != testsA.end(); it++) {
    auto& elem = *it;
    rowWriterAux(0, elem.x(), getHash(yBins, zBins, elem.y(), elem.floatZ()));
  }
  auto tableAux = builderAux.finalize();
  BOOST_REQUIRE_EQUAL(tableAux->num_rows(), 8);
  using TestsAux = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  TestsAux testAux{tableAux};
  BOOST_REQUIRE_EQUAL(8, testAux.size());

  CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<TestA, TestA>>::CombinationsIterator combIt(CombinationsStrictlyUpperIndexPolicy(testsA, testsA));
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combIt))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentPos), 1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combIt))).getIterator().mCurrentChunk, 0);

  auto comb2 = combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA));

  static_assert(std::is_same_v<decltype(comb2.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<TestA, TestA>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2.begin())), CombinationsStrictlyUpperIndexPolicy<TestA, TestA>::CombinationType&>, "Wrong combination type");

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
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endCombination)).getIterator().mCurrentChunk, 0);

  o2::framework::expressions::Filter filter = test::x > 3;
  auto filtered = Filtered<TestA>{{testsA.asArrowTable()}, o2::framework::expressions::createSelection(testsA.asArrowTable(), filter)};

  CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<Filtered<TestA>, Filtered<TestA>>>::CombinationsIterator combItFiltered(CombinationsStrictlyUpperIndexPolicy(filtered, filtered));
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(*(combItFiltered))).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentPos), 5);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(*(combItFiltered))).getIterator().mCurrentChunk, 0);

  auto comb2Filter = combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsA), filter, testsA, testsA);

  static_assert(std::is_same_v<decltype(comb2Filter.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<Filtered<TestA>, Filtered<TestA>>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Filter.begin())), CombinationsStrictlyUpperIndexPolicy<Filtered<TestA>, Filtered<TestA>>::CombinationType&>, "Wrong combination type");

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
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endFilterCombination)).getIterator().mCurrentPos), -1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endFilterCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentPos), -1);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endFilterCombination)).getIterator().mCurrentChunk, 0);

  auto comb2Concat = combinations(CombinationsStrictlyUpperIndexPolicy(concatTests, concatTests));

  static_assert(std::is_same_v<decltype(comb2Concat.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<ConcatTest, ConcatTest>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Concat.begin())), CombinationsStrictlyUpperIndexPolicy<ConcatTest, ConcatTest>::CombinationType&>, "Wrong combination type");

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
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endConcatCombination)).getIterator().mCurrentPos), 12);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endConcatCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentPos), 12);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endConcatCombination)).getIterator().mCurrentChunk, 0);

  auto comb2Diff = combinations(CombinationsFullIndexPolicy(testsA, testsB));

  static_assert(std::is_same_v<decltype(comb2Diff.begin()), CombinationsGenerator<CombinationsFullIndexPolicy<TestA, TestB>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Diff.begin())), CombinationsFullIndexPolicy<TestA, TestB>::CombinationType&>, "Wrong combination type");

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

  static_assert(std::is_same_v<decltype(comb2Bad.begin()), CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<TestB, TestB, TestB, TestB, TestB>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(comb2Bad.begin())), CombinationsStrictlyUpperIndexPolicy<TestB, TestB, TestB, TestB, TestB>::CombinationType&>, "Wrong combination type");

  auto beginBadCombination = *(comb2Bad.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<2>(beginBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<3>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<3>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<3>(beginBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<4>(beginBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<4>(beginBadCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<4>(beginBadCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(comb2Bad.begin() == comb2Bad.end());

  auto endBadCombination = *(comb2Bad.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<2>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<2>(endBadCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<2>(endBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<3>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<3>(endBadCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<3>(endBadCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<4>(endBadCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<4>(endBadCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<4>(endBadCombination)).getIterator().mCurrentChunk, 0);

  auto combBlock = combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("y", 2, -1, testAux, testAux));

  static_assert(std::is_same_v<decltype(combBlock.begin()), CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<int32_t, TestsAux, TestsAux>>::CombinationsIterator>, "Wrong iterator type");
  static_assert(std::is_same_v<decltype(*(combBlock.begin())), CombinationsBlockStrictlyUpperSameIndexPolicy<int32_t, TestsAux, TestsAux>::CombinationType&>, "Wrong combination type");

  auto beginBlockCombination = *(combBlock.begin());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(beginBlockCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(beginBlockCombination)).getIterator().mCurrentPos), 0);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(beginBlockCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(beginBlockCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(beginBlockCombination)).getIterator().mCurrentPos), 4);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(beginBlockCombination)).getIterator().mCurrentChunk, 0);

  BOOST_REQUIRE(combBlock.begin() != combBlock.end());

  auto endBlockCombination = *(combBlock.end());
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<0>(endBlockCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<0>(endBlockCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<0>(endBlockCombination)).getIterator().mCurrentChunk, 0);
  BOOST_REQUIRE_NE(static_cast<test::X>(std::get<1>(endBlockCombination)).getIterator().mCurrentPos, nullptr);
  BOOST_REQUIRE_EQUAL(*(static_cast<test::X>(std::get<1>(endBlockCombination)).getIterator().mCurrentPos), 8);
  BOOST_REQUIRE_EQUAL(static_cast<test::X>(std::get<1>(endBlockCombination)).getIterator().mCurrentChunk, 0);
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
  BOOST_REQUIRE_EQUAL(4, testsB.size());
  BOOST_REQUIRE_EQUAL(4, testsC.size());
  BOOST_REQUIRE_EQUAL(12, concatTests.size());
  int nA = testsA.size();
  int nB = testsB.size();
  int nC = testsC.size();

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

  // Combinations shortcut
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
      j = nA + i;
    }
  }
  BOOST_CHECK_EQUAL(count, 10);

  // Different tables of different size
  count = 0;
  i = 0;
  j = 0 + nA;
  for (auto& [t0, t1] : combinations(CombinationsFullIndexPolicy(testsA, testsB))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA + nB) {
      i++;
      j = 0 + nA;
    }
  }
  BOOST_CHECK_EQUAL(count, 32);

  count = 0;
  i = 0 + nA;
  j = 0;
  for (auto& [t0, t1] : combinations(CombinationsFullIndexPolicy(testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA) {
      i++;
      j = 0;
    }
  }
  BOOST_CHECK_EQUAL(count, 32);

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
  i = 0 + nA + nB;
  j = 0 + nA;
  k = 0;
  for (auto& [t0, t1, t2] : combinations(CombinationsFullIndexPolicy(testsC, testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA) {
      if (j == nA + nB - 1) {
        i++;
        j = 0 + nA;
      } else {
        j++;
      }
      k = 0;
    }
  }
  BOOST_CHECK_EQUAL(count, 128);

  count = 0;
  i = 0;
  j = 0 + nA;
  for (auto& [t0, t1] : combinations(CombinationsUpperIndexPolicy(testsA, testsB))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA + nB) {
      i++;
      j = nA + i;
    }
  }
  BOOST_CHECK_EQUAL(count, 10);

  count = 0;
  i = 0 + nA;
  j = 0;
  for (auto& [t0, t1] : combinations(CombinationsUpperIndexPolicy(testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA) {
      i++;
      j = -nA + i;
    }
  }
  BOOST_CHECK_EQUAL(count, 26);

  count = 0;
  i = 0;
  j = 0 + nA;
  k = 0 + nA + nB;
  for (auto& [t0, t1, t2] : combinations(CombinationsUpperIndexPolicy(testsA, testsB, testsC))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA + nB + nC) {
      if (j == nA + nB - 1) {
        i++;
        j = nA + i;
      } else {
        j++;
      }
      k = nB + j;
    }
  }
  BOOST_CHECK_EQUAL(count, 20);

  count = 0;
  i = 0 + nA + nB;
  j = 0 + nA;
  k = 0;
  for (auto& [t0, t1, t2] : combinations(CombinationsUpperIndexPolicy(testsC, testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA) {
      if (j == nA + nB - 1) {
        i++;
        j = -nB + i;
      } else {
        j++;
      }
      k = -nA + j;
    }
  }
  BOOST_CHECK_EQUAL(count, 60);

  count = 0;
  i = 0;
  j = 0 + nA + 1;
  for (auto& [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsB))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA + nB) {
      i++;
      j = nA + i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 6);

  count = 0;
  i = 0 + nA;
  j = 1;
  for (auto& [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    count++;
    j++;
    if (j == 0 + nA) {
      i++;
      j = -nA + i + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 18);

  count = 0;
  i = 0;
  j = 0 + nA + 1;
  k = 0 + nA + nB + 2;
  for (auto& [t0, t1, t2] : combinations(CombinationsStrictlyUpperIndexPolicy(testsA, testsB, testsC))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA + nB + nC) {
      if (j == nA + nB - 2) {
        i++;
        j = nA + i + 1;
      } else {
        j++;
      }
      k = nB + j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 4);

  count = 0;
  i = 0 + nA + nB;
  j = 0 + nA + 1;
  k = 2;
  for (auto& [t0, t1, t2] : combinations(CombinationsStrictlyUpperIndexPolicy(testsC, testsB, testsA))) {
    BOOST_CHECK_EQUAL(t0.x(), i);
    BOOST_CHECK_EQUAL(t1.x(), j);
    BOOST_CHECK_EQUAL(t2.x(), k);
    count++;
    k++;
    if (k == nA) {
      if (j == nA + nB - 2) {
        i++;
        j = -nB + i + 1;
      } else {
        j++;
      }
      k = -nA + j + 1;
    }
  }
  BOOST_CHECK_EQUAL(count, 16);
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

BOOST_AUTO_TEST_CASE(BlockCombinations)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t, float>({"x", "y", "floatZ"});
  rowWriterA(0, 0, 25, -6.0f);
  rowWriterA(0, 1, 18, 0.0f);
  rowWriterA(0, 2, 48, 8.0f);
  rowWriterA(0, 3, 103, 2.0f);
  rowWriterA(0, 4, 28, -6.0f);
  rowWriterA(0, 5, 102, 2.0f);
  rowWriterA(0, 6, 12, 0.0f);
  rowWriterA(0, 7, 24, -7.0f);
  rowWriterA(0, 8, 41, 8.0f);
  rowWriterA(0, 9, 49, 8.0f);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 10);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y, test::FloatZ>;
  TestA testA{tableA};
  BOOST_REQUIRE_EQUAL(10, testA.size());

  // Grouped data:
  // [3, 5] [0, 4, 7], [1, 6], [2, 8, 9]
  // Assuming bins intervals: [ , )
  std::vector<uint32_t> yBins{0, 5, 10, 20, 30, 40, 50, 101};
  std::vector<float> zBins{-7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f};

  TableBuilder builderAux;
  TableBuilder builderAuxHalf;
  auto rowWriterAux = builderAux.persist<int32_t, int32_t>({"x", "y"});
  auto rowWriterAuxHalf = builderAuxHalf.persist<int32_t, int32_t>({"x", "y"});
  int size = 0;
  for (auto it = testA.begin(); it != testA.end(); it++) {
    auto& elem = *it;
    rowWriterAux(0, elem.x(), getHash(yBins, zBins, elem.y(), elem.floatZ()));
    if (size < 5) {
      rowWriterAuxHalf(0, elem.x(), getHash(yBins, zBins, elem.y(), elem.floatZ()));
    }
    size++;
  }
  auto tableAux = builderAux.finalize();
  auto tableAuxHalf = builderAuxHalf.finalize();
  BOOST_REQUIRE_EQUAL(tableAux->num_rows(), 10);
  BOOST_REQUIRE_EQUAL(tableAuxHalf->num_rows(), 5);

  // Auxiliary table: testsAux with id and hash, hash is the category for grouping
  using TestsAux = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  TestsAux testAux{tableAux};
  TestsAux testAuxHalf{tableAuxHalf};
  BOOST_REQUIRE_EQUAL(10, testAux.size());
  BOOST_REQUIRE_EQUAL(5, testAuxHalf.size());

  // Omitting values outside bins
  TableBuilder builderAuxNoOverflows;
  auto rowWriterAux2 = builderAuxNoOverflows.persist<int32_t, int32_t>({"x", "y"});
  for (auto it = testA.begin(); it != testA.end(); it++) {
    auto& elem = *it;
    rowWriterAux2(0, elem.x(), getHash(yBins, zBins, elem.y(), elem.floatZ(), true));
  }
  auto tableAuxNoOverflows = builderAuxNoOverflows.finalize();
  BOOST_REQUIRE_EQUAL(tableAuxNoOverflows->num_rows(), 10);

  TestsAux testAuxNoOverflows{tableAuxNoOverflows};
  BOOST_REQUIRE_EQUAL(10, testAuxNoOverflows.size());

  // 2, 3, 5, 8, 9 have overflows in testA
  std::vector<std::tuple<int32_t, int32_t>> expectedFullPairsNoOverflows{
    {0, 0}, {0, 4}, {4, 0}, {4, 4}, {4, 7}, {7, 4}, {7, 7}, {1, 1}, {1, 6}, {6, 1}, {6, 6}};
  int count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockFullIndexPolicy("y", 1, -1, testAuxNoOverflows, testAuxNoOverflows))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedFullPairsNoOverflows[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedFullPairsNoOverflows[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullPairsNoOverflows.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedFullPairs{
    {0, 0}, {0, 4}, {0, 7}, {4, 0}, {7, 0}, {4, 4}, {4, 7}, {7, 4}, {7, 7}, {1, 1}, {1, 6}, {6, 1}, {6, 6}, {3, 3}, {3, 5}, {5, 3}, {5, 5}, {2, 2}, {2, 8}, {2, 9}, {8, 2}, {9, 2}, {8, 8}, {8, 9}, {9, 8}, {9, 9}};
  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockFullIndexPolicy("y", 2, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedFullPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedFullPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullPairs.size());

  std::vector<std::tuple<int32_t, int32_t, int32_t>> expectedFullTriples{
    {0, 0, 0}, {0, 0, 4}, {0, 0, 7}, {0, 4, 0}, {0, 4, 4}, {0, 4, 7}, {0, 7, 0}, {0, 7, 4}, {0, 7, 7}, {4, 0, 0}, {4, 0, 4}, {4, 0, 7}, {7, 0, 0}, {7, 0, 4}, {7, 0, 7}, {4, 4, 0}, {4, 7, 0}, {7, 4, 0}, {7, 7, 0}, {4, 4, 4}, {4, 4, 7}, {4, 7, 4}, {4, 7, 7}, {7, 4, 4}, {7, 4, 7}, {7, 7, 4}, {7, 7, 7}, {1, 1, 1}, {1, 1, 6}, {1, 6, 1}, {1, 6, 6}, {6, 1, 1}, {6, 1, 6}, {6, 6, 1}, {6, 6, 6}, {3, 3, 3}, {3, 3, 5}, {3, 5, 3}, {3, 5, 5}, {5, 3, 3}, {5, 3, 5}, {5, 5, 3}, {5, 5, 5}, {2, 2, 2}, {2, 2, 8}, {2, 2, 9}, {2, 8, 2}, {2, 8, 8}, {2, 8, 9}, {2, 9, 2}, {2, 9, 8}, {2, 9, 9}, {8, 2, 2}, {8, 2, 8}, {8, 2, 9}, {9, 2, 2}, {9, 2, 8}, {9, 2, 9}, {8, 8, 2}, {8, 9, 2}, {9, 8, 2}, {9, 9, 2}, {8, 8, 8}, {8, 8, 9}, {8, 9, 8}, {8, 9, 9}, {9, 8, 8}, {9, 8, 9}, {9, 9, 8}, {9, 9, 9}};
  count = 0;
  for (auto& [c0, c1, c2] : combinations(CombinationsBlockFullIndexPolicy("y", 2, -1, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedFullTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedFullTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedFullTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullTriples.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedUpperPairs{
    {0, 0}, {0, 4}, {0, 7}, {4, 4}, {4, 7}, {7, 7}, {1, 1}, {1, 6}, {6, 6}, {3, 3}, {3, 5}, {5, 5}, {2, 2}, {2, 8}, {2, 9}, {8, 8}, {8, 9}, {9, 9}};
  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockUpperIndexPolicy("y", 2, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperPairs.size());

  std::vector<std::tuple<int32_t, int32_t, int32_t>> expectedUpperTriples{
    {0, 0, 0}, {0, 0, 4}, {0, 4, 4}, {4, 4, 4}, {4, 4, 7}, {4, 7, 7}, {7, 7, 7}, {1, 1, 1}, {1, 1, 6}, {1, 6, 6}, {6, 6, 6}, {3, 3, 3}, {3, 3, 5}, {3, 5, 5}, {5, 5, 5}, {2, 2, 2}, {2, 2, 8}, {2, 8, 8}, {8, 8, 8}, {8, 8, 9}, {8, 9, 9}, {9, 9, 9}};
  count = 0;
  for (auto& [c0, c1, c2] : combinations(CombinationsBlockUpperIndexPolicy("y", 1, -1, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedUpperTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperTriples.size());

  std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>> expectedUpperFives{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 4}, {0, 0, 0, 0, 7}, {0, 0, 0, 4, 4}, {0, 0, 0, 4, 7}, {0, 0, 0, 7, 7}, {0, 0, 4, 4, 4}, {0, 0, 4, 4, 7}, {0, 0, 4, 7, 7}, {0, 0, 7, 7, 7}, {0, 4, 4, 4, 4}, {0, 4, 4, 4, 7}, {0, 4, 4, 7, 7}, {0, 4, 7, 7, 7}, {0, 7, 7, 7, 7}, {4, 4, 4, 4, 4}, {4, 4, 4, 4, 7}, {4, 4, 4, 7, 7}, {4, 4, 7, 7, 7}, {4, 7, 7, 7, 7}, {7, 7, 7, 7, 7}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 6}, {1, 1, 1, 6, 6}, {1, 1, 6, 6, 6}, {1, 6, 6, 6, 6}, {6, 6, 6, 6, 6}, {3, 3, 3, 3, 3}, {3, 3, 3, 3, 5}, {3, 3, 3, 5, 5}, {3, 3, 5, 5, 5}, {3, 5, 5, 5, 5}, {5, 5, 5, 5, 5}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 8}, {2, 2, 2, 2, 9}, {2, 2, 2, 8, 8}, {2, 2, 2, 8, 9}, {2, 2, 2, 9, 9}, {2, 2, 8, 8, 8}, {2, 2, 8, 8, 9}, {2, 2, 8, 9, 9}, {2, 2, 9, 9, 9}, {2, 8, 8, 8, 8}, {2, 8, 8, 8, 9}, {2, 8, 8, 9, 9}, {2, 8, 9, 9, 9}, {2, 9, 9, 9, 9}, {8, 8, 8, 8, 8}, {8, 8, 8, 8, 9}, {8, 8, 8, 9, 9}, {8, 8, 9, 9, 9}, {8, 9, 9, 9, 9}, {9, 9, 9, 9, 9}};
  count = 0;
  for (auto& [c0, c1, c2, c3, c4] : combinations(CombinationsBlockUpperIndexPolicy("y", 2, -1, testAux, testAux, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c3.x(), std::get<3>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c4.x(), std::get<4>(expectedUpperFives[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperFives.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedStrictlyUpperPairsSmaller{
    {0, 4}, {4, 7}, {1, 6}, {3, 5}, {2, 8}, {8, 9}};
  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("y", 1, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperPairsSmaller[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperPairsSmaller[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperPairsSmaller.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedStrictlyUpperPairs{
    {0, 4}, {0, 7}, {4, 7}, {1, 6}, {3, 5}, {2, 8}, {2, 9}, {8, 9}};
  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("y", 2, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperPairs.size());

  std::vector<std::tuple<int32_t, int32_t, int32_t>> expectedStrictlyUpperTriples{
    {0, 4, 7}, {2, 8, 9}};
  count = 0;
  for (auto& [c0, c1, c2] : combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("y", 2, -1, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedStrictlyUpperTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperTriples.size());

  count = 0;
  for (auto& [c0, c1, c2, c3, c4] : combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("y", 1, -1, testAux, testAux, testAux, testAux, testAux))) {
    count++;
  }
  BOOST_CHECK_EQUAL(count, 0);

  // Different tables of different size
  std::vector<std::tuple<int32_t, int32_t>> expectedFullPairsFirstSmaller{
    {0, 0}, {0, 4}, {4, 0}, {4, 4}, {4, 7}, {1, 1}, {1, 6}, {3, 3}, {3, 5}, {2, 2}, {2, 8}};
  count = 0;
  for (auto& [x0, x1] : combinations(CombinationsBlockFullIndexPolicy("y", 1, -1, testAuxHalf, testAux))) {
    BOOST_CHECK_EQUAL(x0.x(), std::get<0>(expectedFullPairsFirstSmaller[count]));
    BOOST_CHECK_EQUAL(x1.x(), std::get<1>(expectedFullPairsFirstSmaller[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullPairsFirstSmaller.size());

  count = 0;
  std::vector<std::tuple<int32_t, int32_t>> expectedFullPairsSecondSmaller{
    {0, 0}, {0, 4}, {4, 0}, {4, 4}, {7, 4}, {1, 1}, {6, 1}, {3, 3}, {5, 3}, {2, 2}, {8, 2}};
  for (auto& [x0, x1] : combinations(CombinationsBlockFullIndexPolicy("y", 1, -1, testAux, testAuxHalf))) {
    BOOST_CHECK_EQUAL(x0.x(), std::get<0>(expectedFullPairsSecondSmaller[count]));
    BOOST_CHECK_EQUAL(x1.x(), std::get<1>(expectedFullPairsSecondSmaller[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullPairsSecondSmaller.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedUpperPairsFirstSmaller{
    {0, 0}, {0, 4}, {4, 4}, {4, 7}, {1, 1}, {1, 6}, {3, 3}, {3, 5}, {2, 2}, {2, 8}};
  count = 0;
  for (auto& [x0, x1] : combinations(CombinationsBlockUpperIndexPolicy("y", 1, -1, testAuxHalf, testAux))) {
    BOOST_CHECK_EQUAL(x0.x(), std::get<0>(expectedUpperPairsFirstSmaller[count]));
    BOOST_CHECK_EQUAL(x1.x(), std::get<1>(expectedUpperPairsFirstSmaller[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperPairsFirstSmaller.size());

  count = 0;
  std::vector<std::tuple<int32_t, int32_t>> expectedUpperPairsSecondSmaller{
    {0, 0}, {0, 4}, {4, 4}, {1, 1}, {3, 3}, {2, 2}};
  for (auto& [x0, x1] : combinations(CombinationsBlockUpperIndexPolicy("y", 1, -1, testAux, testAuxHalf))) {
    BOOST_CHECK_EQUAL(x0.x(), std::get<0>(expectedUpperPairsSecondSmaller[count]));
    BOOST_CHECK_EQUAL(x1.x(), std::get<1>(expectedUpperPairsSecondSmaller[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperPairsSecondSmaller.size());

  // Using same index combinations for better performance
  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockFullSameIndexPolicy("y", 2, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedFullPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedFullPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullPairs.size());

  count = 0;
  for (auto& [c0, c1, c2] : combinations(CombinationsBlockFullSameIndexPolicy("y", 2, -1, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedFullTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedFullTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedFullTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedFullTriples.size());

  count = 0;
  for (auto& [c0, c1] : combinations(CombinationsBlockUpperSameIndexPolicy("y", 2, -1, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperPairs.size());

  count = 0;
  for (auto& [c0, c1, c2] : combinations(CombinationsBlockUpperSameIndexPolicy("y", 1, -1, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedUpperTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperTriples.size());

  count = 0;
  for (auto& [c0, c1, c2, c3, c4] : combinations(CombinationsBlockUpperSameIndexPolicy("y", 2, -1, testAux, testAux, testAux, testAux, testAux))) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c3.x(), std::get<3>(expectedUpperFives[count]));
    BOOST_CHECK_EQUAL(c4.x(), std::get<4>(expectedUpperFives[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedUpperFives.size());

  count = 0;
  for (auto& [c0, c1] : selfCombinations("y", 2, -1, testAux, testAux)) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperPairs.size());

  count = 0;
  for (auto& [c0, c1, c2] : selfCombinations("y", 2, -1, testAux, testAux, testAux)) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedStrictlyUpperTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperTriples.size());

  count = 0;
  for (auto& [c0, c1, c2, c3, c4] : selfCombinations("y", 2, -1, testAux, testAux, testAux, testAux, testAux)) {
    count++;
  }
  BOOST_CHECK_EQUAL(count, 0);
}

BOOST_AUTO_TEST_CASE(CombinationsHelpers)
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
  for (auto& [t0, t1] : pairCombinations(testsA)) {
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
  i = 0;
  j = 1;
  int k = 2;
  for (auto& [t0, t1, t2] : tripleCombinations(testsA)) {
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

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t, int32_t, float>({"x", "y", "floatZ"});
  rowWriterB(0, 0, 25, -6.0f);
  rowWriterB(0, 1, 18, 0.0f);
  rowWriterB(0, 2, 48, 8.0f);
  rowWriterB(0, 3, 103, 2.0f);
  rowWriterB(0, 4, 28, -6.0f);
  rowWriterB(0, 5, 102, 2.0f);
  rowWriterB(0, 6, 12, 0.0f);
  rowWriterB(0, 7, 24, -7.0f);
  rowWriterB(0, 8, 41, 8.0f);
  rowWriterB(0, 9, 49, 8.0f);
  auto tableB = builderB.finalize();
  BOOST_REQUIRE_EQUAL(tableB->num_rows(), 10);

  using TestB = o2::soa::Table<o2::soa::Index<>, test::X, test::Y, test::FloatZ>;
  TestB testB{tableB};
  BOOST_REQUIRE_EQUAL(10, testB.size());

  // Grouped data:
  // [3, 5] [0, 4, 7], [1, 6], [2, 8, 9]
  // Assuming bins intervals: [ , )
  std::vector<uint32_t> yBins{0, 5, 10, 20, 30, 40, 50, 101};
  std::vector<float> zBins{-7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f};

  TableBuilder builderAux;
  auto rowWriterAux = builderAux.persist<int32_t, int32_t>({"x", "y"});
  for (auto it = testB.begin(); it != testB.end(); it++) {
    auto& elem = *it;
    rowWriterAux(0, elem.x(), getHash(yBins, zBins, elem.y(), elem.floatZ()));
  }
  auto tableAux = builderAux.finalize();
  BOOST_REQUIRE_EQUAL(tableAux->num_rows(), 10);

  // Auxiliary table: testsAux with id and hash, hash is the category for grouping
  using TestsAux = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  TestsAux testAux{tableAux};
  BOOST_REQUIRE_EQUAL(10, testAux.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedStrictlyUpperPairs{
    {0, 4}, {0, 7}, {4, 7}, {1, 6}, {3, 5}, {2, 8}, {2, 9}, {8, 9}};
  count = 0;
  for (auto& [c0, c1] : selfPairCombinations("y", 2, -1, testAux)) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperPairs[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperPairs[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperPairs.size());

  std::vector<std::tuple<int32_t, int32_t, int32_t>> expectedStrictlyUpperTriples{
    {0, 4, 7}, {2, 8, 9}};
  count = 0;
  for (auto& [c0, c1, c2] : selfTripleCombinations("y", 2, -1, testAux)) {
    BOOST_CHECK_EQUAL(c0.x(), std::get<0>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c1.x(), std::get<1>(expectedStrictlyUpperTriples[count]));
    BOOST_CHECK_EQUAL(c2.x(), std::get<2>(expectedStrictlyUpperTriples[count]));
    count++;
  }
  BOOST_CHECK_EQUAL(count, expectedStrictlyUpperTriples.size());
}
