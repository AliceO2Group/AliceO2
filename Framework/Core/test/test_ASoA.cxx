// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ASoA
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ASoA.h"
#include "Framework/TableBuilder.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

namespace test
{
DECLARE_SOA_COLUMN(X, x, uint64_t, "x");
DECLARE_SOA_COLUMN(Y, y, uint64_t, "y");
DECLARE_SOA_COLUMN(Z, z, uint64_t, "z");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](uint64_t x, uint64_t y) { return x + y; });
} // namespace test

BOOST_AUTO_TEST_CASE(TestTableIteration)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<uint64_t, uint64_t>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 0, 1);
  rowWriter(0, 0, 2);
  rowWriter(0, 0, 3);
  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  auto table = builder.finalize();

  auto i = ColumnIterator<uint64_t>(table->column(0).get());
  int64_t pos = 0;
  i.mCurrentPos = &pos;
  BOOST_CHECK_EQUAL(*i, 0);
  pos++;
  BOOST_CHECK_EQUAL(*i, 0);
  pos++;
  BOOST_CHECK_EQUAL(*i, 0);
  pos++;
  BOOST_CHECK_EQUAL(*i, 0);
  pos++;
  BOOST_CHECK_EQUAL(*i, 1);
  pos++;
  BOOST_CHECK_EQUAL(*i, 1);
  pos++;
  BOOST_CHECK_EQUAL(*i, 1);
  pos++;
  BOOST_CHECK_EQUAL(*i, 1);

  auto rowIndex = std::make_tuple(
    std::pair<test::X*, arrow::Column*>{nullptr, table->column(0).get()},
    std::pair<test::Y*, arrow::Column*>{nullptr, table->column(1).get()});
  RowView<test::X, test::Y> tests(rowIndex, table->num_rows());
  BOOST_CHECK_EQUAL(tests.x(), 0);
  BOOST_CHECK_EQUAL(tests.y(), 0);
  ++tests;
  BOOST_CHECK_EQUAL(tests.x(), 0);
  BOOST_CHECK_EQUAL(tests.y(), 1);
  using Test = o2::soa::Table<test::X, test::Y>;
  Test tests2{table};
  size_t value = 0;
  auto b = tests2.begin();
  auto e = tests2.end();
  BOOST_CHECK(b != e);
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  BOOST_CHECK(b == e);

  b = tests2.begin();
  e = tests2.end();
  BOOST_CHECK(b != e);
  BOOST_CHECK((b + 1) == (b + 1));
  BOOST_CHECK((b + 7) != b);
  BOOST_CHECK((b + 7) != e);
  BOOST_CHECK((b + 8) == e);

  for (auto& t : tests2) {
    BOOST_CHECK_EQUAL(t.x(), value / 4);
    BOOST_CHECK_EQUAL(t.y(), value);
    BOOST_REQUIRE(value < 8);
    value++;
  }

  for (auto t1 = tests2.begin(); t1 != tests2.end() - 1; ++t1) {
    for (auto t2 = t1 + 1; t2 != tests2.end(); ++t2) {
    }
  }
}

BOOST_AUTO_TEST_CASE(TestDynamicColumns)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<uint64_t, uint64_t>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 0, 1);
  rowWriter(0, 0, 2);
  rowWriter(0, 0, 3);
  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X, test::Y, test::Sum<test::X, test::Y>>;

  Test tests{table};
  for (auto& test : tests) {
    BOOST_CHECK_EQUAL(test.sum(), test.x() + test.y());
  }

  using Test2 = o2::soa::Table<test::X, test::Y, test::Sum<test::Y, test::Y>>;

  Test2 tests2{table};
  for (auto& test : tests2) {
    BOOST_CHECK_EQUAL(test.sum(), test.y() + test.y());
  }
}

BOOST_AUTO_TEST_CASE(TestColumnIterators)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<uint64_t, uint64_t>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 0, 1);
  rowWriter(0, 0, 2);
  rowWriter(0, 0, 3);
  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  auto table = builder.finalize();

  int64_t index1 = 0;
  int64_t index2 = 0;
  ColumnIterator<uint64_t> foo{table->column(1).get()};
  foo.mCurrentPos = &index1;
  auto bar{foo};
  bar.mCurrentPos = &index2;
  BOOST_REQUIRE_EQUAL(foo.mCurrent, bar.mCurrent);
  BOOST_REQUIRE_EQUAL(foo.mLast, bar.mLast);
  BOOST_REQUIRE_EQUAL(foo.mColumn, bar.mColumn);
  BOOST_REQUIRE_EQUAL(foo.mFirstIndex, bar.mFirstIndex);
  BOOST_REQUIRE_EQUAL(foo.mCurrentChunk, bar.mCurrentChunk);

  auto foobar = std::move(foo);
  BOOST_REQUIRE_EQUAL(foobar.mCurrent, bar.mCurrent);
  BOOST_REQUIRE_EQUAL(foobar.mLast, bar.mLast);
  BOOST_REQUIRE_EQUAL(foobar.mColumn, bar.mColumn);
  BOOST_REQUIRE_EQUAL(foobar.mFirstIndex, bar.mFirstIndex);
  BOOST_REQUIRE_EQUAL(foobar.mCurrentChunk, bar.mCurrentChunk);
}

BOOST_AUTO_TEST_CASE(TestJoinedTables)
{
  TableBuilder builderX;
  auto rowWriterX = builderX.persist<uint64_t>({"x"});
  rowWriterX(0, 0);
  rowWriterX(0, 1);
  rowWriterX(0, 2);
  rowWriterX(0, 3);
  rowWriterX(0, 4);
  rowWriterX(0, 5);
  rowWriterX(0, 6);
  rowWriterX(0, 7);
  auto tableX = builderX.finalize();

  TableBuilder builderY;
  auto rowWriterY = builderY.persist<uint64_t>({"y"});
  rowWriterY(0, 7);
  rowWriterY(0, 6);
  rowWriterY(0, 5);
  rowWriterY(0, 4);
  rowWriterY(0, 3);
  rowWriterY(0, 2);
  rowWriterY(0, 1);
  rowWriterY(0, 0);
  auto tableY = builderY.finalize();

  using TestX = o2::soa::Table<test::X>;
  using TestY = o2::soa::Table<test::Y>;
  using Test = Join<TestX, TestY>;

  Test tests{tableX, tableY};
  for (auto& test : tests) {
    BOOST_CHECK_EQUAL(7, test.x() + test.y());
  }

  auto tests2 = join(TestX{tableX}, TestY{tableY});
  static_assert(std::is_same_v<Test::table_t, decltype(tests2)>,
                "Joined tables should have the same type, regardless how we construct them");
  for (auto& test : tests2) {
    BOOST_CHECK_EQUAL(7, test.x() + test.y());
  }
}

BOOST_AUTO_TEST_CASE(TestConcatTables)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<uint64_t, uint64_t>({"x", "y"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  rowWriterA(0, 4, 0);
  rowWriterA(0, 5, 0);
  rowWriterA(0, 6, 0);
  rowWriterA(0, 7, 0);
  auto tableA = builderA.finalize();

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<uint64_t>({"x"});
  rowWriterB(0, 8);
  rowWriterB(0, 9);
  rowWriterB(0, 10);
  rowWriterB(0, 11);
  rowWriterB(0, 12);
  rowWriterB(0, 13);
  rowWriterB(0, 14);
  rowWriterB(0, 15);
  auto tableB = builderB.finalize();

  TableBuilder builderC;
  auto rowWriterC = builderC.persist<uint64_t>({"z"});
  rowWriterC(0, 8);
  rowWriterC(0, 9);
  rowWriterC(0, 10);
  rowWriterC(0, 11);
  rowWriterC(0, 12);
  rowWriterC(0, 13);
  rowWriterC(0, 14);
  rowWriterC(0, 15);
  auto tableC = builderC.finalize();

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using TestC = o2::soa::Table<test::Z>;
  using ConcatTest = Concat<TestA, TestB>;
  using JoinedTest = Join<TestA, TestC>;

  static_assert(std::is_same_v<ConcatTest::table_t, o2::soa::Table<o2::soa::Index<>, test::X>>, "Bad intersection of columns");
  ConcatTest tests{tableA, tableB};
  BOOST_REQUIRE_EQUAL(16, tests.size());
  for (auto& test : tests) {
    BOOST_CHECK_EQUAL(test.index(), test.x());
  }

  // Hardcode a selection for the first 5 odd numbers
  using FilteredTest = Filtered<TestA>;
  using namespace o2::framework;
  expressions::Selection selection;
  auto status = gandiva::SelectionVector::MakeInt32(tests.size(), arrow::default_memory_pool(), &selection);
  BOOST_CHECK_EQUAL(status.ok(), true);
  selection->SetIndex(0, 1);
  selection->SetIndex(1, 3);
  selection->SetNumSlots(2);
  TestA testA{tableA};
  FilteredTest filtered{testA.asArrowTable(), selection};
  BOOST_CHECK_EQUAL(2, filtered.size());

  auto i = 0;
  BOOST_CHECK(filtered.begin() != filtered.end());
  for (auto& f : filtered) {
    BOOST_CHECK_EQUAL(i * 2 + 1, f.x());
    BOOST_CHECK_EQUAL(i * 2 + 1, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 2);

  // Hardcode a selection for the first 5 odd numbers
  using FilteredConcatTest = Filtered<ConcatTest::table_t>;
  using namespace o2::framework;
  expressions::Selection selectionConcat;
  status = gandiva::SelectionVector::MakeInt32(tests.size(), arrow::default_memory_pool(), &selectionConcat);
  BOOST_CHECK_EQUAL(status.ok(), true);
  selectionConcat->SetIndex(0, 0);
  selectionConcat->SetIndex(1, 5);
  selectionConcat->SetIndex(2, 10);
  selectionConcat->SetNumSlots(3);
  ConcatTest concatTest{tableA, tableB};
  FilteredConcatTest concatTestTable{concatTest.asArrowTable(), selectionConcat};
  BOOST_CHECK_EQUAL(3, concatTestTable.size());

  i = 0;
  auto b = concatTestTable.begin();
  auto e = concatTestTable.end();

  BOOST_CHECK_EQUAL(b.mRowIndex, 0);
  BOOST_CHECK_EQUAL(e.mRowIndex, -1);
  BOOST_CHECK_EQUAL(b.getSelectionRow(), 0);
  BOOST_CHECK_EQUAL(e.getSelectionRow(), 3);

  BOOST_CHECK(concatTestTable.begin() != concatTestTable.end());
  for (auto& f : concatTestTable) {
    BOOST_CHECK_EQUAL(i * 5, f.x());
    BOOST_CHECK_EQUAL(i * 5, f.index());
    BOOST_CHECK_EQUAL(i, f.filteredIndex());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 3);

  // Test with a Joined table
  using FilteredJoinTest = Filtered<JoinedTest::table_t>;
  expressions::Selection selectionJoin;
  status = gandiva::SelectionVector::MakeInt32(tests.size(), arrow::default_memory_pool(), &selectionJoin);
  BOOST_CHECK_EQUAL(status.ok(), true);
  selectionJoin->SetIndex(0, 0);
  selectionJoin->SetIndex(1, 2);
  selectionJoin->SetIndex(2, 4);
  selectionJoin->SetNumSlots(3);
  JoinedTest testJoin{tableA, tableC};
  FilteredJoinTest filteredJoin{testJoin.asArrowTable(), selectionJoin};

  i = 0;
  BOOST_CHECK(filteredJoin.begin() != filteredJoin.end());
  for (auto& f : filteredJoin) {
    BOOST_CHECK_EQUAL(i * 2, f.x());
    BOOST_CHECK_EQUAL(i * 2, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 3);
}
