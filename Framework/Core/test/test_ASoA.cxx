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
#include "Framework/ASoAHelpers.h"
#include "Framework/Expressions.h"
#include "Framework/AnalysisHelpers.h"
#include "../src/ExpressionHelpers.h"
#include "gandiva/tree_expr_builder.h"
#include "arrow/status.h"
#include "gandiva/filter.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

DECLARE_SOA_STORE();
namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, int32_t, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, int32_t, "y");
DECLARE_SOA_COLUMN_FULL(Z, z, int32_t, "z");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](int32_t x, int32_t y) { return x + y; });
DECLARE_SOA_EXPRESSION_COLUMN(ESum, esum, int32_t, 1 * test::x + test::y);
} // namespace test

DECLARE_SOA_TABLE(Points, "TST", "POINTS", test::X, test::Y);

namespace test
{
DECLARE_SOA_COLUMN_FULL(SomeBool, someBool, bool, "someBool");
DECLARE_SOA_COLUMN_FULL(Color, color, int32_t, "color");
} // namespace test

DECLARE_SOA_TABLE(Infos, "TST", "INFOS", test::Color, test::SomeBool);

namespace test
{
DECLARE_SOA_COLUMN(N, n, int);
DECLARE_SOA_INDEX_COLUMN(Info, info);
DECLARE_SOA_INDEX_COLUMN_FULL(PointA, pointA, int, Points, "_A");
DECLARE_SOA_INDEX_COLUMN_FULL(PointB, pointB, int, Points, "_B");
DECLARE_SOA_COLUMN_FULL(Thickness, thickness, int, "thickness");
} // namespace test

DECLARE_SOA_TABLE(Segments, "TST", "SEGMENTS", test::N, test::PointAId, test::PointBId, test::InfoId);
DECLARE_SOA_TABLE(SegmentsExtras, "TST", "SEGMENTSEX", test::Thickness);

BOOST_AUTO_TEST_CASE(TestTableIteration)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 0, 1);
  rowWriter(0, 0, 2);
  rowWriter(0, 0, 3);
  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  auto table = builder.finalize();

  auto i = ColumnIterator<int32_t>(table->column(0).get());
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

  arrow::ChunkedArray* chunks[2] = {
    table->column(0).get(),
    table->column(1).get()};
  Points::iterator tests(chunks, {table->num_rows(), 0});
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

  for (auto t1 = tests2.begin(); t1 != tests2.end(); ++t1) {
    for (auto t2 = t1 + 1; t2 != tests2.end(); ++t2) {
    }
  }
}

BOOST_AUTO_TEST_CASE(TestDynamicColumns)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"x", "y"});
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
  auto rowWriter = builder.persist<int32_t, int32_t>({"x", "y"});
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
  ColumnIterator<int32_t> foo{table->column(1).get()};
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
  auto rowWriterX = builderX.persist<int32_t>({"x"});
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
  auto rowWriterY = builderY.persist<int32_t>({"y"});
  rowWriterY(0, 7);
  rowWriterY(0, 6);
  rowWriterY(0, 5);
  rowWriterY(0, 4);
  rowWriterY(0, 3);
  rowWriterY(0, 2);
  rowWriterY(0, 1);
  rowWriterY(0, 0);
  auto tableY = builderY.finalize();

  TableBuilder builderZ;
  auto rowWriterZ = builderZ.persist<int32_t>({"z"});
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  auto tableZ = builderZ.finalize();

  using TestX = o2::soa::Table<test::X>;
  using TestY = o2::soa::Table<test::Y>;
  using TestZ = o2::soa::Table<test::Z>;
  using Test = Join<TestX, TestY>;

  Test tests{0, tableX, tableY};
  for (auto& test : tests) {
    BOOST_CHECK_EQUAL(7, test.x() + test.y());
  }

  auto tests2 = join(TestX{tableX}, TestY{tableY});
  static_assert(std::is_same_v<Test::table_t, decltype(tests2)>,
                "Joined tables should have the same type, regardless how we construct them");
  for (auto& test : tests2) {
    BOOST_CHECK_EQUAL(7, test.x() + test.y());
  }

  auto tests3 = join(TestX{tableX}, TestY{tableY}, TestZ{tableZ});

  for (auto& test : tests3) {
    BOOST_CHECK_EQUAL(15, test.x() + test.y() + test.z());
  }
  using TestMoreThanTwo = Join<TestX, TestY, TestZ>;
  TestMoreThanTwo tests4{0, tableX, tableY, tableZ};
  for (auto& test : tests4) {
    BOOST_CHECK_EQUAL(15, test.x() + test.y() + test.z());
  }
}

BOOST_AUTO_TEST_CASE(TestConcatTables)
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
  rowWriterB(0, 12);
  rowWriterB(0, 13);
  rowWriterB(0, 14);
  rowWriterB(0, 15);
  auto tableB = builderB.finalize();

  TableBuilder builderC;
  auto rowWriterC = builderC.persist<int32_t>({"z"});
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
  expressions::Filter testf = (test::x == 1) || (test::x == 3);
  expressions::Selection selection;
  auto status = gandiva::SelectionVector::MakeInt32(tests.size(), arrow::default_memory_pool(), &selection);
  BOOST_REQUIRE(status.ok());

  auto fptr = tableA->schema()->GetFieldByName("x");
  BOOST_REQUIRE(fptr != nullptr);
  BOOST_REQUIRE(fptr->name() == "x");
  BOOST_REQUIRE(fptr->type()->id() == arrow::Type::INT32);

  auto node_x = gandiva::TreeExprBuilder::MakeField(fptr);
  auto literal_1 = gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(1));
  auto literal_3 = gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(3));
  auto equals_to_1 = gandiva::TreeExprBuilder::MakeFunction("equal", {node_x, literal_1}, arrow::boolean());
  auto equals_to_3 = gandiva::TreeExprBuilder::MakeFunction("equal", {node_x, literal_3}, arrow::boolean());
  auto node_or = gandiva::TreeExprBuilder::MakeOr({equals_to_1, equals_to_3});
  auto condition = gandiva::TreeExprBuilder::MakeCondition(node_or);
  BOOST_REQUIRE_EQUAL(condition->ToString(), "bool equal((int32) x, (const int32) 1) || bool equal((int32) x, (const int32) 3)");
  std::shared_ptr<gandiva::Filter> filter;
  status = gandiva::Filter::Make(tableA->schema(), condition, &filter);
  BOOST_REQUIRE_EQUAL(status.ToString(), "OK");

  arrow::TableBatchReader reader(*tableA);
  std::shared_ptr<RecordBatch> batch;
  auto s = reader.ReadNext(&batch);
  BOOST_REQUIRE(s.ok());
  BOOST_REQUIRE(batch != nullptr);
  BOOST_REQUIRE_EQUAL(batch->num_rows(), 8);
  auto st = filter->Evaluate(*batch, selection);
  BOOST_REQUIRE_EQUAL(st.ToString(), "OK");

  expressions::Selection selection_f = expressions::createSelection(tableA, testf);

  TestA testA{tableA};
  FilteredTest filtered{{testA.asArrowTable()}, selection_f};
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
  FilteredConcatTest concatTestTable{{concatTest.asArrowTable()}, selectionConcat};
  BOOST_CHECK_EQUAL(3, concatTestTable.size());

  i = 0;
  auto b = concatTestTable.begin();
  auto e = concatTestTable.end();

  BOOST_CHECK_EQUAL(b.mRowIndex, 0);
  BOOST_CHECK_EQUAL(b.getSelectionRow(), 0);
  BOOST_CHECK_EQUAL(e.index, 3);

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
  JoinedTest testJoin{0, tableA, tableC};
  FilteredJoinTest filteredJoin{{testJoin.asArrowTable()}, selectionJoin};

  i = 0;
  BOOST_CHECK(filteredJoin.begin() != filteredJoin.end());
  for (auto& f : filteredJoin) {
    BOOST_CHECK_EQUAL(i * 2, f.x());
    BOOST_CHECK_EQUAL(i * 2, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 3);
}

BOOST_AUTO_TEST_CASE(TestDereference)
{
  TableBuilder builderA;
  auto pointsWriter = builderA.cursor<Points>();
  pointsWriter(0, 0, 0);
  pointsWriter(0, 3, 4);
  auto pointsT = builderA.finalize();
  Points points{pointsT};
  BOOST_REQUIRE_EQUAL(pointsT->num_rows(), 2);

  TableBuilder builderA2;
  auto infoWriter = builderA2.cursor<Infos>();
  infoWriter(0, 0, true);
  infoWriter(0, 1, false);
  infoWriter(0, 4, true);
  auto infosT = builderA2.finalize();
  Infos infos{infosT};
  BOOST_REQUIRE_EQUAL(infos.begin().someBool(), true);
  BOOST_REQUIRE_EQUAL((infos.begin() + 1).someBool(), false);
  BOOST_REQUIRE_EQUAL((infos.begin() + 2).someBool(), true);
  BOOST_REQUIRE_EQUAL((infos.begin() + 2).color(), 4);
  BOOST_REQUIRE_EQUAL(infosT->num_rows(), 3);

  TableBuilder builderB;
  auto segmentsWriter = builderB.cursor<Segments>();
  segmentsWriter(0, 10, 0, 1, 2);
  auto segmentsT = builderB.finalize();
  Segments segments{segmentsT};
  BOOST_REQUIRE_EQUAL(segmentsT->num_rows(), 1);

  TableBuilder builderC;
  auto segmentsExtraWriter = builderC.cursor<SegmentsExtras>();
  segmentsExtraWriter(0, 1);
  auto segmentsExtraT = builderC.finalize();
  SegmentsExtras segmentsExtras{segmentsExtraT};
  BOOST_REQUIRE_EQUAL(segmentsExtraT->num_rows(), 1);

  BOOST_CHECK_EQUAL(segments.begin().pointAId(), 0);
  BOOST_CHECK_EQUAL(segments.begin().pointBId(), 1);
  static_assert(std::is_same_v<decltype(segments.begin().pointA()), Points::iterator>);
  auto i = segments.begin();
  using namespace o2::framework;
  i.bindExternalIndices(&points, &infos);
  BOOST_CHECK_EQUAL(i.n(), 10);
  BOOST_CHECK_EQUAL(i.info().color(), 4);
  BOOST_CHECK_EQUAL(i.info().someBool(), true);
  BOOST_CHECK_EQUAL(i.pointA().x(), 0);
  BOOST_CHECK_EQUAL(i.pointA().y(), 0);
  BOOST_CHECK_EQUAL(i.pointB().x(), 3);
  BOOST_CHECK_EQUAL(i.pointB().y(), 4);

  segments.bindExternalIndices(&points, &infos);
  auto j = segments.begin();
  BOOST_CHECK_EQUAL(j.n(), 10);
  BOOST_CHECK_EQUAL(j.info().color(), 4);
  BOOST_CHECK_EQUAL(j.info().someBool(), true);
  BOOST_CHECK_EQUAL(j.pointA().x(), 0);
  BOOST_CHECK_EQUAL(j.pointA().y(), 0);
  BOOST_CHECK_EQUAL(j.pointB().x(), 3);
  BOOST_CHECK_EQUAL(j.pointB().y(), 4);

  auto joined = join(segments, segmentsExtras);
  joined.bindExternalIndices(&points, &infos);
  auto se = joined.begin();
  BOOST_CHECK_EQUAL(se.n(), 10);
  BOOST_CHECK_EQUAL(se.info().color(), 4);
  BOOST_CHECK_EQUAL(se.pointA().x(), 0);
  BOOST_CHECK_EQUAL(se.pointA().y(), 0);
  BOOST_CHECK_EQUAL(se.pointB().x(), 3);
  BOOST_CHECK_EQUAL(se.pointB().y(), 4);
  BOOST_CHECK_EQUAL(se.thickness(), 1);
}

BOOST_AUTO_TEST_CASE(TestSchemaCreation)
{
  auto schema = createSchemaFromColumns(Points::persistent_columns_t{});
  BOOST_CHECK_EQUAL(schema->num_fields(), 2);
  BOOST_CHECK_EQUAL(schema->field(0)->name(), "x");
  BOOST_CHECK_EQUAL(schema->field(1)->name(), "y");
}

BOOST_AUTO_TEST_CASE(TestFilteredOperators)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 8);
  rowWriterA(0, 1, 9);
  rowWriterA(0, 2, 10);
  rowWriterA(0, 3, 11);
  rowWriterA(0, 4, 12);
  rowWriterA(0, 5, 13);
  rowWriterA(0, 6, 14);
  rowWriterA(0, 7, 15);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using FilteredTest = Filtered<TestA>;
  using NestedFilteredTest = Filtered<Filtered<TestA>>;
  using namespace o2::framework;

  expressions::Filter f1 = test::x < 4;
  expressions::Filter f2 = test::y > 13;

  TestA testA{tableA};
  FilteredTest filtered1{{testA.asArrowTable()}, expressions::createSelection(testA.asArrowTable(), f1)};
  BOOST_CHECK_EQUAL(4, filtered1.size());
  BOOST_CHECK(filtered1.begin() != filtered1.end());

  FilteredTest filtered2{{testA.asArrowTable()}, expressions::createSelection(testA.asArrowTable(), f2)};
  BOOST_CHECK_EQUAL(2, filtered2.size());
  BOOST_CHECK(filtered2.begin() != filtered2.end());

  FilteredTest filteredUnion = filtered1 + filtered2;
  BOOST_CHECK_EQUAL(6, filteredUnion.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedUnion{
    {0, 8}, {1, 9}, {2, 10}, {3, 11}, {6, 14}, {7, 15}};
  auto i = 0;
  for (auto& f : filteredUnion) {
    BOOST_CHECK_EQUAL(std::get<0>(expectedUnion[i]), f.x());
    BOOST_CHECK_EQUAL(std::get<1>(expectedUnion[i]), f.y());
    BOOST_CHECK_EQUAL(std::get<0>(expectedUnion[i]), f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 6);

  FilteredTest filteredIntersection = filtered1 * filtered2;
  BOOST_CHECK_EQUAL(0, filteredIntersection.size());

  i = 0;
  for (auto& f : filteredIntersection) {
    i++;
  }
  BOOST_CHECK_EQUAL(i, 0);

  expressions::Filter f3 = test::x < 3;
  FilteredTest filtered3{{testA.asArrowTable()}, expressions::createSelection(testA.asArrowTable(), f3)};
  BOOST_CHECK_EQUAL(3, filtered3.size());
  BOOST_CHECK(filtered3.begin() != filtered3.end());

  FilteredTest unionIntersection = (filtered1 + filtered2) * filtered3;
  BOOST_CHECK_EQUAL(3, unionIntersection.size());

  i = 0;
  for (auto& f : unionIntersection) {
    BOOST_CHECK_EQUAL(i, f.x());
    BOOST_CHECK_EQUAL(i + 8, f.y());
    BOOST_CHECK_EQUAL(i, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 3);
}

BOOST_AUTO_TEST_CASE(TestNestedFiltering)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"x", "y"});
  rowWriterA(0, 0, 8);
  rowWriterA(0, 1, 9);
  rowWriterA(0, 2, 10);
  rowWriterA(0, 3, 11);
  rowWriterA(0, 4, 12);
  rowWriterA(0, 5, 13);
  rowWriterA(0, 6, 14);
  rowWriterA(0, 7, 15);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using FilteredTest = Filtered<TestA>;
  using NestedFilteredTest = Filtered<Filtered<TestA>>;
  using TripleNestedFilteredTest = Filtered<Filtered<Filtered<TestA>>>;
  using namespace o2::framework;

  expressions::Filter f1 = test::x < 4;
  expressions::Filter f2 = test::y > 9;
  expressions::Filter f3 = test::x < 3;

  TestA testA{tableA};
  FilteredTest filtered{{testA.asArrowTable()}, expressions::createSelection(testA.asArrowTable(), f1)};
  BOOST_CHECK_EQUAL(4, filtered.size());
  BOOST_CHECK(filtered.begin() != filtered.end());

  NestedFilteredTest nestedFiltered{{filtered}, expressions::createSelection(filtered.asArrowTable(), f2)};
  BOOST_CHECK_EQUAL(2, nestedFiltered.size());
  auto i = 0;
  for (auto& f : nestedFiltered) {
    BOOST_CHECK_EQUAL(i + 2, f.x());
    BOOST_CHECK_EQUAL(i + 10, f.y());
    BOOST_CHECK_EQUAL(i + 2, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 2);

  TripleNestedFilteredTest tripleFiltered{{nestedFiltered}, expressions::createSelection(nestedFiltered.asArrowTable(), f3)};
  BOOST_CHECK_EQUAL(1, tripleFiltered.size());
  i = 0;
  for (auto& f : tripleFiltered) {
    BOOST_CHECK_EQUAL(i + 2, f.x());
    BOOST_CHECK_EQUAL(i + 10, f.y());
    BOOST_CHECK_EQUAL(i + 2, f.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 1);
}

BOOST_AUTO_TEST_CASE(TestEmptyTables)
{
  TableBuilder bPoints;
  auto pwriter = bPoints.cursor<Points>();
  auto pempty = bPoints.finalize();

  TableBuilder bInfos;
  auto iwriter = bInfos.cursor<Infos>();
  auto iempty = bInfos.finalize();

  Points p{pempty};
  Infos i{iempty};

  using PI = Join<Points, Infos>;
  PI pi{0, pempty, iempty};
  BOOST_CHECK_EQUAL(pi.size(), 0);
  auto spawned = Extend<Points, test::ESum>(p);
  BOOST_CHECK_EQUAL(spawned.size(), 0);
}
