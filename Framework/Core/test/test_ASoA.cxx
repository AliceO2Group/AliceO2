// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ASoA.h"
#include "Framework/Expressions.h"
#include "Framework/AnalysisHelpers.h"
#include "gandiva/tree_expr_builder.h"
#include "arrow/status.h"
#include "gandiva/filter.h"
#include <catch_amalgamated.hpp>
#include <arrow/util/key_value_metadata.h>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(X, x, int);
DECLARE_SOA_COLUMN(Y, y, int);
DECLARE_SOA_COLUMN(Z, z, int);
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](int x, int y) { return x + y; });
DECLARE_SOA_EXPRESSION_COLUMN(ESum, esum, int, test::x + test::y);
} // namespace test

DECLARE_SOA_TABLE(Points, "TEST", "POINTS", test::X, test::Y);
DECLARE_SOA_TABLE(Points3Ds, "TEST", "PTS3D", o2::soa::Index<>, test::X, test::Y, test::Z);

DECLARE_SOA_TABLE(Points3DsMk1, "TEST", "PTS3D_1", o2::soa::Index<>, o2::soa::Marker<1>, test::X, test::Y, test::Z);
DECLARE_SOA_TABLE(Points3DsMk2, "TEST", "PTS3D_2", o2::soa::Index<>, o2::soa::Marker<2>, test::X, test::Y, test::Z);
DECLARE_SOA_TABLE(Points3DsMk3, "TEST", "PTS3D_3", o2::soa::Index<>, o2::soa::Marker<3>, test::X, test::Y, test::Z);

namespace test
{
DECLARE_SOA_COLUMN_FULL(SomeBool, someBool, bool, "someBool");
DECLARE_SOA_COLUMN_FULL(Color, color, int32_t, "color");
} // namespace test

DECLARE_SOA_TABLE(Infos, "TEST", "INFOS", test::Color, test::SomeBool);

namespace test
{
DECLARE_SOA_COLUMN(N, n, int);
DECLARE_SOA_INDEX_COLUMN(Info, info);
DECLARE_SOA_INDEX_COLUMN_FULL(PointA, pointA, int, Points, "_A");
DECLARE_SOA_INDEX_COLUMN_FULL(PointB, pointB, int, Points, "_B");
DECLARE_SOA_COLUMN_FULL(Thickness, thickness, int, "thickness");
} // namespace test

DECLARE_SOA_TABLE(Segments, "TEST", "SEGMENTS", test::N, test::PointAId, test::PointBId, test::InfoId);
DECLARE_SOA_TABLE(SegmentsExtras, "TEST", "SEGMENTSEX", test::Thickness);

namespace test
{
DECLARE_SOA_COLUMN(L1, l1, std::vector<float>);
DECLARE_SOA_COLUMN(L2, l2, std::vector<int>);
} // namespace test

DECLARE_SOA_TABLE(Lists, "TEST", "LISTS", o2::soa::Index<>, test::L1, test::L2);
} // namespace o2::aod

TEST_CASE("TestMarkers")
{
  TableBuilder b1;
  auto pwriter = b1.cursor<o2::aod::Points3Ds>();
  for (auto i = 0; i < 20; ++i) {
    pwriter(0, -1 * i, (int)(i / 2), 2 * i);
  }
  auto t1 = b1.finalize();

  auto pt = o2::aod::Points3Ds{t1};
  auto pt1 = o2::aod::Points3DsMk1{t1};
  auto pt2 = o2::aod::Points3DsMk2{t1};
  auto pt3 = o2::aod::Points3DsMk3{t1};
  REQUIRE(pt1.begin().mark() == (size_t)1);
  REQUIRE(pt2.begin().mark() == (size_t)2);
  REQUIRE(pt3.begin().mark() == (size_t)3);
}

TEST_CASE("TestTableIteration")
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"fX", "fY"});
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
  REQUIRE(*i == 0);
  pos++;
  REQUIRE(*i == 0);
  pos++;
  REQUIRE(*i == 0);
  pos++;
  REQUIRE(*i == 0);
  pos++;
  REQUIRE(*i == 1);
  pos++;
  REQUIRE(*i == 1);
  pos++;
  REQUIRE(*i == 1);
  pos++;
  REQUIRE(*i == 1);

  arrow::ChunkedArray* chunks[2] = {
    table->column(0).get(),
    table->column(1).get()};
  o2::aod::Points::iterator tests(chunks, {table->num_rows(), 0});
  REQUIRE(tests.x() == 0);
  REQUIRE(tests.y() == 0);
  ++tests;
  REQUIRE(tests.x() == 0);
  REQUIRE(tests.y() == 1);
  using Test = o2::soa::Table<o2::aod::test::X, o2::aod::test::Y>;
  Test tests2{table};
  size_t value = 0;
  auto b = tests2.begin();
  auto e = tests2.end();
  REQUIRE(b != e);
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  ++b;
  REQUIRE(b == e);

  b = tests2.begin();
  REQUIRE(b != e);
  REQUIRE(((b + 1) == (b + 1)));
  REQUIRE(((b + 7) != b));
  REQUIRE(((b + 7) != e));
  REQUIRE(((b + 8) == e));

  for (auto& t : tests2) {
    REQUIRE(t.x() == value / 4);
    REQUIRE(t.y() == value);
    REQUIRE(value < 8);
    value++;
  }

  for (auto t1 = tests2.begin(); t1 != tests2.end(); ++t1) {
    for (auto t2 = t1 + 1; t2 != tests2.end(); ++t2) {
    }
  }
}

TEST_CASE("TestDynamicColumns")
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"fX", "fY"});
  rowWriter(0, 0, 0);
  rowWriter(0, 0, 1);
  rowWriter(0, 0, 2);
  rowWriter(0, 0, 3);
  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  auto table = builder.finalize();

  using Test = o2::soa::Table<o2::aod::test::X, o2::aod::test::Y, o2::aod::test::Sum<o2::aod::test::X, o2::aod::test::Y>>;

  Test tests{table};
  for (auto& test : tests) {
    REQUIRE(test.sum() == test.x() + test.y());
  }

  using Test2 = o2::soa::Table<o2::aod::test::X, o2::aod::test::Y, o2::aod::test::Sum<o2::aod::test::Y, o2::aod::test::Y>>;

  Test2 tests2{table};
  for (auto& test : tests2) {
    REQUIRE(test.sum() == test.y() + test.y());
  }
}

TEST_CASE("TestColumnIterators")
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"fX", "fY"});
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
  REQUIRE(foo.mCurrent == bar.mCurrent);
  REQUIRE(foo.mLast == bar.mLast);
  REQUIRE(foo.mColumn == bar.mColumn);
  REQUIRE(foo.mFirstIndex == bar.mFirstIndex);
  REQUIRE(foo.mCurrentChunk == bar.mCurrentChunk);

  auto foobar = std::move(foo);
  REQUIRE(foobar.mCurrent == bar.mCurrent);
  REQUIRE(foobar.mLast == bar.mLast);
  REQUIRE(foobar.mColumn == bar.mColumn);
  REQUIRE(foobar.mFirstIndex == bar.mFirstIndex);
  REQUIRE(foobar.mCurrentChunk == bar.mCurrentChunk);
}

TEST_CASE("TestJoinedTables")
{
  TableBuilder builderX;
  auto rowWriterX = builderX.persist<int32_t>({"fX"});
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
  auto rowWriterY = builderY.persist<int32_t>({"fY"});
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
  auto rowWriterZ = builderZ.persist<int32_t>({"fZ"});
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  rowWriterZ(0, 8);
  auto tableZ = builderZ.finalize();

  using TestX = o2::soa::Table<o2::aod::test::X>;
  using TestY = o2::soa::Table<o2::aod::test::Y>;
  using TestZ = o2::soa::Table<o2::aod::test::Z>;
  using Test = Join<TestX, TestY>;

  REQUIRE(Test::contains<TestX>());
  REQUIRE(Test::contains<TestY>());
  REQUIRE(!Test::contains<TestZ>());

  Test tests{0, tableX, tableY};

  REQUIRE(tests.contains<TestX>());
  REQUIRE(tests.contains<TestY>());
  REQUIRE(!tests.contains<TestZ>());

  for (auto& test : tests) {
    REQUIRE(7 == test.x() + test.y());
  }

  auto tests2 = join(TestX{tableX}, TestY{tableY});
  static_assert(std::is_same_v<Test::table_t, decltype(tests2)>,
                "Joined tables should have the same type, regardless how we construct them");
  for (auto& test : tests2) {
    REQUIRE(7 == test.x() + test.y());
  }

  auto tests3 = join(TestX{tableX}, TestY{tableY}, TestZ{tableZ});

  for (auto& test : tests3) {
    REQUIRE(15 == test.x() + test.y() + test.z());
  }
  using TestMoreThanTwo = Join<TestX, TestY, TestZ>;
  TestMoreThanTwo tests4{0, tableX, tableY, tableZ};
  for (auto& test : tests4) {
    REQUIRE(15 == test.x() + test.y() + test.z());
  }
}

TEST_CASE("TestConcatTables")
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"fX", "fY"});
  rowWriterA(0, 0, 0);
  rowWriterA(0, 1, 0);
  rowWriterA(0, 2, 0);
  rowWriterA(0, 3, 0);
  rowWriterA(0, 4, 0);
  rowWriterA(0, 5, 0);
  rowWriterA(0, 6, 0);
  rowWriterA(0, 7, 0);
  auto tableA = builderA.finalize();
  REQUIRE(tableA->num_rows() == 8);

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"fX"});
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
  auto rowWriterC = builderC.persist<int32_t>({"fZ"});
  rowWriterC(0, 8);
  rowWriterC(0, 9);
  rowWriterC(0, 10);
  rowWriterC(0, 11);
  rowWriterC(0, 12);
  rowWriterC(0, 13);
  rowWriterC(0, 14);
  rowWriterC(0, 15);
  auto tableC = builderC.finalize();

  TableBuilder builderD;
  auto rowWriterD = builderD.persist<int32_t, int32_t>({"fX", "fZ"});
  rowWriterD(0, 16, 8);
  rowWriterD(0, 17, 9);
  rowWriterD(0, 18, 10);
  rowWriterD(0, 19, 11);
  rowWriterD(0, 20, 12);
  rowWriterD(0, 21, 13);
  rowWriterD(0, 22, 14);
  rowWriterD(0, 23, 15);
  auto tableD = builderD.finalize();

  using TestA = o2::soa::Table<o2::soa::Index<>, o2::aod::test::X, o2::aod::test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, o2::aod::test::X>;
  using TestC = o2::soa::Table<o2::aod::test::Z>;
  using TestD = o2::soa::Table<o2::aod::test::X, o2::aod::test::Z>;
  using ConcatTest = Concat<TestA, TestB>;
  using JoinedTest = Join<TestA, TestC>;
  using NestedJoinTest = Join<JoinedTest, TestD>;
  using NestedConcatTest = Concat<Join<TestA, TestB>, TestD>;

  static_assert(std::is_same_v<NestedJoinTest::table_t, o2::soa::Table<o2::soa::Index<>, o2::aod::test::Y, o2::aod::test::X, o2::aod::test::Z>>, "Bad nested join");

  static_assert(std::is_same_v<ConcatTest::table_t, o2::soa::Table<o2::soa::Index<>, o2::aod::test::X>>, "Bad intersection of columns");
  ConcatTest tests{tableA, tableB};
  REQUIRE(16 == tests.size());
  for (auto& test : tests) {
    REQUIRE(test.index() == test.x());
  }

  static_assert(std::is_same_v<NestedConcatTest::table_t, o2::soa::Table<o2::aod::test::X>>, "Bad nested concat");

  // Hardcode a selection for the first 5 odd numbers
  using FilteredTest = Filtered<TestA>;
  using namespace o2::framework;
  expressions::Filter testf = (o2::aod::test::x == 1) || (o2::aod::test::x == 3);
  gandiva::Selection selection;
  auto status = gandiva::SelectionVector::MakeInt64(tests.size(), arrow::default_memory_pool(), &selection);
  REQUIRE(status.ok());

  auto fptr = tableA->schema()->GetFieldByName("fX");
  REQUIRE(fptr != nullptr);
  REQUIRE(fptr->name() == "fX");
  REQUIRE(fptr->type()->id() == arrow::Type::INT32);

  auto node_x = gandiva::TreeExprBuilder::MakeField(fptr);
  auto literal_1 = gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(1));
  auto literal_3 = gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(3));
  auto equals_to_1 = gandiva::TreeExprBuilder::MakeFunction("equal", {node_x, literal_1}, arrow::boolean());
  auto equals_to_3 = gandiva::TreeExprBuilder::MakeFunction("equal", {node_x, literal_3}, arrow::boolean());
  auto node_or = gandiva::TreeExprBuilder::MakeOr({equals_to_1, equals_to_3});
  auto condition = gandiva::TreeExprBuilder::MakeCondition(node_or);
  REQUIRE(condition->ToString() == "bool equal((int32) fX, (const int32) 1) || bool equal((int32) fX, (const int32) 3)");
  std::shared_ptr<gandiva::Filter> filter;
  status = gandiva::Filter::Make(tableA->schema(), condition, &filter);
  REQUIRE(status.ToString() == "OK");

  arrow::TableBatchReader reader(*tableA);
  std::shared_ptr<RecordBatch> batch;
  auto s = reader.ReadNext(&batch);
  REQUIRE(s.ok());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->num_rows() == 8);
  auto st = filter->Evaluate(*batch, selection);
  REQUIRE(st.ToString() == "OK");

  gandiva::Selection selection_f = expressions::createSelection(tableA, testf);

  TestA testA{tableA};
  FilteredTest filtered{{testA.asArrowTable()}, selection_f};
  REQUIRE(2 == filtered.size());

  auto i = 0;
  REQUIRE(filtered.begin() != filtered.end());
  for (auto& f : filtered) {
    REQUIRE(i * 2 + 1 == f.x());
    REQUIRE(i * 2 + 1 == f.index());
    i++;
  }
  REQUIRE(i == 2);

  // Hardcode a selection for the first 5 odd numbers
  using FilteredConcatTest = Filtered<ConcatTest::table_t>;
  using namespace o2::framework;
  gandiva::Selection selectionConcat;
  status = gandiva::SelectionVector::MakeInt64(tests.size(), arrow::default_memory_pool(), &selectionConcat);
  REQUIRE(status.ok() == true);
  selectionConcat->SetIndex(0, 0);
  selectionConcat->SetIndex(1, 5);
  selectionConcat->SetIndex(2, 10);
  selectionConcat->SetNumSlots(3);
  ConcatTest concatTest{tableA, tableB};
  FilteredConcatTest concatTestTable{{concatTest.asArrowTable()}, selectionConcat};
  REQUIRE(3 == concatTestTable.size());

  i = 0;
  auto b = concatTestTable.begin();
  auto e = concatTestTable.end();

  REQUIRE(b.mRowIndex == 0);
  REQUIRE(b.getSelectionRow() == 0);
  REQUIRE(e.index == 3);

  REQUIRE(concatTestTable.begin() != concatTestTable.end());
  for (auto& f : concatTestTable) {
    REQUIRE(i * 5 == f.x());
    REQUIRE(i * 5 == f.index());
    REQUIRE(i == f.filteredIndex());
    i++;
  }
  REQUIRE(i == 3);

  // Test with a Joined table
  using FilteredJoinTest = Filtered<JoinedTest::table_t>;
  gandiva::Selection selectionJoin;
  status = gandiva::SelectionVector::MakeInt64(tests.size(), arrow::default_memory_pool(), &selectionJoin);
  REQUIRE(status.ok() == true);
  selectionJoin->SetIndex(0, 0);
  selectionJoin->SetIndex(1, 2);
  selectionJoin->SetIndex(2, 4);
  selectionJoin->SetNumSlots(3);
  JoinedTest testJoin{0, tableA, tableC};
  FilteredJoinTest filteredJoin{{testJoin.asArrowTable()}, selectionJoin};

  i = 0;
  REQUIRE(filteredJoin.begin() != filteredJoin.end());
  for (auto& f : filteredJoin) {
    REQUIRE(i * 2 == f.x());
    REQUIRE(i * 2 == f.index());
    i++;
  }
  REQUIRE(i == 3);
}

TEST_CASE("TestDereference")
{
  TableBuilder builderA;
  auto pointsWriter = builderA.cursor<o2::aod::Points>();
  pointsWriter(0, 0, 0);
  pointsWriter(0, 3, 4);
  auto pointsT = builderA.finalize();
  o2::aod::Points points{pointsT};
  REQUIRE(pointsT->num_rows() == 2);

  TableBuilder builderA2;
  auto infoWriter = builderA2.cursor<o2::aod::Infos>();
  infoWriter(0, 0, true);
  infoWriter(0, 1, false);
  infoWriter(0, 4, true);
  auto infosT = builderA2.finalize();
  o2::aod::Infos infos{infosT};
  REQUIRE(infos.begin().someBool() == true);
  REQUIRE((infos.begin() + 1).someBool() == false);
  REQUIRE((infos.begin() + 2).someBool() == true);
  REQUIRE((infos.begin() + 2).color() == 4);
  REQUIRE(infosT->num_rows() == 3);

  TableBuilder builderB;
  auto segmentsWriter = builderB.cursor<o2::aod::Segments>();
  segmentsWriter(0, 10, 0, 1, 2);
  auto segmentsT = builderB.finalize();
  o2::aod::Segments segments{segmentsT};
  REQUIRE(segmentsT->num_rows() == 1);

  TableBuilder builderC;
  auto segmentsExtraWriter = builderC.cursor<o2::aod::SegmentsExtras>();
  segmentsExtraWriter(0, 1);
  auto segmentsExtraT = builderC.finalize();
  o2::aod::SegmentsExtras segmentsExtras{segmentsExtraT};
  REQUIRE(segmentsExtraT->num_rows() == 1);

  REQUIRE(segments.begin().pointAId() == 0);
  REQUIRE(segments.begin().pointBId() == 1);
  static_assert(std::is_same_v<decltype(segments.begin().pointA()), o2::aod::Points::iterator>);
  auto i = segments.begin();
  using namespace o2::framework;
  i.bindExternalIndices(&points, &infos);
  REQUIRE(i.n() == 10);
  REQUIRE(i.info().color() == 4);
  REQUIRE(i.info().someBool() == true);
  REQUIRE(i.pointA().x() == 0);
  REQUIRE(i.pointA().y() == 0);
  REQUIRE(i.pointB().x() == 3);
  REQUIRE(i.pointB().y() == 4);

  segments.bindExternalIndices(&points, &infos);
  auto j = segments.begin();
  REQUIRE(j.n() == 10);
  REQUIRE(j.info().color() == 4);
  REQUIRE(j.info().someBool() == true);
  REQUIRE(j.pointA().x() == 0);
  REQUIRE(j.pointA().y() == 0);
  REQUIRE(j.pointB().x() == 3);
  REQUIRE(j.pointB().y() == 4);

  auto joined = join(segments, segmentsExtras);
  joined.bindExternalIndices(&points, &infos);
  auto se = joined.begin();
  REQUIRE(se.n() == 10);
  REQUIRE(se.info().color() == 4);
  REQUIRE(se.pointA().x() == 0);
  REQUIRE(se.pointA().y() == 0);
  REQUIRE(se.pointB().x() == 3);
  REQUIRE(se.pointB().y() == 4);
  REQUIRE(se.thickness() == 1);
}

TEST_CASE("TestSchemaCreation")
{
  auto schema = std::make_shared<arrow::Schema>(createFieldsFromColumns(o2::aod::Points::persistent_columns_t{}));
  REQUIRE(schema->num_fields() == 2);
  REQUIRE(schema->field(0)->name() == "fX");
  REQUIRE(schema->field(1)->name() == "fY");
}

TEST_CASE("TestFilteredOperators")
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"fX", "fY"});
  rowWriterA(0, 0, 8);
  rowWriterA(0, 1, 9);
  rowWriterA(0, 2, 10);
  rowWriterA(0, 3, 11);
  rowWriterA(0, 4, 12);
  rowWriterA(0, 5, 13);
  rowWriterA(0, 6, 14);
  rowWriterA(0, 7, 15);
  auto tableA = builderA.finalize();
  REQUIRE(tableA->num_rows() == 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, o2::aod::test::X, o2::aod::test::Y>;
  using FilteredTest = Filtered<TestA>;
  using NestedFilteredTest = Filtered<Filtered<TestA>>;
  using namespace o2::framework;

  expressions::Filter f1 = o2::aod::test::x < 4;
  expressions::Filter f2 = o2::aod::test::y > 13;

  TestA testA{tableA};
  auto s1 = expressions::createSelection(testA.asArrowTable(), f1);
  FilteredTest filtered1{{testA.asArrowTable()}, s1};
  REQUIRE(4 == filtered1.size());
  REQUIRE(filtered1.begin() != filtered1.end());

  auto s2 = expressions::createSelection(testA.asArrowTable(), f2);
  FilteredTest filtered2{{testA.asArrowTable()}, s2};
  REQUIRE(2 == filtered2.size());
  REQUIRE(filtered2.begin() != filtered2.end());

  FilteredTest filteredUnion = filtered1 + filtered2;
  REQUIRE(6 == filteredUnion.size());

  std::vector<std::tuple<int32_t, int32_t>> expectedUnion{
    {0, 8}, {1, 9}, {2, 10}, {3, 11}, {6, 14}, {7, 15}};
  auto i = 0;
  for (auto& f : filteredUnion) {
    REQUIRE(std::get<0>(expectedUnion[i]) == f.x());
    REQUIRE(std::get<1>(expectedUnion[i]) == f.y());
    REQUIRE(std::get<0>(expectedUnion[i]) == f.index());
    i++;
  }
  REQUIRE(i == 6);

  FilteredTest filteredIntersection = filtered1 * filtered2;
  REQUIRE(0 == filteredIntersection.size());

  i = 0;
  for (auto& f : filteredIntersection) {
    i++;
  }
  REQUIRE(i == 0);

  expressions::Filter f3 = o2::aod::test::x < 3;
  auto s3 = expressions::createSelection(testA.asArrowTable(), f3);
  FilteredTest filtered3{{testA.asArrowTable()}, s3};
  REQUIRE(3 == filtered3.size());
  REQUIRE(filtered3.begin() != filtered3.end());

  FilteredTest unionIntersection = (filtered1 + filtered2) * filtered3;
  REQUIRE(3 == unionIntersection.size());

  i = 0;
  for (auto& f : unionIntersection) {
    REQUIRE(i == f.x());
    REQUIRE(i + 8 == f.y());
    REQUIRE(i == f.index());
    i++;
  }
  REQUIRE(i == 3);
}

TEST_CASE("TestNestedFiltering")
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<int32_t, int32_t>({"fX", "fY"});
  rowWriterA(0, 0, 8);
  rowWriterA(0, 1, 9);
  rowWriterA(0, 2, 10);
  rowWriterA(0, 3, 11);
  rowWriterA(0, 4, 12);
  rowWriterA(0, 5, 13);
  rowWriterA(0, 6, 14);
  rowWriterA(0, 7, 15);
  auto tableA = builderA.finalize();
  REQUIRE(tableA->num_rows() == 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, o2::aod::test::X, o2::aod::test::Y>;
  using FilteredTest = Filtered<TestA>;
  using NestedFilteredTest = Filtered<Filtered<TestA>>;
  using TripleNestedFilteredTest = Filtered<Filtered<Filtered<TestA>>>;
  using namespace o2::framework;

  expressions::Filter f1 = o2::aod::test::x < 4;
  expressions::Filter f2 = o2::aod::test::y > 9;
  expressions::Filter f3 = o2::aod::test::x < 3;

  TestA testA{tableA};
  auto s1 = expressions::createSelection(testA.asArrowTable(), f1);
  FilteredTest filtered{{testA.asArrowTable()}, s1};
  REQUIRE(4 == filtered.size());
  REQUIRE(filtered.begin() != filtered.end());

  auto s2 = expressions::createSelection(filtered.asArrowTable(), f2);
  NestedFilteredTest nestedFiltered{{filtered}, s2};
  REQUIRE(2 == nestedFiltered.size());
  auto i = 0;
  for (auto& f : nestedFiltered) {
    REQUIRE(i + 2 == f.x());
    REQUIRE(i + 10 == f.y());
    REQUIRE(i + 2 == f.index());
    i++;
  }
  REQUIRE(i == 2);

  auto s3 = expressions::createSelection(nestedFiltered.asArrowTable(), f3);
  TripleNestedFilteredTest tripleFiltered{{nestedFiltered}, s3};
  REQUIRE(1 == tripleFiltered.size());
  i = 0;
  for (auto& f : tripleFiltered) {
    REQUIRE(i + 2 == f.x());
    REQUIRE(i + 10 == f.y());
    REQUIRE(i + 2 == f.index());
    i++;
  }
  REQUIRE(i == 1);
}

TEST_CASE("TestEmptyTables")
{
  TableBuilder bPoints;
  auto pwriter = bPoints.cursor<o2::aod::Points>();
  auto pempty = bPoints.finalize();

  TableBuilder bInfos;
  auto iwriter = bInfos.cursor<o2::aod::Infos>();
  auto iempty = bInfos.finalize();

  o2::aod::Points p{pempty};
  o2::aod::Infos i{iempty};

  using PI = Join<o2::aod::Points, o2::aod::Infos>;
  PI pi{0, pempty, iempty};
  REQUIRE(pi.size() == 0);
  auto spawned = Extend<o2::aod::Points, o2::aod::test::ESum>(p);
  REQUIRE(spawned.size() == 0);
}

namespace o2::aod
{
DECLARE_SOA_TABLE(Origints, "TEST", "ORIG", o2::soa::Index<>, test::X, test::SomeBool);
namespace test
{
DECLARE_SOA_INDEX_COLUMN(Origint, origint);
DECLARE_SOA_INDEX_COLUMN_FULL(AltOrigint, altOrigint, int, Origints, "_alt");
DECLARE_SOA_ARRAY_INDEX_COLUMN(Origint, origints);
} // namespace test
DECLARE_SOA_TABLE(References, "TEST", "REFS", o2::soa::Index<>, test::OrigintId);
DECLARE_SOA_TABLE(OtherReferences, "TEST", "OREFS", o2::soa::Index<>, test::AltOrigintId);
DECLARE_SOA_TABLE(ManyReferences, "TEST", "MREFS", o2::soa::Index<>, test::OrigintIds);
} // namespace o2::aod
TEST_CASE("TestIndexToFiltered")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::Origints>();
  for (auto i = 0; i < 20; ++i) {
    writer(0, i, i % 3 == 0);
  }
  auto origins = b.finalize();
  o2::aod::Origints o{origins};

  TableBuilder z;
  auto writer_z = z.cursor<o2::aod::ManyReferences>();
  std::vector<int> ids;
  for (auto i = 0; i < 5; ++i) {
    ids.clear();
    for (auto j = 0; j < 20; ++j) {
      ids.push_back(j);
    }
    writer_z(0, ids);
  }
  auto mrefs = z.finalize();
  o2::aod::ManyReferences m{mrefs};

  TableBuilder w;
  auto writer_w = w.cursor<o2::aod::References>();
  for (auto i = 0; i < 5 * 20; ++i) {
    writer_w(0, i % 20);
  }
  auto refs = w.finalize();
  o2::aod::References r{refs};
  expressions::Filter flt = o2::aod::test::someBool == true;
  using Flt = o2::soa::Filtered<o2::aod::Origints>;
  auto selection = expressions::createSelection(o.asArrowTable(), flt);
  Flt f{{o.asArrowTable()}, selection};
  r.bindExternalIndices(&f);
  auto it = r.begin();
  it.moveByIndex(23);
  REQUIRE(it.origint().globalIndex() == 3);
  it++;
  REQUIRE(it.origint().globalIndex() == 4);
  it++;
  REQUIRE(it.origint().globalIndex() == 5);

  m.bindExternalIndices(&f);
  for (auto const& row : m) {
    auto os = row.origints_as<Flt>();
    auto fos = row.filtered_origints_as<Flt>();
    REQUIRE(os.size() == 20);
    REQUIRE(fos.size() == 6);
  }
}
namespace o2::aod
{
namespace test
{
DECLARE_SOA_INDEX_COLUMN_FULL(SinglePoint, singlePoint, int32_t, Points3Ds, "");
DECLARE_SOA_ARRAY_INDEX_COLUMN(Points3D, pointGroup);
DECLARE_SOA_SLICE_INDEX_COLUMN(Points3D, pointSlice);
DECLARE_SOA_SELF_INDEX_COLUMN(OtherPoint, otherPoint);
DECLARE_SOA_SELF_SLICE_INDEX_COLUMN(PointSeq, pointSeq);
DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN(PointSet, pointSet);
} // namespace test

DECLARE_SOA_TABLE(PointsRef, "TEST", "PTSREF", test::Points3DIdSlice, test::Points3DIds);
DECLARE_SOA_TABLE(PointsRefF, "TEST", "PTSREFF", test::SinglePointId, test::Points3DIdSlice, test::Points3DIds);
DECLARE_SOA_TABLE(PointsSelfIndex, "TEST", "PTSSLF", o2::soa::Index<>, test::X, test::Y, test::Z, test::OtherPointId,
                  test::PointSeqIdSlice, test::PointSetIds);
} // namespace o2::aod

TEST_CASE("TestAdvancedIndices")
{
  TableBuilder b1;
  auto pwriter = b1.persist<int, int, int>({"fX", "fY", "fZ"});
  for (auto i = 0; i < 20; ++i) {
    pwriter(0, -1 * i, (int)(i / 2), 2 * i);
  }
  auto tpts1 = b1.finalize();

  TableBuilder b2;
  auto prwriter = b2.cursor<o2::aod::PointsRef>();
  auto a = std::array{0, 1};
  auto aa = std::vector{2, 3, 4};
  prwriter(0, &a[0], aa);
  a = {4, 10};
  aa = {12, 2, 19};
  prwriter(0, &a[0], aa);
  auto t2 = b2.finalize();

  auto pt = o2::aod::Points3Ds{tpts1};
  auto prt = o2::aod::PointsRef{t2};
  prt.bindExternalIndices(&pt);

  auto it = prt.begin();
  auto s1 = it.pointSlice();
  auto g1 = it.pointGroup();
  auto bb = std::is_same_v<decltype(s1), o2::aod::Points3Ds>;
  REQUIRE(bb);
  REQUIRE(s1.size() == 2);
  aa = {2, 3, 4};
  for (int i = 0; i < 3; ++i) {
    REQUIRE(g1[i].globalIndex() == aa[i]);
  }

  // Check the X coordinate of the points in the pointGroup
  // for the first point.
  for (auto& p : it.pointGroup_as<o2::aod::Points3Ds>()) {
    REQUIRE(p.x() == -1 * p.globalIndex());
  }

  ++it;
  auto s2 = it.pointSlice();
  auto g2 = it.pointGroup();
  REQUIRE(s2.size() == 7);
  aa = {12, 2, 19};
  for (int i = 0; i < 3; ++i) {
    REQUIRE(g2[i].globalIndex() == aa[i]);
  }

  using Flt = o2::soa::Filtered<o2::aod::Points3Ds>;
  expressions::Filter fltx = (o2::aod::test::x <= -6);
  Flt f{{tpts1}, expressions::createSelection(tpts1, fltx)};
  prt.bindExternalIndices(&f);

  auto it2 = prt.begin();
  auto s1f = it2.pointSlice_as<Flt>();
  auto g1f = it2.pointGroup_as<Flt>();
  REQUIRE(s1f.size() == 2);
  aa = {2, 3, 4};
  for (int i = 0; i < 3; ++i) {
    REQUIRE(g1f[i].globalIndex() == aa[i]);
  }

  ++it2;
  auto s2f = it2.pointSlice_as<Flt>();
  auto g2f = it2.pointGroup_as<Flt>();
  REQUIRE(s2f.size() == 7);
  aa = {12, 2, 19};
  for (int i = 0; i < 3; ++i) {
    REQUIRE(g2f[i].globalIndex() == aa[i]);
  }

  TableBuilder b3;
  auto pswriter = b3.cursor<o2::aod::PointsSelfIndex>();
  int references[] = {19, 2, 0, 13, 4, 6, 5, 5, 11, 9, 3, 8, 16, 14, 1, 18, 12, 18, 2, 7};
  int slice[2] = {-1, -1};
  std::vector<int> pset;
  std::array<int, 4> withSlices = {3, 6, 13, 19};
  std::array<std::pair<int, int>, 4> bounds = {std::pair{1, 5}, std::pair{3, 3}, std::pair{11, 11}, std::pair{10, 18}};
  std::array<int, 4> withSets = {0, 1, 13, 14};
  unsigned int sizes[] = {3, 1, 5, 4};
  unsigned int c1 = 0;
  unsigned int c2 = 0;
  for (auto i = 0; i < 20; ++i) {
    pset.clear();
    slice[0] = -1;
    slice[1] = -1;
    if (c1 < withSlices.size() && i == withSlices[c1]) {
      slice[0] = bounds[c1].first;
      slice[1] = bounds[c1].second;
      ++c1;
    }
    if (c2 < withSets.size() && i == withSets[c2]) {
      for (auto z = 0U; z < sizes[c2]; ++z) {
        pset.push_back(i + 1 + z);
      }
      ++c2;
    }
    pswriter(0, -1 * i, 0.5 * i, 2 * i, references[i], slice, pset);
  }
  auto t3 = b3.finalize();
  auto pst = o2::aod::PointsSelfIndex{t3};
  pst.bindInternalIndicesTo(&pst);
  auto i = 0;
  c1 = 0;
  c2 = 0;
  for (auto& p : pst) {
    auto op = p.otherPoint_as<o2::aod::PointsSelfIndex>();
    auto bbb = std::is_same_v<decltype(op), o2::aod::PointsSelfIndex::iterator>;
    REQUIRE(bbb);
    REQUIRE(op.globalIndex() == references[i]);

    auto ops = p.pointSeq_as<o2::aod::PointsSelfIndex>();
    if (i == withSlices[c1]) {
      auto it = ops.begin();
      REQUIRE(ops.size() == bounds[c1].second - bounds[c1].first + 1);
      REQUIRE(it.globalIndex() == bounds[c1].first);
      for (auto j = 1; j < ops.size(); ++j) {
        ++it;
      }
      REQUIRE(it.globalIndex() == bounds[c1].second);
      ++c1;
    } else {
      REQUIRE(ops.size() == 0);
    }
    auto opss = p.pointSet_as<o2::aod::PointsSelfIndex>();
    auto opss_ids = p.pointSetIds();
    if (c2 < withSets.size() && i == withSets[c2]) {
      REQUIRE(opss.size() == sizes[c2]);
      REQUIRE(opss.begin()->globalIndex() == i + 1);
      REQUIRE(opss.back().globalIndex() == i + sizes[c2]);
      int c3 = 0;
      for (auto& id : opss_ids) {
        REQUIRE(id == i + 1 + c3);
        ++c3;
      }
      ++c2;
    } else {
      REQUIRE(opss.size() == 0);
    }
    ++i;
  }
}

TEST_CASE("TestListColumns")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::Lists>();
  std::vector<float> floats;
  std::vector<int> ints;
  for (auto i = 1; i < 11; ++i) {
    floats.clear();
    ints.clear();
    for (auto j = 0; j < i; ++j) {
      floats.push_back(0.1231233f * (float)j + 0.1982798f);
      ints.push_back(j + 10);
    }

    writer(0, floats, ints);
  }
  auto lt = b.finalize();
  o2::aod::Lists tbl{lt};
  int s = 1;
  for (auto& row : tbl) {
    auto f = row.l1();
    auto i = row.l2();
    auto constexpr bf = std::is_same_v<decltype(f), gsl::span<const float, (size_t)-1>>;
    auto constexpr bi = std::is_same_v<decltype(i), gsl::span<const int, (size_t)-1>>;
    REQUIRE(bf);
    REQUIRE(bi);
    REQUIRE(f.size() == s);
    REQUIRE(i.size() == s);

    for (auto j = 0u; j < f.size(); ++j) {
      REQUIRE(f[j] == 0.1231233f * (float)j + 0.1982798f);
      REQUIRE(i[j] == j + 10);
    }
    ++s;
  }
}

TEST_CASE("TestSliceByCached")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::Origints>();
  for (auto i = 0; i < 20; ++i) {
    writer(0, i, i % 3 == 0);
  }
  auto origins = b.finalize();
  o2::aod::Origints o{origins};

  TableBuilder w;
  auto writer_w = w.cursor<o2::aod::References>();
  auto step = -1;
  for (auto i = 0; i < 5 * 20; ++i) {
    if (i % 5 == 0) {
      ++step;
    }
    writer_w(0, step);
  }
  auto refs = w.finalize();
  o2::aod::References r{refs};

  ArrowTableSlicingCache atscache({{o2::soa::getLabelFromType<o2::aod::References>(), "fIndex" + o2::framework::cutString(o2::soa::getLabelFromType<o2::aod::Origints>())}});
  auto s = atscache.updateCacheEntry(0, refs);
  SliceCache cache{&atscache};

  for (auto& oi : o) {
    auto cachedSlice = r.sliceByCached(o2::aod::test::origintId, oi.globalIndex(), cache);
    REQUIRE(cachedSlice.size() == 5);
    for (auto& ri : cachedSlice) {
      REQUIRE(ri.origintId() == oi.globalIndex());
    }
  }
}

TEST_CASE("TestSliceByCachedMismatched")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::Origints>();
  for (auto i = 0; i < 20; ++i) {
    writer(0, i, i % 3 == 0);
  }
  auto origins = b.finalize();
  o2::aod::Origints o{origins};

  TableBuilder w;
  auto writer_w = w.cursor<o2::aod::References>();
  auto step = -1;
  for (auto i = 0; i < 5 * 20; ++i) {
    if (i % 5 == 0) {
      ++step;
    }
    writer_w(0, step);
  }
  auto refs = w.finalize();
  o2::aod::References r{refs};

  TableBuilder w2;
  auto writer_w2 = w2.cursor<o2::aod::OtherReferences>();
  step = -1;
  for (auto i = 0; i < 5 * 20; ++i) {
    if (i % 3 == 0) {
      ++step;
    }
    writer_w2(0, step);
  }
  auto refs2 = w2.finalize();
  o2::aod::OtherReferences r2{refs2};

  using J = o2::soa::Join<o2::aod::References, o2::aod::OtherReferences>;
  J rr{{refs, refs2}};

  auto key = "fIndex" + o2::framework::cutString(o2::soa::getLabelFromType<o2::aod::Origints>()) + "_alt";
  ArrowTableSlicingCache atscache({{o2::soa::getLabelFromTypeForKey<J>(key), key}});
  auto s = atscache.updateCacheEntry(0, refs2);
  SliceCache cache{&atscache};

  for (auto& oi : o) {
    auto cachedSlice = rr.sliceByCached(o2::aod::test::altOrigintId, oi.globalIndex(), cache);
    REQUIRE(cachedSlice.size() == 3);
    for (auto& ri : cachedSlice) {
      REQUIRE(ri.altOrigintId() == oi.globalIndex());
    }
  }
}

TEST_CASE("TestIndexUnboundExceptions")
{
  TableBuilder b;
  auto prwriter = b.cursor<o2::aod::PointsRefF>();
  auto a = std::array{0, 1};
  auto aa = std::vector{2, 3, 4};
  prwriter(0, 0, &a[0], aa);
  a = {4, 10};
  aa = {12, 2, 19};
  prwriter(0, 1, &a[0], aa);
  auto t = b.finalize();
  auto prt = o2::aod::PointsRefF{t};

  for (auto& row : prt) {
    try {
      auto sp = row.singlePoint();
    } catch (RuntimeErrorRef ref) {
      REQUIRE(std::string{error_from_ref(ref).what} == "Index pointing to Points3Ds is not bound! Did you subscribe to the table?");
    }
    try {
      auto ps = row.pointSlice();
    } catch (RuntimeErrorRef ref) {
      REQUIRE(std::string{error_from_ref(ref).what} == "Index pointing to Points3Ds is not bound! Did you subscribe to the table?");
    }
    try {
      auto pg = row.pointGroup();
    } catch (RuntimeErrorRef ref) {
      REQUIRE(std::string{error_from_ref(ref).what} == "Index pointing to Points3Ds is not bound! Did you subscribe to the table?");
    }
  }
}

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(SmallIntArray, smallIntArray, int8_t[32]);
DECLARE_SOA_BITMAP_COLUMN(BoolArray, boolArray, 32);
} // namespace test

DECLARE_SOA_TABLE(BILists, "TEST", "BILISTS", o2::soa::Index<>, test::SmallIntArray, test::BoolArray);
} // namespace o2::aod

TEST_CASE("TestArrayColumns")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::BILists>();
  int8_t ii[32];
  uint32_t bb;
  for (auto i = 0; i < 20; ++i) {
    bb = 0;
    for (auto j = 0; j < 32; ++j) {
      ii[j] = j;
      if (j % 2 == 0) {
        bb |= 1 << j;
      }
    }
    writer(0, ii, bb);
  }
  auto t = b.finalize();

  o2::aod::BILists li{t};
  for (auto const& row : li) {
    auto iir = row.smallIntArray();
    auto bbrr = row.boolArray_raw();
    REQUIRE(std::is_same_v<std::decay_t<decltype(iir)>, int8_t const*>);
    for (auto i = 0; i < 32; ++i) {
      REQUIRE(iir[i] == i);
      REQUIRE(row.boolArray_bit(i) == (i % 2 == 0));
    }
  }
}

namespace o2::aod
{
namespace table
{
DECLARE_SOA_COLUMN(One, one, int);
DECLARE_SOA_COLUMN(Two, two, float);
DECLARE_SOA_COLUMN(Three, three, double);
DECLARE_SOA_COLUMN(Four, four, int[2]);
DECLARE_SOA_DYNAMIC_COLUMN(Five, five, [](const int in[2]) -> float { return (float)in[0] / (float)in[1]; });
} // namespace table
DECLARE_SOA_TABLE(MixTest, "AOD", "MIXTST",
                  table::One, table::Two, table::Three, table::Four,
                  table::Five<table::Four>);
} // namespace o2::aod
TEST_CASE("TestCombinedGetter")
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::MixTest>();
  int f[2];
  for (auto i = 0; i < 20; ++i) {
    f[0] = i;
    f[1] = i + 1;
    writer(0, i, o2::constants::math::PI * i, o2::constants::math::Almost0 * i, f);
  }
  auto t = b.finalize();
  o2::aod::MixTest mt{t};
  auto count = 0;
  for (auto const& row : mt) {
    auto features1 = row.getValues<float, o2::aod::table::One, o2::aod::table::Three>();
    auto features2 = row.getValues<double, o2::aod::table::One, o2::aod::table::Two, o2::aod::table::Three>();
    auto features3 = row.getValues<float, o2::aod::table::Two, o2::aod::table::Five<o2::aod::table::Four>>();
    auto b1 = std::is_same_v<std::array<float, 2>, decltype(features1)>;
    REQUIRE(b1);
    auto b2 = std::is_same_v<std::array<double, 3>, decltype(features2)>;
    REQUIRE(b2);
    auto b3 = std::is_same_v<std::array<float, 2>, decltype(features3)>;
    REQUIRE(b3);
    REQUIRE(features1[0] == (float)count);
    REQUIRE(features1[1] == (float)(o2::constants::math::Almost0 * count));

    REQUIRE(features2[0] == (double)count);
    REQUIRE(features2[1] == (double)(o2::constants::math::PI * count));
    REQUIRE(features2[2] == (double)(o2::constants::math::Almost0 * count));

    REQUIRE(features3[0] == (float)(o2::constants::math::PI * count));
    REQUIRE(features3[1] == (float)((float)count / (float)(count + 1)));
    ++count;
  }
}
