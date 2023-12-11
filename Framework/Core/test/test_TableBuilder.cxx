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

#include <catch_amalgamated.hpp>

#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputRoute.h"
#include <arrow/table.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include "../src/ArrowDebugHelpers.h"

using namespace o2::framework;

// We use a different namespace to avoid clashes with the
// test_ASoA.cxx test.
namespace test2
{
DECLARE_SOA_COLUMN_FULL(X, x, uint64_t, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, uint64_t, "y");
DECLARE_SOA_COLUMN_FULL(Pos, pos, int[4], "pos");
} // namespace test

using TestTable = o2::soa::Table<test2::X, test2::Y>;
using ArrayTable = o2::soa::Table<test2::Pos>;

TEST_CASE("TestTableBuilder")
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<uint64_t, uint64_t>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 10, 1);
  rowWriter(0, 20, 2);
  rowWriter(0, 30, 3);
  rowWriter(0, 40, 4);
  rowWriter(0, 50, 5);
  rowWriter(0, 60, 6);
  rowWriter(0, 70, 7);
  auto table = builder.finalize();
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 8);
  REQUIRE(table->schema()->field(0)->name() == "x");
  REQUIRE(table->schema()->field(1)->name() == "y");
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::uint64()->id());
  REQUIRE(table->schema()->field(1)->type()->id() == arrow::uint64()->id());

  auto readBack = TestTable{table};

  auto readBackTable = readBack.asArrowTable();
  REQUIRE(readBackTable->num_columns() == 2);
  REQUIRE(readBackTable->num_rows() == 8);
  REQUIRE(readBackTable->schema()->field(0)->name() == "x");
  REQUIRE(readBackTable->schema()->field(1)->name() == "y");
  REQUIRE(readBackTable->schema()->field(0)->type()->id() == arrow::uint64()->id());
  REQUIRE(readBackTable->schema()->field(1)->type()->id() == arrow::uint64()->id());
  size_t i = 0;
  SECTION("Check")
  {
    for (auto const& row : readBack) {
      REQUIRE(row.x() == i * 10);
      REQUIRE(row.y() == i);
      ++i;
    }
  }
}

TEST_CASE("TestTableBuilderArray")
{
  using namespace o2::framework;
  TableBuilder builder;
  const int numElem = 4;
  auto rowWriter = builder.persist<int[numElem]>({"pos"});
  int a[numElem] = {1, 10, 300, 350};
  int b[numElem] = {0, 20, 30, 40};
  rowWriter(0, a);
  rowWriter(0, b);
  using v3 = std::array<int, numElem>;
  rowWriter(0, v3{0, 11, 123, 256}.data());
  auto table = builder.finalize();

  REQUIRE(table->num_columns() == 1);
  REQUIRE(table->num_rows() == 3);
  REQUIRE(table->schema()->field(0)->name() == "pos");
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::fixed_size_list(arrow::int32(), numElem)->id());

  auto chunkToUse = table->column(0)->chunk(0);
  chunkToUse = std::static_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
  auto data = chunkToUse->data();

  REQUIRE(data->GetValues<int>(1)[0] == 1);
  REQUIRE(data->GetValues<int>(1)[1] == 10);
  REQUIRE(data->GetValues<int>(1)[2] == 300);
  REQUIRE(data->GetValues<int>(1)[3] == 350);
  REQUIRE(data->GetValues<int>(1)[4] == 0);
  REQUIRE(data->GetValues<int>(1)[5] == 20);
  REQUIRE(data->GetValues<int>(1)[6] == 30);
  REQUIRE(data->GetValues<int>(1)[7] == 40);

  auto readBack = ArrayTable{table};
  auto row = readBack.begin();

  REQUIRE(row.pos()[0] == 1);
  REQUIRE(row.pos()[1] == 10);
  REQUIRE(row.pos()[2] == 300);
  REQUIRE(row.pos()[3] == 350);

  row++;
  REQUIRE(row.pos()[0] == 0);
  REQUIRE(row.pos()[1] == 20);
  REQUIRE(row.pos()[2] == 30);
  REQUIRE(row.pos()[3] == 40);

  row++;
  REQUIRE(row.pos()[0] == 0);
  REQUIRE(row.pos()[1] == 11);
  REQUIRE(row.pos()[2] == 123);
  REQUIRE(row.pos()[3] == 256);
}

TEST_CASE("TestTableBuilderStruct")
{
  using namespace o2::framework;
  TableBuilder builder;
  struct Foo {
    uint64_t x;
    uint64_t y;
  };
  auto rowWriter = builder.persist<Foo>({"x", "y"});
  rowWriter(0, Foo{0, 0});
  rowWriter(0, Foo{10, 1});
  rowWriter(0, Foo{20, 2});
  rowWriter(0, Foo{30, 3});
  rowWriter(0, Foo{40, 4});
  rowWriter(0, Foo{50, 5});
  rowWriter(0, Foo{60, 6});
  rowWriter(0, Foo{70, 7});
  auto table = builder.finalize();
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 8);
  REQUIRE(table->schema()->field(0)->name() == "x");
  REQUIRE(table->schema()->field(1)->name() == "y");
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::uint64()->id());
  REQUIRE(table->schema()->field(1)->type()->id() == arrow::uint64()->id());

  auto readBack = TestTable{table};

  size_t i = 0;
  for (auto& row : readBack) {
    REQUIRE(row.x() == i * 10);
    REQUIRE(row.y() == i);
    ++i;
  }
}

TEST_CASE("TestTableBuilderBulk")
{
  using namespace o2::framework;
  TableBuilder builder;
  auto bulkWriter = builder.bulkPersist<int, int>({"x", "y"}, 10);
  int x[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int y[] = {0, 1, 2, 3, 4, 5, 6, 7};

  bulkWriter(0, 8, x, y);

  auto table = builder.finalize();
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 8);
  REQUIRE(table->schema()->field(0)->name() == "x");
  REQUIRE(table->schema()->field(1)->name() == "y");
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::int32()->id());
  REQUIRE(table->schema()->field(1)->type()->id() == arrow::int32()->id());

  for (size_t i = 0; i < 8; ++i) {
    auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(table->column(0)->chunk(0));
    REQUIRE(p->Value(i) == i);
  }
}

TEST_CASE("TestTableBuilderMore")
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, std::string, bool>({"x", "y", "s", "b"});
  builder.reserve(pack<int, float, std::string, bool>{}, 5);
  rowWriter(0, 0, 0., "foo", true);
  rowWriter(0, 1, 1., "bar", false);
  rowWriter(0, 2, 2., "fbr", false);
  rowWriter(0, 3, 3., "bar", false);
  rowWriter(0, 4, 4., "abr", true);
  rowWriter(0, 5, 5., "aaa", false);
  rowWriter(0, 6, 6., "bbb", true);
  rowWriter(0, 7, 7., "ccc", false);
  auto table = builder.finalize();
  REQUIRE(table->num_columns() == 4);
  REQUIRE(table->num_rows() == 8);
  REQUIRE(table->schema()->field(0)->name() == "x");
  REQUIRE(table->schema()->field(1)->name() == "y");
  REQUIRE(table->schema()->field(2)->name() == "s");
  REQUIRE(table->schema()->field(3)->name() == "b");
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::int32()->id());
  REQUIRE(table->schema()->field(1)->type()->id() == arrow::float32()->id());
  REQUIRE(table->schema()->field(2)->type()->id() == arrow::utf8()->id());
  REQUIRE(table->schema()->field(3)->type()->id() == arrow::boolean()->id());
}

TEST_CASE("TestSoAIntegration")
{
  TableBuilder builder;
  auto rowWriter = builder.cursor<TestTable>();
  rowWriter(0, 0, 0);
  rowWriter(0, 10, 1);
  rowWriter(0, 20, 2);
  rowWriter(0, 30, 3);
  rowWriter(0, 40, 4);
  rowWriter(0, 50, 5);
  auto table = builder.finalize();
  auto readBack = TestTable{table};

  size_t i = 0;
  for (auto& row : readBack) {
    REQUIRE(row.x() == i * 10);
    REQUIRE(row.y() == i);
    ++i;
  }
}

TEST_CASE("TestDataAllocatorReturnType")
{
  const Output output{"TST", "DUMMY", 0};
}

TEST_CASE("TestPodInjestion")
{
  struct A {
    uint64_t x;
    uint64_t y;
  };
  TableBuilder builder;
  auto rowWriter = builder.cursor<TestTable, A>();
  rowWriter(0, A{0, 0});
  rowWriter(0, A{10, 1});
  rowWriter(0, A{20, 2});
  rowWriter(0, A{30, 3});
  rowWriter(0, A{40, 4});
  rowWriter(0, A{50, 5});
  auto table = builder.finalize();
  auto readBack = TestTable{table};

  size_t i = 0;
  for (auto& row : readBack) {
    REQUIRE(row.x() == i * 10);
    REQUIRE(row.y() == i);
    ++i;
  }
}

TEST_CASE("TestColumnCount")
{
  struct Foo {
    int x;
    int y;
  };
  struct Bar {
    int x;
    int y;
    std::string s;
  };
  struct FooBar {
    int x;
    int y;
    float f;
  };
  REQUIRE(TableBuilder::countColumns<Foo>() == 2);
  REQUIRE(TableBuilder::countColumns<Bar>() == 3);
  REQUIRE(TableBuilder::countColumns<FooBar>() == 3);
  int count = TableBuilder::countColumns<float, int>();
  REQUIRE(count == 2);
  int count2 = TableBuilder::countColumns<float, int, char[3]>();
  REQUIRE(count2 == 3);
}

TEST_CASE("TestMakeFields")
{
  auto fields = TableBuilderHelpers::makeFields<int, float>({ "i", "f" });
  REQUIRE(fields.size() == 2);
  REQUIRE(fields[0]->name() == "i");
  REQUIRE(fields[1]->name() == "f");
  REQUIRE(fields[0]->type()->name() == "int32");
  REQUIRE(fields[1]->type()->name() == "float");
}
