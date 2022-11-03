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

#define BOOST_TEST_MODULE Test Framework TableBuilder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputRoute.h"
#include "Framework/FairMQResizableBuffer.h"
#include <arrow/table.h>
#include "Framework/RCombinedDS.h"
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include "../src/ArrowDebugHelpers.h"

using namespace o2::framework;

namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, uint64_t, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, uint64_t, "y");
DECLARE_SOA_COLUMN_FULL(Pos, pos, int[4], "pos");
} // namespace test

using TestTable = o2::soa::Table<test::X, test::Y>;
using ArrayTable = o2::soa::Table<test::Pos>;

BOOST_AUTO_TEST_CASE(TestTableBuilder)
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
  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 8);
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->type()->id(), arrow::uint64()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->type()->id(), arrow::uint64()->id());

  auto readBack = TestTable{table};

  size_t i = 0;
  for (auto& row : readBack) {
    BOOST_CHECK_EQUAL(row.x(), i * 10);
    BOOST_CHECK_EQUAL(row.y(), i);
    ++i;
  }
}

BOOST_AUTO_TEST_CASE(TestTableBuilderArray)
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

  BOOST_REQUIRE_EQUAL(table->num_columns(), 1);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 3);
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->name(), "pos");
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->type()->id(), arrow::fixed_size_list(arrow::int32(), numElem)->id());

  auto chunkToUse = table->column(0)->chunk(0);
  chunkToUse = std::static_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
  auto data = chunkToUse->data();

  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[0], 1);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[1], 10);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[2], 300);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[3], 350);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[4], 0);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[5], 20);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[6], 30);
  BOOST_REQUIRE_EQUAL(data->GetValues<int>(1)[7], 40);

  auto readBack = ArrayTable{table};
  auto row = readBack.begin();

  BOOST_CHECK_EQUAL(row.pos()[0], 1);
  BOOST_CHECK_EQUAL(row.pos()[1], 10);
  BOOST_CHECK_EQUAL(row.pos()[2], 300);
  BOOST_CHECK_EQUAL(row.pos()[3], 350);

  row++;
  BOOST_CHECK_EQUAL(row.pos()[0], 0);
  BOOST_CHECK_EQUAL(row.pos()[1], 20);
  BOOST_CHECK_EQUAL(row.pos()[2], 30);
  BOOST_CHECK_EQUAL(row.pos()[3], 40);

  row++;
  BOOST_CHECK_EQUAL(row.pos()[0], 0);
  BOOST_CHECK_EQUAL(row.pos()[1], 11);
  BOOST_CHECK_EQUAL(row.pos()[2], 123);
  BOOST_CHECK_EQUAL(row.pos()[3], 256);
}

BOOST_AUTO_TEST_CASE(TestTableBuilderStruct)
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
  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 8);
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->type()->id(), arrow::uint64()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->type()->id(), arrow::uint64()->id());

  auto readBack = TestTable{table};

  size_t i = 0;
  for (auto& row : readBack) {
    BOOST_CHECK_EQUAL(row.x(), i * 10);
    BOOST_CHECK_EQUAL(row.y(), i);
    ++i;
  }
}

BOOST_AUTO_TEST_CASE(TestTableBuilderBulk)
{
  using namespace o2::framework;
  TableBuilder builder;
  auto bulkWriter = builder.bulkPersist<int, int>({"x", "y"}, 10);
  int x[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int y[] = {0, 1, 2, 3, 4, 5, 6, 7};

  bulkWriter(0, 8, x, y);

  auto table = builder.finalize();
  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 8);
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->type()->id(), arrow::int32()->id());

  for (size_t i = 0; i < 8; ++i) {
    auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(table->column(0)->chunk(0));
    BOOST_CHECK_EQUAL(p->Value(i), i);
  }
}

BOOST_AUTO_TEST_CASE(TestTableBuilderMore)
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
  BOOST_REQUIRE_EQUAL(table->num_columns(), 4);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 8);
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->schema()->field(2)->name(), "s");
  BOOST_REQUIRE_EQUAL(table->schema()->field(3)->name(), "b");
  BOOST_REQUIRE_EQUAL(table->schema()->field(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(1)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(2)->type()->id(), arrow::utf8()->id());
  BOOST_REQUIRE_EQUAL(table->schema()->field(3)->type()->id(), arrow::boolean()->id());
}

// Use RDataFrame to build the table
// BOOST_AUTO_TEST_CASE(TestRDataFrame)
//{
//  using namespace o2::framework;
//  TableBuilder builder;
//  ROOT::RDataFrame rdf(100);
//  auto t = rdf.Define("x", "1")
//             .Define("y", "2")
//             .Define("z", "x+y");
//  t.ForeachSlot(builder.persist<int, int>({"x", "z"}), {"x", "z"});
//
//  auto table = builder.finalize();
//  BOOST_REQUIRE_EQUAL(table->num_rows(), 100);
//  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
//  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::int32()->id());
//  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::int32()->id());
//
//  /// Writing to a stream
//  std::shared_ptr<arrow::io::BufferOutputStream> stream;
//  auto streamOk = arrow::io::BufferOutputStream::Create(100000, arrow::default_memory_pool(), &stream);
//  BOOST_REQUIRE_EQUAL(streamOk.ok(), true);
//  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
//  auto outBatch = arrow::ipc::RecordBatchStreamWriter::Open(stream.get(), table->schema(), &writer);
//  auto outStatus = writer->WriteTable(*table);
//  BOOST_REQUIRE_EQUAL(writer->Close().ok(), true);
//
//  std::shared_ptr<arrow::Buffer> inBuffer;
//  BOOST_REQUIRE_EQUAL(stream->Finish(&inBuffer).ok(), true);
//
//  BOOST_REQUIRE_EQUAL(outStatus.ok(), true);
//
//  /// Reading back from the stream
//  TableConsumer consumer(inBuffer->data(), inBuffer->size());
//  std::shared_ptr<arrow::Table> inTable = consumer.asArrowTable();
//
//  BOOST_REQUIRE_EQUAL(inTable->num_columns(), 2);
//  BOOST_REQUIRE_EQUAL(inTable->num_rows(), 100);
//
//  auto source = std::make_unique<ROOT::RDF::RArrowDS>(inTable, std::vector<std::string>{});
//  ROOT::RDataFrame finalDF{std::move(source)};
//  BOOST_REQUIRE_EQUAL(*finalDF.Count(), 100);
//}

BOOST_AUTO_TEST_CASE(TestCombinedDS)
{
  using namespace o2::framework;
  TableBuilder builder1;
  auto rowWriter1 = builder1.persist<int, int, int>({"x", "y", "event"});
  for (size_t i = 0; i < 8; ++i) {
    rowWriter1(0, i, i, i / 4);
  }
  auto table1 = builder1.finalize();

  TableBuilder builder2;
  auto rowWriter2 = builder2.persist<int, int>({"x", "y"});
  for (size_t i = 0; i < 8; ++i) {
    rowWriter2(0, i, i);
  }
  auto table2 = builder2.finalize();
  BOOST_REQUIRE_EQUAL(table2->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table2->num_rows(), 8);
  for (size_t i = 0; i < 8; ++i) {
    auto p2 = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(table2->column(0)->chunk(0));
    BOOST_CHECK_EQUAL(p2->Value(i), i);
  }

  auto source1 = std::make_unique<ROOT::RDF::RArrowDS>(table1, std::vector<std::string>{});
  auto source2 = std::make_unique<ROOT::RDF::RArrowDS>(table2, std::vector<std::string>{});
  auto cross = ROOT::RDF::MakeCrossProductDataFrame(std::move(source1), std::move(source2));
  auto source3 = std::make_unique<ROOT::RDF::RArrowDS>(table1, std::vector<std::string>{});
  auto source4 = std::make_unique<ROOT::RDF::RArrowDS>(table2, std::vector<std::string>{});
  auto indexed = ROOT::RDF::MakeColumnIndexedDataFrame(std::move(source3), std::move(source4), "x");
  auto source5 = std::make_unique<ROOT::RDF::RArrowDS>(table1, std::vector<std::string>{});
  auto source6 = std::make_unique<ROOT::RDF::RArrowDS>(table2, std::vector<std::string>{});
  auto unionDS = ROOT::RDF::MakeFriendDataFrame(std::move(source5), std::move(source6));
  auto source7 = std::make_unique<ROOT::RDF::RArrowDS>(table1, std::vector<std::string>{});
  auto source8 = std::make_unique<ROOT::RDF::RArrowDS>(table1, std::vector<std::string>{}); // Notice the table needs to be the same
  auto blockDS = ROOT::RDF::MakeBlockAntiDataFrame(std::move(source7), std::move(source8), "event");

  ROOT::RDataFrame finalDF{std::move(cross)};
  ROOT::RDataFrame indexedDF{std::move(indexed)};
  ROOT::RDataFrame unionDF{std::move(unionDS)};
  ROOT::RDataFrame blockDF{std::move(blockDS)};

  BOOST_CHECK_EQUAL(*finalDF.Count(), 64);  // Full cross product of 8x8 rows, 64 entries
  BOOST_CHECK_EQUAL(*indexedDF.Count(), 8); // Indexing the left table using a column of the right table
                                            // the number of rows remains as the right table ones: 8
  BOOST_CHECK_EQUAL(*unionDF.Count(), 8);   // Pairing one by one the rows of the two tables, still 8
  BOOST_CHECK_EQUAL(*blockDF.Count(), 24);  // The entries of the table are categorized by event:
                                            // 4 in event 0 and 4 in event 1. So the total number
                                            // of row is given by the cross product of the two parts, minus
                                            // the diagonal (4*4) - 4 + (4*4) - 4
  auto sum = [](int lx, int rx) { return lx + rx; };
  auto left = [](int lx, int) { return lx; };
  auto right = [](int, int rx) { return rx; };

  BOOST_CHECK_EQUAL(*finalDF.Define("s1", sum, { "left_x", "left_y" }).Sum("s1"), 448);
  BOOST_CHECK_EQUAL(*finalDF.Define("s4", sum, { "right_x", "left_x" }).Sum("s4"), 448);
  BOOST_CHECK_EQUAL(*finalDF.Define("s2", left, { "left_x", "left_y" }).Sum("s2"), 224);
  BOOST_CHECK_EQUAL(*finalDF.Define("s3", right, { "right_x", "left_x" }).Sum("s3"), 224);
  BOOST_CHECK_EQUAL(*indexedDF.Define("s4", sum, {"right_x", "left_x"}).Sum("s4"), 56);
  BOOST_CHECK_EQUAL(*unionDF.Define("s5", sum, {"right_x", "left_x"}).Sum("s5"), 56);
  BOOST_CHECK_EQUAL(*blockDF.Define("s5", sum, {"right_x", "left_x"}).Sum("s5"), 168);
}

BOOST_AUTO_TEST_CASE(TestSoAIntegration)
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
    BOOST_CHECK_EQUAL(row.x(), i * 10);
    BOOST_CHECK_EQUAL(row.y(), i);
    ++i;
  }
}

BOOST_AUTO_TEST_CASE(TestDataAllocatorReturnType)
{
  const Output output{"TST", "DUMMY", 0, Lifetime::Timeframe};
  // we require reference to object owned by allocator contexallocatort
  static_assert(std::is_lvalue_reference<decltype(std::declval<DataAllocator>().make<TableBuilder>(output))>::value);
}

BOOST_AUTO_TEST_CASE(TestPodInjestion)
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
    BOOST_CHECK_EQUAL(row.x(), i * 10);
    BOOST_CHECK_EQUAL(row.y(), i);
    ++i;
  }
}

BOOST_AUTO_TEST_CASE(TestColumnCount)
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
  BOOST_REQUIRE_EQUAL(TableBuilder::countColumns<Foo>(), 2);
  BOOST_REQUIRE_EQUAL(TableBuilder::countColumns<Bar>(), 3);
  BOOST_REQUIRE_EQUAL(TableBuilder::countColumns<FooBar>(), 3);
  int count = TableBuilder::countColumns<float, int>();
  BOOST_REQUIRE_EQUAL(count, 2);
  int count2 = TableBuilder::countColumns<float, int, char[3]>();
  BOOST_REQUIRE_EQUAL(count2, 3);
}

BOOST_AUTO_TEST_CASE(TestMakeFields)
{
  auto fields = TableBuilderHelpers::makeFields<int, float>({"i", "f"});
  BOOST_REQUIRE_EQUAL(fields.size(), 2);
  BOOST_REQUIRE_EQUAL(fields[0]->name(), "i");
  BOOST_REQUIRE_EQUAL(fields[1]->name(), "f");
  BOOST_REQUIRE_EQUAL(fields[0]->type()->name(), "int32");
  BOOST_REQUIRE_EQUAL(fields[1]->type()->name(), "float");
}

BOOST_AUTO_TEST_CASE(TestStreaming)
{
  using namespace o2::framework;
  /// Create a dummy table
  TableBuilder builder;
  ROOT::RDataFrame rdf(100);
  auto t = rdf.Define("x", "1")
             .Define("y", "2")
             .Define("z", "x+y");
  t.ForeachSlot(builder.persist<int, int>({"x", "z"}), {"x", "z"});

  auto table = builder.finalize();
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto creator = [&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(size);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);
  /// Writing to a stream
  auto stream = std::make_shared<arrow::io::BufferOutputStream>(buffer);
  auto outBatch = arrow::ipc::MakeStreamWriter(stream.get(), table->schema());
  auto outStatus = outBatch.ValueOrDie()->WriteTable(*table);
  if (outStatus.ok() == false) {
    throw std::runtime_error("Unable to Write table");
  }

  std::unique_ptr<fair::mq::Message> payload = buffer->Finalise();
}
