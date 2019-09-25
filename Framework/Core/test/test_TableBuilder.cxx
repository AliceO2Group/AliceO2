// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework TableBuilder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputRoute.h"
#include <arrow/table.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include "Framework/RCombinedDS.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include "../src/ArrowDebugHelpers.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestTableBuilder)
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<int, int>({"x", "y"});
  rowWriter(0, 0, 0);
  rowWriter(0, 1, 1);
  rowWriter(0, 2, 2);
  rowWriter(0, 3, 3);
  rowWriter(0, 4, 4);
  rowWriter(0, 5, 5);
  rowWriter(0, 6, 6);
  rowWriter(0, 7, 7);
  auto table = builder.finalize();
  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table->num_rows(), 8);
  BOOST_REQUIRE_EQUAL(table->column(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->column(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::int32()->id());
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
  BOOST_REQUIRE_EQUAL(table->column(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->column(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::int32()->id());

  for (size_t i = 0; i < 8; ++i) {
    auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(table->column(0)->data()->chunk(0));
    BOOST_CHECK_EQUAL(p->Value(i), i);
  }
}

BOOST_AUTO_TEST_CASE(TestTableBuilderMore)
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, std::string, bool>({"x", "y", "s", "b"});
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
  BOOST_REQUIRE_EQUAL(table->column(0)->name(), "x");
  BOOST_REQUIRE_EQUAL(table->column(1)->name(), "y");
  BOOST_REQUIRE_EQUAL(table->column(2)->name(), "s");
  BOOST_REQUIRE_EQUAL(table->column(3)->name(), "b");
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(2)->type()->id(), arrow::utf8()->id());
  BOOST_REQUIRE_EQUAL(table->column(3)->type()->id(), arrow::boolean()->id());
}

// Use RDataFrame to build the table
BOOST_AUTO_TEST_CASE(TestRDataFrame)
{
  using namespace o2::framework;
  TableBuilder builder;
  ROOT::RDataFrame rdf(100);
  auto t = rdf.Define("x", "1")
             .Define("y", "2")
             .Define("z", "x+y");
  t.ForeachSlot(builder.persist<int, int>({"x", "z"}), {"x", "z"});

  auto table = builder.finalize();
  BOOST_REQUIRE_EQUAL(table->num_rows(), 100);
  BOOST_REQUIRE_EQUAL(table->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::int32()->id());

  /// Writing to a stream
  std::shared_ptr<arrow::io::BufferOutputStream> stream;
  auto streamOk = arrow::io::BufferOutputStream::Create(100000, arrow::default_memory_pool(), &stream);
  BOOST_REQUIRE_EQUAL(streamOk.ok(), true);
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  auto outBatch = arrow::ipc::RecordBatchStreamWriter::Open(stream.get(), table->schema(), &writer);
  auto outStatus = writer->WriteTable(*table);
  BOOST_REQUIRE_EQUAL(writer->Close().ok(), true);

  std::shared_ptr<arrow::Buffer> inBuffer;
  BOOST_REQUIRE_EQUAL(stream->Finish(&inBuffer).ok(), true);

  BOOST_REQUIRE_EQUAL(outStatus.ok(), true);

  /// Reading back from the stream
  TableConsumer consumer(inBuffer->data(), inBuffer->size());
  std::shared_ptr<arrow::Table> inTable = consumer.asArrowTable();

  BOOST_REQUIRE_EQUAL(inTable->num_columns(), 2);
  BOOST_REQUIRE_EQUAL(inTable->num_rows(), 100);

  auto source = std::make_unique<ROOT::RDF::RArrowDS>(inTable, std::vector<std::string>{});
  ROOT::RDataFrame finalDF{std::move(source)};
  BOOST_REQUIRE_EQUAL(*finalDF.Count(), 100);
}

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
    auto p2 = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(table2->column(0)->data()->chunk(0));
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
  // FIXME: this is currently affected by a bug in RArrowDS which does not work properly when
  //        doing a rewind. Uncomment once we have a build with a ROOT which includes:
  //
  //        https://github.com/root-project/root/pull/3277
  //        https://github.com/root-project/root/pull/3428
  //
  auto sum = [](int lx, int rx) { return lx + rx; };
  auto left = [](int lx, int) { return lx; };
  auto right = [](int, int rx) { return rx; };

  //BOOST_CHECK_EQUAL(*finalDF.Define("s1", sum, { "left_x", "left_y" }).Sum("s1"), 448);
  //BOOST_CHECK_EQUAL(*finalDF.Define("s4", sum, { "right_x", "left_x" }).Sum("s4"), 448);
  //BOOST_CHECK_EQUAL(*finalDF.Define("s2", left, { "left_x", "left_y" }).Sum("s2"), 224);
  //BOOST_CHECK_EQUAL(*finalDF.Define("s3", right, { "right_x", "left_x" }).Sum("s3"), 224);
  //BOOST_CHECK_EQUAL(*indexedDF.Define("s4", sum, {"right_x", "left_x"}).Sum("s4"), 56);
  //BOOST_CHECK_EQUAL(*unionDF.Define("s5", sum, {"right_x", "left_x"}).Sum("s5"), 56);
  //BOOST_CHECK_EQUAL(*blockDF.Define("s5", sum, {"right_x", "left_x"}).Sum("s5"), 168);
}

namespace test
{
DECLARE_SOA_COLUMN(X, x, uint64_t, "x");
DECLARE_SOA_COLUMN(Y, y, uint64_t, "y");
} // namespace test

using TestTable = o2::soa::Table<test::X, test::Y>;

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
  TimingInfo* timingInfo = nullptr;
  ContextRegistry* contextes = nullptr;
  std::vector<OutputRoute> routes;
  DataAllocator allocator(timingInfo, contextes, routes);
  const Output output{"TST", "DUMMY", 0, Lifetime::Timeframe};
  // we require reference to object owned by allocator context
  static_assert(std::is_lvalue_reference<decltype(allocator.make<TableBuilder>(output))>::value);
}
