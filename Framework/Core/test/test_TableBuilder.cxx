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
#include <arrow/table.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include "Framework/RCombinedDS.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>

using namespace o2::framework;

template class std::shared_ptr<arrow::Schema>;
template class std::shared_ptr<arrow::Column>;
template class std::vector<std::shared_ptr<arrow::Column>>;
template class std::shared_ptr<arrow::Array>;
template class std::vector<std::shared_ptr<arrow::Field>>;
template class std::shared_ptr<arrow::ChunkedArray>;
template class std::shared_ptr<arrow::Table>;
template class std::shared_ptr<arrow::Field>;

BOOST_AUTO_TEST_CASE(TestTableBuilder)
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<int, int>({ "x", "y" });
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

BOOST_AUTO_TEST_CASE(TestTableBuilderMore)
{
  using namespace o2::framework;
  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, std::string, bool>({ "x", "y", "s", "b" });
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
  t.ForeachSlot(builder.persist<int, int>({ "x", "z" }), { "x", "z" });

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
  ROOT::RDataFrame finalDF{ std::move(source) };
  BOOST_REQUIRE_EQUAL(*finalDF.Count(), 100);
}

BOOST_AUTO_TEST_CASE(TestCombinedDS)
{
  using namespace o2::framework;
  TableBuilder builder1;
  auto rowWriter1 = builder1.persist<int, int>({ "x", "y" });
  for (size_t i = 0; i < 8; ++i) {
    rowWriter1(0, i, i);
  }
  auto table1 = builder1.finalize();

  TableBuilder builder2;
  auto rowWriter2 = builder2.persist<int, int>({ "x", "y" });
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
  auto combined = ROOT::RDF::MakeCrossProductDataFrame(std::move(source1), std::move(source2));
  ROOT::RDataFrame finalDF{ std::move(combined) };
  BOOST_CHECK_EQUAL(*finalDF.Count(), 64);
  // FIXME: this is currently affected by a bug in RArrowDS which does not work properly when 
  //        doing a rewind. Uncomment once we have a build with a ROOT which includes:
  //        
  //        https://github.com/root-project/root/pull/3428
  //
  //BOOST_CHECK_EQUAL(*finalDF.Define("s1", [](int lx, int rx) { return lx + rx; }, { "left_x", "left_y" }).Sum("s1"), 448);
  //BOOST_CHECK_EQUAL(*finalDF.Define("s4", [](int lx, int rx) { return lx + rx; }, { "right_x", "left_x" }).Sum("s4"), 448);
  BOOST_CHECK_EQUAL(*finalDF.Define("s2", [](int lx, int rx) { return lx; }, { "left_x", "left_y" }).Sum("s2"), 224);
  BOOST_CHECK_EQUAL(*finalDF.Define("s3", [](int lx, int rx) { return rx; }, { "right_x", "left_x" }).Sum("s3"), 224);
}
