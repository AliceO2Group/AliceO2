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

BOOST_AUTO_TEST_CASE(TestTableBuilderHelper)
{
  using namespace o2::framework;

  auto fields = TableBuilderHelper::makeFields<int, float>({ "x", "y" });
  BOOST_REQUIRE_EQUAL(fields.size(), 2);
  BOOST_REQUIRE_EQUAL(fields[0]->name(), "x");
  BOOST_REQUIRE_EQUAL(fields[1]->name(), "y");
  BOOST_REQUIRE_EQUAL(fields[0]->type()->id(), arrow::int32()->id());
  BOOST_REQUIRE_EQUAL(fields[1]->type()->id(), arrow::float32()->id());
}

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
