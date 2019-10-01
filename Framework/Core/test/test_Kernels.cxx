// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework AlgorithmSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/AnalysisDataModel.h"
#include "Framework/Kernels.h"
#include "Framework/TableBuilder.h"
#include <arrow/compute/context.h>
#include <arrow/compute/kernels/hash.h>
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace arrow::compute;

BOOST_AUTO_TEST_CASE(TestHashByColumnKernel)
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
  rowWriter(0, 2, 8);
  auto table = builder.finalize();

  arrow::compute::FunctionContext ctx;
  HashByColumnKernel kernel{{"x"}};
  std::shared_ptr<arrow::Array> out;
  auto outDatum = arrow::compute::Datum(out);
  BOOST_CHECK_EQUAL(kernel.Call(&ctx, arrow::compute::Datum(table), &outDatum).ok(), true);
  auto indices = arrow::util::get<std::shared_ptr<arrow::ChunkedArray>>(outDatum.value);
  BOOST_CHECK_EQUAL(indices->length(), table->num_rows());

  std::shared_ptr<arrow::Array> uniqueValues;
  BOOST_CHECK_EQUAL(Unique(&ctx, arrow::compute::Datum(indices), &uniqueValues).ok(), true);
  BOOST_REQUIRE(uniqueValues.get() != nullptr);
  BOOST_CHECK_EQUAL(uniqueValues->length(), 3);

  arrow::compute::Datum outRanges;
  SortedGroupByKernel groupBy{{"x"}};
  BOOST_CHECK_EQUAL(groupBy.Call(&ctx, arrow::compute::Datum(table), &outRanges).ok(), true);
  auto result = arrow::util::get<std::shared_ptr<arrow::Table>>(outRanges.value);
  BOOST_REQUIRE(result.get() != nullptr);
  BOOST_CHECK_EQUAL(result->num_rows(), 3);

  std::vector<Datum> splitted;
  BOOST_CHECK_EQUAL(sliceByColumn(&ctx, "x", arrow::compute::Datum(table), &splitted).ok(), true);
  BOOST_REQUIRE_EQUAL(splitted.size(), 3);
  BOOST_CHECK_EQUAL(util::get<std::shared_ptr<Table>>(splitted[0].value)->num_rows(), 4);
  BOOST_CHECK_EQUAL(util::get<std::shared_ptr<Table>>(splitted[1].value)->num_rows(), 4);
  BOOST_CHECK_EQUAL(util::get<std::shared_ptr<Table>>(splitted[2].value)->num_rows(), 1);
}

BOOST_AUTO_TEST_CASE(TestWithSOATables)
{
  using namespace o2;
  TableBuilder builder1;
  auto collisionsCursor = builder1.cursor<aod::Collisions>();
  collisionsCursor(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
  collisionsCursor(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
  auto collisions = builder1.finalize();
  TableBuilder builder2;
  auto tracksCursor = builder2.cursor<aod::Tracks>();
  tracksCursor(0, 0, 2, 3, 4, 5, 6, 7, 8);
  tracksCursor(0, 1, 2, 3, 4, 5, 6, 7, 8);
  tracksCursor(0, 1, 2, 3, 4, 5, 6, 7, 8);
  auto tracks = builder2.finalize();

  arrow::compute::FunctionContext ctx;
  arrow::compute::Datum outRanges;
  SortedGroupByKernel groupBy{{"fID4Collisions"}};
  BOOST_CHECK_EQUAL(groupBy.Call(&ctx, arrow::compute::Datum(tracks), &outRanges).ok(), true);
  auto result = arrow::util::get<std::shared_ptr<arrow::Table>>(outRanges.value);
  BOOST_REQUIRE(result.get() != nullptr);
  BOOST_CHECK_EQUAL(result->num_rows(), 2);

  std::vector<Datum> splitted;
  BOOST_CHECK_EQUAL(sliceByColumn(&ctx, "fID4Collisions", arrow::compute::Datum(tracks), &splitted).ok(), true);
  BOOST_REQUIRE_EQUAL(splitted.size(), 2);
  BOOST_CHECK_EQUAL(util::get<std::shared_ptr<Table>>(splitted[0].value)->num_rows(), 1);
  BOOST_CHECK_EQUAL(util::get<std::shared_ptr<Table>>(splitted[1].value)->num_rows(), 2);
}
