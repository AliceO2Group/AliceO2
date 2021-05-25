// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Kernels
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/Kernels.h"
#include "Framework/TableBuilder.h"
#include "Framework/Pack.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace arrow::compute;

BOOST_AUTO_TEST_CASE(TestSlicing)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"x", "y"});

  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  rowWriter(0, 2, 7);
  rowWriter(0, 4, 8);
  rowWriter(0, 5, 9);
  rowWriter(0, 5, 10);
  auto table = builder.finalize();

  auto options = arrow::compute::CountOptions::Defaults();
  auto value_counts = arrow::compute::CallFunction("value_counts", {table->GetColumnByName("x")}, &options).ValueOrDie();
  auto array = static_cast<arrow::StructArray>(value_counts.array());

  auto arr0 = static_cast<NumericArray<Int32Type>>(array.field(0)->data());
  auto arr1 = static_cast<NumericArray<Int64Type>>(array.field(1)->data());

  std::array<int, 4> v{1, 2, 4, 5};
  std::array<int, 4> c{4, 1, 1, 2};

  for (auto i = 0; i < arr0.length(); ++i) {
    BOOST_REQUIRE_EQUAL(arr0.Value(i), v[i]);
    BOOST_REQUIRE_EQUAL(arr1.Value(i), c[i]);
  }
}

BOOST_AUTO_TEST_CASE(TestSlicingFramework)
{
  TableBuilder builder;
  auto rowWriter = builder.persist<int32_t, int32_t>({"x", "y"});

  rowWriter(0, 1, 4);
  rowWriter(0, 1, 5);
  rowWriter(0, 1, 6);
  rowWriter(0, 1, 7);
  rowWriter(0, 2, 7);
  rowWriter(0, 4, 8);
  rowWriter(0, 5, 9);
  rowWriter(0, 5, 10);
  auto table = builder.finalize();

  std::vector<uint64_t> offsets;
  std::vector<arrow::Datum> slices;
  auto status = sliceByColumn<int32_t>("x", table, 12, &slices, nullptr, &offsets);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(slices.size(), 12);
  std::array<int, 12> sizes{0, 4, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0};
  for (auto i = 0u; i < slices.size(); ++i) {
    BOOST_REQUIRE_EQUAL(slices[i].table()->num_rows(), sizes[i]);
  }
}
