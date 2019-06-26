// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework WorkflowHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <arrow/builder.h>
#include <arrow/array.h>
#include <arrow/compute/kernel.h>
#include <arrow/compute/context.h>
#include <memory>
#include <iostream>

class ARROW_EXPORT PrintingKernel : public arrow::compute::UnaryKernel
{
 public:
  arrow::Status Call(arrow::compute::FunctionContext* ctx, const arrow::compute::Datum& input,
                     arrow::compute::Datum* output) override
  {
    return arrow::Status::OK();
  }

#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#endif // __clang__
  virtual std::shared_ptr<arrow::DataType> out_type() const
  {
    return mType;
  }
#pragma GCC diagnostic pop

  std::shared_ptr<arrow::DataType> mType;
};

BOOST_AUTO_TEST_CASE(TestArrow01)
{
  arrow::Int64Builder builder;
  BOOST_REQUIRE(builder.Append(1).ok());
  BOOST_REQUIRE(builder.Append(2).ok());
  BOOST_REQUIRE(builder.Append(3).ok());
  BOOST_REQUIRE(builder.AppendNull().ok());
  BOOST_REQUIRE(builder.Append(5).ok());
  BOOST_REQUIRE(builder.Append(6).ok());
  BOOST_REQUIRE(builder.Append(7).ok());
  BOOST_REQUIRE(builder.Append(8).ok());

  std::shared_ptr<arrow::Array> input;
  BOOST_REQUIRE(builder.Finish(&input).ok());

  std::shared_ptr<arrow::Array> output;
  PrintingKernel kernel;
  arrow::compute::FunctionContext ctx(arrow::default_memory_pool());
  auto status = kernel.Call(&ctx, arrow::compute::Datum(input), nullptr);
  assert(status.ok() == true);
}
