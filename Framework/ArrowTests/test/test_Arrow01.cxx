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
  virtual arrow::Status Call(arrow::compute::FunctionContext* ctx, const arrow::compute::Datum& input,
                             arrow::compute::Datum* output)
  {
    return arrow::Status::OK();
  }
};

BOOST_AUTO_TEST_CASE(TestArrow01)
{
  arrow::Int64Builder builder;
  builder.Append(1);
  builder.Append(2);
  builder.Append(3);
  builder.AppendNull();
  builder.Append(5);
  builder.Append(6);
  builder.Append(7);
  builder.Append(8);

  std::shared_ptr<arrow::Array> input;
  builder.Finish(&input);

  std::shared_ptr<arrow::Array> output;
  PrintingKernel kernel;
  arrow::compute::FunctionContext ctx(arrow::default_memory_pool());
  auto status = kernel.Call(&ctx, arrow::compute::Datum(input), nullptr);
  assert(status.ok() == true);
}
