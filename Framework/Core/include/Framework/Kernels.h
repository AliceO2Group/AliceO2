// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_KERNELS_H_
#define O2_FRAMEWORK_KERNELS_H_

#include "arrow/compute/kernel.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"
#include <arrow/util/variant.h>

#include <string>

namespace arrow
{
class Array;
class DataType;

namespace compute
{
class FunctionContext;
} // namespace compute
} // namespace arrow

namespace o2
{
namespace framework
{

struct ARROW_EXPORT HashByColumnOptions {
  std::string columnName;
};

/// A kernel which provides a unique hash based on the contents of a given
/// column
/// * The input datum has to be a table like object.
/// * The output datum will be a column of integers which define the
///   category.
class ARROW_EXPORT HashByColumnKernel : public arrow::compute::UnaryKernel
{
 public:
  HashByColumnKernel(HashByColumnOptions options = {});
  arrow::Status Call(arrow::compute::FunctionContext* ctx,
                     arrow::compute::Datum const& table,
                     arrow::compute::Datum* hashes) override;

#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#endif // __clang__

  std::shared_ptr<arrow::DataType> out_type() const final
  {
    return mType;
  }
#pragma GCC diagnostic pop

 private:
  HashByColumnOptions mOptions;
  std::shared_ptr<arrow::DataType> mType;
};

struct ARROW_EXPORT GroupByOptions {
  std::string columnName;
};

/// Build ranges
class ARROW_EXPORT SortedGroupByKernel : public arrow::compute::UnaryKernel
{
 public:
  explicit SortedGroupByKernel(GroupByOptions options = {});
  arrow::Status Call(arrow::compute::FunctionContext* ctx,
                     arrow::compute::Datum const& table,
                     arrow::compute::Datum* outputRanges) override;
#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#endif // __clang__
  std::shared_ptr<arrow::DataType> out_type() const final
  {
    return mType;
  }
#pragma GCC diagnostic pop

 private:
  std::shared_ptr<arrow::DataType> mType;
  GroupByOptions mOptions;
};

/// Slice a given table is a vector of tables each containing a slice.
arrow::Status sliceByColumn(arrow::compute::FunctionContext* context,
                            std::string const& key,
                            arrow::compute::Datum const& inputTable,
                            std::vector<arrow::compute::Datum>* outputSlices);

} // namespace framework
} // namespace o2

#endif // O2_FRAMEWORK_KERNELS_H_
