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

#include "Framework/BasicOps.h"
#include "Framework/TableBuilder.h"

#include <arrow/compute/kernel.h>
#include <arrow/status.h>
#include <arrow/util/visibility.h>
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

namespace o2::framework
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

template <typename T>
struct ARROW_EXPORT GroupByOptions {
  std::string columnName;
  T size;
};

/// Build ranges
template <typename T, typename ARRAY>
class ARROW_EXPORT SortedGroupByKernel : public arrow::compute::UnaryKernel
{
 public:
  explicit SortedGroupByKernel(GroupByOptions<T> options = {}) : mOptions(options){};
  arrow::Status Call(arrow::compute::FunctionContext* ctx,
                     arrow::compute::Datum const& table,
                     arrow::compute::Datum* outputRanges) override
  {
    using namespace arrow;
    if (table.kind() == arrow::compute::Datum::TABLE) {
      auto atable = util::get<std::shared_ptr<arrow::Table>>(table.value);
      auto columnIndex = atable->schema()->GetFieldIndex(mOptions.columnName);
      auto chunkedArray = atable->column(columnIndex);
      return doGrouping(chunkedArray, outputRanges);
    };
    return arrow::Status::OK();
  };
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
  arrow::Status doGrouping(std::shared_ptr<arrow::ChunkedArray> chunkedArray, arrow::compute::Datum* outputRanges)
  {
    o2::framework::TableBuilder builder;
    auto writer = builder.persist<T, T, T>({"start", "count", "index"});
    auto zeroChunk = std::static_pointer_cast<ARRAY>(chunkedArray->chunk(0));
    if (zeroChunk->length() == 0) {
      *outputRanges = std::move(builder.finalize());
      return arrow::Status::OK();
    }
    T currentIndex = 0;
    T currentCount = 0;
    T currentOffset = 0;
    for (auto ci = 0; ci < chunkedArray->num_chunks(); ++ci) {
      auto chunk = chunkedArray->chunk(ci);
      T const* data = std::static_pointer_cast<ARRAY>(chunk)->raw_values();
      for (auto ai = 0; ai < chunk->length(); ++ai) {
        if (currentIndex == data[ai]) {
          currentCount++;
        } else {
          writer(0, currentOffset, currentCount, currentIndex);
          currentOffset += currentCount;
          while (data[ai] - currentIndex > 1) {
            writer(0, currentOffset, 0, ++currentIndex);
          }
          currentIndex++;
          currentCount = 1;
        }
      }
    }
    writer(0, currentOffset, currentCount, currentIndex);
    while (currentIndex < mOptions.size - 1) {
      writer(0, currentOffset, 0, ++currentIndex);
    }
    *outputRanges = std::move(builder.finalize());
    return arrow::Status::OK();
  }
  std::shared_ptr<arrow::DataType> mType;
  GroupByOptions<T> mOptions;
};

/// Slice a given table is a vector of tables each containing a slice.
/// @a outputSlices the arrow tables in which the original @a inputTable
/// is split into.
/// @a offset the offset in the original table at which the corresponding
/// slice was split.
/// Slice a given table is a vector of tables each containing a slice.
template <typename T>
arrow::Status sliceByColumn(arrow::compute::FunctionContext* context,
                            std::string const& key,
                            T size,
                            arrow::compute::Datum const& inputTable,
                            std::vector<arrow::compute::Datum>* outputSlices,
                            std::vector<uint64_t>* offsets = nullptr)
{
  if (inputTable.kind() != arrow::compute::Datum::TABLE) {
    return arrow::Status::Invalid("Input Datum was not a table");
  }
  // build all the ranges on the fly.
  arrow::compute::Datum outRanges;
  auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
  o2::framework::SortedGroupByKernel<T, soa::arrow_array_for_t<T>> kernel{GroupByOptions<T>{key, size}};

  ARROW_RETURN_NOT_OK(kernel.Call(context, inputTable, &outRanges));
  auto ranges = arrow::util::get<std::shared_ptr<arrow::Table>>(outRanges.value);
  outputSlices->reserve(ranges->num_rows());
  if (offsets) {
    offsets->reserve(ranges->num_rows());
  }

  auto startChunks = ranges->column(0);
  assert(startChunks->num_chunks() == 1);
  auto countChunks = ranges->column(1);
  assert(countChunks->num_chunks() == 1);
  auto startData = std::static_pointer_cast<soa::arrow_array_for_t<T>>(startChunks->chunk(0))->raw_values();
  auto countData = std::static_pointer_cast<soa::arrow_array_for_t<T>>(countChunks->chunk(0))->raw_values();

  for (auto ri = 0; ri < ranges->num_rows(); ++ri) {
    auto start = startData[ri];
    auto count = countData[ri];
    auto schema = table->schema();
    std::vector<std::shared_ptr<arrow::ChunkedArray>> slicedColumns;
    slicedColumns.reserve(schema->num_fields());
    //    if (count != 0) {
    for (auto ci = 0; ci < schema->num_fields(); ++ci) {
      slicedColumns.emplace_back(table->column(ci)->Slice(start, count));
    }
    //    }
    outputSlices->emplace_back(arrow::compute::Datum(arrow::Table::Make(table->schema(), slicedColumns)));
    if (offsets) {
      offsets->emplace_back(start);
    }
  }
  return arrow::Status::OK();
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_KERNELS_H_
