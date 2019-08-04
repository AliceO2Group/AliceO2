// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/Kernels.h"
#include "Framework/TableBuilder.h"
#include "ArrowDebugHelpers.h"

#include <arrow/builder.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/variant.h>
#include <memory>
#include <iostream>

using namespace arrow;
using namespace arrow::compute;

namespace o2
{
namespace framework
{

HashByColumnKernel::HashByColumnKernel(HashByColumnOptions options)
  : mOptions{options}
{
}

Status HashByColumnKernel::Call(FunctionContext* ctx, Datum const& inputTable, Datum* hashes)
{
  if (inputTable.kind() == Datum::TABLE) {
    auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
    auto columnIndex = table->schema()->GetFieldIndex(mOptions.columnName);
    auto chunkedArray = table->column(columnIndex)->data();
    *hashes = std::move(chunkedArray);
    return arrow::Status::OK();
  }
  return Status::Invalid("Input Datum was not a table");
}

SortedGroupByKernel::SortedGroupByKernel(GroupByOptions options)
  : mOptions(options)
{
}

template <typename T, typename ARRAY>
Status doGrouping(std::shared_ptr<ChunkedArray> chunkedArray, Datum* outputRanges)
{
  TableBuilder builder;
  auto writer = builder.persist<uint64_t, uint64_t>({"start", "count"});
  T currentIndex = std::static_pointer_cast<ARRAY>(chunkedArray->chunk(0))->raw_values()[0];
  T currentCount = 0;
  for (size_t ci = 0; ci < chunkedArray->num_chunks(); ++ci) {
    auto chunk = chunkedArray->chunk(ci);
    T const* data = std::static_pointer_cast<ARRAY>(chunk)->raw_values();
    for (size_t ai = 0; ai < chunk->length(); ++ai) {
      if (currentIndex == data[ai]) {
        currentCount++;
      } else if (currentIndex != -1) {
        writer(0, currentIndex, currentCount);
        currentIndex = data[ai];
        currentCount = 1;
      }
    }
  }
  writer(0, currentIndex, currentCount);
  *outputRanges = std::move(builder.finalize());
  return arrow::Status::OK();
}

Status SortedGroupByKernel::Call(FunctionContext* ctx, Datum const& inputTable, Datum* outputRanges)
{
  using namespace arrow;
  if (inputTable.kind() == Datum::TABLE) {
    auto table = util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
    auto columnIndex = table->schema()->GetFieldIndex(mOptions.columnName);
    auto dataType = table->column(columnIndex)->type();
    auto chunkedArray = table->column(columnIndex)->data();
    if (dataType->id() == Type::UINT64) {
      return doGrouping<uint64_t, arrow::UInt64Array>(chunkedArray, outputRanges);
    } else if (dataType->id() == Type::INT64) {
      return doGrouping<int64_t, arrow::Int64Array>(chunkedArray, outputRanges);
    } else if (dataType->id() == Type::UINT32) {
      return doGrouping<uint32_t, arrow::UInt32Array>(chunkedArray, outputRanges);
    } else if (dataType->id() == Type::INT32) {
      return doGrouping<int32_t, arrow::Int32Array>(chunkedArray, outputRanges);
    }
  }
  return arrow::Status::OK();
}

/// Slice a given table is a vector of tables each containing a slice.
arrow::Status sliceByColumn(FunctionContext* context, std::string const& key,
                            Datum const& inputTable, std::vector<Datum>* outputSlices)
{
  if (inputTable.kind() != Datum::TABLE) {
    return Status::Invalid("Input Datum was not a table");
  }

  // build all the ranges on the fly.
  Datum outRanges;
  SortedGroupByKernel groupBy({GroupByOptions{key}});
  ARROW_RETURN_NOT_OK(groupBy.Call(context, inputTable, &outRanges));
  auto ranges = util::get<std::shared_ptr<arrow::Table>>(outRanges.value);
  outputSlices->reserve(ranges->num_rows());

  auto startChunks = ranges->column(0)->data();
  assert(startChunks->num_chunks() == 1);
  auto countChunks = ranges->column(1)->data();
  assert(countChunks->num_chunks() == 1);
  auto startData = std::static_pointer_cast<UInt64Array>(startChunks->chunk(0))->raw_values();
  auto countData = std::static_pointer_cast<UInt64Array>(countChunks->chunk(0))->raw_values();

  auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
  for (size_t ri = 0; ri < ranges->num_rows(); ++ri) {
    auto start = startData[ri];
    auto count = countData[ri];
    auto schema = table->schema();
    std::vector<std::shared_ptr<Column>> slicedColumns;
    slicedColumns.reserve(schema->num_fields());
    for (size_t ci = 0; ci < schema->num_fields(); ++ci) {
      slicedColumns.emplace_back(table->column(ci)->Slice(start, count));
    }
    outputSlices->emplace_back(Datum(arrow::Table::Make(table->schema(), slicedColumns)));
  }
  return arrow::Status::OK();
}

} // namespace framework
} // namespace o2
