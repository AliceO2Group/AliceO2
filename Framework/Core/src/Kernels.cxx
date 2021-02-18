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
    auto chunkedArray = getBackendColumnData(table->column(columnIndex));
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
  auto writer = builder.persist<uint64_t, uint64_t, uint64_t>({"start", "count", "index"});
  auto zeroChunk = std::static_pointer_cast<ARRAY>(chunkedArray->chunk(0));
  if (zeroChunk->length() == 0) {
    *outputRanges = std::move(builder.finalize());
    return arrow::Status::OK();
  }
  uint64_t currentIndex = 0;
  uint64_t currentCount = 0;
  uint64_t currentOffset = 0;
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
  *outputRanges = std::move(builder.finalize());
  return arrow::Status::OK();
}

Status SortedGroupByKernel::Call(FunctionContext*, Datum const& inputTable, Datum* outputRanges)
{
  using namespace arrow;
  if (inputTable.kind() == Datum::TABLE) {
    auto table = util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
    auto columnIndex = table->schema()->GetFieldIndex(mOptions.columnName);
    auto dataType = table->column(columnIndex)->type();
    auto chunkedArray = table->column(columnIndex)->data();
    switch (dataType->id()) {
      case Type::INT32:
        return doGrouping<int32_t, arrow::Int32Array>(chunkedArray, outputRanges);
      case Type::UINT32:
        return doGrouping<uint32_t, arrow::UInt32Array>(chunkedArray, outputRanges);
      case Type::INT64:
        return doGrouping<int64_t, arrow::Int64Array>(chunkedArray, outputRanges);
      case Type::UINT64:
        return doGrouping<uint64_t, arrow::UInt64Array>(chunkedArray, outputRanges);
      default:
        return arrow::Status::TypeError("Unsupported index type");
    }
  }
  return arrow::Status::OK();
}

/// Slice a given table is a vector of tables each containing a slice.
arrow::Status sliceByColumn(FunctionContext* context, std::string const& key,
                            Datum const& inputTable, std::vector<Datum>* outputSlices,
                            std::vector<uint64_t>* offsets)
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
  if (offsets) {
    offsets->reserve(ranges->num_rows());
  }

  auto startChunks = getBackendColumnData(ranges->column(0));
  assert(startChunks->num_chunks() == 1);
  auto countChunks = getBackendColumnData(ranges->column(1));
  assert(countChunks->num_chunks() == 1);
  auto startData = std::static_pointer_cast<UInt64Array>(startChunks->chunk(0))->raw_values();
  auto countData = std::static_pointer_cast<UInt64Array>(countChunks->chunk(0))->raw_values();

  auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
  for (size_t ri = 0; ri < ranges->num_rows(); ++ri) {
    auto start = startData[ri];
    auto count = countData[ri];
    auto schema = table->schema();
    std::vector<std::shared_ptr<BackendColumnType>> slicedColumns;
    slicedColumns.reserve(schema->num_fields());
    for (size_t ci = 0; ci < schema->num_fields(); ++ci) {
      slicedColumns.emplace_back(table->column(ci)->Slice(start, count));
    }
    outputSlices->emplace_back(Datum(arrow::Table::Make(table->schema(), slicedColumns)));
    if (offsets) {
      offsets->emplace_back(start);
    }
  }
  return arrow::Status::OK();
}

} // namespace framework
} // namespace o2
