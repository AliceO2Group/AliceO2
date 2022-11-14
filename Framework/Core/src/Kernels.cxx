// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Kernels.h"
#include "Framework/BasicOps.h"
#include "Framework/RuntimeError.h"
#include <arrow/compute/kernel.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/status.h>
#include <arrow/util/visibility.h>
#include <arrow/util/config.h>
#if ARROW_VERSION_MAJOR < 10
#include <arrow/util/variant.h>
#endif
#include <arrow/util/config.h>

#include <string>

namespace o2::framework
{
arrow::Status getSlices(
  const char* key,
  std::shared_ptr<arrow::Table> const& input,
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>>& values,
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>>& counts)
{
  arrow::Datum value_counts;
  auto options = arrow::compute::ScalarAggregateOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {input->GetColumnByName(key)},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  values = std::make_shared<arrow::NumericArray<arrow::Int32Type>>(pair.field(0)->data());
  counts = std::make_shared<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());
  return arrow::Status::OK();
}

arrow::Status getSliceFor(
  int value,
  char const* key,
  std::shared_ptr<arrow::Table> const& input,
  std::shared_ptr<arrow::Table>& output,
  uint64_t& offset)
{
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>> values;
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> counts;

  auto status = getSlices(key, input, values, counts);

  for (auto slice = 0; slice < values->length(); ++slice) {
    if (values->Value(slice) == value) {
      output = input->Slice(offset, counts->Value(slice));
      return arrow::Status::OK();
    }
    offset += counts->Value(slice);
  }
  output = input->Slice(offset, 0);
  return arrow::Status::OK();
}

void sliceByColumnGeneric(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  ListVector* groups,
  ListVector* unassigned)
{
  groups->resize(fullSize);
  auto column = input->GetColumnByName(key);
  int32_t row = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (v >= groups->size()) {
          throw runtime_error_f("Table %s has an entry with index (%d) that is larger than the grouping table size (%d)", target, v, fullSize);
        }
        (*groups)[v].push_back(row);
      } else if (unassigned != nullptr) {
        auto av = std::abs(v);
        if (unassigned->size() < av + 1) {
          unassigned->resize(av + 1);
        }
        (*unassigned)[av].push_back(row);
      }
      ++row;
    }
  }
}

arrow::Status sliceByColumn(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  std::vector<arrow::Datum>* slices,
  std::vector<uint64_t>* offsets,
  std::vector<int>* sizes,
  std::vector<arrow::Datum>* unassignedSlices,
  std::vector<uint64_t>* unassignedOffsets)
{
  arrow::Datum value_counts;
  auto column = input->GetColumnByName(key);
  auto array0 = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(0)->data());
  int32_t prev = 0;
  int32_t cur = array0.Value(0);
  int32_t lastNeg = cur < 0 ? cur : 0;
  int32_t lastPos = cur < 0 ? -1 : cur;
  for (auto i = 0; i < column->num_chunks(); ++i) {
    auto array = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(i)->data());
    for (auto e = 0; e < array.length(); ++e) {
      prev = cur;
      if (prev >= 0) {
        lastPos = prev;
      } else {
        lastNeg = prev;
      }
      cur = array.Value(e);
      if (cur >= 0) {
        if (lastPos > cur) {
          throw runtime_error_f("Table %s index %s is not sorted: next value %d < previous value %d!", target, key, cur, lastPos);
        }
        if (lastPos == cur && prev < 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target, key, cur, prev);
        }
      } else {
        if (lastNeg < cur) {
          throw runtime_error_f("Table %s index %s is not sorted: next negative value %d > previous negative value %d!", target, key, cur, lastNeg);
        }
        if (lastNeg == cur && prev >= 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target, key, cur, prev);
        }
      }
    }
  }
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>> values;
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> counts;

  auto status = getSlices(key, input, values, counts);

  // create slices and offsets
  uint64_t offset = 0;
  int64_t count = 0;
  auto size = values->length();

  auto makeSlice = [&](uint64_t offset_, int64_t count_) {
    slices->emplace_back(arrow::Datum{input->Slice(offset_, count_)});
    if (offsets) {
      offsets->emplace_back(offset_);
    }
    if (sizes) {
      sizes->emplace_back(count_);
    }
  };

  auto makeUnassignedSlice = [&](uint64_t offset_, int64_t count_) {
    if (unassignedSlices) {
      unassignedSlices->emplace_back(arrow::Datum{input->Slice(offset_, count_)});
    }
    if (unassignedOffsets) {
      unassignedOffsets->emplace_back(offset_);
    }
  };

  auto v = 0;
  auto vprev = v;
  auto nzeros = 0;

  for (auto i = 0; i < size; ++i) {
    count = counts->Value(i);
    if (v >= 0) {
      vprev = v;
    }
    v = values->Value(i);
    if (v < 0) {
      makeUnassignedSlice(offset, count);
      offset += count;
      continue;
    }
    nzeros = v - vprev - ((i == 0 || slices->empty() == true) ? 0 : 1);
    for (auto z = 0; z < nzeros; ++z) {
      makeSlice(offset, 0);
    }
    makeSlice(offset, count);
    offset += count;
  }
  v = values->Value(size - 1);
  if (v >= 0) {
    vprev = v;
  }
  if (vprev < fullSize - 1) {
    for (auto v = vprev + 1; v < fullSize; ++v) {
      makeSlice(offset, 0);
    }
  }

  return arrow::Status::OK();
}
} // namespace o2::framework
