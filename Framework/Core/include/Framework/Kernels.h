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

#ifndef O2_FRAMEWORK_KERNELS_H_
#define O2_FRAMEWORK_KERNELS_H_

#include "Framework/BasicOps.h"
#include "Framework/TableBuilder.h"

#include <arrow/compute/kernel.h>
#include <arrow/status.h>
#include <arrow/util/visibility.h>
#include <arrow/util/variant.h>
#include <arrow/util/config.h>

#include <string>

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;
template <typename T>
auto sliceByColumnGeneric(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  T fullSize,
  ListVector* groups,
  ListVector* unassigned = nullptr)
{
  groups->resize(fullSize);
  auto column = input->GetColumnByName(key);
  int64_t row = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<typename detail::ConversionTraits<T>::ArrowType>>(column->chunk(iChunk)->data());
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

/// Slice a given table in a vector of tables each containing a slice.
/// @a slices the arrow tables in which the original @a input
/// is split into.
/// @a offset the offset in the original table at which the corresponding
/// slice was split.
template <typename T>
auto sliceByColumn(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  T fullSize,
  std::vector<arrow::Datum>* slices,
  std::vector<uint64_t>* offsets = nullptr,
  std::vector<int>* sizes = nullptr,
  std::vector<arrow::Datum>* unassignedSlices = nullptr,
  std::vector<uint64_t>* unassignedOffsets = nullptr)
{
  arrow::Datum value_counts;
  auto column = input->GetColumnByName(key);
  for (auto i = 0; i < column->num_chunks(); ++i) {
    T prev = 0;
    T cur = 0;
    T lastNeg = -1;
    T lastPos = 0;

    auto array = static_cast<arrow::NumericArray<typename detail::ConversionTraits<T>::ArrowType>>(column->chunk(i)->data());
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
        } else if (lastPos == cur && prev < 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target, key, cur, prev);
        }
      } else {
        if (lastNeg < cur) {
          throw runtime_error_f("Table %s index %s is not sorted: next negative value %d > previous negative value %d!", target, key, cur, lastNeg);
        } else if (lastNeg == cur && prev >= 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target, key, cur, prev);
        }
      }
    }
  }
  auto options = arrow::compute::ScalarAggregateOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {column},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  auto values = static_cast<arrow::NumericArray<typename detail::ConversionTraits<T>::ArrowType>>(pair.field(0)->data());
  auto counts = static_cast<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());

  // create slices and offsets
  uint64_t offset = 0;
  auto count = 0;
  auto size = values.length();

  auto makeSlice = [&](uint64_t offset_, T count_) {
    slices->emplace_back(arrow::Datum{input->Slice(offset_, count_)});
    if (offsets) {
      offsets->emplace_back(offset_);
    }
    if (sizes) {
      sizes->emplace_back(count_);
    }
  };

  auto makeUnassignedSlice = [&](uint64_t offset_, T count_) {
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
    count = counts.Value(i);
    if (v >= 0) {
      vprev = v;
    }
    v = values.Value(i);
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
  v = values.Value(size - 1);
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

#endif // O2_FRAMEWORK_KERNELS_H_
