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

#include <string>

namespace o2::framework
{
/// Slice a given table in a vector of tables each containing a slice.
/// @a slices the arrow tables in which the original @a input
/// is split into.
/// @a offset the offset in the original table at which the corresponding
/// slice was split.
template <typename T>
auto sliceByColumn(
  char const* key,
  std::shared_ptr<arrow::Table> const& input,
  T fullSize,
  std::vector<arrow::Datum>* slices,
  std::vector<int32_t>* vals = nullptr,
  std::vector<uint64_t>* offsets = nullptr)
{
  arrow::Datum value_counts;
  auto options = arrow::compute::CountOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {input->GetColumnByName(key)},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  auto values = static_cast<arrow::NumericArray<typename detail::ConversionTraits<T>::ArrowType>>(pair.field(0)->data());
  auto counts = static_cast<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());

  // create slices and offsets
  uint64_t offset = 0;
  auto count = 0;

  auto size = values.length();
  if (vals != nullptr) {
    for (auto i = 0; i < size; ++i) {
      vals->push_back(values.Value(i));
    }
  }

  auto makeSlice = [&](uint64_t offset_, T count) {
    slices->emplace_back(arrow::Datum{input->Slice(offset_, count)});
    if (offsets) {
      offsets->emplace_back(offset_);
    }
  };

  auto v = values.Value(0);
  auto vprev = v;
  auto vnext = v;
  for (auto i = 0; i < size - 1; ++i) {
    vprev = v;
    v = values.Value(i);
    count = counts.Value(i);
    if (v < 0) {
      vnext = values.Value(i + 1);
      while (vnext - vprev > 1) {
        makeSlice(offset, 0);
        ++vprev;
      }
      makeSlice(offset, count);
      offset += count;
      continue;
    }
    if (vprev < 0 || (v - vprev) == 1) {
      makeSlice(offset, count);
      offset += count;
      continue;
    } else {
      while (v - vprev > 1) {
        makeSlice(offset, 0);
        ++vprev;
      }
      makeSlice(offset, count);
      offset += count;
      continue;
    }
  }

  makeSlice(offset, counts.Value(size - 1));
  offset += counts.Value(size - 1);

  if (values.Value(size - 1) < fullSize - 1) {
    for (auto v = values.Value(size - 1) + 1; v < fullSize; ++v) {
      makeSlice(offset, 0);
    }
  }

  return arrow::Status::OK();
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_KERNELS_H_
