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

namespace o2::framework
{
/// Slice a given table in a vector of tables each containing a slice.
/// @a slices the arrow tables in which the original @a input
/// is split into.
/// @a offset the offset in the original table at which the corresponding
/// slice was split.
template <typename T>
auto sliceByColumn(char const* key,
                   arrow::Table const* input,
                   T fullSize,
                   std::vector<arrow::Datum>* slices,
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
  auto offset = 0;
  auto count = 0;
  auto size = values.length();
  for (auto r = 0; r < size; ++r) {
    count = counts.Value(r);
    std::shared_ptr<arrow::Schema> schema(input->schema());
    std::vector<std::shared_ptr<arrow::ChunkedArray>> sliceArray;
    sliceArray.reserve(schema->num_fields());
    for (auto ci = 0; ci < schema->num_fields(); ++ci) {
      sliceArray.emplace_back(input->column(ci)->Slice(offset, count));
    }
    slices->emplace_back(arrow::Datum(arrow::Table::Make(schema, sliceArray)));
    if (offsets) {
      offsets->emplace_back(offset);
    }
    offset += count;
    sliceArray.clear();
    if (r < size - 1) {
      auto nextValue = values.Value(r + 1);
      while (nextValue - values.Value(r) > 1) {
        for (auto ci = 0; ci < schema->num_fields(); ++ci) {
          sliceArray.emplace_back(input->column(ci)->Slice(offset, 0));
        }
        slices->emplace_back(arrow::Datum(arrow::Table::Make(schema, sliceArray)));
        if (offsets) {
          offsets->emplace_back(offset);
        }
        sliceArray.clear();
        nextValue -= 1;
      }
    }
  }
  if (values.Value(size - 1) < fullSize) {
    for (auto v = values.Value(size - 1) + 1; v < fullSize; ++v) {
      std::shared_ptr<arrow::Schema> schema(input->schema());
      std::vector<std::shared_ptr<arrow::ChunkedArray>> sliceArray;
      for (auto ci = 0; ci < schema->num_fields(); ++ci) {
        sliceArray.emplace_back(input->column(ci)->Slice(offset, 0));
      }
      slices->emplace_back(arrow::Datum(arrow::Table::Make(schema, sliceArray)));
      if (offsets) {
        offsets->emplace_back(offset);
      }
    }
  }

  return arrow::Status::OK();
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_KERNELS_H_
