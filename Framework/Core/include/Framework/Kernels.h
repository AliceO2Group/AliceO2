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

#include <arrow/table.h>

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;
/// Slice a given table uncheked, filling slice caches
arrow::Status getSlices(
  const char* key,
  std::shared_ptr<arrow::Table> const& input,
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>>& values,
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>>& counts);

/// Slice a given table unchecked
arrow::Status getSliceFor(
  int value,
  char const* key,
  std::shared_ptr<arrow::Table> const& input,
  std::shared_ptr<arrow::Table>& output,
  uint64_t& offset);

/// Slice a given table checked, for grouping association
void sliceByColumnGeneric(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  ListVector* groups,
  ListVector* unassigned = nullptr);

/// Slice a given table checked, fast, for grouping association assuming
/// the index is properly sorted
arrow::Status sliceByColumn(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  std::vector<arrow::Datum>* slices,
  std::vector<uint64_t>* offsets = nullptr,
  std::vector<int>* sizes = nullptr,
  std::vector<arrow::Datum>* unassignedSlices = nullptr,
  std::vector<uint64_t>* unassignedOffsets = nullptr);
} // namespace o2::framework

#endif // O2_FRAMEWORK_KERNELS_H_
