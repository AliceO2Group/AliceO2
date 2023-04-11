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

#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/RuntimeError.h"

#include <arrow/compute/api_aggregate.h>
#include <arrow/compute/kernel.h>
#include <arrow/table.h>

namespace o2::framework
{

std::pair<int64_t, int64_t> SliceInfoPtr::getSliceFor(int value) const
{
  int64_t offset = 0;
  if (values.empty()) {
    return {offset, 0};
  }
  for (auto i = 0; i < values.size(); ++i) {
    if (values[i] == value) {
      return {offset, counts[i]};
    }
    offset += counts[i];
  }
  return {offset, 0};
}

void ArrowTableSlicingCacheDef::setCaches(std::vector<std::pair<std::string, std::string>>&& bsks)
{
  bindingsKeys = bsks;
}

ArrowTableSlicingCache::ArrowTableSlicingCache(std::vector<std::pair<std::string, std::string>>&& bsks)
  : bindingsKeys{bsks}
{
  values.resize(bindingsKeys.size());
  counts.resize(bindingsKeys.size());
}

void ArrowTableSlicingCache::setCaches(std::vector<std::pair<std::string, std::string>>&& bsks)
{
  bindingsKeys = bsks;
  values.clear();
  values.resize(bindingsKeys.size());
  counts.clear();
  counts.resize(bindingsKeys.size());
}

arrow::Status ArrowTableSlicingCache::updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table)
{
  if (table->num_rows() == 0) {
    values[pos].reset();
    counts[pos].reset();
    return arrow::Status::OK();
  }
  validateOrder(bindingsKeys[pos], table);
  arrow::Datum value_counts;
  auto options = arrow::compute::ScalarAggregateOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {table->GetColumnByName(bindingsKeys[pos].second)},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  values[pos].reset();
  counts[pos].reset();
  values[pos] = std::make_shared<arrow::NumericArray<arrow::Int32Type>>(pair.field(0)->data());
  counts[pos] = std::make_shared<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());
  return arrow::Status::OK();
}

SliceInfoPtr ArrowTableSlicingCache::getCacheFor(std::pair<std::string, std::string> const& bindingKey) const
{
  auto locate = std::find_if(bindingsKeys.begin(), bindingsKeys.end(), [&](std::pair<std::string, std::string> const& bk) { return (bindingKey.first == bk.first) && (bindingKey.second == bk.second); });
  if (locate == bindingsKeys.end()) {
    throw runtime_error_f("Slicing cache miss for %s/%s", bindingKey.first.c_str(), bindingKey.second.c_str());
  }
  auto i = std::distance(bindingsKeys.begin(), locate);

  if (values[i] == nullptr && counts[i] == nullptr) {
    return {
      {},
      {} //
    };
  }

  return {
    {reinterpret_cast<int const*>(values[i]->values()->data()), static_cast<size_t>(values[i]->length())},
    {reinterpret_cast<int64_t const*>(counts[i]->values()->data()), static_cast<size_t>(counts[i]->length())} //
  };
}

void ArrowTableSlicingCache::validateOrder(const std::pair<std::string, std::string>& bindingKey, const std::shared_ptr<arrow::Table>& input)
{
  auto& [target, key] = bindingKey;
  auto column = input->GetColumnByName(key.c_str());
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
          throw runtime_error_f("Table %s index %s is not sorted: next value %d < previous value %d!", target.c_str(), key.c_str(), cur, lastPos);
        }
        if (lastPos == cur && prev < 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target.c_str(), key.c_str(), cur, prev);
        }
      } else {
        if (lastNeg < cur) {
          throw runtime_error_f("Table %s index %s is not sorted: next negative value %d > previous negative value %d!", target.c_str(), key.c_str(), cur, lastNeg);
        }
        if (lastNeg == cur && prev >= 0) {
          throw runtime_error_f("Table %s index %s has a group with index %d that is split by %d", target.c_str(), key.c_str(), cur, prev);
        }
      }
    }
  }
}
} // namespace o2::framework
