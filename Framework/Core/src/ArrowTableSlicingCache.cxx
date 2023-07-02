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
  int64_t p = static_cast<int64_t>(values.size()) - 1;
  while (values[p] < 0) {
    --p;
    if (p < 0) {
      return {offset, 0};
    }
  }

  if (value > values[p]) {
    return {offset, 0};
  }

  for (auto i = 0U; i < values.size(); ++i) {
    if (values[i] == value) {
      return {offset, counts[i]};
    }
    offset += counts[i];
  }
  return {offset, 0};
}

gsl::span<const int64_t> SliceInfoUnsortedPtr::getSliceFor(int value) const
{
  if (values.empty()) {
    return {};
  }
  if (value > values[values.size() - 1]) {
    return {};
  }

  return {(*groups)[value].data(), (*groups)[value].size()};
}

void ArrowTableSlicingCacheDef::setCaches(std::vector<StringPair>&& bsks)
{
  bindingsKeys = bsks;
}

void ArrowTableSlicingCacheDef::setCachesUnsorted(std::vector<StringPair>&& bsks)
{
  bindingsKeysUnsorted = bsks;
}

ArrowTableSlicingCache::ArrowTableSlicingCache(std::vector<StringPair>&& bsks, std::vector<StringPair>&& bsksUnsorted)
  : bindingsKeys{bsks},
    bindingsKeysUnsorted{bsksUnsorted}
{
  values.resize(bindingsKeys.size());
  counts.resize(bindingsKeys.size());

  valuesUnsorted.resize(bindingsKeysUnsorted.size());
  groups.resize(bindingsKeysUnsorted.size());
}

void ArrowTableSlicingCache::setCaches(std::vector<StringPair>&& bsks, std::vector<StringPair>&& bsksUnsorted)
{
  bindingsKeys = bsks;
  bindingsKeysUnsorted = bsksUnsorted;
  values.clear();
  values.resize(bindingsKeys.size());
  counts.clear();
  counts.resize(bindingsKeys.size());
  valuesUnsorted.clear();
  valuesUnsorted.resize(bindingsKeysUnsorted.size());
  groups.clear();
  groups.resize(bindingsKeysUnsorted.size());
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

arrow::Status ArrowTableSlicingCache::updateCacheEntryUnsorted(int pos, const std::shared_ptr<arrow::Table>& table)
{
  valuesUnsorted[pos].clear();
  groups[pos].clear();
  if (table->num_rows() == 0) {
    return arrow::Status::OK();
  }
  auto& [b, k] = bindingsKeysUnsorted[pos];
  auto column = table->GetColumnByName(k);
  auto row = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (std::find(valuesUnsorted[pos].begin(), valuesUnsorted[pos].end(), v) == valuesUnsorted[pos].end()) {
          valuesUnsorted[pos].push_back(v);
        }
        if (groups[pos].size() <= v) {
          groups[pos].resize(v + 1);
        }
        (groups[pos])[v].push_back(row);
      }
      ++row;
    }
  }
  std::sort(valuesUnsorted[pos].begin(), valuesUnsorted[pos].end());
  return arrow::Status::OK();
}

std::pair<int, bool> ArrowTableSlicingCache::getCachePos(const StringPair& bindingKey) const
{
  auto pos = getCachePosSortedFor(bindingKey);
  if (pos != -1) {
    return {pos, true};
  }
  pos = getCachePosUnsortedFor(bindingKey);
  if (pos != -1) {
    return {pos, false};
  }
  throw runtime_error_f("%s/%s not found neither in sorted or unsorted cache", bindingKey.first.c_str(), bindingKey.second.c_str());
}

int ArrowTableSlicingCache::getCachePosSortedFor(StringPair const& bindingKey) const
{
  auto locate = std::find_if(bindingsKeys.begin(), bindingsKeys.end(), [&](StringPair const& bk) { return (bindingKey.first == bk.first) && (bindingKey.second == bk.second); });
  if (locate != bindingsKeys.end()) {
    return std::distance(bindingsKeys.begin(), locate);
  }
  return -1;
}

int ArrowTableSlicingCache::getCachePosUnsortedFor(StringPair const& bindingKey) const
{
  auto locate_unsorted = std::find_if(bindingsKeysUnsorted.begin(), bindingsKeysUnsorted.end(), [&](StringPair const& bk) { return (bindingKey.first == bk.first) && (bindingKey.second == bk.second); });
  if (locate_unsorted != bindingsKeysUnsorted.end()) {
    return std::distance(bindingsKeysUnsorted.begin(), locate_unsorted);
  }
  return -1;
}
SliceInfoPtr ArrowTableSlicingCache::getCacheFor(StringPair const& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (!s) {
    throw runtime_error_f("%s/%s is found in unsorted cache", bindingKey.first.c_str(), bindingKey.second.c_str());
  }

  return getCacheForPos(p);
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedFor(const StringPair& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (s) {
    throw runtime_error_f("%s/%s is found in sorted cache", bindingKey.first.c_str(), bindingKey.second.c_str());
  }

  return getCacheUnsortedForPos(p);
}

SliceInfoPtr ArrowTableSlicingCache::getCacheForPos(int pos) const
{
  if (values[pos] == nullptr && counts[pos] == nullptr) {
    return {
      {},
      {} //
    };
  }

  return {
    {reinterpret_cast<int const*>(values[pos]->values()->data()), static_cast<size_t>(values[pos]->length())},
    {reinterpret_cast<int64_t const*>(counts[pos]->values()->data()), static_cast<size_t>(counts[pos]->length())} //
  };
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedForPos(int pos) const
{
  return {
    {reinterpret_cast<int const*>(valuesUnsorted[pos].data()), valuesUnsorted[pos].size()},
    &(groups[pos]) //
  };
}

void ArrowTableSlicingCache::validateOrder(StringPair const& bindingKey, const std::shared_ptr<arrow::Table>& input)
{
  auto const& [target, key] = bindingKey;
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
