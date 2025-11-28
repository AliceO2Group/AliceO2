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

namespace {
std::shared_ptr<arrow::ChunkedArray> GetColumnByNameCI(std::shared_ptr<arrow::Table> const& table, std::string const& key)
{
  auto const& fields = table->schema()->fields();
  auto target = std::find_if(fields.begin(), fields.end(), [&key](std::shared_ptr<arrow::Field> const& field){
    return [](std::string_view const& s1, std::string_view const& s2){
      return std::ranges::equal(
        s1, s2,
        [](char c1, char c2){
          return std::tolower(static_cast<unsigned char>(c1)) == std::tolower(static_cast<unsigned char>(c2));
        }
        );
    }(field->name(), key);
  });
  return table->column(std::distance(fields.begin(), target));
}
}

void updatePairList(Cache& list, std::string const& binding, std::string const& key, bool enabled = true)
{
  auto locate = std::find_if(list.begin(), list.end(), [&binding, &key](auto const& entry) { return (entry.binding == binding) && (entry.key == key); });
  if (locate == list.end()) {
    list.emplace_back(binding, key, enabled);
  } else if (!locate->enabled && enabled) {
    locate->enabled = true;
  }
}

std::pair<int64_t, int64_t> SliceInfoPtr::getSliceFor(int value) const
{
  if ((size_t)value >= offsets.size()) {
    return {0, 0};
  }

  return {offsets[value], sizes[value]};
}

std::span<const int64_t> SliceInfoUnsortedPtr::getSliceFor(int value) const
{
  if (values.empty()) {
    return {};
  }
  if (value > values[values.size() - 1]) {
    return {};
  }

  return {(*groups)[value].data(), (*groups)[value].size()};
}

void ArrowTableSlicingCacheDef::setCaches(Cache&& bsks)
{
  bindingsKeys = bsks;
}

void ArrowTableSlicingCacheDef::setCachesUnsorted(Cache&& bsks)
{
  bindingsKeysUnsorted = bsks;
}

ArrowTableSlicingCache::ArrowTableSlicingCache(Cache&& bsks, Cache&& bsksUnsorted)
  : bindingsKeys{bsks},
    bindingsKeysUnsorted{bsksUnsorted}
{
  offsets.resize(bindingsKeys.size());
  sizes.resize(bindingsKeys.size());

  valuesUnsorted.resize(bindingsKeysUnsorted.size());
  groups.resize(bindingsKeysUnsorted.size());
}

void ArrowTableSlicingCache::setCaches(Cache&& bsks, Cache&& bsksUnsorted)
{
  bindingsKeys = bsks;
  bindingsKeysUnsorted = bsksUnsorted;
  offsets.clear();
  offsets.resize(bindingsKeys.size());
  sizes.clear();
  sizes.resize(bindingsKeys.size());
  valuesUnsorted.clear();
  valuesUnsorted.resize(bindingsKeysUnsorted.size());
  groups.clear();
  groups.resize(bindingsKeysUnsorted.size());
}

arrow::Status ArrowTableSlicingCache::updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table)
{
  offsets[pos].clear();
  sizes[pos].clear();
  if (table->num_rows() == 0) {
    return arrow::Status::OK();
  }
  auto& [b, k, e] = bindingsKeys[pos];
  if (!e) {
    throw runtime_error_f("Disabled cache %s/%s update requested", b.c_str(), k.c_str());
  }
  validateOrder(bindingsKeys[pos], table);

  int maxValue = -1;
  auto column = GetColumnByNameCI(table, k);

  // starting from the end, find the first positive value, in a sorted column it is the largest index
  for (auto iChunk = column->num_chunks() - 1; iChunk >= 0; --iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = chunk.length() - 1; iElement >= 0; --iElement) {
      auto value = chunk.Value(iElement);
      if (value < 0) {
        continue;
      } else {
        maxValue = value;
        break;
      }
    }
    if (maxValue >= 0) {
      break;
    }
  }

  offsets[pos].resize(maxValue + 1);
  sizes[pos].resize(maxValue + 1);

  // loop over the index and collect size/offset
  int lastValue = std::numeric_limits<int>::max();
  int globalRow = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (v == lastValue) {
          ++sizes[pos][v];
        } else {
          lastValue = v;
          ++sizes[pos][v];
          offsets[pos][v] = globalRow;
        }
      }
      ++globalRow;
    }
  }

  return arrow::Status::OK();
}

arrow::Status ArrowTableSlicingCache::updateCacheEntryUnsorted(int pos, const std::shared_ptr<arrow::Table>& table)
{
  valuesUnsorted[pos].clear();
  groups[pos].clear();
  if (table->num_rows() == 0) {
    return arrow::Status::OK();
  }
  auto& [b, k, e] = bindingsKeysUnsorted[pos];
  if (!e) {
    throw runtime_error_f("Disabled unsorted cache %s/%s update requested", b.c_str(), k.c_str());
  }
  auto column = GetColumnByNameCI(table, k);
  auto row = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (std::find(valuesUnsorted[pos].begin(), valuesUnsorted[pos].end(), v) == valuesUnsorted[pos].end()) {
          valuesUnsorted[pos].push_back(v);
        }
        if ((int)groups[pos].size() <= v) {
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

std::pair<int, bool> ArrowTableSlicingCache::getCachePos(const Entry& bindingKey) const
{
  auto pos = getCachePosSortedFor(bindingKey);
  if (pos != -1) {
    return {pos, true};
  }
  pos = getCachePosUnsortedFor(bindingKey);
  if (pos != -1) {
    return {pos, false};
  }
  throw runtime_error_f("%s/%s not found neither in sorted or unsorted cache", bindingKey.binding.c_str(), bindingKey.key.c_str());
}

int ArrowTableSlicingCache::getCachePosSortedFor(Entry const& bindingKey) const
{
  auto locate = std::find_if(bindingsKeys.begin(), bindingsKeys.end(), [&](Entry const& bk) { return (bindingKey.binding == bk.binding) && (bindingKey.key == bk.key); });
  if (locate != bindingsKeys.end()) {
    return std::distance(bindingsKeys.begin(), locate);
  }
  return -1;
}

int ArrowTableSlicingCache::getCachePosUnsortedFor(Entry const& bindingKey) const
{
  auto locate_unsorted = std::find_if(bindingsKeysUnsorted.begin(), bindingsKeysUnsorted.end(), [&](Entry const& bk) { return (bindingKey.binding == bk.binding) && (bindingKey.key == bk.key); });
  if (locate_unsorted != bindingsKeysUnsorted.end()) {
    return std::distance(bindingsKeysUnsorted.begin(), locate_unsorted);
  }
  return -1;
}
SliceInfoPtr ArrowTableSlicingCache::getCacheFor(Entry const& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (!s) {
    throw runtime_error_f("%s/%s is found in unsorted cache", bindingKey.binding.c_str(), bindingKey.key.c_str());
  }
  if (!bindingsKeys[p].enabled) {
    throw runtime_error_f("Disabled cache %s/%s is requested", bindingKey.binding.c_str(), bindingKey.key.c_str());
  }

  return getCacheForPos(p);
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedFor(const Entry& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (s) {
    throw runtime_error_f("%s/%s is found in sorted cache", bindingKey.binding.c_str(), bindingKey.key.c_str());
  }
  if (!bindingsKeysUnsorted[p].enabled) {
    throw runtime_error_f("Disabled unsorted cache %s/%s is requested", bindingKey.binding.c_str(), bindingKey.key.c_str());
  }

  return getCacheUnsortedForPos(p);
}

SliceInfoPtr ArrowTableSlicingCache::getCacheForPos(int pos) const
{
  return {
    gsl::span{offsets[pos].data(), offsets[pos].size()}, //
    gsl::span(sizes[pos].data(), sizes[pos].size())      //
  };
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedForPos(int pos) const
{
  return {
    {reinterpret_cast<int const*>(valuesUnsorted[pos].data()), valuesUnsorted[pos].size()},
    &(groups[pos]) //
  };
}

void ArrowTableSlicingCache::validateOrder(Entry const& bindingKey, const std::shared_ptr<arrow::Table>& input)
{
  auto const& [target, key, enabled] = bindingKey;
  auto column = o2::framework::GetColumnByNameCI(input, key);
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
