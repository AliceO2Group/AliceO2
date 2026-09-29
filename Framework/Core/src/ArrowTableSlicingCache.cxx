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
#include "Framework/DataSpecUtils.h"

#include <arrow/compute/api_aggregate.h>
#include <arrow/compute/kernel.h>
#include <arrow/table.h>

#include <numeric>

namespace o2::framework
{

namespace
{
// ASCII-only lowercase. Column labels are plain identifiers, so we deliberately
// avoid the locale-aware std::tolower: it goes through the C locale facet on
// every character and dominated getIndexFromLabel in profiles.
constexpr inline char asciiToLower(char c)
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
}

arrow::ChunkedArray* getIndexFromLabel(arrow::Table* table, std::string_view label)
{
  auto field = std::ranges::find_if(table->schema()->fields(), [label](std::shared_ptr<arrow::Field> const& field) {
    std::string_view name = field->name();
    return name == label ||
           std::ranges::equal(label, name, [](char c1, char c2) {
             return asciiToLower(c1) == asciiToLower(c2);
           });
  });
  if (field == table->schema()->fields().end()) {
    throw runtime_error_f("Unable to find column with label %s.", label);
  }
  return table->column(std::distance(table->schema()->fields().begin(), field)).get();
}

// collect offset and size of each group of a sorted index column
void fillSorted(arrow::ChunkedArray* column, std::vector<int64_t>& offsets, std::vector<int64_t>& sizes)
{
  int maxValue = -1;
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

  offsets.resize(maxValue + 1);
  sizes.resize(maxValue + 1);

  // loop over the index and collect size/offset
  int lastValue = std::numeric_limits<int>::max();
  int globalRow = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (v == lastValue) {
          ++sizes[v];
        } else {
          lastValue = v;
          ++sizes[v];
          offsets[v] = globalRow;
        }
      }
      ++globalRow;
    }
  }
}
} // namespace

InputSpec inputForEntry(Entry const& entry, bool sorted)
{
  // the slice info table inherits the sliced table binding and origin, while using hash
  // of original description and normalized column name as a new description
  auto& [origin, description, version] = entry.matcher;
  auto newdescription = std::string{description.str} + "/" + entry.key;
  auto hash = runtime_hash(newdescription.c_str());
  auto d = header::DataDescription{"initial"};
  d.runtimeInit(std::to_string(hash).c_str());
  InputSpec result{entry.binding + "_Slice", origin, d, version};
  // add metadata to retrieve the original table
  result.metadata.emplace_back(
    o2::framework::ConfigParamSpec{fmt::format("slice-source:{}", entry.binding),
                                   framework::VariantType::String,
                                   fmt::format("{}/{}/{}/{}", entry.binding, origin.as<std::string>(), description.as<std::string>(), version),
                                   {"\"\""}}
    );
  result.metadata.emplace_back(
    o2::framework::ConfigParamSpec{"slice-key", framework::VariantType::String, entry.key, {"\"\""}}
    );
  result.metadata.emplace_back(
    o2::framework::ConfigParamSpec{"sorted", framework::VariantType::Bool, sorted, {"\"\""}}
    );

  return result;
}

ConcreteDataMatcher matcherForEntry(Entry const& entry)
{
  return matcherForMatcherAndKey(entry.matcher, entry.key);
}

ConcreteDataMatcher matcherForMatcherAndKey(ConcreteDataMatcher const& matcher, std::string const& key)
{
  auto& [origin, description, version] = matcher;
  auto newdescription = std::string{description.str} + "/" + key;
  auto hash = runtime_hash(newdescription.c_str());
  auto d = header::DataDescription{"initial"};
  d.runtimeInit(std::to_string(hash).c_str());
  return {origin, d, version};
}

void updatePairList(Cache& list, Entry& entry)
{
  auto locate = std::find(list.begin(), list.end(), entry);
  if (locate == list.end()) {
    list.emplace_back(entry);
  } else if (!locate->enabled && entry.enabled) {
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

std::span<int64_t const> SliceInfoUnsortedPtr::getSliceFor(int value) const
{
  if (value < 0 || (size_t)value + 1 >= offsets.size()) {
    return {};
  }
  return rows.subspan(offsets[value], offsets[value + 1] - offsets[value]);
}

std::shared_ptr<arrow::Schema> SliceInfo::sortedSchema()
{
  return arrow::schema({arrow::field(offsetsLabel, arrow::int64()), arrow::field(sizesLabel, arrow::int64())});
}

std::shared_ptr<arrow::Schema> SliceInfo::unsortedSchema()
{
  return arrow::schema({arrow::field(rowsLabel, arrow::list(arrow::int64()))});
}

std::shared_ptr<arrow::Table> SliceInfo::makeSorted(Entry const& entry, std::shared_ptr<arrow::Table> const& source)
{
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  if (source->num_rows() != 0) {
    ArrowTableSlicingCache::validateOrder(entry, source);
    fillSorted(getIndexFromLabel(source.get(), entry.key), offsets, sizes);
  }
  auto length = static_cast<int64_t>(offsets.size());
  return arrow::Table::Make(sortedSchema(),
                            {std::make_shared<arrow::Int64Array>(length, arrow::Buffer::FromVector(std::move(offsets))),
                             std::make_shared<arrow::Int64Array>(length, arrow::Buffer::FromVector(std::move(sizes)))},
                            length);
}

std::shared_ptr<arrow::Table> SliceInfo::makeUnsorted(Entry const& entry, std::shared_ptr<arrow::Table> const& source)
{
  std::vector<int32_t> offsets{0};
  std::vector<int64_t> rows;
  if (source->num_rows() != 0) {
    auto column = getIndexFromLabel(source.get(), entry.key);
    // count the rows in each group
    std::vector<int32_t> counts;
    for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
      auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
      for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
        auto v = chunk.Value(iElement);
        if (v >= 0) {
          if ((int)counts.size() <= v) {
            counts.resize(v + 1);
          }
          ++counts[v];
        }
      }
    }
    offsets.resize(counts.size() + 1);
    std::inclusive_scan(counts.begin(), counts.end(), offsets.begin() + 1);
    rows.resize(offsets.back());

    // place the row numbers of each group, reusing counts as fill positions
    std::copy(offsets.begin(), offsets.end() - 1, counts.begin());
    int64_t row = 0;
    for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
      auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
      for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
        auto v = chunk.Value(iElement);
        if (v >= 0) {
          rows[counts[v]++] = row;
        }
        ++row;
      }
    }
  }
  auto length = static_cast<int64_t>(offsets.size()) - 1;
  auto nRows = static_cast<int64_t>(rows.size());
  auto values = std::make_shared<arrow::Int64Array>(nRows, arrow::Buffer::FromVector(std::move(rows)));
  return arrow::Table::Make(unsortedSchema(),
                            {std::make_shared<arrow::ListArray>(arrow::list(arrow::int64()), length, arrow::Buffer::FromVector(std::move(offsets)), values)},
                            length);
}

SliceInfoPtr SliceInfo::readSorted(std::shared_ptr<arrow::Table> const& table)
{
  if (table->num_rows() == 0) {
    return {};
  }
  auto offsets = std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
  auto sizes = std::static_pointer_cast<arrow::Int64Array>(table->column(1)->chunk(0));
  return {
    gsl::span{offsets->raw_values(), (size_t)offsets->length()}, //
    gsl::span{sizes->raw_values(), (size_t)sizes->length()}      //
  };
}

SliceInfoUnsortedPtr SliceInfo::readUnsorted(std::shared_ptr<arrow::Table> const& table)
{
  if (table->num_rows() == 0) {
    return {};
  }
  auto list = std::static_pointer_cast<arrow::ListArray>(table->column(0)->chunk(0));
  auto values = std::static_pointer_cast<arrow::Int64Array>(list->values());
  return {
    {list->raw_value_offsets(), (size_t)list->length() + 1}, //
    {values->raw_values(), (size_t)values->length()}         //
  };
}

void ArrowTableSlicingCacheDef::setCaches(Cache&& bsks)
{
  bindingsKeys = bsks;
}

void ArrowTableSlicingCacheDef::setCachesUnsorted(Cache&& bsks)
{
  bindingsKeysUnsorted = bsks;
}

ArrowTableSlicingCache::ArrowTableSlicingCache(Cache&& bsks, Cache&& bsksUnsorted, header::DataOrigin newOrigin_)
  : bindingsKeys{bsks},
    bindingsKeysUnsorted{bsksUnsorted},
    newOrigin{newOrigin_}
{
  clearCacheEntries();
}

void ArrowTableSlicingCache::setCaches(Cache&& bsks, Cache&& bsksUnsorted)
{
  bindingsKeys = bsks;
  bindingsKeysUnsorted = bsksUnsorted;
  clearCacheEntries();
}

void ArrowTableSlicingCache::clearCacheEntries()
{
  sliceInfos.assign(bindingsKeys.size(), nullptr);
  sliceInfoPtrs.assign(bindingsKeys.size(), {});
  sliceInfosUnsorted.assign(bindingsKeysUnsorted.size(), nullptr);
  sliceInfoPtrsUnsorted.assign(bindingsKeysUnsorted.size(), {});
}

void ArrowTableSlicingCache::setCacheEntry(int pos, std::shared_ptr<arrow::Table> sliceInfo)
{
  sliceInfoPtrs[pos] = SliceInfo::readSorted(sliceInfo);
  sliceInfos[pos] = std::move(sliceInfo);
}

void ArrowTableSlicingCache::setCacheEntryUnsorted(int pos, std::shared_ptr<arrow::Table> sliceInfo)
{
  sliceInfoPtrsUnsorted[pos] = SliceInfo::readUnsorted(sliceInfo);
  sliceInfosUnsorted[pos] = std::move(sliceInfo);
}

arrow::Status ArrowTableSlicingCache::updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table)
{
  auto& [b, m, k, e] = bindingsKeys[pos];
  if (!e) {
    throw runtime_error_f("Disabled cache (%s) %s/%s update requested", DataSpecUtils::describe(m).c_str(), b.c_str(), k.c_str());
  }
  setCacheEntry(pos, SliceInfo::makeSorted(bindingsKeys[pos], table));
  return arrow::Status::OK();
}

arrow::Status ArrowTableSlicingCache::updateCacheEntryUnsorted(int pos, std::shared_ptr<arrow::Table> const& table)
{
  auto& [b, m, k, e] = bindingsKeysUnsorted[pos];
  if (!e) {
    throw runtime_error_f("Disabled unsorted cache (%s) %s/%s update requested", DataSpecUtils::describe(m).c_str(), b.c_str(), k.c_str());
  }
  setCacheEntryUnsorted(pos, SliceInfo::makeUnsorted(bindingsKeysUnsorted[pos], table));
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
  throw runtime_error_f("(%s) %s/%s not found neither in sorted or unsorted cache", DataSpecUtils::describe(bindingKey.matcher).c_str(), bindingKey.binding.c_str(), bindingKey.key.c_str());
}

int ArrowTableSlicingCache::getCachePosSortedFor(Entry const& bindingKey) const
{
  auto locate = std::ranges::find(bindingsKeys, bindingKey);
  if (locate != bindingsKeys.end()) {
    return std::distance(bindingsKeys.begin(), locate);
  }
  return -1;
}

int ArrowTableSlicingCache::getCachePosUnsortedFor(Entry const& bindingKey) const
{
  auto locate_unsorted = std::ranges::find(bindingsKeysUnsorted, bindingKey);
  if (locate_unsorted != bindingsKeysUnsorted.end()) {
    return std::distance(bindingsKeysUnsorted.begin(), locate_unsorted);
  }
  return -1;
}
SliceInfoPtr ArrowTableSlicingCache::getCacheFor(Entry const& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (!s) {
    throw runtime_error_f("(%s) %s/%s is found in unsorted cache", DataSpecUtils::describe(bindingKey.matcher).c_str(), bindingKey.binding.c_str(), bindingKey.key.c_str());
  }
  if (!bindingsKeys[p].enabled) {
    throw runtime_error_f("Disabled cache (%s) %s/%s is requested", DataSpecUtils::describe(bindingKey.matcher).c_str(), bindingKey.binding.c_str(), bindingKey.key.c_str());
  }

  return getCacheForPos(p);
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedFor(const Entry& bindingKey) const
{
  auto [p, s] = getCachePos(bindingKey);
  if (s) {
    throw runtime_error_f("(%s) %s/%s is found in sorted cache", DataSpecUtils::describe(bindingKey.matcher).c_str(), bindingKey.binding.c_str(), bindingKey.key.c_str());
  }
  if (!bindingsKeysUnsorted[p].enabled) {
    throw runtime_error_f("Disabled unsorted cache (%s) %s/%s is requested", DataSpecUtils::describe(bindingKey.matcher).c_str(), bindingKey.binding.c_str(), bindingKey.key.c_str());
  }

  return getCacheUnsortedForPos(p);
}

SliceInfoPtr ArrowTableSlicingCache::getCacheForPos(int pos) const
{
  return sliceInfoPtrs[pos];
}

SliceInfoUnsortedPtr ArrowTableSlicingCache::getCacheUnsortedForPos(int pos) const
{
  return sliceInfoPtrsUnsorted[pos];
}

void ArrowTableSlicingCache::validateOrder(Entry const& bindingKey, const std::shared_ptr<arrow::Table>& input)
{
  auto const& [target, matcher, key, enabled] = bindingKey;
  if (!enabled) {
    return;
  }
  auto column = getIndexFromLabel(input.get(), key);
  auto array = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(0)->data());
  int32_t cur = array.Value(0);
  int32_t lastNeg = cur < 0 ? cur : 0;
  int32_t lastPos = cur < 0 ? -1 : cur;
  for (auto i = 0; i < column->num_chunks(); ++i) {
    array = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(i)->data());
    for (auto e = 0; e < array.length(); ++e) {
      int32_t prev = cur;
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
