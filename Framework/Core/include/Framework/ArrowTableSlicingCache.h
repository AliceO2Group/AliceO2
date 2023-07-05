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

#ifndef ARROWTABLESLICINGCACHE_H
#define ARROWTABLESLICINGCACHE_H

#include "Framework/ServiceHandle.h"
#include <arrow/array.h>
#include <string_view>
#include <gsl/span>

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;

struct SliceInfoPtr {
  gsl::span<int const> values;
  gsl::span<int64_t const> counts;

  std::pair<int64_t, int64_t> getSliceFor(int value) const;
};

struct SliceInfoUnsortedPtr {
  gsl::span<int const> values;
  ListVector const* groups;

  gsl::span<int64_t const> getSliceFor(int value) const;
};

using StringPair = std::pair<std::string, std::string>;

struct ArrowTableSlicingCacheDef {
  constexpr static ServiceKind service_kind = ServiceKind::Global;
  std::vector<StringPair> bindingsKeys;
  std::vector<StringPair> bindingsKeysUnsorted;

  void setCaches(std::vector<StringPair>&& bsks);
  void setCachesUnsorted(std::vector<StringPair>&& bsks);
};

struct ArrowTableSlicingCache {
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  std::vector<StringPair> bindingsKeys;
  std::vector<std::shared_ptr<arrow::NumericArray<arrow::Int32Type>>> values;
  std::vector<std::shared_ptr<arrow::NumericArray<arrow::Int64Type>>> counts;

  std::vector<StringPair> bindingsKeysUnsorted;
  std::vector<std::vector<int>> valuesUnsorted;
  std::vector<ListVector> groups;

  ArrowTableSlicingCache(std::vector<StringPair>&& bsks, std::vector<StringPair>&& bsksUnsorted = {});

  // set caching information externally
  void setCaches(std::vector<StringPair>&& bsks, std::vector<StringPair>&& bsksUnsorted = {});

  // update slicing info cache entry (assumes it is already present)
  arrow::Status updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table);
  arrow::Status updateCacheEntryUnsorted(int pos, std::shared_ptr<arrow::Table> const& table);

  // helper to locate cache position
  std::pair<int, bool> getCachePos(StringPair const& bindingKey) const;
  int getCachePosSortedFor(StringPair const& bindingKey) const;
  int getCachePosUnsortedFor(StringPair const& bindingKey) const;

  // get slice from cache for a given value
  SliceInfoPtr getCacheFor(StringPair const& bindingKey) const;
  SliceInfoUnsortedPtr getCacheUnsortedFor(StringPair const& bindingKey) const;
  SliceInfoPtr getCacheForPos(int pos) const;
  SliceInfoUnsortedPtr getCacheUnsortedForPos(int pos) const;

  static void validateOrder(StringPair const& bindingKey, std::shared_ptr<arrow::Table> const& input);
};
} // namespace o2::framework

#endif // ARROWTABLESLICINGCACHE_H
