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

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ServiceHandle.h"
#include <arrow/array.h>
#include <gsl/span>

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;

struct SliceInfoPtr {
  gsl::span<int64_t const> offsets;
  gsl::span<int64_t const> sizes;

  std::pair<int64_t, int64_t> getSliceFor(int value) const;
};

struct SliceInfoUnsortedPtr {
  std::span<int const> values;
  ListVector const* groups;

  std::span<int64_t const> getSliceFor(int value) const;
};

struct Entry {
  std::string binding;
  ConcreteDataMatcher matcher;
  std::string key;
  bool enabled;

  Entry(std::string b, ConcreteDataMatcher m, std::string k, bool e = true)
    : binding{b},
      matcher{m},
      key{k},
      enabled{e}
  {
  }

  friend bool operator==(Entry const& lhs, Entry const& rhs)
  {
    return (lhs.matcher == rhs.matcher) &&
           (lhs.key == rhs.key);
  }
};

using Cache = std::vector<Entry>;

void updatePairList(Cache& list, Entry& entry);

struct ArrowTableSlicingCacheDef {
  constexpr static ServiceKind service_kind = ServiceKind::Global;
  Cache bindingsKeys;
  Cache bindingsKeysUnsorted;
  header::DataOrigin newOrigin = header::DataOrigin{"AOD"};

  void setCaches(Cache&& bsks);
  void setCachesUnsorted(Cache&& bsks);
  void setOrigin(header::DataOrigin newOrigin_ = header::DataOrigin{"AOD"})
  {
    newOrigin = newOrigin_;
  }
};

struct ArrowTableSlicingCache {
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  Cache bindingsKeys;
  std::vector<std::vector<int64_t>> offsets;
  std::vector<std::vector<int64_t>> sizes;

  Cache bindingsKeysUnsorted;
  std::vector<std::vector<int>> valuesUnsorted;
  std::vector<ListVector> groups;

  header::DataOrigin newOrigin = header::DataOrigin{"AOD"};

  ArrowTableSlicingCache(Cache&& bsks, Cache&& bsksUnsorted = {}, header::DataOrigin newOrigin_ = header::DataOrigin{"AOD"});

  // set caching information externally
  void setCaches(Cache&& bsks, Cache&& bsksUnsorted = {});

  // update slicing info cache entry (assumes it is already present)
  arrow::Status updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table);
  arrow::Status updateCacheEntryUnsorted(int pos, std::shared_ptr<arrow::Table> const& table);

  // helper to locate cache position
  std::pair<int, bool> getCachePos(Entry const& bindingKey) const;
  int getCachePosSortedFor(Entry const& bindingKey) const;
  int getCachePosUnsortedFor(Entry const& bindingKey) const;

  // get slice from cache for a given value
  SliceInfoPtr getCacheFor(Entry const& bindingKey) const;
  SliceInfoUnsortedPtr getCacheUnsortedFor(Entry const& bindingKey) const;
  SliceInfoPtr getCacheForPos(int pos) const;
  SliceInfoUnsortedPtr getCacheUnsortedForPos(int pos) const;

  static void validateOrder(Entry const& bindingKey, std::shared_ptr<arrow::Table> const& input);
};
} // namespace o2::framework

#endif // ARROWTABLESLICINGCACHE_H
