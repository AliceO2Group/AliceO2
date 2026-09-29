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
struct SliceInfoPtr {
  gsl::span<int64_t const> offsets;
  gsl::span<int64_t const> sizes;

  std::pair<int64_t, int64_t> getSliceFor(int value) const;
};

/// view of an unsorted slice-info table: rows of group v are rows[offsets[v], offsets[v + 1])
struct SliceInfoUnsortedPtr {
  std::span<int32_t const> offsets;
  std::span<int64_t const> rows;

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

/// Layout of the slice-info tables produced by the internal slicer device.
/// Row v describes the group with index value v, for v in [0, max index value];
/// rows with negative index values do not belong to any group.
struct SliceInfo {
  /// sorted: group v is the contiguous range [fOffset, fOffset + fSize)
  static constexpr const char* offsetsLabel = "fOffset"; // int64
  static constexpr const char* sizesLabel = "fSize";     // int64
  /// unsorted: group v is the list of row numbers in fRows
  static constexpr const char* rowsLabel = "fRows"; // list<int64>

  static std::shared_ptr<arrow::Schema> sortedSchema();
  static std::shared_ptr<arrow::Schema> unsortedSchema();

  /// build slice-info tables for the index column entry.key of the source table
  static std::shared_ptr<arrow::Table> makeSorted(Entry const& entry, std::shared_ptr<arrow::Table> const& source);
  static std::shared_ptr<arrow::Table> makeUnsorted(Entry const& entry, std::shared_ptr<arrow::Table> const& source);

  /// non-owning views of the slice-info tables, valid as long as the table is alive
  static SliceInfoPtr readSorted(std::shared_ptr<arrow::Table> const& table);
  static SliceInfoUnsortedPtr readUnsorted(std::shared_ptr<arrow::Table> const& table);
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

  // slice-info tables (see SliceInfo) for the current timeframe and views into them
  Cache bindingsKeys;
  std::vector<std::shared_ptr<arrow::Table>> sliceInfos;
  std::vector<SliceInfoPtr> sliceInfoPtrs;

  Cache bindingsKeysUnsorted;
  std::vector<std::shared_ptr<arrow::Table>> sliceInfosUnsorted;
  std::vector<SliceInfoUnsortedPtr> sliceInfoPtrsUnsorted;

  header::DataOrigin newOrigin = header::DataOrigin{"AOD"};

  ArrowTableSlicingCache(Cache&& bsks, Cache&& bsksUnsorted = {}, header::DataOrigin newOrigin_ = header::DataOrigin{"AOD"});

  // set caching information externally
  void setCaches(Cache&& bsks, Cache&& bsksUnsorted = {});

  // store slice-info table received for the cache entry (assumes it is already present)
  void setCacheEntry(int pos, std::shared_ptr<arrow::Table> sliceInfo);
  void setCacheEntryUnsorted(int pos, std::shared_ptr<arrow::Table> sliceInfo);
  // drop all slice-info tables, e.g. at the start of a new timeframe
  void clearCacheEntries();

  // compute slice-info table for the cache entry locally from the sliced table (assumes it is already present)
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
