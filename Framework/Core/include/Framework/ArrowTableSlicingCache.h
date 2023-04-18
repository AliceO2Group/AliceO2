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
struct SliceInfoPtr {
  gsl::span<int const> values;
  gsl::span<int64_t const> counts;

  std::pair<int64_t, int64_t> getSliceFor(int value) const;
};

struct ArrowTableSlicingCacheDef {
  constexpr static ServiceKind service_kind = ServiceKind::Global;
  std::vector<std::pair<std::string, std::string>> bindingsKeys;

  void setCaches(std::vector<std::pair<std::string, std::string>>&& bsks);
};

struct ArrowTableSlicingCache {
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  std::vector<std::pair<std::string, std::string>> bindingsKeys;
  std::vector<std::shared_ptr<arrow::NumericArray<arrow::Int32Type>>> values;
  std::vector<std::shared_ptr<arrow::NumericArray<arrow::Int64Type>>> counts;

  ArrowTableSlicingCache(std::vector<std::pair<std::string, std::string>>&& bsks);

  // set caching information externally
  void setCaches(std::vector<std::pair<std::string, std::string>>&& bsks);

  // update slicing info cache entry (assumes it is already present)
  arrow::Status updateCacheEntry(int pos, std::shared_ptr<arrow::Table> const& table);

  // get slice from cache for a given value
  SliceInfoPtr getCacheFor(std::pair<std::string, std::string> const& bindingKey) const;

  static void validateOrder(std::pair<std::string, std::string> const& bindingKey, std::shared_ptr<arrow::Table> const& input);
};
} // namespace o2::framework

#endif // ARROWTABLESLICINGCACHE_H
