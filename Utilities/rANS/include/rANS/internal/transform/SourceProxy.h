// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   algorithm.h
/// @author Michael Lettrich
/// @brief  helper functionalities useful for packing operations

#ifndef RANS_INTERNAL_TRANSFORM_SOURCEPROXY_H_
#define RANS_INTERNAL_TRANSFORM_SOURCEPROXY_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <optional>
#include <variant>

#include <gsl/span>

#include "rANS/internal/common/utils.h"

namespace o2::rans
{

template <typename source_T>
class SourceCache
{
 public:
  using source_type = source_T;
  using pointer = source_type*;
  using const_pointer = const source_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  SourceCache() = default;

 public:
  template <typename IT>
  SourceCache(IT begin, IT end)
  {
    const size_t size = std::distance(begin, end);
    mBuffer.reserve(size);

    for (size_t i = 0; i < size; ++i) {
      auto value = begin[i];
      mMin = std::min(mMin, value);
      mMax = std::max(mMax, value);
      mBuffer.push_back(value);
    };
  }

  inline const_iterator begin() const noexcept { return mBuffer.data(); };
  inline const_iterator end() const noexcept { return begin() + mBuffer.size(); };
  inline size_t size() const noexcept { return mBuffer.size(); };
  inline bool empty() const noexcept { return mBuffer.empty(); };
  inline source_type getMin() const noexcept { return mMin; };
  inline source_type getMax() const noexcept { return mMax; };
  inline size_t getAlphabetRangeBits() const noexcept { return utils::getRangeBits(getMin(), getMax()); };

 private:
  std::vector<source_type> mBuffer;
  source_type mMin{std::numeric_limits<source_type>::min()};
  source_type mMax{std::numeric_limits<source_type>::max()};
};

template <typename IT>
class RangeWrapper
{
  using iterator = IT;

 public:
  RangeWrapper() = default;
  RangeWrapper(IT begin, IT end) : mBegin{begin}, mEnd{end} {};

  iterator begin() const { return mBegin; };
  iterator end() const { return mEnd; };

 private:
  iterator mBegin{};
  iterator mEnd{};
};

template <typename IT>
class SourceProxy
{
  using iterator = IT;
  using source_type = typename std::iterator_traits<iterator>::value_type;
  using pointer = source_type*;
  using const_pointer = const source_type*;
  using cache_type = SourceCache<source_type>;
  using range_type = RangeWrapper<iterator>;

 public:
  template <typename F>
  SourceProxy(IT begin, IT end, F functor)
  {
    if (functor(begin, end)) {
      mProxy.template emplace<0>(begin, end);
      mIsCached = true;
      LOGP(info, "Caching enabled");
    } else {
      mProxy.template emplace<1>(begin, end);
      mIsCached = false;
      LOGP(info, "Caching disabled");
    }
  }

  inline const_pointer beginCache() const { return std::get<0>(mProxy).begin(); };
  inline const_pointer endCache() const { return std::get<0>(mProxy).end(); };
  inline iterator beginIter() const { return std::get<1>(mProxy).begin(); };
  inline iterator endIter() const { return std::get<1>(mProxy).end(); };
  inline source_type getMin() const { return std::get<0>(mProxy).getMin(); };
  inline source_type getMax() const { return std::get<0>(mProxy).getMax(); };
  inline size_t getAlphabetRangeBits() const { return std::get<0>(mProxy).getAlphabetRangeBits(); };

  bool isCached() const { return mIsCached; };

  inline size_t size() const noexcept
  {
    if (std::holds_alternative<cache_type>(mProxy)) {
      return std::get<0>(mProxy).size();
    } else {
      return std::get<1>(mProxy).size();
    }
  };

  inline bool empty() const noexcept
  {
    if (std::holds_alternative<cache_type>(mProxy)) {
      return std::get<0>(mProxy).empty();
    } else {
      return std::get<1>(mProxy).empty();
    }
  };

 private:
  std::variant<cache_type, range_type> mProxy{};
  bool mIsCached{true};
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_TRANSFORM_SOURCEPROXY_H_ */