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

/// @file   OrderedSet.h
/// @author Michael Lettrich
/// @brief  Vector Wrapper with contiguous but shifted, integer indexing. Base of all frequency container classes.

#ifndef RANS_INTERNAL_CONTAINER_ORDEREDSET_H_
#define RANS_INTERNAL_CONTAINER_ORDEREDSET_H_

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <typename P>
class OrderedSetIterator;

enum class OrderedSetState { ordered,
                             unordered };

template <class source_T, class value_T>
class OrderedSet
{
 public:
  using source_type = source_T;
  using value_type = value_T;
  using tuple_type = std::pair<source_T, value_T>;
  using container_type = std::vector<tuple_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = OrderedSetIterator<tuple_type*>;
  using const_iterator = OrderedSetIterator<const tuple_type*>;

  OrderedSet() = default;
  OrderedSet(value_type nullElement) : mNullElement{std::move(nullElement)} {};
  OrderedSet(container_type container, value_type nullElement, OrderedSetState state = OrderedSetState::unordered) : mContainer{std::move(container)}, mNullElement{std::move(nullElement)}
  {
    if (state == OrderedSetState::unordered) {
      std::sort(mContainer.begin(), mContainer.end(), [](const tuple_type& a, const tuple_type& b) {
        return a.first < b.first;
      });
    }

#if !defined(NDEBUG)
    if (!container.empty()) {
      auto iter = mContainer.begin();
      auto value = iter->first;
      for (++iter; iter != mContainer.end(); ++iter) {
        assert(value < iter->first);
        value = iter->first;
      }
    }
#endif
  };

  [[nodiscard]] inline const_reference getNullElement() const { return mNullElement; };

  [[nodiscard]] inline const_reference operator[](source_type sourceSymbol) const
  {
    auto iter = findImpl(sourceSymbol);
    if (iter != mContainer.end()) {
      return iter->second;
    } else {
      return getNullElement();
    }
  };

  [[nodiscard]] inline reference operator[](source_type sourceSymbol)
  {
    auto iter = findImpl(sourceSymbol);
    if (iter != mContainer.end()) {
      return iter->second;
    } else {
      throw Exception(fmt::format("sourceSymbol {} is not contained in data-structure", sourceSymbol));
    }
  };

  [[nodiscard]] inline const_iterator find(source_type sourceSymbol) const { return {findImpl(sourceSymbol)}; };

  [[nodiscard]] inline iterator find(source_type sourceSymbol) { return {findImpl(sourceSymbol)}; };

  [[nodiscard]] inline size_type size() const noexcept { return mContainer.size(); };

  [[nodiscard]] inline bool empty() const noexcept { return mContainer.empty(); };

  [[nodiscard]] inline const_iterator cbegin() const noexcept { return {mContainer.cbegin().base()}; };

  [[nodiscard]] inline const_iterator cend() const noexcept { return {mContainer.cend().base()}; };

  [[nodiscard]] inline const_iterator begin() const noexcept { return cbegin(); };

  [[nodiscard]] inline const_iterator end() const noexcept { return cend(); };

  [[nodiscard]] inline iterator begin() noexcept { return {mContainer.begin().base()}; };

  [[nodiscard]] inline iterator end() noexcept { return {mContainer.end().base()}; };

  [[nodiscard]] inline container_type release() && noexcept { return std::move(this->mContainer); };

  friend void swap(OrderedSet& a, OrderedSet& b) noexcept
  {
    using std::swap;
    swap(a.mContainer, b.mContainer);
    swap(a.mNullElement, b.mNullElement);
  };

 protected:
  struct Comparator {
    bool operator()(const tuple_type& a, const source_type& b) const
    {
      return a.first < b;
    }

    bool operator()(const source_type& a, const tuple_type& b) const
    {
      return a < b.first;
    }
  };

  [[nodiscard]] inline decltype(auto) findImpl(source_type sourceSymbol)
  {
    auto iter = std::lower_bound(mContainer.begin(), mContainer.end(), sourceSymbol, Comparator());
    if (iter != mContainer.end()) {
      if (iter->first != sourceSymbol) {
        return mContainer.end();
      }
    }
    return iter;
  };

  [[nodiscard]] inline decltype(auto) findImpl(source_type sourceSymbol) const
  {
    auto iter = std::lower_bound(mContainer.begin(), mContainer.end(), sourceSymbol, Comparator());
    if (iter != mContainer.end()) {
      if (iter->first != sourceSymbol) {
        return mContainer.end();
      }
    }
    return iter;
  };

  container_type mContainer{};
  value_type mNullElement{};
};

template <typename P>
class OrderedSetIterator
{

  using pair_type = std::remove_cv_t<std::remove_pointer_t<P>>;

 public:
  class PtrHelper;

  using source_type = typename pair_type::first_type;
  using difference_type = std::ptrdiff_t;
  using value_type = std::pair<const source_type, std::add_lvalue_reference_t<std::conditional_t<std::is_const_v<std::remove_pointer_t<P>>,
                                                                                                 std::add_const_t<typename pair_type::second_type>,
                                                                                                 typename pair_type::second_type>>>;
  using pointer = PtrHelper;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  inline constexpr OrderedSetIterator() noexcept = default;

  inline constexpr OrderedSetIterator(P ptr) noexcept : mPtr{ptr} {};
  inline constexpr OrderedSetIterator(const OrderedSetIterator& iter) noexcept = default;
  inline constexpr OrderedSetIterator(OrderedSetIterator&& iter) noexcept = default;
  inline constexpr OrderedSetIterator& operator=(const OrderedSetIterator& other) noexcept = default;
  inline constexpr OrderedSetIterator& operator=(OrderedSetIterator&& other) noexcept = default;
  inline ~OrderedSetIterator() noexcept = default;

  // pointer arithmetics
  inline constexpr OrderedSetIterator& operator++() noexcept
  {
    ++mPtr;
    return *this;
  };

  inline constexpr OrderedSetIterator operator++(int) noexcept
  {
    auto res = *this;
    ++(*this);
    return res;
  };

  inline constexpr OrderedSetIterator& operator--() noexcept
  {
    --mPtr;
    return *this;
  };

  inline constexpr OrderedSetIterator operator--(int) noexcept
  {
    auto res = *this;
    --(*this);
    return res;
  };

  inline constexpr OrderedSetIterator& operator+=(difference_type i) noexcept
  {
    mPtr += i;
    return *this;
  };

  inline constexpr OrderedSetIterator operator+(difference_type i) const noexcept
  {
    auto tmp = *const_cast<OrderedSetIterator*>(this);
    return tmp += i;
  }

  inline constexpr OrderedSetIterator& operator-=(difference_type i) noexcept
  {
    mPtr -= i;
    return *this;
  };

  inline constexpr OrderedSetIterator operator-(difference_type i) const noexcept
  {
    auto tmp = *const_cast<OrderedSetIterator*>(this);
    return tmp -= i;
  };

  inline constexpr difference_type operator-(const OrderedSetIterator& other) const noexcept
  {
    return this->mPtr - other.mPtr;
  };

  // comparison
  inline constexpr bool operator==(const OrderedSetIterator& other) const noexcept { return this->mPtr == other.mPtr; };
  inline constexpr bool operator!=(const OrderedSetIterator& other) const noexcept { return this->mPtr != other.mPtr; };
  inline constexpr bool operator<(const OrderedSetIterator& other) const noexcept { return this->mPtr < other->mPtr; };
  inline constexpr bool operator>(const OrderedSetIterator& other) const noexcept { return this->mPtr > other->mPtr; };
  inline constexpr bool operator>=(const OrderedSetIterator& other) const noexcept { return this->mPtr >= other->mPtr; };
  inline constexpr bool operator<=(const OrderedSetIterator& other) const noexcept { return this->mPtr <= other->mPtr; };

  // dereference
  inline constexpr value_type operator*() const { return {mPtr->first, mPtr->second}; };

  inline constexpr pointer operator->() const { return {operator*()}; };

  inline constexpr value_type operator[](difference_type i) const
  {
    auto& val = mPtr[i];
    return {val.first, val.second};
  };

  class PtrHelper
  {
   public:
    PtrHelper() = default;
    PtrHelper(value_type value) : mValue{std::move(value)} {};

    value_type* operator->() const { return &mValue; }
    value_type* operator->() { return &mValue; }

   private:
    value_type mValue{};
  };

 private:
  value_type get() { return {mPtr->first, mPtr->second}; };
  P mPtr;
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINER_ORDEREDSET_H_ */
