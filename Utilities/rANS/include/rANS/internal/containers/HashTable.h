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

/// @file   HashTable.h
/// @author Michael Lettrich
/// @brief  Wrapper around absl::flat_hash_map to be used as a container for building histograms and LUTs

#ifndef RANS_INTERNAL_CONTAINER_HASHTABLE_H_
#define RANS_INTERNAL_CONTAINER_HASHTABLE_H_

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

#include <fairlogger/Logger.h>
#include <absl/container/flat_hash_map.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <class source_T, class value_T>
class HashTable
{
 public:
  using source_type = source_T;
  using value_type = value_T;
  using container_type = absl::flat_hash_map<source_type, value_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  HashTable() = default;
  HashTable(value_type nullElement) : mNullElement{std::move(nullElement)} {};
  HashTable(container_type container, value_type nullElement) : mContainer{std::move(container)}, mNullElement{std::move(nullElement)} {};

  [[nodiscard]] inline const_reference getNullElement() const { return mNullElement; };

  [[nodiscard]] inline const_reference operator[](source_type sourceSymbol) const
  {
    auto iter = mContainer.find(sourceSymbol);
    if (iter != mContainer.end()) {
      return iter->second;
    } else {
      return getNullElement();
    }
  };

  [[nodiscard]] inline reference operator[](source_type sourceSymbol) { return mContainer[sourceSymbol]; };

  [[nodiscard]] inline const_iterator find(source_type sourceSymbol) const { return mContainer.find(sourceSymbol); };

  [[nodiscard]] inline size_type size() const noexcept { return mContainer.size(); };

  [[nodiscard]] inline bool empty() const noexcept { return mContainer.empty(); };

  [[nodiscard]] inline const_iterator cbegin() const noexcept { return mContainer.cbegin(); };

  [[nodiscard]] inline const_iterator cend() const noexcept { return mContainer.cend(); };

  [[nodiscard]] inline const_iterator begin() const noexcept { return cbegin(); };

  [[nodiscard]] inline const_iterator end() const noexcept { return cend(); };

  [[nodiscard]] inline iterator begin() noexcept { return mContainer.begin(); };

  [[nodiscard]] inline iterator end() noexcept { return mContainer.end(); };

  [[nodiscard]] inline container_type release() && noexcept { return std::move(this->mContainer); };

  friend void swap(HashTable& a, HashTable& b) noexcept
  {
    using std::swap;
    swap(a.mContainer, b.mContainer);
    swap(a.mNullElement, b.mNullElement);
  };

 protected:
  container_type mContainer{};
  value_type mNullElement{};
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINER_HASHTABLE_H_ */
