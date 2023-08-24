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

#ifndef RANS_INTERNAL_TRANSFORM_ALGORITHM_H_
#define RANS_INTERNAL_TRANSFORM_ALGORITHM_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/containertraits.h"
#include "rANS/internal/transform/algorithmImpl.h"

namespace o2::rans::internal
{

template <class container_T>
class SparseVectorIterator;

template <class IT, std::enable_if_t<isPair_v<typename std::iterator_traits<IT>::value_type>, bool> = true>
inline auto getValue(IT iter) -> typename std::iterator_traits<IT>::value_type::second_type
{
  return iter->second;
}

template <typename source_T, typename value_T>
inline auto getValue(const std::pair<source_T, value_T>& pair) -> value_T
{
  return pair.second;
}

template <class IT, std::enable_if_t<std::is_pointer_v<std::remove_reference_t<IT>>, bool> = true>
inline auto getValue(IT iter) -> typename std::iterator_traits<IT>::value_type
{
  return *iter;
}

template <class IT, std::enable_if_t<isPair_v<typename std::iterator_traits<IT>::value_type>, bool> = true>
inline void setValue(IT iter, const typename std::iterator_traits<IT>::value_type::second_type& value)
{
  return iter->second = value;
}

template <class IT, std::enable_if_t<std::is_pointer_v<std::remove_reference_t<IT>>, bool> = true>
inline void setValue(IT iter, std::add_lvalue_reference_t<std::add_const_t<typename std::iterator_traits<IT>::value_type>> value)
{
  *iter = value;
}

template <typename container_T, std::enable_if_t<isDenseContainer_v<container_T>, bool> = true>
inline constexpr auto getIndex(const container_T& container, typename container_T::const_iterator iter) -> typename container_T::source_type
{
  return container.getOffset() + std::distance(container.begin(), iter);
};

template <typename container_T, std::enable_if_t<isAdaptiveContainer_v<container_T> ||
                                                   isHashContainer_v<container_T> ||
                                                   isSetContainer_v<container_T>,
                                                 bool> = true>
inline constexpr auto getIndex(const container_T& container, typename container_T::const_iterator iter) -> typename container_T::source_type
{
  return iter->first;
};

template <typename container_T, class F>
inline void forEachIndexValue(const container_T& container, typename container_T::const_iterator begin, typename container_T::const_iterator end, F functor)
{
  algorithmImpl::forEachIndexValue(container, begin, end, functor);
};

template <typename container_T, class F, std::enable_if_t<isStorageContainer_v<container_T>, bool> = true>
inline void forEachIndexValue(container_T& container, typename container_T::iterator begin, typename container_T::iterator end, F functor)
{
  algorithmImpl::forEachIndexValue(container, begin, end, functor);
};

template <typename container_T, class F>
inline void forEachIndexValue(const container_T& container, F functor)
{
  forEachIndexValue(container, container.begin(), container.end(), functor);
};

template <typename container_T, class F>
inline void forEachIndexValue(container_T& container, F functor)
{
  forEachIndexValue(container, container.begin(), container.end(), functor);
};

template <typename container_T>
inline auto trim(typename container_T::iterator begin, typename container_T::iterator end, typename container_T::const_reference zeroElem = {})
  -> std::pair<typename container_T::iterator, typename container_T::iterator>
{
  return algorithmImpl::trim<container_T, typename container_T::iterator>(begin, end, zeroElem);
};

template <typename container_T>
inline auto trim(typename container_T::const_iterator begin, typename container_T::const_iterator end, typename container_T::const_reference zeroElem = {})
  -> std::pair<typename container_T::const_iterator, typename container_T::const_iterator>
{
  return algorithmImpl::trim<container_T, typename container_T::const_iterator>(begin, end, zeroElem);
}

template <typename container_T, std::enable_if_t<isStorageContainer_v<container_T>, bool> = true>
inline decltype(auto) trim(container_T& container, const typename container_T::value_type& zeroElem = {})
{
  return algorithmImpl::trim<container_T, typename container_T::iterator>(container.begin(), container.end(), zeroElem);
};

template <typename container_T, std::enable_if_t<isContainer_v<container_T>, bool> = true>
inline decltype(auto) trim(const container_T& container, const typename container_T::value_type& zeroElem = {})
{
  return algorithmImpl::trim<container_T, typename container_T::const_iterator>(container.begin(), container.end(), zeroElem);
};

template <class container_T,
          std::enable_if_t<isDenseContainer_v<container_T> ||
                             isAdaptiveContainer_v<container_T> ||
                             isSetContainer_v<container_T>,
                           bool> = true>
auto getMinMax(const container_T& container,
               typename container_T::const_iterator begin,
               typename container_T::const_iterator end,
               typename container_T::const_reference zeroElem = {})
  -> std::pair<typename container_T::source_type, typename container_T::source_type>
{
  auto [trimmedBegin, trimmedEnd] = trim<container_T>(begin, end, zeroElem);

  if (trimmedBegin != trimmedEnd) {
    const auto min = getIndex(container, trimmedBegin);
    const auto max = getIndex(container, --trimmedEnd);
    assert(max >= min);
    return {min, max};
  }
  return {container.getOffset(), container.getOffset()};
};

template <typename container_T, std::enable_if_t<isHashContainer_v<container_T>, bool> = true>
auto getMinMax(const container_T& container,
               typename container_T::const_iterator begin,
               typename container_T::const_iterator end,
               typename container_T::const_reference zeroElem = {})
  -> std::pair<typename container_T::source_type, typename container_T::source_type>
{
  using iterator_type = typename container_T::const_iterator;
  using value_type = typename std::iterator_traits<iterator_type>::value_type::second_type;
  using return_type = std::pair<value_type, value_type>;

  bool empty = container.empty();

  if constexpr (isRenormedHistogram_v<container_T>) {
    empty = container.getNumSamples() == container.getIncompressibleSymbolFrequency();
  }

  if (empty) {
    return return_type{container.getOffset(), container.getOffset()};
  };

  const auto [minIter, maxIter] = std::minmax_element(begin, end, [](const auto& a, const auto& b) { return a.first < b.first; });
  return return_type{minIter->first, maxIter->first};
};

template <typename container_T>
auto getMinMax(const container_T& container, typename container_T::const_reference zeroElem = {})
  -> std::pair<typename container_T::source_type, typename container_T::source_type>
{
  return getMinMax(container, container.begin(), container.end(), zeroElem);
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_TRANSFORM_ALGORITHM_H_ */