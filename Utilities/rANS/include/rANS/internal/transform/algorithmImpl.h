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

#ifndef RANS_INTERNAL_TRANSFORM_ALGORITHMIMPL_H_
#define RANS_INTERNAL_TRANSFORM_ALGORITHMIMPL_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <gsl/span>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/containertraits.h"

namespace o2::rans::internal::algorithmImpl
{

template <typename container_T, typename IT, std::enable_if_t<isDenseContainer_v<container_T>, bool> = true>
inline auto trim(IT begin, IT end, typename container_T::const_reference zeroElem) -> std::pair<IT, IT>
{
  using value_type = typename std::iterator_traits<IT>::value_type;

  auto isZero = [&zeroElem](value_type i) { return i == zeroElem; };
  auto nonZeroBegin = std::find_if_not(begin, end, isZero);
  auto nonZeroEnd = nonZeroBegin == end ? end : std::find_if_not(std::make_reverse_iterator(end), std::make_reverse_iterator(begin), isZero).base();

  return {nonZeroBegin, nonZeroEnd};
};

template <typename container_T, typename IT, std::enable_if_t<isAdaptiveContainer_v<container_T>, bool> = true>
inline auto trim(IT begin, IT end, typename container_T::const_reference zeroElem) -> std::pair<IT, IT>
{
  using value_type = typename container_T::value_type;
  using iterator_type = IT;
  using lut_iterator = typename iterator_type::lut_iterator;
  using bucket_iterator = typename iterator_type::bucket_iterator;

  auto isZero = [&zeroElem](const auto& i) { return i == zeroElem; };

  // no range
  if (begin == end) {
    return {end, end};
  }

  iterator_type nonZeroBegin = [&]() -> iterator_type {
    auto lutIter = begin.getLUTIterator();
    if (lutIter != end.getLUTIterator()) {
      // finish first incomplete bucket
      auto nonZeroBegin = std::find_if_not(begin.getBucketIterator(), lutIter->end(), isZero);
      if (nonZeroBegin != lutIter->end()) {
        return {begin.getContainer(), lutIter, nonZeroBegin};
      }

      // go over all remaining buckets
      for (++lutIter; lutIter != end.getLUTIterator(); ++lutIter) {
        // and process each element in each bucket
        auto nonZeroBegin = std::find_if_not(lutIter->begin(), lutIter->end(), isZero);
        if (nonZeroBegin != lutIter->end()) {
          return {begin.getContainer(), lutIter, nonZeroBegin};
        }
      }
    }
    // go over the tail, i.e. the last, possibly incomplete or empty bucket
    if (end.getBucketIterator() != bucket_iterator{}) {
      auto iter = (begin.getLUTIterator() == end.getLUTIterator()) ? begin.getBucketIterator() : lutIter->begin();
      auto nonZeroBegin = std::find_if_not(iter, end.getBucketIterator(), isZero);
      if (nonZeroBegin != lutIter->end()) {
        return {begin.getContainer(), lutIter, nonZeroBegin};
      }
    }
    return end;
  }();

  // empty
  if (nonZeroBegin == end) {
    return {end, end};
  }

  iterator_type nonZeroEnd = [&]() -> iterator_type {
    auto lutIter = end.getLUTIterator();
    // start at the tail, i.e. the last, possibly incomplete or empty bucket
    if (end.getBucketIterator() != bucket_iterator{}) {
      // if a tail exists, process it
      auto nonZeroEnd = std::find_if_not(std::make_reverse_iterator(end.getBucketIterator()), lutIter->rend(), isZero);
      if (nonZeroEnd != lutIter->rend()) {
        return {begin.getContainer(), lutIter, nonZeroEnd.base()};
      }
    }

    // go over all full buckets, appart from the last one.
    for (; lutIter-- > begin.getLUTIterator();) {
      // and process each element in each bucket
      auto nonZeroEnd = std::find_if_not(lutIter->rbegin(), lutIter->rend(), isZero);
      if (nonZeroEnd != lutIter->rend()) {
        return {begin.getContainer(), lutIter, nonZeroEnd.base()};
      }
    }

    // finish at first ,possibly incomplete bucket
    assert(lutIter == begin.getLUTIterator());
    auto bucketREnd = std::make_reverse_iterator(begin.getBucketIterator());
    auto nonZeroEnd = std::find_if_not(lutIter->rbegin(), bucketREnd, isZero);
    if (nonZeroEnd != bucketREnd) {
      return {begin.getContainer(), lutIter, nonZeroEnd.base()};
    }
    return begin;
  }();

  return {nonZeroBegin, nonZeroEnd};
};

template <typename container_T, typename IT, std::enable_if_t<isHashContainer_v<container_T>, bool> = true>
inline auto trim(IT begin, IT end, const typename container_T::const_reference zeroElem) -> std::pair<IT, IT>
{
  return {begin, end};
};

template <typename container_T, typename IT, std::enable_if_t<isSetContainer_v<container_T>, bool> = true>
inline auto trim(IT begin, IT end, typename container_T::const_reference zeroElem) -> std::pair<IT, IT>
{
  using value_type = typename std::iterator_traits<IT>::value_type;

  auto isZero = [&zeroElem](const value_type& keyValuePair) { return keyValuePair.second == zeroElem; };
  auto nonZeroBegin = std::find_if_not(begin, end, isZero);

  // workaround, because we cannot use reverse iterators with this type
  IT nonZeroEnd = [&]() {
    if (nonZeroBegin == end) {
      return end;
    }
    size_t size = std::distance(begin, end);

    for (size_t i = size; i-- > 0;) {
      if ((begin + i)->second != zeroElem) {
        return ++(begin + i); // don't forget the +1 for one-past the end iterator
      }
    }
    return begin;
  }();

  return {nonZeroBegin, nonZeroEnd};
};

template <typename container_T, typename IT, class F, std::enable_if_t<isDenseContainer_v<container_T>, bool> = true>
inline void forEachIndexValue(container_T&& container, IT begin, IT end, F functor)
{
  using container_type = removeCVRef_t<container_T>;

  typename container_type::source_type index = container.getOffset() + std::distance(container.begin(), begin);
  for (std::ptrdiff_t i = 0; i < std::distance(begin, end); ++i) {
    functor(index++, begin[i]);
  }
}

template <typename container_T, typename IT, class F, std::enable_if_t<isAdaptiveContainer_v<container_T>, bool> = true>
inline void forEachIndexValue(container_T&& container, IT begin, IT end, F functor)
{
  using container_type = removeCVRef_t<container_T>;

  using source_type = typename container_type::source_type;
  using value_type = typename container_type::value_type;
  using storage_type = SparseVector<source_type, value_type>;
  using iterator_type = IT;
  using lut_iterator = typename iterator_type::lut_iterator;
  using bucket_iterator = typename iterator_type::bucket_iterator;

  auto& sparseContainer = begin.getContainer();

  // empty
  if (begin == end) {
    return;
  }

  auto lutIter = begin.getLUTIterator();
  size_t lut = std::distance<>(sparseContainer.data(), lutIter.base());
  auto incLut = [&lutIter, &lut]() {
    ++lutIter;
    ++lut;
  };

  if (lutIter != end.getLUTIterator()) {
    // go over first bucket
    auto bucketRange = gsl::make_span(lutIter->begin().base(), lutIter->end().base());
    // but start at the bucket position indicated by begin
    for (size_t bucket = std::distance(bucketRange.data(), begin.getBucketIterator().base()); bucket < bucketRange.size(); ++bucket) {
      functor(storage_type::joinIndex(lut, bucket), bucketRange.data()[bucket]);
    }

    // go over all remaining buckets
    for (incLut(); lutIter != end.getLUTIterator(); incLut()) {
      // and process each element in each bucket
      bucketRange = gsl::make_span(*lutIter);
      for (size_t bucket = 0; bucket < bucketRange.size(); ++bucket) {
        functor(storage_type::joinIndex(lut, bucket), bucketRange.data()[bucket]);
      }
    }
  }
  // go over the tail, i.e. the last, possibly incomplete or empty bucket
  if (end.getBucketIterator() != bucket_iterator{}) {
    // start at the begining of the last bucket and go till bucket position indicated at "end"
    // special case if there is only a single, incomplete bucket, we must use the bucket indicated by "begin", otherwise we can use 0
    size_t bucket = (begin.getLUTIterator() == end.getLUTIterator()) ? std::distance(begin.getLUTIterator()->begin(), begin.getBucketIterator()) : 0;
    auto bucketRange = gsl::make_span(lutIter->begin().base(), end.getBucketIterator().base());
    for (; bucket < bucketRange.size(); ++bucket) {
      functor(storage_type::joinIndex(lut, bucket), bucketRange.data()[bucket]);
    }
  }
};

template <typename container_T, typename IT, class F, std::enable_if_t<isHashContainer_v<container_T>, bool> = true>
inline void forEachIndexValue(container_T&& container, IT begin, IT end, F functor)
{
  using iterator_type = IT;

  std::vector<iterator_type> orderedIterators{};
  orderedIterators.reserve(container.size());
  for (auto iter = begin; iter != end; ++iter) {
    orderedIterators.push_back(iter);
  };

  std::sort(orderedIterators.begin(), orderedIterators.end(), [](const iterator_type& a, const iterator_type& b) {
    return a->first < b->first;
  });

  for (const auto& iter : orderedIterators) {
    functor(iter->first, iter->second);
  }
};

template <typename container_T, typename IT, class F, std::enable_if_t<isSetContainer_v<container_T>, bool> = true>
inline void forEachIndexValue(container_T&& container, IT begin, IT end, F functor)
{
  using container_type = removeCVRef_t<container_T>;

  for (auto iter = begin; iter != end; ++iter) {
    functor(iter->first, iter->second);
  }
}

} // namespace o2::rans::internal::algorithmImpl

#endif /* RANS_INTERNAL_TRANSFORM_ALGORITHMIMPL_H_ */