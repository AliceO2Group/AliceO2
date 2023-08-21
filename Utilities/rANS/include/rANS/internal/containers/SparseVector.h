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

/// @file   SparseVector.h
/// @author Michael Lettrich
/// @brief  Vector Wrapper with contiguous but shifted, integer indexing. Base of all frequency container classes.

#ifndef RANS_INTERNAL_CONTAINER_SPARSEVECTOR_H_
#define RANS_INTERNAL_CONTAINER_SPARSEVECTOR_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <class container_T>
class SparseVectorIterator;

template <class source_T, class value_T>
class SparseVector
{
  using this_type = SparseVector<source_T, value_T>;

 public:
  using source_type = source_T;
  using value_type = value_T;
  using bucket_type = std::vector<value_type>;
  using container_type = std::vector<bucket_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = SparseVectorIterator<this_type>;
  using const_iterator = SparseVectorIterator<const this_type>;

 private:
  friend iterator;
  friend const_iterator;

  using lut_iterator = typename container_type::iterator;

 public:
  explicit SparseVector(const_reference neutralElement = {}) : mNullElement{neutralElement},
                                                               mContainer(mNBuckets),
                                                               mBegin{mContainer.end()},
                                                               mEnd{mContainer.end()} {};

  SparseVector(const SparseVector& other) : mNullElement{other.mNullElement},
                                            mUsedBuckets{other.mUsedBuckets},
                                            mContainer{other.mContainer},
                                            mBegin{mContainer.end()},
                                            mEnd{mContainer.end()}
  {
    if (!this->empty()) {
      mBegin = std::find_if_not(mContainer.begin(), mContainer.end(), [](const auto& vec) { return vec.empty(); });
      mEnd = std::find_if_not(mContainer.rbegin(), mContainer.rend(), [](const auto& vec) { return vec.empty(); }).base();
    }
  };

  SparseVector(SparseVector&& other) = default;

  SparseVector& operator=(const SparseVector& other)
  {
    SparseVector tmp = other;
    std::swap(tmp, *this);
    return *this;
  };

  SparseVector& operator=(SparseVector&& other) = default;
  ~SparseVector() = default;

  [[nodiscard]] inline const_reference operator[](source_type sourceSymbol) const
  {
    const auto [lutIndex, bucketIndex] = splitIndex(sourceSymbol);
    return mContainer[lutIndex][bucketIndex];
  };

  [[nodiscard]] inline reference operator[](source_type sourceSymbol)
  {
    auto [lutIndex, bucketIndex] = splitIndex(sourceSymbol);
    bucket_type& bucket = mContainer[lutIndex];
    if (bucket.empty()) {
      bucket = makeBucket(lutIndex);
      return bucket[bucketIndex];
    } else {
      return bucket[bucketIndex];
    }
  };

  [[nodiscard]] inline const_reference at(source_type sourceSymbol) const
  {
    auto [lutIndex, bucketIndex] = splitIndex(sourceSymbol);
    const bucket_type& bucket = mContainer[lutIndex];
    if (bucket.empty()) {
      return mNullElement;
    } else {
      return bucket[bucketIndex];
    }
  };

  [[nodiscard]] inline reference at(source_type sourceSymbol) { return this->operator[](sourceSymbol); };

  [[nodiscard]] inline size_type size() const noexcept { return mUsedBuckets * mBucketSize; };

  [[nodiscard]] inline bool empty() const noexcept { return mUsedBuckets == 0; };

  [[nodiscard]] inline static constexpr source_type getOffset() noexcept { return mOffset; };

  [[nodiscard]] inline bucket_type* data() noexcept { return mContainer.data(); };

  [[nodiscard]] inline const bucket_type* data() const noexcept { return mContainer.data(); };

  [[nodiscard]] inline const_iterator cbegin() const noexcept { return empty() ? const_iterator{*this} : const_iterator{*this, mBegin}; };

  [[nodiscard]] inline const_iterator cend() const noexcept { return empty() ? const_iterator{*this} : const_iterator{*this, mEnd}; };

  [[nodiscard]] inline const_iterator begin() const noexcept { return cbegin(); };

  [[nodiscard]] inline const_iterator end() const noexcept { return cend(); };

  [[nodiscard]] inline iterator begin() noexcept { return empty() ? iterator{*this} : iterator{*this, mBegin}; };

  [[nodiscard]] inline iterator end() noexcept { return empty() ? iterator{*this} : iterator{*this, mEnd}; };

  [[nodiscard]] inline container_type release() && noexcept { return std::move(this->mContainer); };

  [[nodiscard]] inline static size_type getBucketSize() noexcept { return mBucketSize; };

  [[nodiscard]] inline static size_type getNBuckets() noexcept { return mNBuckets; };

  [[nodiscard]] inline static constexpr std::pair<uint32_t, uint32_t> splitIndex(source_type sourceSymbol) noexcept
  {
    if constexpr (sizeof(source_type) < 4) {
      difference_type idx = static_cast<difference_type>(sourceSymbol) - mOffset;
      return {0, idx};
    } else {

      if constexpr (std::is_signed_v<source_type>) {
        difference_type idx = static_cast<difference_type>(sourceSymbol) - mOffset;
        return {static_cast<uint32_t>(idx >> mBucketWidth), static_cast<uint16_t>(idx)};
      } else {
        return {static_cast<uint32_t>(sourceSymbol) >> mBucketWidth, static_cast<uint16_t>(sourceSymbol)};
      }
    }
  }

  [[nodiscard]] inline static constexpr source_type joinIndex(std::uint32_t lutID, std::uint32_t bucketID) noexcept
  {
    auto index = static_cast<difference_type>((lutID << mBucketWidth) | bucketID) + mOffset;
    assert(index >= static_cast<difference_type>(std::numeric_limits<source_type>::min()));
    assert(index <= static_cast<difference_type>(std::numeric_limits<source_type>::max()));
    return index;
  };

  inline void swap(SparseVector& a, SparseVector& b)
  {
    using std::swap;
    swap(a.mNullElement, b.mNullElement);
    swap(a.mUsedBuckets, b.mUsedBuckets);
    swap(a.mContainer, b.mContainer);
    swap(a.mBegin, b.mBegin);
    swap(a.mEnd, b.mEnd);
  };

 private:
  inline bucket_type& makeBucket(size_type index)
  {
    mContainer[index] = bucket_type(mBucketSize, mNullElement);
    updateIterators(index);
    ++mUsedBuckets;
    return mContainer[index];
  };

  inline void updateIterators(size_type newPos)
  {
    lut_iterator newBucketIter = mContainer.begin() + newPos;
    // init if empty
    if (empty()) {
      mBegin = newBucketIter;
      mEnd = ++newBucketIter;
    } else {
      mBegin = newBucketIter < mBegin ? newBucketIter : mBegin;
      mEnd = ++newBucketIter > mEnd ? newBucketIter : mEnd;
    }
  };

  value_type mNullElement{};
  size_type mUsedBuckets{};
  container_type mContainer{};
  lut_iterator mBegin{};
  lut_iterator mEnd{};

  inline static constexpr difference_type mOffset{std::numeric_limits<source_type>::min()};
  inline static constexpr size_type mBucketWidth = sizeof(source_type) > 1 ? 16 : 8;
  inline static constexpr size_type mBucketSize = utils::pow2(mBucketWidth);
  inline static constexpr size_type mNBuckets = utils::pow2(utils::toBits<source_type>() - mBucketWidth);
};

/*
Iterator conventions:
* mContainer->empty() is never true
* mUsedBuckets determines how many buckets are currently occupied
* if there are no used buckets, begin() and end() must point to mContainer.end();
* if a find_next operation does not find any further allocated buckets, we are at the end.
* trying to increment a one-past the end or singular iterator is undefined behavior
* "end() is not a valid position indicator for a bucket unless mUsedBuckets==0, in which case begin()==end()"
*/

template <class container_T>
class SparseVectorIterator
{
 public:
  using container_value_type = std::conditional_t<std::is_const_v<container_T>, const typename container_T::value_type, typename container_T::value_type>;

  using lut_iterator = std::conditional_t<std::is_const_v<container_value_type>,
                                          typename container_T::container_type::const_iterator,
                                          typename container_T::container_type::iterator>;
  using bucket_iterator = std::conditional_t<std::is_const_v<container_value_type>,
                                             typename std::iterator_traits<lut_iterator>::value_type::const_iterator,
                                             typename std::iterator_traits<lut_iterator>::value_type::iterator>;

 public:
  class PtrHelper;

  using source_type = typename container_T::source_type;
  using difference_type = std::ptrdiff_t;
  using value_type = std::pair<source_type, container_value_type&>;
  using pointer = PtrHelper;
  using reference = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;

  inline SparseVectorIterator() noexcept = default;

  inline SparseVectorIterator(container_T& container) noexcept : mContainer{&container}, mLutIter{mContainer->mContainer.end()} {};

  inline SparseVectorIterator(container_T& container, lut_iterator lutIter) noexcept : mContainer{&container}, mLutIter{lutIter}
  {
    if (mLutIter != mContainer->mContainer.end()) {
      mBucketIter = mLutIter->begin();
    }
  };

  inline SparseVectorIterator(container_T& container, lut_iterator lutIter, bucket_iterator bucketIter) noexcept : mContainer{&container}, mLutIter{lutIter}, mBucketIter{bucketIter} {};

  // pointer arithmetics
  inline SparseVectorIterator& operator++() noexcept
  {
    // this is always legitimate. Incrementing a singular and one-past the end SparseVectorIterator is UB.
    ++mBucketIter;
    // handle end of current bucket
    if (mBucketIter == mLutIter->end()) {
      // find next, non-empty bucket.
      auto newEndIter = std::find_if(++mLutIter, mContainer->mContainer.end(), [](const auto& container) { return !container.empty(); });
      // if no non-empty buckets can be found, we point to one-past our current bucket which is always a legal iterator, otherwise to the one that has been found.
      mLutIter = newEndIter == mContainer->mContainer.end() ? mLutIter : newEndIter;
      // point to the first element of the new bucket (if that is legal)
      mBucketIter = mLutIter != mContainer->mContainer.end() ? mLutIter->begin() : bucket_iterator{};
    }
    return *this;
  };

  inline SparseVectorIterator operator++(int) noexcept
  {
    auto res = *this;
    ++(*this);
    return res;
  };

  // pointer arithmetics
  inline SparseVectorIterator& operator--() noexcept
  {
    // base case: not at the beginning of an allocated bucket.
    if ((mBucketIter != mLutIter->begin()) && (mBucketIter != bucket_iterator{})) {
      --mBucketIter;
    } else {
      // if not, we need to find the nearest allocated  bucket before this.
      auto nextRBucket = std::find_if(std::make_reverse_iterator(mLutIter), mContainer->mContainer.rend(), [](const auto& container) { return !container.empty(); });
      // if there are no more allocated buckets before this bucket, we stay where we are, otherwise we point to the new location
      mLutIter = nextRBucket != mContainer->mContainer.rend() ? --nextRBucket.base() : mLutIter;
      // we can always point to the end. decrementing a singular SparseVectorIterator or the first element is UB.
      mBucketIter = --mLutIter->end();
    }
    return *this;
  };

  inline SparseVectorIterator operator--(int) noexcept
  {
    auto res = *this;
    --(*this);
    return res;
  };

  // comparison
  inline bool operator==(const SparseVectorIterator& other) const noexcept { return this->mLutIter == other.mLutIter &&
                                                                                    this->mBucketIter == other.mBucketIter; };
  inline bool operator!=(const SparseVectorIterator& other) const noexcept { return !(*this == other); };

  // dereference
  inline value_type operator*() const noexcept
  {
    size_t lut = std::distance(mContainer->mContainer.begin(), mLutIter);
    size_t bucket = std::distance(mLutIter->begin(), mBucketIter);
    source_type index = container_T::joinIndex(lut, bucket);
    assert(mBucketIter != bucket_iterator{});
    return {index, *mBucketIter};
  };

  inline pointer operator->() noexcept
  {
    return {operator*()};
  };

  inline pointer operator->() const noexcept
  {
    return {operator*()};
  };

  // convert to const iter
  inline operator SparseVectorIterator<const container_T>() { return {*mContainer, mLutIter, mBucketIter}; };

  class PtrHelper
  {
   public:
    PtrHelper() = default;
    PtrHelper(value_type value) : mValue{std::move(value)} {};

    value_type* operator->() { return &mValue; }
    value_type* operator->() const { return &mValue; }

   private:
    value_type mValue{};
  };

  inline container_T& getContainer() const { return *mContainer; };

  inline lut_iterator getLUTIterator() const { return mLutIter; };
  inline bucket_iterator getBucketIterator() const { return mBucketIter; };

 private:
  container_T* mContainer{};
  lut_iterator mLutIter{};
  bucket_iterator mBucketIter{};
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINER_SPARSEVECTOR_H_ */
