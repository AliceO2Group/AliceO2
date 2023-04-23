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

/// @file   iterators.h
/// @author michael.lettrich@cern.ch
/// @brief  iterators that allow lazy merging or spliting of data using a custom functor

#ifndef RANS_INTERNAL_TRANSFORM_ITERATOR_H_
#define RANS_INTERNAL_TRANSFORM_ITERATOR_H_

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <iostream>
#include <iterator>

#include <fairlogger/Logger.h>

namespace o2::rans
{

namespace internal
{

template <typename IT, typename tag_T>
struct hasIteratorTag : public std::bool_constant<std::is_same_v<typename std::iterator_traits<IT>::iterator_category, tag_T>> {
};
template <typename IT>
inline constexpr bool isBidirectionalIterator_v = hasIteratorTag<IT, std::bidirectional_iterator_tag>::value;
template <typename IT>
inline constexpr bool isRandomAccessIterator_v = hasIteratorTag<IT, std::random_access_iterator_tag>::value;

template <typename iterA_T, typename iterB_T>
inline constexpr bool areBothRandomAccessIterators_v = std::bool_constant<isRandomAccessIterator_v<iterA_T> && isRandomAccessIterator_v<iterB_T>>::value;

template <typename iterA_T, typename iterB_T>
struct getIteratorTag {
};

template <>
struct getIteratorTag<std::bidirectional_iterator_tag, std::bidirectional_iterator_tag> {
  using value_type = std::bidirectional_iterator_tag;
};

template <>
struct getIteratorTag<std::bidirectional_iterator_tag, std::random_access_iterator_tag> {
  using value_type = std::bidirectional_iterator_tag;
};

template <>
struct getIteratorTag<std::random_access_iterator_tag, std::bidirectional_iterator_tag> {
  using value_type = std::bidirectional_iterator_tag;
};

template <>
struct getIteratorTag<std::random_access_iterator_tag, std::random_access_iterator_tag> {
  using value_type = std::random_access_iterator_tag;
};

template <typename iterA_T, typename iterB_T>
using getIteratorTag_t = typename getIteratorTag<iterA_T, iterB_T>::value_type;

} // namespace internal

template <class iterA_T, class iterB_T, class F>
class CombinedInputIterator
{
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::invoke_result_t<F, iterA_T, iterB_T>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = internal::getIteratorTag_t<typename std::iterator_traits<iterA_T>::iterator_category, typename std::iterator_traits<iterB_T>::iterator_category>;

  CombinedInputIterator() = default;
  CombinedInputIterator(iterA_T iterA, iterB_T iterB, F functor);
  CombinedInputIterator(const CombinedInputIterator& iter) = default;
  CombinedInputIterator(CombinedInputIterator&& iter) = default;
  CombinedInputIterator& operator=(const CombinedInputIterator& other);
  CombinedInputIterator& operator=(CombinedInputIterator&& other) = default;
  ~CombinedInputIterator() = default;

  // pointer arithmetics
  CombinedInputIterator& operator++();
  CombinedInputIterator operator++(int);
  CombinedInputIterator& operator--();
  CombinedInputIterator operator--(int);

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  CombinedInputIterator& operator+=(difference_type i);

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  CombinedInputIterator operator+(difference_type i) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  CombinedInputIterator& operator-=(difference_type i);

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  CombinedInputIterator operator-(difference_type i) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  difference_type operator-(const CombinedInputIterator& other) const;

  // comparison
  bool operator==(const CombinedInputIterator& other) const;
  bool operator!=(const CombinedInputIterator& other) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  bool operator<(const CombinedInputIterator& other) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  bool operator>(const CombinedInputIterator& other) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  bool operator>=(const CombinedInputIterator& other) const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  bool operator<=(const CombinedInputIterator& other) const;

  // dereference
  auto operator*() const;

  template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool> = true>
  value_type operator[](difference_type i) const;

 private:
  iterA_T mIterA{};
  iterB_T mIterB{};
  F mFunctor{};

 public:
  friend std::ostream& operator<<(std::ostream& o, const CombinedInputIterator& iter)
  {
    o << "CombinedInputIterator{iterA: " << &(iter.mIterA) << ", iterB: " << &(iter.mIterB) << "}";
    return o;
  }

  friend CombinedInputIterator operator+(CombinedInputIterator::difference_type i, const CombinedInputIterator& iter)
  {
    return iter + i;
  }
};

template <class input_T, class iterA_T, class iterB_T, class F>
class CombinedOutputIterator
{

  class Proxy
  {
   public:
    Proxy(CombinedOutputIterator& iter);

    Proxy& operator=(input_T value);

   private:
    CombinedOutputIterator* mIter{};
  };

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = input_T;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::input_iterator_tag;

  CombinedOutputIterator(iterA_T iterA, iterB_T iterB, F functor);
  CombinedOutputIterator(const CombinedOutputIterator& iter) = default;
  CombinedOutputIterator(CombinedOutputIterator&& iter) = default;
  CombinedOutputIterator& operator=(const CombinedOutputIterator& other);
  CombinedOutputIterator& operator=(CombinedOutputIterator&& other) = default;
  ~CombinedOutputIterator() = default;

  // pointer arithmetics
  CombinedOutputIterator& operator++();
  CombinedOutputIterator operator++(int);

  // dereference
  Proxy& operator*();

 private:
  iterA_T mIterA{};
  iterB_T mIterB{};
  F mFunctor{};
  Proxy mProxy{*this};

 public:
  friend std::ostream& operator<<(std::ostream& o, const CombinedOutputIterator& iter)
  {
    o << "CombinedOutputIterator{iterA: " << &(iter.mIterA) << ", iterB: " << &(iter.mIterB) << "}";
    return o;
  }
};

template <typename input_T>
struct CombinedOutputIteratorFactory {

  template <class iterA_T, class iterB_T, class F>
  static inline auto makeIter(iterA_T iterA, iterB_T iterB, F functor) -> CombinedOutputIterator<input_T, iterA_T, iterB_T, F>
  {
    return {iterA, iterB, functor};
  }
};

template <class iterA_T, class iterB_T, class F>
CombinedInputIterator<iterA_T, iterB_T, F>::CombinedInputIterator(iterA_T iterA, iterB_T iterB, F functor) : mIterA{iterA}, mIterB{iterB}, mFunctor{functor}
{
}

template <class iterA_T, class iterB_T, class F>
auto CombinedInputIterator<iterA_T, iterB_T, F>::operator=(const CombinedInputIterator& other) -> CombinedInputIterator&
{
  mIterA = other.mIterA;
  mIterB = other.mIterB;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator++() -> CombinedInputIterator&
{
  ++mIterA;
  ++mIterB;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator++(int) -> CombinedInputIterator
{
  auto res = *this;
  ++(*this);
  return res;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator--() -> CombinedInputIterator&
{
  --mIterA;
  --mIterB;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator--(int) -> CombinedInputIterator
{
  auto res = *this;
  --(*this);
  return res;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator+=(difference_type i) -> CombinedInputIterator&
{
  mIterA += i;
  mIterB += i;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator+(difference_type i) const -> CombinedInputIterator
{
  auto tmp = *const_cast<CombinedInputIterator*>(this);
  return tmp += i;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator-=(difference_type i) -> CombinedInputIterator&
{
  mIterA -= i;
  mIterB -= i;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator-(difference_type i) const -> CombinedInputIterator
{
  auto tmp = *const_cast<CombinedInputIterator*>(this);
  return tmp -= i;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator-(const CombinedInputIterator& other) const -> difference_type
{
  return this->mIterA - other.mIterA;
}

template <class iterA_T, class iterB_T, class F>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator==(const CombinedInputIterator& other) const
{
  return (mIterA == other.mIterA) && (mIterB == other.mIterB);
}

template <class iterA_T, class iterB_T, class F>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator!=(const CombinedInputIterator& other) const
{
  return !(*this == other);
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator<(const CombinedInputIterator& other) const
{
  return other - *this > 0;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator>(const CombinedInputIterator& other) const
{
  return other < *this;
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator>=(const CombinedInputIterator& other) const
{
  return !(*this < other);
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline bool CombinedInputIterator<iterA_T, iterB_T, F>::operator<=(const CombinedInputIterator& other) const
{
  return !(*this > other);
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator*() const
{
  return mFunctor(mIterA, mIterB);
}

template <class iterA_T, class iterB_T, class F>
template <std::enable_if_t<internal::areBothRandomAccessIterators_v<iterA_T, iterB_T>, bool>>
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator[](difference_type i) const -> value_type
{
  return *(*this + i);
}

template <typename input_T, class iterA_T, class iterB_T, class F>
CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::CombinedOutputIterator(iterA_T iterA, iterB_T iterB, F functor) : mIterA{iterA},
                                                                                                                        mIterB{iterB},
                                                                                                                        mFunctor{functor}
{
}

template <typename input_T, class iterA_T, class iterB_T, class F>
auto CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::operator=(const CombinedOutputIterator& other) -> CombinedOutputIterator&
{
  mIterA = other.mIterA;
  mIterB = other.mIterB;
  return *this;
}

template <typename input_T, class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::operator++() -> CombinedOutputIterator&
{
  ++mIterA;
  ++mIterB;
  return *this;
}

template <typename input_T, class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::operator++(int) -> CombinedOutputIterator
{
  auto res = *this;
  ++(*this);
  return res;
}

template <typename input_T, class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::operator*() -> Proxy&
{
  mProxy = {*this};
  return mProxy;
}

template <typename input_T, class iterA_T, class iterB_T, class F>
CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::Proxy::Proxy(CombinedOutputIterator& iter) : mIter{&iter}
{
}

template <typename input_T, class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<input_T, iterA_T, iterB_T, F>::Proxy::operator=(input_T value) -> Proxy&
{
  mIter->mFunctor(mIter->mIterA, mIter->mIterB, value);
  return *this;
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_TRANSFORM_ITERATOR_H_ */
