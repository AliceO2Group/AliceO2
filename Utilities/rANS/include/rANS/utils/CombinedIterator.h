// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CombinedIterator.h
/// \brief
/// \author michael.lettrich@cern.ch

#ifndef INCLUDE_RANS_UTILS_COMBINEDITERATOR_H_
#define INCLUDE_RANS_UTILS_COMBINEDITERATOR_H_

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <iostream>
#include <iterator>

namespace o2
{
namespace rans
{
namespace utils
{

template <class iterA_T, class iterB_T, class F>
class CombinedInputIterator
{
  using difference_type = std::ptrdiff_t;
  using value_type = std::result_of<F(iterA_T, iterB_T)>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::input_iterator_tag;

 public:
  CombinedInputIterator(iterA_T iterA, iterB_T iterB, F functor);
  CombinedInputIterator(const CombinedInputIterator& iter) = default;
  CombinedInputIterator(CombinedInputIterator&& iter) = default;
  CombinedInputIterator& operator=(const CombinedInputIterator& other);
  CombinedInputIterator& operator=(CombinedInputIterator&& other) = default;
  ~CombinedInputIterator() = default;

  //comparison
  bool operator==(const CombinedInputIterator& other) const;
  bool operator!=(const CombinedInputIterator& other) const;

  //pointer arithmetics
  CombinedInputIterator& operator++();
  CombinedInputIterator operator++(int);

  // dereference
  auto operator*() const;

 private:
  iterA_T mIterA;
  iterB_T mIterB;
  F mFunctor;

 public:
  friend std::ostream& operator<<(std::ostream& o, const CombinedInputIterator& iter)
  {
    o << "CombinedInputIterator{iterA: " << &(iter.mIterA) << ", iterB: " << &(iter.mIterB) << "}";
    return o;
  }
};

template <class iterA_T, class iterB_T, class F>
class CombinedOutputIterator
{

  class Proxy
  {
   public:
    Proxy(CombinedOutputIterator& iter);

    template <typename value_T>
    Proxy& operator=(value_T value);

   private:
    CombinedOutputIterator& mIter;
  };

  using difference_type = std::ptrdiff_t;
  using value_type = Proxy;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::input_iterator_tag;

 public:
  CombinedOutputIterator(iterA_T iterA, iterB_T iterB, F functor);
  CombinedOutputIterator(const CombinedOutputIterator& iter) = default;
  CombinedOutputIterator(CombinedOutputIterator&& iter) = default;
  CombinedOutputIterator& operator=(const CombinedOutputIterator& other);
  CombinedOutputIterator& operator=(CombinedOutputIterator&& other) = default;
  ~CombinedOutputIterator() = default;

  //pointer arithmetics
  CombinedOutputIterator& operator++();
  CombinedOutputIterator operator++(int);

  // dereference
  value_type operator*();

 private:
  iterA_T mIterA;
  iterB_T mIterB;
  F mFunctor;

 public:
  friend std::ostream& operator<<(std::ostream& o, const CombinedOutputIterator& iter)
  {
    o << "CombinedOutputIterator{iterA: " << &(iter.mIterA) << ", iterB: " << &(iter.mIterB) << "}";
    return o;
  }
};

template <class iterA_T, class iterB_T, class F>
CombinedInputIterator<iterA_T, iterB_T, F>::CombinedInputIterator(iterA_T iterA, iterB_T iterB, F functor) : mIterA(iterA), mIterB(iterB), mFunctor(functor)
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
inline auto CombinedInputIterator<iterA_T, iterB_T, F>::operator*() const
{
  return mFunctor(mIterA, mIterB);
}

template <class iterA_T, class iterB_T, class F>
CombinedOutputIterator<iterA_T, iterB_T, F>::CombinedOutputIterator(iterA_T iterA, iterB_T iterB, F functor) : mIterA(iterA), mIterB(iterB), mFunctor(functor)
{
}

template <class iterA_T, class iterB_T, class F>
auto CombinedOutputIterator<iterA_T, iterB_T, F>::operator=(const CombinedOutputIterator& other) -> CombinedOutputIterator&
{
  mIterA = other.mIterA;
  mIterB = other.mIterB;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<iterA_T, iterB_T, F>::operator++() -> CombinedOutputIterator&
{
  ++mIterA;
  ++mIterB;
  return *this;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<iterA_T, iterB_T, F>::operator++(int) -> CombinedOutputIterator
{
  auto res = *this;
  ++(*this);
  return res;
}

template <class iterA_T, class iterB_T, class F>
inline auto CombinedOutputIterator<iterA_T, iterB_T, F>::operator*() -> value_type
{
  return Proxy(*this);
}

template <class iterA_T, class iterB_T, class F>
CombinedOutputIterator<iterA_T, iterB_T, F>::Proxy::Proxy(CombinedOutputIterator& iter) : mIter(iter)
{
}

template <class iterA_T, class iterB_T, class F>
template <typename value_T>
inline auto CombinedOutputIterator<iterA_T, iterB_T, F>::Proxy::operator=(value_T value) -> CombinedOutputIterator::Proxy&
{
  mIter.mFunctor(mIter.mIterA, mIter.mIterB, value);
  return *this;
}

} // namespace utils
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_UTILS_COMBINEDITERATOR_H_ */
