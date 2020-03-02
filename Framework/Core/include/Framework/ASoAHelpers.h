// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_ASOAHELPERS_H_
#define O2_FRAMEWORK_ASOAHELPERS_H_

#include "Framework/ASoA.h"

#include <iterator>
#include <tuple>
#include <utility>

namespace o2::soa
{

template <typename T2, std::size_t K2>
void addOne(std::array<T2, K2>& array, const T2& maxOffset, bool& isEnd)
{
  for (int i = 0; i < K2; i++) {
    array[K2 - i - 1]++;
    if (array[K2 - i - 1] != maxOffset + K2 - i - 1) { // no < operator for RowViewBase
      for (int j = K2 - i; j < K2; j++) {
        array[j].setCursor(array[j - 1].mRowIndex + 1);
      }
      isEnd = false;
      return;
    }
  }
  isEnd = true;
}

/// @return next K-combination of the rows of the table T.
/// FIXME: move to coroutines once we have C++20
template <typename T, int K>
class CombinationsGenerator
{
 public:
  using IteratorType = typename T::iterator;
  using CombinationType = std::array<IteratorType, K>;
  using FunctionType = std::function<bool(const CombinationType&)>;

  class CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>
  {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const IteratorType& begin, int n, const FunctionType& condition)
      : mN(n), mIsEnd(false), mTableBegin(begin), mMaxOffset(begin + n - K + 1), mCondition(condition)
    {
      initIterators();
    }

    ~CombinationsIterator() = default;

    // prefix increment
    CombinationsIterator& operator++()
    {
      if (!mIsEnd) {
        addOne(mCurrent, mMaxOffset, mIsEnd);
      }
      while (!mIsEnd && !mCondition(mCurrent)) {
        addOne(mCurrent, mMaxOffset, mIsEnd);
      }
      return *this;
    }
    // postfix increment
    CombinationsIterator operator++(int /*unused*/)
    {
      CombinationsIterator copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return mCurrent;
    }
    bool operator==(const CombinationsIterator& rh)
    {
      return (mIsEnd && rh.mIsEnd) || (mCurrent == rh.mCurrent);
    }
    bool operator!=(const CombinationsIterator& rh)
    {
      return !(*this == rh);
    }

    void initIterators()
    {
      for (int i = 0; i < K; i++) {
        mCurrent[i] = mTableBegin + i;
      }
      if (!mCondition(mCurrent)) {
        operator++();
      }
    }

    void goToEnd()
    {
      for (int i = 0; i < K; i++) {
        mCurrent[i].setCursor(mN - K + i);
      }
      operator++();
    }

   private:
    CombinationType mCurrent;
    int mN;                   // number of elements
    bool mIsEnd;              // whether there are any more tuples available
    FunctionType mCondition;  // only tuples satisfying the condition will be outputed
    IteratorType mTableBegin; // start of the table for which tuples are generated
    IteratorType mMaxOffset;  // one position past maximum acceptable position for 0th element of combination
  };

  using iterator = CombinationsIterator;
  using const_iterator = CombinationsIterator;

  inline iterator begin()
  {
    return iterator(mTableBegin, mN, mCondition);
  }
  inline iterator end()
  {
    auto it = iterator(mTableBegin, mN, mCondition);
    it.goToEnd();
    return it;
  }

  CombinationsGenerator() = delete;
  CombinationsGenerator(const T& table, const FunctionType& condition)
    : mTableBegin(table.begin()), mN(table.size()), mCondition(condition)
  {
    static_assert(K > 0);
  }
  ~CombinationsGenerator() = default;

 private:
  IteratorType mTableBegin;
  int mN;
  FunctionType mCondition;
};

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
