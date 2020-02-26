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

using namespace o2::framework;

namespace o2::soa
{

template <std::size_t K2, typename T2>
void addOne(std::array<T2, K2>& array, const std::array<T2, K2>& maxOffset, bool& isEnd)
{
  for (int i = 0; i < K2; i++) {
    // FIXME: FilteredIndexPolicy does not check for too big index, gets spurious values from the selection
    array[K2 - i - 1]++;
    // FIXME: End of Filtered is marked with mRowIndex = -1 but for DefaultIndexPolicy it is (max + 1).
    // How to set maxOffset properly without the knowledge of the index policy being used?
    if (array[K2 - i - 1] != maxOffset[K2 - i - 1]) { // no < operator for RowViewBase
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
template <typename T, std::size_t K>
class CombinationsGenerator
{
 public:
  using IteratorType = typename T::iterator;
  using CombinationType = std::array<IteratorType, K>;

  class CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>
  {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const std::array<IteratorType, K>& begin, std::array<int64_t, K> n)
      : mN(n), mTableBegin(begin)
    {
      for (int i = 0; i < K; i++) {
        mMaxOffset[i] = mTableBegin[i] + mN[i] - K + i + 1;
        mCurrent[i] = mTableBegin[i] + i;
      }
      mIsEnd = std::any_of(mN.begin(), mN.end(), [](auto i) { return i <= K; });
    }

    ~CombinationsIterator() = default;

    // prefix increment
    CombinationsIterator& operator++()
    {
      if (!mIsEnd) {
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

    void goToEnd()
    {
      for (int i = 0; i < K; i++) {
        mCurrent[i].setCursor(mN[i] - K + i);
      }
      operator++();
    }

   private:
    CombinationType mCurrent;
    std::array<int64_t, K> mN;               // numbers of elements in tables
    bool mIsEnd;                             // whether there are any more tuples available
    std::array<IteratorType, K> mTableBegin; // start of the tables for which tuples are generated
    std::array<IteratorType, K> mMaxOffset;  // one position past maximum acceptable position for each element of combination
  };

  using iterator = CombinationsIterator;
  using const_iterator = CombinationsIterator;

  inline iterator begin()
  {
    return iterator(mTableBegin, mN);
  }
  inline iterator end()
  {
    auto it = iterator(mTableBegin, mN);
    it.goToEnd();
    return it;
  }

  CombinationsGenerator() = delete;
  CombinationsGenerator(const std::array<T, K>& tables, const std::array<IteratorType, K>& tableBegin, const std::array<int64_t, K>& n) : mTables(tables), mTableBegin(tableBegin), mN(n)
  {
    static_assert(K > 0);
  }
  ~CombinationsGenerator() = default;

 private:
  std::array<IteratorType, K> mTableBegin;
  std::array<int64_t, K> mN;
  std::array<T, K> mTables; // the (filtered) tables need to be kept somewhere so as not to be destroyed on return from combinations()
};

template <typename... T2>
CombinationsGenerator<std::common_type_t<T2...>, sizeof...(T2)> combinations(const T2&... tables)
{
  static_assert(sizeof...(T2) > 0);
  static_assert(std::conjunction_v<std::is_same<T2, T2>...>);

  using commonType = std::common_type_t<T2...>;

  std::array<typename commonType::iterator, sizeof...(T2)> beginArray{tables.begin()...};
  std::array<int64_t, sizeof...(T2)> nArray{tables.size()...};
  std::array<commonType, sizeof...(T2)> tablesArray{tables...};
  return CombinationsGenerator<commonType, sizeof...(T2)>(tablesArray, beginArray, nArray);
}

template <typename... T2>
CombinationsGenerator<Filtered<std::common_type_t<T2...>>, sizeof...(T2)> combinations(const expressions::Filter& filter, const T2&... tables)
{
  static_assert(sizeof...(T2) > 0);
  static_assert(std::conjunction_v<std::is_same<T2, T2>...>);

  using commonType = std::common_type_t<T2...>;

  std::array<Filtered<commonType>, sizeof...(T2)> filtered{Filtered<commonType>{{tables.asArrowTable()}, expressions::createSelection(tables.asArrowTable(), filter)}...};
  std::array<typename Filtered<commonType>::iterator, sizeof...(T2)> beginArray;
  std::array<int64_t, sizeof...(T2)> nArray;
  for (int i = 0; i < sizeof...(T2); i++) {
    beginArray[i] = filtered[i].begin();
    nArray[i] = filtered[i].size();
  }
  return CombinationsGenerator<Filtered<commonType>, sizeof...(T2)>(filtered, beginArray, nArray);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
