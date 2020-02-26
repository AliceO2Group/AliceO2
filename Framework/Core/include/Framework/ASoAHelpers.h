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
    array[K2 - i - 1]++;
    if (array[K2 - i - 1] != maxOffset[K2 - i - 1]) { // no < operator for RowViewBase
      for (int j = K2 - i; j < K2; j++) {
        array[j].setCursor(*std::get<1>(array[j - 1].getIndices()) + 1);
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

    CombinationsIterator(const std::array<T, K>& tables)
    {
      mIsEnd = false;
      for (int i = 0; i < K; i++) {
        if (tables[i].size() <= K) {
          mIsEnd = true;
        }
        mMaxOffset[i] = tables[i].end() - K + i + 1;
        mCurrent[i] = tables[i].begin() + i;
      }
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
      for (int i = 0; i < K - 1; i++) {
        mCurrent[i].setCursor(*std::get<1>(mMaxOffset[i].getIndices()));
      }
      mCurrent[K - 1].moveToEnd();
      mIsEnd = true;
    }

   private:
    CombinationType mCurrent;
    bool mIsEnd;                             // whether there are any more tuples available
    std::array<IteratorType, K> mMaxOffset;  // one position past maximum acceptable position for each element of combination
  };

  using iterator = CombinationsIterator;
  using const_iterator = CombinationsIterator;

  inline iterator begin()
  {
    return iterator(mTables);
  }
  inline iterator end()
  {
    auto it = iterator(mTables);
    it.goToEnd();
    return it;
  }

  CombinationsGenerator() = delete;
  CombinationsGenerator(const std::array<T, K>& tables) : mTables(tables)
  {
    static_assert(K > 0);
  }
  ~CombinationsGenerator() = default;

 private:
  std::array<T, K> mTables; // the (filtered) tables need to be kept somewhere so as not to be destroyed on return from combinations()
};

template <typename... T2>
CombinationsGenerator<std::common_type_t<T2...>, sizeof...(T2)> combinations(const T2&... tables)
{
  static_assert(sizeof...(T2) > 0);
  static_assert(std::conjunction_v<std::is_same<T2, T2>...>);

  using commonType = std::common_type_t<T2...>;

  std::array<commonType, sizeof...(T2)> tablesArray{tables...};
  return CombinationsGenerator<commonType, sizeof...(T2)>(tablesArray);
}

template <typename... T2>
CombinationsGenerator<Filtered<std::common_type_t<T2...>>, sizeof...(T2)> combinations(const expressions::Filter& filter, const T2&... tables)
{
  static_assert(sizeof...(T2) > 0);
  static_assert(std::conjunction_v<std::is_same<T2, T2>...>);

  using commonType = std::common_type_t<T2...>;

  std::array<Filtered<commonType>, sizeof...(T2)> filtered{Filtered<commonType>{{tables.asArrowTable()}, expressions::createSelection(tables.asArrowTable(), filter)}...};
  return CombinationsGenerator<Filtered<commonType>, sizeof...(T2)>(filtered);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
