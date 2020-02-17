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

template <typename T2, unsigned N, typename... REST>
struct generateTupleType {
  typedef typename generateTupleType<T2, N - 1, T2, REST...>::type type;
};

template <typename T2, typename... REST>
struct generateTupleType<T2, 0, REST...> {
  typedef std::tuple<REST...> type;
};

template <std::size_t... Is>
constexpr auto indexSequenceReverse(std::index_sequence<Is...> const&)
  -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template <std::size_t N>
using makeIndexSequenceReverse = decltype(indexSequenceReverse(std::make_index_sequence<N>{}));

template <std::size_t V>
struct num {
  static const constexpr auto value = V;
};

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

template <std::size_t N, typename F>
void reverseFor(F func)
{
  for_(func, makeIndexSequenceReverse<N>());
}

template <typename... Ts>
void addOne(std::tuple<Ts...>& tuple, int maxOffset, bool& isEnd)
{
  constexpr auto size = std::tuple_size_v<std::tuple<Ts...>>;
  bool modify = true;
  reverseFor<size>([&](auto i) {
    if (modify) {
      std::get<i.value>(tuple)++;
      if (std::get<i.value>(tuple) < maxOffset + i.value + 1) {
        modify = false;
      }
    }
  });
  for_<size>([&](auto i) {
    if constexpr (i.value != 0) {
      if (std::get<i.value>(tuple) == maxOffset + i.value + 1) {
        std::get<i.value>(tuple) = std::get<i.value - 1>(tuple) + 1;
      }
    }
  });
  isEnd = std::get<0>(tuple) == maxOffset + 1;
}

template <typename... Ts>
void updateTuple(std::tuple<Ts...>& tuple, std::function<int(size_t index)> func, int maxOffset, bool& isEnd)
{
  constexpr auto size = std::tuple_size_v<std::tuple<Ts...>>;
  for_<size>([&](auto i) {
    std::get<i.value>(tuple) = func(i.value);
  });
  isEnd = std::get<0>(tuple) == maxOffset + 1;
}

template <typename T2, typename... Is, typename... Ts>
void indicesToIterators(std::tuple<Is...>& tuple, std::tuple<Ts...>& iteratorsTuple, T2 tableBegin)
{
  constexpr auto size = std::tuple_size_v<std::tuple<Ts...>>;
  for_<size>([&](auto i) {
    std::get<i.value>(iteratorsTuple) = tableBegin + std::get<i.value>(tuple);
  });
}

/// @return a vector of K-tuples with all the possible
/// combinations of the rows of the table T.
/// FIXME: move to coroutines once we have C++20
template <typename T, int K>
class TuplesGenerator
{
 public:
  using IteratorType = typename T::iterator;
  using IndexTupleType = typename generateTupleType<int, K>::type;
  using TupleType = typename generateTupleType<IteratorType, K>::type;
  using FunctionType = std::function<bool(const TupleType&)>;

  class TuplesIterator : public std::iterator<std::forward_iterator_tag, TupleType>
  {
   public:
    using reference = TupleType&;
    using value_type = TupleType;
    using pointer = TupleType*;
    using iterator_category = std::forward_iterator_tag;

    TuplesIterator() = delete;

    TuplesIterator(IteratorType begin, IteratorType end, int n, FunctionType condition)
      : mN(n), mK(K), mIsEnd(false), mTableBegin(begin), mTableEnd(end), mCondition(condition)
    {
      // First permutation
      goToBegin();
    }

    ~TuplesIterator() = default;

    // prefix increment
    TuplesIterator& operator++()
    {
      if (!mIsEnd) {
        addOne(mCurrentIndices, mN - mK, mIsEnd);
        indicesToIterators(mCurrentIndices, mCurrent, mTableBegin);
      }
      while (!mIsEnd && !mCondition(mCurrent)) {
        addOne(mCurrentIndices, mN - mK, mIsEnd);
        indicesToIterators(mCurrentIndices, mCurrent, mTableBegin);
      }
      return *this;
    }
    // postfix increment
    TuplesIterator operator++(int /*unused*/)
    {
      TuplesIterator copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return mCurrent;
    }
    // Assuming iterators with the same permutation to be at the same "position"
    bool operator==(const TuplesIterator& rh)
    {
      return (mIsEnd && rh.mIsEnd) || (mCurrent == rh.mCurrent);
    }
    bool operator!=(const TuplesIterator& rh)
    {
      return !(*this == rh);
    }

    void goToBegin()
    {
      updateTuple(
        mCurrentIndices, [](size_t index) { return index; }, mN - mK, mIsEnd);
      indicesToIterators(mCurrentIndices, mCurrent, mTableBegin);
      if (!mIsEnd && !mCondition(mCurrent)) {
        operator++();
      }
    }

    void goToEnd()
    {
      updateTuple(
        mCurrentIndices, [this](size_t index) { return mN - mK + index + 1; }, mN - mK, mIsEnd);
      indicesToIterators(mCurrentIndices, mCurrent, mTableBegin);
    }

   private:
    TupleType mCurrent;
    IndexTupleType mCurrentIndices;
    int mN;                   // number of elements
    int mK;                   // tuple size, number of elements to choose
    bool mIsEnd;              // whether there are any more tuples available
    FunctionType mCondition;  // only tuples satisfying the condition will be outputed
    IteratorType mTableBegin; // start of the table for which tuples are generated
    IteratorType mTableEnd;   // end of the table for which tuples are generated
  };

  using iterator = TuplesIterator;
  using const_iterator = TuplesIterator;

  inline iterator begin()
  {
    return iterator(mTableBegin, mTableEnd, mN, mCondition);
  }
  inline iterator end()
  {
    auto it = iterator(mTableBegin, mTableEnd, mN, mCondition);
    it.goToEnd();
    return it;
  }

  TuplesGenerator() = delete;
  TuplesGenerator(const T& table, FunctionType condition)
    : mTableBegin(table.begin()), mTableEnd(table.end()), mN(table.size()), mCondition(condition)
  {
    static_assert(K > 0);
  }
  ~TuplesGenerator() = default;

 private:
  IteratorType mTableBegin;
  IteratorType mTableEnd;
  int mN;
  FunctionType mCondition;
};

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
