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

template <typename... T2s>
void addOne(std::tuple<T2s...>& tuple, const std::tuple<T2s...>& maxOffset, bool& isEnd)
{
  constexpr auto size = std::tuple_size_v<std::tuple<T2s...>>;
  bool modify = true;
  for_<size>([&](auto i) {
    if (modify) {
      std::get<size - i.value - 1>(tuple)++;
      if (std::get<size - i.value - 1>(tuple) != std::get<size - i.value - 1>(maxOffset)) {
        for_<i.value>([&](auto j) {
          std::get<size - i.value + j.value>(tuple).setCursor(*std::get<1>(std::get<size - i.value + j.value - 1>(tuple).getIndices()) + 1);
        });
        modify = false;
      }
    }
  });
  isEnd = modify;
}

/// @return next combination of rows of tables.
/// FIXME: move to coroutines once we have C++20
template <typename... Ts>
class CombinationsGenerator
{
 public:
  using CombinationType = std::tuple<typename Ts::iterator...>;

  class CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>
  {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const Ts&... tables) : mIsEnd(false),
                                                mMaxOffset(tables.end()...),
                                                mCurrent(tables.begin()...)
    {
      if (((tables.size() <= sizeof...(Ts)) || ...)) {
        mIsEnd = true;
        return;
      }
      for_<sizeof...(Ts) - 1>([&](auto i) {
        std::get<i.value>(mMaxOffset).moveByIndex(-sizeof...(Ts) + i.value + 1);
        std::get<i.value>(mCurrent).moveByIndex(i.value);
      });
      std::get<sizeof...(Ts) - 1>(mCurrent).moveByIndex(sizeof...(Ts) - 1);
    }

    CombinationsIterator(CombinationsIterator const&) = default;
    CombinationsIterator& operator=(CombinationsIterator const&) = default;
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

    void moveToEnd()
    {
      for_<sizeof...(Ts) - 1>([&](auto i) {
        std::get<i.value>(mCurrent).setCursor(*std::get<1>(std::get<i.value>(mMaxOffset).getIndices()));
      });
      std::get<sizeof...(Ts) - 1>(mCurrent).moveToEnd();
      mIsEnd = true;
    }

   private:
    CombinationType mCurrent;
    CombinationType mMaxOffset; // one position past maximum acceptable position for each element of combination
    bool mIsEnd;                // whether there are any more tuples available
  };

  using iterator = CombinationsIterator;
  using const_iterator = CombinationsIterator;

  inline iterator begin()
  {
    return iterator(mBegin);
  }
  inline iterator end()
  {
    return iterator(mEnd);
  }
  inline const_iterator begin() const
  {
    return iterator(mBegin);
  }
  inline const_iterator end() const
  {
    return iterator(mEnd);
  }

  CombinationsGenerator() = delete;
  CombinationsGenerator(const Ts&... tables) : mBegin(tables...), mEnd(tables...)
  {
    static_assert(sizeof...(Ts) > 0);
    mEnd.moveToEnd();
  }
  ~CombinationsGenerator() = default;

 private:
  iterator mBegin;
  iterator mEnd;
};

template <typename... T2s>
CombinationsGenerator<T2s...> combinations(const T2s&... tables)
{
  return CombinationsGenerator<T2s...>(tables...);
}

template <typename... T2s>
CombinationsGenerator<Filtered<T2s>...> combinations(const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  return CombinationsGenerator<Filtered<T2s>...>({{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
