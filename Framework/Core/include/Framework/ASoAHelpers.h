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

template <typename... Ts>
struct CombinationsIndexPolicyBase {
  using CombinationType = std::tuple<typename Ts::iterator...>;

  CombinationsIndexPolicyBase(const Ts&... tables) : mIsEnd(false),
                                                     mMaxOffset(tables.end()...),
                                                     mCurrent(tables.begin()...)
  {
    constexpr auto k = sizeof...(Ts);
    if (((tables.size() < k) || ...)) {
      mIsEnd = true;
    }
  }

  void moveToEnd() {}
  void addOne() {}

  CombinationType mCurrent;
  CombinationType mMaxOffset; // one position past maximum acceptable position for each element of combination
  bool mIsEnd;                // whether there are any more tuples available
};

template <typename... Ts>
struct CombinationsUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  CombinationsUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset).moveByIndex(-k + i.value + 1);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(*std::get<1>(std::get<i.value>(this->mMaxOffset).getIndices()));
    });
    std::get<k - 1>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (std::get<curInd>(this->mCurrent) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices()));
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsStrictlyUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  CombinationsStrictlyUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset).moveByIndex(-k + i.value + 1);
      std::get<i.value>(this->mCurrent).moveByIndex(i.value);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(*std::get<1>(std::get<i.value>(this->mMaxOffset).getIndices()));
    });
    std::get<k - 1>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (std::get<curInd>(this->mCurrent) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices()) + 1);
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsFullIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  CombinationsFullIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (std::get<curInd>(this->mCurrent) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(0);
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

/// @return next combination of rows of tables.
/// FIXME: move to coroutines once we have C++20
template <template <typename...> typename P, typename... Ts>
struct CombinationsGenerator {
 public:
  using CombinationType = std::tuple<typename Ts::iterator...>;

  struct CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>, public P<Ts...> {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const P<Ts...>& policy) : P<Ts...>(policy) {}

    CombinationsIterator(CombinationsIterator const&) = default;
    CombinationsIterator& operator=(CombinationsIterator const&) = default;
    ~CombinationsIterator() = default;

    // prefix increment
    CombinationsIterator& operator++()
    {
      if (!this->mIsEnd) {
        this->addOne();
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
      return this->mCurrent;
    }
    bool operator==(const CombinationsIterator& rh)
    {
      return (this->mIsEnd && rh.mIsEnd) || (this->mCurrent == rh.mCurrent);
    }
    bool operator!=(const CombinationsIterator& rh)
    {
      return !(*this == rh);
    }
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
  CombinationsGenerator(const P<Ts...>& policy) : mBegin(policy), mEnd(policy)
  {
    static_assert(sizeof...(Ts) > 0);
    mEnd.moveToEnd();
  }
  ~CombinationsGenerator() = default;

 private:
  iterator mBegin;
  iterator mEnd;
};

template <typename T2, typename... T2s>
auto combinations(const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, T2, T2s...>(CombinationsStrictlyUpperIndexPolicy(table, tables...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy, T2, T2s...>(CombinationsFullIndexPolicy(table, tables...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, Filtered<T2>, Filtered<T2s>...>(CombinationsStrictlyUpperIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy, Filtered<T2>, Filtered<T2s>...>(CombinationsFullIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  }
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, T2s...> combinations(const P2<T2s...>& policy, const T2s&... tables)
{
  return CombinationsGenerator<P2, T2s...>(P2<T2s...>(tables...));
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, Filtered<T2s>...> combinations(const P2<T2s...>& policy, const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  return CombinationsGenerator<P2, Filtered<T2s>...>(P2<Filtered<T2s>...>({{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
}

// This shortened version cannot be used for Filtered
// (unless users create filtered tables themselves before policy creation)
template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, T2s...> combinations(const P2<T2s...>& policy)
{
  return CombinationsGenerator<P2, T2s...>(policy);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
