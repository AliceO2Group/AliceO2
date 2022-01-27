// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_GROUPEDCOMBINATIONS_H
#define FRAMEWORK_GROUPEDCOMBINATIONS_H

#include "Framework/ASoAHelpers.h"
#include "Framework/GroupSlicer.h"
#include "Framework/Pack.h"
#include <optional>

namespace o2::framework
{

// Create an instance of a tuple interleaved from given tuples
template <typename... T1s, typename... T2s, std::size_t... Is>
auto interleaveTuplesImpl(std::tuple<T1s...>& t1, std::tuple<T2s...>& t2, std::index_sequence<Is...>)
{
  return std::tuple_cat(std::make_tuple(std::get<Is>(t1), std::get<Is>(t2))...);
}

template <typename... T1s, typename... T2s>
auto interleaveTuples(std::tuple<T1s...>& t1, std::tuple<T2s...>& t2)
{
  return interleaveTuplesImpl(t1, t2, std::index_sequence_for<T1s...>());
}

// Functions to create a tuple from N runs of a function that returns a value
template <std::size_t I, typename R, typename C, typename... Args>
R execFunctionWithDummyIndex(R (C::*f)(Args...), C& obj, Args... args)
{
  return (obj.*f)(args...);
}

template <typename R, typename C, typename... Args, std::size_t... Is>
auto functionToTupleImpl(R (C::*f)(Args...), C& obj, Args... args, std::index_sequence<Is...>)
{
  return std::make_tuple((execFunctionWithDummyIndex<Is>(f, obj, args...))...);
}

template <std::size_t N, typename R, typename C, typename... Args>
auto functionToTuple(R (C::*f)(Args...), C& obj, Args... args)
{
  return functionToTupleImpl(f, obj, args..., std::make_index_sequence<N>());
}

template <typename T1, typename GroupingPolicy, typename H, typename G, typename... Ts>
struct GroupedCombinationsGenerator {
};

template <typename T1, typename GroupingPolicy, typename H, typename G, typename... Us, typename... As>
struct GroupedCombinationsGenerator<T1, GroupingPolicy, H, G, pack<Us...>, As...> {
  using joinIterator = typename soa::Join<H, G>::table_t::iterator;
  using GroupedIteratorType = pack_to_tuple_t<interleaved_pack_t<repeated_type_pack_t<joinIterator, sizeof...(As)>, pack<As...>>>;

  struct GroupedIterator : public std::iterator<std::forward_iterator_tag, GroupedIteratorType>, public GroupingPolicy {
   public:
    using reference = GroupedIteratorType&;
    using value_type = GroupedIteratorType;
    using pointer = GroupedIteratorType*;
    using iterator_category = std::forward_iterator_tag;

    GroupedIterator(const GroupingPolicy& groupingPolicy) : GroupingPolicy(groupingPolicy) {}
    GroupedIterator(const GroupingPolicy& groupingPolicy, const H& hashes, const G& grouping, const std::shared_ptr<GroupSlicer<G, Us...>>&& slicer_ptr) : GroupingPolicy(groupingPolicy), mSlicer{std::move(slicer_ptr)}, mGrouping{std::make_shared<G>(std::vector{grouping.asArrowTable()})}
    {
      GroupingPolicy::setTables(join(hashes, grouping), join(hashes, grouping));
      if (!this->mIsEnd) {
        setCurrentGroupedCombination();
      }
    }

    GroupedIterator(GroupedIterator const&) = default;
    GroupedIterator& operator=(GroupedIterator const&) = default;
    ~GroupedIterator() = default;

    void setTables(const H& hashes, const G& grouping, std::shared_ptr<GroupSlicer<G, Us...>> slicer_ptr)
    {
      mGrouping = std::make_shared<G>(std::vector{grouping.asArrowTable()});
      mSlicer = slicer_ptr;
      setMultipleGroupingTables<sizeof...(As)>(join(hashes, grouping));
      if (!this->mIsEnd) {
        setCurrentGroupedCombination();
      }
    }

    template <std::size_t N, typename T, typename... Args>
    void setMultipleGroupingTables(const T& param, const Args&... args)
    {
      if constexpr (N == 1) {
        GroupingPolicy::setTables(param, args...);
      } else {
        setMultipleGroupingTables<N - 1>(param, param, args...);
      }
    }

    void moveToEnd()
    {
      GroupingPolicy::moveToEnd();
    }

    // prefix increment
    GroupedIterator& operator++()
    {
      if (!this->mIsEnd) {
        this->addOne();
        setCurrentGroupedCombination();
      }
      return *this;
    }
    // postfix increment
    GroupedIterator operator++(int /*unused*/)
    {
      GroupedIterator copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return *mCurrentGrouped;
    }
    bool operator==(const GroupedIterator& rh)
    {
      return (this->mIsEnd && rh.mIsEnd) || (this->mCurrent == rh.mCurrent);
    }
    bool operator!=(const GroupedIterator& rh)
    {
      return !(*this == rh);
    }

   private:
    std::tuple<As...> getAssociatedTables()
    {
      auto& currentGrouping = GroupingPolicy::mCurrent;
      constexpr auto k = sizeof...(As);
      auto slicerIterators = functionToTuple<k>(&GroupSlicer<G, Us...>::begin, *mSlicer);
      o2::soa::for_<k>([&](auto i) {
        auto col = std::get<i.value>(currentGrouping);
        for (auto& slice : *mSlicer) {
          if (slice.groupingElement().globalIndex() == col.globalIndex()) {
            std::get<i.value>(slicerIterators) = slice;
            break;
          }
        }
      });

      return getSlices(slicerIterators, std::index_sequence_for<As...>());
    }

    template <std::size_t I, typename T, typename... Ts>
    auto getSliceAt(std::tuple<Ts...>& t)
    {
      auto it = std::get<I>(t); // Get the tables corresponding to the grouping at index I
      auto associatedType = it.template prepareArgument<T>();
      return associatedType;
    }

    template <typename... Ts, std::size_t... Is>
    std::tuple<As...> getSlices(std::tuple<Ts...>& t, std::index_sequence<Is...> is)
    {
      return std::make_tuple(getSliceAt<Is, As, Ts...>(t)...);
    }

    void setCurrentGroupedCombination()
    {
      std::tuple<As...> initAssociatedTables = getAssociatedTables();
      constexpr auto k = sizeof...(As);
      bool moveForward = false;
      o2::soa::for_<k>([&](auto i) {
        if (std::get<i.value>(initAssociatedTables).size() == 0) {
          moveForward = true;
        }
      });
      while (!this->mIsEnd && moveForward) {
        GroupingPolicy::addOne();
        std::tuple<As...> temp = getAssociatedTables();
        moveForward = false;
        o2::soa::for_<k>([&](auto i) {
          if (std::get<i.value>(temp).size() == 0) {
            moveForward = true;
          }
        });
      }
      std::tuple<As...> associatedTables = getAssociatedTables();

      if (!this->mIsEnd) {
        auto& currentGrouping = GroupingPolicy::mCurrent;
        o2::soa::for_<k>([&](auto i) {
          std::get<i.value>(associatedTables).bindExternalIndices(mGrouping.get());
        });

        mCurrentGrouped.emplace(interleaveTuples(currentGrouping, associatedTables));
      }
    }

    std::shared_ptr<GroupSlicer<G, Us...>> mSlicer = nullptr;
    std::shared_ptr<G> mGrouping;
    std::optional<GroupedIteratorType> mCurrentGrouped;
  };

  using iterator = GroupedIterator;
  using const_iterator = GroupedIterator;

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

  GroupedCombinationsGenerator(const char* category, int catNeighbours, const T1& outsider) : mBegin(GroupingPolicy(category, catNeighbours, outsider)), mEnd(GroupingPolicy(category, catNeighbours, outsider)), mCategory(category), mCatNeighbours(catNeighbours), mOutsider(outsider) {}
  GroupedCombinationsGenerator(const char* category, int catNeighbours, const T1& outsider, H& hashes, G& grouping, std::tuple<Us...>& associated) : GroupedCombinationsGenerator(category, catNeighbours, outsider)
  {
    setTables(hashes, grouping, associated);
  }
  ~GroupedCombinationsGenerator() = default;

  void setTables(H& hashes, G& grouping, std::tuple<Us...>& associated)
  {
    if (mSlicer == nullptr) {
      mSlicer = std::make_shared<GroupSlicer<G, Us...>>(grouping, associated);
      mBegin.setTables(hashes, grouping, mSlicer);
      mEnd.setTables(hashes, grouping, mSlicer);
      mEnd.moveToEnd();
    }
  }

 private:
  iterator mBegin;
  iterator mEnd;
  const char* mCategory;
  const int mCatNeighbours;
  const T1 mOutsider;
  std::shared_ptr<GroupSlicer<G, Us...>> mSlicer = nullptr;
};

// Aliases for 2-particle correlations
// 'Pair' and 'Triple' can be used for same kind pair/triple, too, just specify the same type twice
template <typename H, typename G>
using joinedCollisions = typename soa::Join<H, G>::table_t;
template <typename H, typename G, typename A1, typename A2, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<T1, joinedCollisions<H, G>, joinedCollisions<H, G>>>
using Pair = GroupedCombinationsGenerator<T1, GroupingPolicy, H, G, unique_pack_t<pack<A1, A2>>>;
template <typename H, typename G, typename A, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<T1, joinedCollisions<H, G>, joinedCollisions<H, G>>>
using SameKindPair = GroupedCombinationsGenerator<T1, GroupingPolicy, H, G, pack<A>, A, A>;

// Aliases for 3-particle correlations
template <typename H, typename G, typename A1, typename A2, typename A3, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<T1, joinedCollisions<H, G>, joinedCollisions<H, G>, joinedCollisions<H, G>>>
using Triple = GroupedCombinationsGenerator<T1, GroupingPolicy, H, G, unique_pack_t<pack<A1, A2, A3>>>;
template <typename H, typename G, typename A, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<T1, joinedCollisions<H, G>, joinedCollisions<H, G>, joinedCollisions<H, G>>>
using SameKindTriple = GroupedCombinationsGenerator<T1, GroupingPolicy, H, G, pack<A>, A, A, A>;

} // namespace o2::framework
#endif // FRAMEWORK_GROUPEDCOMBINATIONS_H_
