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
#include "Framework/BinningPolicy.h"
#include "Framework/Pack.h"
#include "Framework/SliceCache.h"
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

template <typename T, typename G>
using is_index_to_g_t = typename std::conditional<o2::soa::is_binding_compatible_v<G, typename T::binding_t>(), std::true_type, std::false_type>::type;

template <typename G, typename A>
expressions::BindingNode getMatchingIndexNode()
{
  using external_index_columns_pack = typename A::external_index_columns_t;
  using selected_indices_t = selected_pack_multicondition<is_index_to_g_t, pack<G>, external_index_columns_pack>;
  static_assert(pack_size(selected_indices_t{}) == 1, "No matching index column from associated to grouping");
  using index_column_t = pack_head_t<selected_indices_t>;
  auto label = std::string(index_column_t::mLabel);
  return expressions::BindingNode{label, typeid(typename index_column_t::column_t).hash_code(), expressions::selectArrowType<typename index_column_t::type>()};
}

template <typename T1, typename GroupingPolicy, typename BP, typename G, typename... As>
struct GroupedCombinationsGenerator {
  using GroupedIteratorType = pack_to_tuple_t<interleaved_pack_t<repeated_type_pack_t<typename G::iterator, sizeof...(As)>, pack<As...>>>;

  struct GroupedIterator : public GroupingPolicy {
   public:
    using reference = GroupedIteratorType&;
    using value_type = GroupedIteratorType;
    using pointer = GroupedIteratorType*;
    using iterator_category = std::forward_iterator_tag;

    GroupedIterator(const GroupingPolicy& groupingPolicy, SliceCache* cache_)
      : GroupingPolicy(groupingPolicy), mIndexColumns{getMatchingIndexNode<G, As>()...}, cache{cache_}
    {
    }
    template <typename... T2s>
    GroupedIterator(const GroupingPolicy& groupingPolicy, const G& grouping, const std::tuple<T2s...>& associated, SliceCache* cache_)
      : GroupingPolicy(groupingPolicy),
        mGrouping{std::make_shared<G>(std::vector{grouping.asArrowTable()})},
        mAssociated{std::make_shared<std::tuple<As...>>(std::make_tuple(std::get<has_type_at<As>(pack<T2s...>{})>(associated)...))},
        mIndexColumns{getMatchingIndexNode<G, As>()...},
        cache{cache_}
    {
      if constexpr (soa::is_soa_filtered_v<std::decay_t<G>>) {
        mGrouping = std::make_shared<G>(std::vector{grouping.asArrowTable()}, grouping.getSelectedRows());
      } else {
        mGrouping = std::make_shared<G>(std::vector{grouping.asArrowTable()});
      }
      setMultipleGroupingTables<sizeof...(As)>(grouping);
      if (!this->mIsEnd) {
        setCurrentGroupedCombination();
      }
    }

    GroupedIterator(GroupedIterator const&) = default;
    GroupedIterator& operator=(GroupedIterator const&) = default;
    ~GroupedIterator() = default;

    template <typename... T2s>
    void setTables(const G& grouping, const std::tuple<T2s...>& associated)
    {
      if constexpr (soa::is_soa_filtered_v<std::decay_t<G>>) {
        mGrouping = std::make_shared<G>(std::vector{grouping.asArrowTable()}, grouping.getSelectedRows());
      } else {
        mGrouping = std::make_shared<G>(std::vector{grouping.asArrowTable()});
      }
      mAssociated = std::make_shared<std::tuple<As...>>(std::make_tuple(std::get<has_type_at<As>(pack<T2s...>{})>(associated)...));
      setMultipleGroupingTables<sizeof...(As)>(grouping);
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
    bool operator==(const GroupedIterator& rh) const
    {
      return (this->mIsEnd && rh.mIsEnd) || (this->mCurrent == rh.mCurrent);
    }
    bool operator!=(const GroupedIterator& rh) const
    {
      return !(*this == rh);
    }

   private:
    std::tuple<As...> getAssociatedTables()
    {
      return doGetAssociatedTables(std::make_index_sequence<sizeof...(As)>());
    }

    template <std::size_t... Is>
    std::tuple<As...> doGetAssociatedTables(std::index_sequence<Is...>)
    {
      return std::make_tuple(getAssociatedTable<Is>()...);
    }

    template <std::size_t I>
    auto getAssociatedTable()
    {
      uint64_t ind = *std::get<0>(std::get<I>(this->mCurrent).getIndices());
      if (std::get<I>(*mAssociated).size() == 0) {
        return std::get<I>(*mAssociated);
      }
      return std::get<I>(*mAssociated).sliceByCached(mIndexColumns[I], ind, *cache);
    }

    void setCurrentGroupedCombination()
    {
      if (!this->mIsEnd) {
        auto& currentGrouping = GroupingPolicy::mCurrent;
        mSlices.emplace(getAssociatedTables());
        mCurrentGrouped.emplace(interleaveTuples(currentGrouping, *mSlices));
      }
    }

    std::array<expressions::BindingNode, sizeof...(As)> mIndexColumns;
    std::shared_ptr<G> mGrouping;
    std::shared_ptr<std::tuple<As...>> mAssociated;
    std::optional<std::tuple<As...>> mSlices;
    std::optional<GroupedIteratorType> mCurrentGrouped;
    SliceCache* cache = nullptr;
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

  GroupedCombinationsGenerator(const BP& binningPolicy, int catNeighbours, const T1& outsider, SliceCache* cache)
    : mBegin(GroupingPolicy(binningPolicy, catNeighbours, outsider), cache),
      mEnd(GroupingPolicy(binningPolicy, catNeighbours, outsider), cache) {}
  template <typename... T2s>
  GroupedCombinationsGenerator(const BP& binningPolicy, int catNeighbours, const T1& outsider, const G& grouping, const std::tuple<T2s...>& associated, SliceCache* cache)
    : GroupedCombinationsGenerator(binningPolicy, catNeighbours, outsider, cache)
  {
    setTables(grouping, associated);
  }
  GroupedCombinationsGenerator(GroupedCombinationsGenerator const&) = default;
  GroupedCombinationsGenerator& operator=(GroupedCombinationsGenerator const&) = default;
  ~GroupedCombinationsGenerator() = default;

  template <typename... T2s>
  void setTables(const G& grouping, const std::tuple<T2s...>& associated)
  {
    mBegin.setTables(grouping, associated);
    mEnd.setTables(grouping, associated);
    mEnd.moveToEnd();
  }

 private:
  iterator mBegin;
  iterator mEnd;
};

// Aliases for 2-particle correlations
// 'Pair' and 'Triple' can be used for same kind pair/triple, too, just specify the same type twice
template <typename G, typename A1, typename A2, typename BP, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, G, G>>
using Pair = GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, A1, A2>;
template <typename G, typename A, typename BP, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, G, G>>
using SameKindPair = GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, A, A>;

// Aliases for 3-particle correlations
template <typename G, typename A1, typename A2, typename A3, typename BP, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, G, G, G>>
using Triple = GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, A1, A2, A3>;
template <typename G, typename A, typename BP, typename T1 = int, typename GroupingPolicy = o2::soa::CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, G, G, G>>
using SameKindTriple = GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, A, A, A>;

} // namespace o2::framework
#endif // FRAMEWORK_GROUPEDCOMBINATIONS_H_
