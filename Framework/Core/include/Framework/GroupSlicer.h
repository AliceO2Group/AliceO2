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

#ifndef FRAMEWORK_GROUP_SLICER_H_
#define FRAMEWORK_GROUP_SLICER_H_

#include "Framework/Pack.h"
#include "Framework/ASoA.h"
#include "Framework/AnalysisHelpers.h"

#include <arrow/util/config.h>
#include <arrow/util/key_value_metadata.h>
#include <type_traits>
#include <string>
namespace
{
template <typename T>
auto getMatcherFor(std::string const& columnName, o2::header::DataOrigin newOrigin = o2::header::DataOrigin{"AOD"})
{
  auto matcher = o2::soa::getMatcherFromTypeForKey<std::decay_t<T>>(columnName);
  if ((matcher.origin == o2::header::DataOrigin{"AOD"}) && (newOrigin != o2::header::DataOrigin{"AOD"})) {
    matcher = o2::framework::replaceOrigin(matcher, newOrigin);
  }
  return matcher;
}
} // namespace

namespace o2::framework
{
template <typename G, typename... A>
struct GroupSlicer {
  using grouping_t = std::decay_t<G>;
  GroupSlicer(G& gt, std::tuple<A...>& at, ArrowTableSlicingCache& slices, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
    : max{gt.size()},
      mBegin{GroupSlicerIterator(gt, at, slices, newOrigin)}
  {
  }

  struct GroupSlicerSentinel {
    int64_t position;
  };

  struct GroupSlicerIterator {
    using associated_pack_t = framework::pack<A...>;

    GroupSlicerIterator() = default;
    GroupSlicerIterator(GroupSlicerIterator const&) = default;
    GroupSlicerIterator(GroupSlicerIterator&&) = default;
    GroupSlicerIterator& operator=(GroupSlicerIterator const&) = default;
    GroupSlicerIterator& operator=(GroupSlicerIterator&&) = default;

    template <typename T>
    auto splittingFunction(T&&)
    {
    }

    template <soa::is_table T>
      requires(o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<T>>() && !soa::is_smallgroups<T>)
    auto splittingFunction(T&& table)
    {
      if (table.size() == 0) {
        return;
      }
      sliceInfos[framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{})] = mSlices->getCacheFor(Entry("", getMatcherFor<T>(mIndexColumnName, replacementOrigin), mIndexColumnName));
    }

    template <soa::is_smallgroups T>
      requires(o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<T>>())
    auto splittingFunction(T&& table)
    {
      if (table.tableSize() == 0) {
        return;
      }
      sliceInfosUnsorted[framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{})] = mSlices->getCacheUnsortedFor(Entry("", getMatcherFor<T>(mIndexColumnName, replacementOrigin), mIndexColumnName));
    }

    template <typename T>
    auto extractingFunction(T&&)
    {
    }

    template <soa::is_filtered_table T>
    auto extractingFunction(T&& table)
    {
      constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
      selections[index] = &table.getSelectedRows();
      starts[index] = selections[index]->begin();
    }

    GroupSlicerIterator(G& gt, std::tuple<A...>& at, ArrowTableSlicingCache& slices, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
      : mIndexColumnName{std::string("fIndex") + o2::framework::cutString(o2::soa::getLabelFromType<G>())},
        mGt{&gt},
        mAt{&at},
        mGroupingElement{gt.begin()},
        position{0},
        mSlices{&slices},
        replacementOrigin{newOrigin}
    {
      if constexpr (soa::is_filtered_table<std::decay_t<G>>) {
        groupSelection = mGt->getSelectedRows();
      }

      /// prepare slices and offsets for all associated tables that have index
      /// to grouping table
      /// extract selections from filtered associated tables

      [this]<size_t... Is>(std::tuple<A...>& at, std::index_sequence<Is...>) {
        (splittingFunction(std::get<Is>(at)), ...);
        (extractingFunction(std::get<Is>(at)), ...);
      }(*mAt, std::make_index_sequence<sizeof...(A)>());
    }

    GroupSlicerIterator& operator++()
    {
      ++position;
      ++mGroupingElement;
      return *this;
    }

    GroupSlicerIterator operator+(int64_t inc) const
    {
      GroupSlicerIterator copy = *this;
      copy.position += inc;
      copy.mGroupingElement = copy.mGroupingElement + inc;
      return copy;
    }

    GroupSlicerIterator operator+=(int64_t inc)
    {
      position += inc;
      mGroupingElement += inc;
      return *this;
    }

    bool operator==(GroupSlicerSentinel const& other)
    {
      return O2_BUILTIN_UNLIKELY(position == other.position);
    }

    bool operator!=(GroupSlicerSentinel const& other)
    {
      return O2_BUILTIN_LIKELY(position != other.position);
    }

    auto& groupingElement()
    {
      return mGroupingElement;
    }

    GroupSlicerIterator& operator*()
    {
      return *this;
    }

    auto associatedTables()
    {
      return std::make_tuple(prepareArgument<A>()...);
    }

    template <soa::is_smallgroups A1>
      requires(o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<A1>>() && soa::is_filtered_table<A1>)
    auto prepareArgument()
    {
      constexpr auto index = framework::has_type_at_v<A1>(associated_pack_t{});
      auto& originalTable = std::get<A1>(*mAt);
      uint64_t pos;
      if constexpr (soa::is_filtered_table<std::decay_t<G>>) {
        pos = groupSelection[position];
      } else {
        pos = position;
      }
      // generic split
      auto selection = sliceInfosUnsorted[index].getSliceFor(pos);
      // intersect selections
      o2::soa::SelectionVector s;
      if (selections[index]->empty()) {
        if (!selection.empty()) {
          std::copy(selection.begin(), selection.end(), std::back_inserter(s));
        }
      } else {
        if (!selection.empty()) {
          if constexpr (std::decay_t<A1>::applyFilters) {
            std::set_intersection(selection.begin(), selection.end(), selections[index]->begin(), selections[index]->end(), std::back_inserter(s));
          } else {
            std::copy(selection.begin(), selection.end(), std::back_inserter(s));
          }
        }
      }
      std::decay_t<A1> typedTable{{originalTable.asArrowTable()}, std::move(s)};
      typedTable.bindInternalIndicesTo(&originalTable);
      return typedTable;
    }

    template <soa::is_filtered_table A1>
      requires(o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<A1>>() && !soa::is_smallgroups<A1>)
    auto prepareArgument()
    {
      constexpr auto index = framework::has_type_at_v<A1>(associated_pack_t{});
      auto& originalTable = std::get<A1>(*mAt);
      if (originalTable.size() == 0) {
        return originalTable;
      }
      uint64_t pos;
      if constexpr (soa::is_filtered_table<std::decay_t<G>>) {
        pos = groupSelection[position];
      } else {
        pos = position;
      }
      // optimized split
      auto oc = sliceInfos[index].getSliceFor(pos);
      uint64_t offset = oc.first;
      auto count = oc.second;
      auto groupedElementsTable = originalTable.asArrowTable()->Slice(offset, count);
      if (count == 0) {
        return std::decay_t<A1>{{groupedElementsTable}, soa::SelectionVector{}};
      }

      // for each grouping element we need to slice the selection vector
      auto start_iterator = std::lower_bound(starts[index], selections[index]->end(), offset);
      auto stop_iterator = std::lower_bound(start_iterator, selections[index]->end(), offset + count);
      starts[index] = stop_iterator;
      soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
      std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                     [&offset](int64_t idx) {
                       return idx - static_cast<int64_t>(offset);
                     });

      std::decay_t<A1> typedTable{{groupedElementsTable}, std::move(slicedSelection), offset};
      typedTable.bindInternalIndicesTo(&originalTable);
      return typedTable;
    }

    template <soa::is_table A1>
      requires(o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<A1>>() && !soa::is_smallgroups<A1> && !soa::is_filtered_table<A1>)
    auto prepareArgument()
    {
      constexpr auto index = framework::has_type_at_v<A1>(associated_pack_t{});
      auto& originalTable = std::get<A1>(*mAt);
      if (originalTable.size() == 0) {
        return originalTable;
      }
      uint64_t pos;
      if constexpr (soa::is_filtered_table<std::decay_t<G>>) {
        pos = groupSelection[position];
      } else {
        pos = position;
      }
      // optimized split
      auto [offset, count] = sliceInfos[index].getSliceFor(pos);
      auto groupedElementsTable = originalTable.rawSlice(offset, offset + count - 1);
      groupedElementsTable.bindInternalIndicesTo(&originalTable);
      return groupedElementsTable;
    }

    template <soa::is_table A1>
      requires(!o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<A1>>() && !soa::is_smallgroups<A1>)
    auto prepareArgument()
    {
      return std::get<A1>(*mAt);
    }

    std::string mIndexColumnName;
    G const* mGt;
    std::tuple<A...>* mAt;
    typename grouping_t::iterator mGroupingElement;
    uint64_t position = 0;
    std::span<int64_t const> groupSelection;
    std::array<std::span<int64_t const> const*, sizeof...(A)> selections;
    std::array<std::span<int64_t const>::iterator, sizeof...(A)> starts;

    std::array<SliceInfoPtr, sizeof...(A)> sliceInfos;
    std::array<SliceInfoUnsortedPtr, sizeof...(A)> sliceInfosUnsorted;
    ArrowTableSlicingCache* mSlices;
    header::DataOrigin replacementOrigin;
  };

  GroupSlicerIterator& begin()
  {
    return mBegin;
  }

  GroupSlicerSentinel end()
  {
    return GroupSlicerSentinel{max};
  }
  int64_t max;
  GroupSlicerIterator mBegin;
};

} // namespace o2::framework
#endif // FRAMEWORK_GROUP_SLICER_H_
