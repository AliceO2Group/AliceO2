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

#include <arrow/util/config.h>
#include <arrow/util/key_value_metadata.h>
#include <type_traits>
#include <string>

namespace o2::framework
{

template <typename G, typename... A>
struct GroupSlicer {
  using grouping_t = std::decay_t<G>;
  GroupSlicer(G& gt, std::tuple<A...>& at, ArrowTableSlicingCache& slices)
    : max{gt.size()},
      mBegin{GroupSlicerIterator(gt, at, slices)}
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
    auto splittingFunction(T&& table)
    {
      constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
      if constexpr (o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<T>>()) {
        auto binding = o2::soa::getLabelFromTypeForKey<std::decay_t<T>>(mIndexColumnName);
        auto bk = std::make_pair(binding, mIndexColumnName);
        if constexpr (!o2::soa::is_smallgroups_v<std::decay_t<T>>) {
          if (table.size() == 0) {
            return;
          }
          sliceInfos[index] = mSlices->getCacheFor(bk);
        } else {
          if (table.tableSize() == 0) {
            return;
          }
          sliceInfosUnsorted[index] = mSlices->getCacheUnsortedFor(bk);
        }
      }
    }

    template <typename T>
    auto extractingFunction(T&& table)
    {
      if constexpr (soa::is_soa_filtered_v<std::decay_t<T>>) {
        constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
        selections[index] = &table.getSelectedRows();
        starts[index] = selections[index]->begin();
      }
    }

    GroupSlicerIterator(G& gt, std::tuple<A...>& at, ArrowTableSlicingCache& slices)
      : mIndexColumnName{std::string("fIndex") + o2::framework::cutString(o2::soa::getLabelFromType<G>())},
        mGt{&gt},
        mAt{&at},
        mGroupingElement{gt.begin()},
        position{0},
        mSlices{&slices}
    {
      if constexpr (soa::is_soa_filtered_v<std::decay_t<G>>) {
        groupSelection = mGt->getSelectedRows();
      }

      /// prepare slices and offsets for all associated tables that have index
      /// to grouping table
      ///
      std::apply(
        [&](auto&&... x) -> void {
          (splittingFunction(x), ...);
        },
        at);
      /// extract selections from filtered associated tables
      std::apply(
        [&](auto&&... x) -> void {
          (extractingFunction(x), ...);
        },
        at);
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

    template <typename A1>
    auto prepareArgument()
    {
      constexpr auto index = framework::has_type_at_v<A1>(associated_pack_t{});
      auto& originalTable = std::get<A1>(*mAt);

      if constexpr (o2::soa::relatedByIndex<std::decay_t<G>, std::decay_t<A1>>()) {
        uint64_t pos;
        if constexpr (soa::is_soa_filtered_v<std::decay_t<G>>) {
          pos = groupSelection[position];
        } else {
          pos = position;
        }

        if constexpr (!o2::soa::is_smallgroups_v<std::decay_t<A1>>) {
          // optimized split
          if (originalTable.size() == 0) {
            return originalTable;
          }
          auto oc = sliceInfos[index].getSliceFor(pos);
          uint64_t offset = oc.first;
          auto count = oc.second;
          if constexpr (soa::is_soa_filtered_v<std::decay_t<A1>>) {
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

          } else {
            auto groupedElementsTable = originalTable.rawSlice(offset, offset + count - 1);
            groupedElementsTable.bindInternalIndicesTo(&originalTable);
            return groupedElementsTable;
          }
        } else {
          // generic split
          if constexpr (soa::is_soa_filtered_v<std::decay_t<A1>>) {
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
          } else {
            throw runtime_error("Unsorted grouped table needs to be used with soa::SmallGroups<>");
          }
        }
      } else {
        static_assert(!o2::soa::is_smallgroups_v<std::decay_t<A1>>, "SmallGroups used with a table that is not related by index to the gouping table");
        return originalTable;
      }
    }

    std::string mIndexColumnName;
    G const* mGt;
    std::tuple<A...>* mAt;
    typename grouping_t::iterator mGroupingElement;
    uint64_t position = 0;
    gsl::span<int64_t const> groupSelection;
    std::array<gsl::span<int64_t const> const*, sizeof...(A)> selections;
    std::array<gsl::span<int64_t const>::iterator, sizeof...(A)> starts;

    std::array<SliceInfoPtr, sizeof...(A)> sliceInfos;
    std::array<SliceInfoUnsortedPtr, sizeof...(A)> sliceInfosUnsorted;
    ArrowTableSlicingCache* mSlices;
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
