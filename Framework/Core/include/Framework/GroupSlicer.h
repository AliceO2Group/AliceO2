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
#include "Framework/Kernels.h"

#include <arrow/util/key_value_metadata.h>
#include <type_traits>
#include <string>

namespace o2::framework
{

template <typename G, typename... A>
struct GroupSlicer {
  using grouping_t = std::decay_t<G>;
  GroupSlicer(G& gt, std::tuple<A...>& at)
    : max{gt.size()},
      mBegin{GroupSlicerIterator(gt, at)}
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

    template <typename Z>
    std::string getLabelFromType()
    {
      auto cutString = [](std::string&& str) -> std::string {
        auto pos = str.find('_');
        if (pos != std::string::npos) {
          str.erase(pos);
        }
        return str;
      };

      if constexpr (soa::is_soa_index_table_t<std::decay_t<Z>>::value) {
        using T = typename std::decay_t<Z>::first_t;
        if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
          using O = typename framework::pack_element_t<0, typename std::decay_t<Z>::originals>;
          using groupingMetadata = typename aod::MetadataTrait<O>::metadata;
          return cutString(std::string{groupingMetadata::tableLabel()});
        } else {
          using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
          return cutString(std::string{groupingMetadata::tableLabel()});
        }
      } else if constexpr (soa::is_type_with_originals_v<std::decay_t<Z>>) {
        using T = typename framework::pack_element_t<0, typename std::decay_t<Z>::originals>;
        using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
        return cutString(std::string{groupingMetadata::tableLabel()});
      } else {
        using groupingMetadata = typename aod::MetadataTrait<std::decay_t<Z>>::metadata;
        return cutString(std::string{groupingMetadata::tableLabel()});
      }
    }

    template <typename T>
    auto splittingFunction(T&& table)
    {
      constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
      if constexpr (relatedByIndex<std::decay_t<G>, std::decay_t<T>>()) {
        auto name = getLabelFromType<std::decay_t<T>>();
        if constexpr (!framework::is_specialization_v<std::decay_t<T>, soa::SmallGroups>) {
          if (table.size() == 0) {
            return;
          }
          // use presorted splitting approach
          auto result = o2::framework::sliceByColumn(mIndexColumnName.c_str(),
                                                     name.c_str(),
                                                     table.asArrowTable(),
                                                     static_cast<int32_t>(mGt->tableSize()),
                                                     &groups[index],
                                                     &offsets[index],
                                                     &sizes[index]);
          if (result.ok() == false) {
            throw runtime_error("Cannot split collection");
          }
          if (groups[index].size() > mGt->tableSize()) {
            throw runtime_error_f("Splitting collection %s resulted in a larger group number (%d) than there is rows in the grouping table (%d).", name.c_str(), groups[index].size(), mGt->tableSize());
          };
        } else {
          if (table.tableSize() == 0) {
            return;
          }
          // use generic splitting approach
          o2::framework::sliceByColumnGeneric(mIndexColumnName.c_str(),
                                              name.c_str(),
                                              table.asArrowTable(),
                                              static_cast<int32_t>(mGt->tableSize()),
                                              &filterGroups[index]);
        }
      }
    }

    template <typename T>
    auto extractingFunction(T&& table)
    {
      if constexpr (soa::is_soa_filtered_t<std::decay_t<T>>::value) {
        constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
        selections[index] = &table.getSelectedRows();
        starts[index] = selections[index]->begin();
        offsets[index].push_back(table.tableSize());
      }
    }

    GroupSlicerIterator(G& gt, std::tuple<A...>& at)
      : mIndexColumnName{std::string("fIndex") + getLabelFromType<G>()},
        mGt{&gt},
        mAt{&at},
        mGroupingElement{gt.begin()},
        position{0}
    {
      if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
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

    template <typename B, typename... C>
    constexpr static bool hasIndexTo(framework::pack<C...>&&)
    {
      return (o2::soa::is_binding_compatible_v<B, typename C::binding_t>() || ...);
    }

    template <typename B, typename... C>
    constexpr static bool hasSortedIndexTo(framework::pack<C...>&&)
    {
      return ((C::sorted && o2::soa::is_binding_compatible_v<B, typename C::binding_t>()) || ...);
    }

    template <typename B, typename Z>
    constexpr static bool relatedByIndex()
    {
      return hasIndexTo<B>(typename Z::table_t::external_index_columns_t{});
    }

    template <typename B, typename Z>
    constexpr static bool relatedBySortedIndex()
    {
      return hasSortedIndexTo<B>(typename Z::table_t::external_index_columns_t{});
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

      if constexpr (relatedByIndex<std::decay_t<G>, std::decay_t<A1>>()) {
        uint64_t pos;
        if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
          pos = groupSelection[position];
        } else {
          pos = position;
        }

        if constexpr (!framework::is_specialization_v<std::decay_t<A1>, soa::SmallGroups>) {
          // optimized split
          if (originalTable.size() == 0) {
            return originalTable;
          }
          if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
            if (groups[index].empty()) {
              return std::decay_t<A1>{{makeEmptyTable<A1>("empty")}, soa::SelectionVector{}};
            }
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);

            // for each grouping element we need to slice the selection vector
            auto start_iterator = std::lower_bound(starts[index], selections[index]->end(), (offsets[index])[pos]);
            auto stop_iterator = std::lower_bound(start_iterator, selections[index]->end(), (offsets[index])[pos] + (sizes[index])[pos]);
            starts[index] = stop_iterator;
            soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
            std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                           [&](int64_t idx) {
                             return idx - static_cast<int64_t>((offsets[index])[pos]);
                           });

            std::decay_t<A1> typedTable{{groupedElementsTable}, std::move(slicedSelection), (offsets[index])[pos]};
            typedTable.bindInternalIndicesTo(&originalTable);
            return typedTable;

          } else {
            if (groups[index].empty()) {
              return std::decay_t<A1>{{makeEmptyTable<A1>("empty")}};
            }
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);
            std::decay_t<A1> typedTable{{groupedElementsTable}, (offsets[index])[pos]};
            typedTable.bindInternalIndicesTo(&originalTable);
            return typedTable;
          }
        } else {
          //generic split
          if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
            // intersect selections
            o2::soa::SelectionVector s;
            if (selections[index]->empty()) {
              std::copy((filterGroups[index])[pos].begin(), (filterGroups[index])[pos].end(), std::back_inserter(s));
            } else {
              std::set_intersection((filterGroups[index])[pos].begin(), (filterGroups[index])[pos].end(), selections[index]->begin(), selections[index]->end(), std::back_inserter(s));
            }
            std::decay_t<A1> typedTable{{originalTable.asArrowTable()}, std::move(s)};
            typedTable.bindInternalIndicesTo(&originalTable);
            return typedTable;
          } else {
            throw runtime_error("Unsorted grouped table needs to be used with soa::SmallGroups<>");
          }
        }
      } else {
        return originalTable;
      }
    }

    std::string mIndexColumnName;
    G const* mGt;
    std::tuple<A...>* mAt;
    typename grouping_t::iterator mGroupingElement;
    uint64_t position = 0;
    gsl::span<int64_t const> groupSelection;
    std::array<std::vector<arrow::Datum>, sizeof...(A)> groups;
    std::array<ListVector, sizeof...(A)> filterGroups;
    std::array<std::vector<uint64_t>, sizeof...(A)> offsets;
    std::array<std::vector<int>, sizeof...(A)> sizes;
    std::array<gsl::span<int64_t const> const*, sizeof...(A)> selections;
    std::array<gsl::span<int64_t const>::iterator, sizeof...(A)> starts;
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
