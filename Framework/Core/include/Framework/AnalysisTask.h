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

#ifndef FRAMEWORK_ANALYSIS_TASK_H_
#define FRAMEWORK_ANALYSIS_TASK_H_

#include "Framework/AnalysisManagers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigContext.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Logger.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Traits.h"
#include "Framework/VariantHelpers.h"
#include "Framework/RuntimeError.h"
#include "Framework/TypeIdHelpers.h"

#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <gandiva/node.h>
#include <type_traits>
#include <utility>
#include <memory>
#include <sstream>
#include <iomanip>
namespace o2::framework
{
/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
// FIXME: for the moment this needs to stay outside AnalysisTask
//        because we cannot inherit from it due to a C++17 bug
//        in GCC 7.3. We need to move to 7.4+
struct AnalysisTask {
};

namespace
{
template <typename B, typename C>
constexpr static bool isIndexTo()
{
  if constexpr (soa::is_type_with_binding_v<C>) {
    if constexpr (soa::is_soa_index_table_t<std::decay_t<B>>::value) {
      using T = typename std::decay_t<B>::first_t;
      if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
        using TT = typename framework::pack_element_t<0, typename std::decay_t<T>::originals>;
        return std::is_same_v<typename C::binding_t, TT>;
      } else {
        using TT = std::decay_t<T>;
        return std::is_same_v<typename C::binding_t, TT>;
      }
    } else {
      if constexpr (soa::is_type_with_originals_v<std::decay_t<B>>) {
        using TT = typename framework::pack_element_t<0, typename std::decay_t<B>::originals>;
        return std::is_same_v<typename C::binding_t, TT>;
      } else {
        using TT = std::decay_t<B>;
        return std::is_same_v<typename C::binding_t, TT>;
      }
    }
  }
  return false;
}

template <typename B, typename C>
constexpr static bool isSortedIndexTo()
{
  if constexpr (soa::is_type_with_binding_v<C>) {
    if constexpr (soa::is_soa_index_table_t<std::decay_t<B>>::value) {
      using T = typename std::decay_t<B>::first_t;
      if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
        using TT = typename framework::pack_element_t<0, typename std::decay_t<T>::originals>;
        return std::is_same_v<typename C::binding_t, TT> && C::sorted;
      } else {
        using TT = std::decay_t<T>;
        return std::is_same_v<typename C::binding_t, TT> && C::sorted;
      }
    } else {
      if constexpr (soa::is_type_with_originals_v<std::decay_t<B>>) {
        using TT = typename framework::pack_element_t<0, typename std::decay_t<B>::originals>;
        return std::is_same_v<typename C::binding_t, TT> && C::sorted;
      } else {
        using TT = std::decay_t<B>;
        return std::is_same_v<typename C::binding_t, TT> && C::sorted;
      }
    }
  }
  return false;
}

template <typename B, typename... C>
constexpr static bool hasIndexTo(framework::pack<C...>&&)
{
  return (isIndexTo<B, C>() || ...);
}

template <typename B, typename... C>
constexpr static bool hasSortedIndexTo(framework::pack<C...>&&)
{
  return (isSortedIndexTo<B, C>() || ...);
}

template <typename B, typename Z>
constexpr static bool relatedByIndex()
{
  return hasIndexTo<B>(typename Z::persistent_columns_t{});
}

template <typename B, typename Z>
constexpr static bool relatedBySortedIndex()
{
  return hasSortedIndexTo<B>(typename Z::persistent_columns_t{});
}
} // namespace

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...

struct AnalysisDataProcessorBuilder {
  template <typename T>
  static ConfigParamSpec getSpec()
  {
    if constexpr (soa::is_type_with_metadata_v<aod::MetadataTrait<T>>) {
      return ConfigParamSpec{std::string{"input:"} + aod::MetadataTrait<T>::metadata::tableLabel(), VariantType::String, aod::MetadataTrait<T>::metadata::sourceSpec(), {"\"\""}};
    } else {
      using O1 = framework::pack_element_t<0, typename T::originals>;
      return ConfigParamSpec{std::string{"input:"} + aod::MetadataTrait<T>::metadata::tableLabel(), VariantType::String, aod::MetadataTrait<O1>::metadata::sourceSpec(), {"\"\""}};
    }
  }

  template <typename... T>
  static std::vector<ConfigParamSpec> getInputSpecs(framework::pack<T...>)
  {
    return std::vector{getSpec<T>()...};
  }

  template <typename T>
  static std::vector<ConfigParamSpec> getIndexSources()
  {
    static_assert(soa::is_soa_index_table_t<T>::value, "Can only be used with IndexTable");
    return getInputSpecs(typename T::sources_t{});
  }

  template <typename Arg>
  static void doAppendInputWithMetadata(const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    std::vector<ConfigParamSpec> inputMetadata;
    inputMetadata.emplace_back(ConfigParamSpec{std::string{"control:"} + name, VariantType::Bool, value, {"\"\""}});
    if constexpr (soa::is_soa_index_table_t<std::decay_t<Arg>>::value) {
      auto inputSources = getIndexSources<std::decay_t<Arg>>();
      std::sort(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name < b.name; });
      auto last = std::unique(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name == b.name; });
      inputSources.erase(last, inputSources.end());
      inputMetadata.insert(inputMetadata.end(), inputSources.begin(), inputSources.end());
    }
    auto locate = std::find_if(inputs.begin(), inputs.end(), [](InputSpec& input) { return input.binding == metadata::tableLabel(); });
    if (locate != inputs.end()) {
      // amend entry
      auto& entryMetadata = locate->metadata;
      entryMetadata.insert(entryMetadata.end(), inputMetadata.begin(), inputMetadata.end());
      std::sort(entryMetadata.begin(), entryMetadata.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name < b.name; });
      auto new_end = std::unique(entryMetadata.begin(), entryMetadata.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name == b.name; });
      entryMetadata.erase(new_end, entryMetadata.end());
    } else {
      // add entry
      inputs.push_back(InputSpec{metadata::tableLabel(), metadata::origin(), metadata::description(), Lifetime::Timeframe, inputMetadata});
    }
  }

  template <typename... Args>
  static void doAppendInputWithMetadata(framework::pack<Args...>, const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    (doAppendInputWithMetadata<Args>(name, value, inputs), ...);
  }

  template <typename T, int AI>
  static void appendSomethingWithMetadata(const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos, size_t hash)
  {
    using dT = std::decay_t<T>;
    if constexpr (soa::is_soa_filtered_t<dT>::value) {
      eInfos.push_back({AI, hash, dT::hashes(), o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
    } else if constexpr (soa::is_soa_iterator_t<dT>::value) {
      if constexpr (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
        eInfos.push_back({AI, hash, dT::parent_t::hashes(), o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
      }
    }
    doAppendInputWithMetadata(soa::make_originals_from_type<dT>(), name, value, inputs);
  }

  template <typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    (appendSomethingWithMetadata<Args, o2::framework::has_type_at_v<Args>(pack<Args...>{})>(name, value, inputs, eInfos, typeHash<R (C::*)(Args...)>()), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord&, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
  {
    return extractSomethingFromRecord<Grouping, 0>(record, infos, typeHash<R (C::*)(Grouping, Args...)>());
  }

  template <typename R, typename C>
  static auto bindGroupingTable(InputRecord&, R (C::*)(), std::vector<ExpressionInfo>&)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return o2::soa::Table<>{nullptr};
  }

  template <typename T>
  static auto extractTableFromRecord(InputRecord& record)
  {
    if constexpr (soa::is_type_with_metadata_v<aod::MetadataTrait<T>>) {
      auto table = record.get<TableConsumer>(aod::MetadataTrait<T>::metadata::tableLabel())->asArrowTable();
      if (table->num_rows() == 0) {
        table = makeEmptyTable<T>(aod::MetadataTrait<T>::metadata::tableLabel());
      }
      return table;
    } else if constexpr (soa::is_type_with_originals_v<T>) {
      return extractFromRecord<T>(record, typename T::originals{});
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename T, typename... Os>
  static auto extractFromRecord(InputRecord& record, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t{{extractTableFromRecord<Os>(record)...}};
    } else {
      return T{{extractTableFromRecord<Os>(record)...}};
    }
  }

  template <typename T, typename... Os>
  static auto extractFilteredFromRecord(InputRecord& record, ExpressionInfo& info, pack<Os...> const&)
  {
    auto table = o2::soa::ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...});
    if (info.tree != nullptr && info.filter == nullptr) {
      info.filter = framework::expressions::createFilter(table->schema(), framework::expressions::makeCondition(info.tree));
    }
    if (info.tree != nullptr && info.filter != nullptr && info.resetSelection == true) {
      info.selection = framework::expressions::createSelection(table, info.filter);
      info.resetSelection = false;
    }
    if constexpr (!framework::is_base_of_template<soa::SmallGroups, std::decay_t<T>>::value) {
      if (info.selection == nullptr) {
        throw runtime_error_f("Null selection for %d (arg %d), missing Filter declaration?", info.processHash, info.argumentIndex);
      }
    }
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t({table}, info.selection);
    } else {
      return T({table}, info.selection);
    }
  }

  template <typename T, int AI>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo>& infos, size_t phash)
  {
    using decayed = std::decay_t<T>;

    if constexpr (soa::is_soa_filtered_t<decayed>::value) {
      return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
    } else if constexpr (soa::is_soa_iterator_t<decayed>::value) {
      if constexpr (std::is_same_v<typename decayed::policy_t, soa::FilteredIndexPolicy>) {
        return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
      } else {
        return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
      }
    } else {
      return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
  {
    return std::make_tuple(extractSomethingFromRecord<Args, has_type_at_v<Args>(pack<Args...>{}) + 1>(record, infos, typeHash<R (C::*)(Grouping, Args...)>())...);
  }

  template <typename R, typename C>
  static auto bindAssociatedTables(InputRecord&, R (C::*)(), std::vector<ExpressionInfo>&)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return std::tuple<>{};
  }

  template <typename T, typename C>
  using is_external_index_to_t = std::is_same<typename C::binding_t, T>;

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
        if constexpr (soa::is_soa_index_table_t<std::decay_t<Z>>::value) {
          using T = typename std::decay_t<Z>::first_t;
          if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
            using O = typename framework::pack_element_t<0, typename std::decay_t<Z>::originals>;
            using groupingMetadata = typename aod::MetadataTrait<O>::metadata;
            return groupingMetadata::tableLabel();
          } else {
            using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
            return groupingMetadata::tableLabel();
          }
        } else if constexpr (soa::is_type_with_originals_v<std::decay_t<Z>>) {
          using T = typename framework::pack_element_t<0, typename std::decay_t<Z>::originals>;
          using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
          return groupingMetadata::tableLabel();
        } else {
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<Z>>::metadata;
          return groupingMetadata::tableLabel();
        }
      }

      template <typename T>
      auto splittingFunction(T&& table)
      {
        constexpr auto index = framework::has_type_at_v<std::decay_t<T>>(associated_pack_t{});
        if constexpr (relatedByIndex<std::decay_t<G>, std::decay_t<T>>()) {
          auto name = getLabelFromType<std::decay_t<T>>();
          if constexpr (!framework::is_specialization<std::decay_t<T>, soa::SmallGroups>::value) {
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

      GroupSlicerIterator& operator++()
      {
        ++position;
        ++mGroupingElement;
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
          if constexpr (!framework::is_specialization<std::decay_t<A1>, soa::SmallGroups>::value) {
            if (originalTable.size() == 0) {
              return originalTable;
            }
            // optimized split
            if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
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
              auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);
              std::decay_t<A1> typedTable{{groupedElementsTable}, (offsets[index])[pos]};
              typedTable.bindInternalIndicesTo(&originalTable);
              return typedTable;
            }
          } else {
            //generic split
            if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
              if (originalTable.tableSize() == 0) {
                return originalTable;
              }
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
          return std::get<A1>(*mAt);
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

  template <typename Task, typename... T>
  static void invokeProcessTuple(Task& task, InputRecord& inputs, std::tuple<T...> const& processTuple, std::vector<ExpressionInfo>& infos)
  {
    (invokeProcess<o2::framework::has_type_at_v<T>(pack<T...>{})>(task, inputs, std::get<T>(processTuple), infos), ...);
  }

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*processingFunction)(Grouping, Associated...), std::vector<ExpressionInfo>& infos)
  {
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, processingFunction, infos);

    // set filtered tables for partitions with grouping
    homogeneous_apply_refs([&groupingTable](auto& x) {
      PartitionManager<std::decay_t<decltype(x)>>::setPartition(x, groupingTable);
      PartitionManager<std::decay_t<decltype(x)>>::bindInternalIndices(x, &groupingTable);
      return true;
    },
                           task);

    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      homogeneous_apply_refs([&groupingTable](auto& x) {
        PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable);
        return true;
      },
                             task);
      if constexpr (soa::is_soa_iterator_t<G>::value) {
        for (auto& element : groupingTable) {
          std::invoke(processingFunction, task, *element);
        }
      } else {
        static_assert(soa::is_soa_table_like_t<G>::value,
                      "Single argument of process() should be a table-like or an iterator");
        std::invoke(processingFunction, task, groupingTable);
      }
    } else {
      // multiple arguments to process
      static_assert(((soa::is_soa_iterator_t<std::decay_t<Associated>>::value == false) && ...),
                    "Associated arguments of process() should not be iterators");
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, processingFunction, infos);
      //pre-bind self indices
      std::apply(
        [&](auto&... t) {
          (homogeneous_apply_refs(
             [&](auto& p) {
               PartitionManager<std::decay_t<decltype(p)>>::bindInternalIndices(p, &t);
               return true;
             },
             task),
           ...);
        },
        associatedTables);

      auto binder = [&](auto&& x) {
        x.bindExternalIndices(&groupingTable, &std::get<std::decay_t<Associated>>(associatedTables)...);
        homogeneous_apply_refs([&x](auto& t) {
          PartitionManager<std::decay_t<decltype(t)>>::setPartition(t, x);
          PartitionManager<std::decay_t<decltype(t)>>::bindExternalIndices(t, &x);
          return true;
        },
                               task);
      };
      groupingTable.bindExternalIndices(&std::get<std::decay_t<Associated>>(associatedTables)...);

      // always pre-bind full tables to support index hierarchy
      std::apply(
        [&](auto&&... x) {
          (binder(x), ...);
        },
        associatedTables);

      if constexpr (soa::is_soa_iterator_t<std::decay_t<G>>::value) {
        // grouping case
        auto slicer = GroupSlicer(groupingTable, associatedTables);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();

          std::apply(
            [&](auto&&... x) {
              (binder(x), ...);
            },
            associatedSlices);

          // bind partitions and grouping table
          homogeneous_apply_refs([&groupingTable](auto& x) {
            PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable);
            return true;
          },
                                 task);

          invokeProcessWithArgsGeneric(task, processingFunction, slice.groupingElement(), associatedSlices);
        }
      } else {
        // non-grouping case

        // bind partitions and grouping table
        homogeneous_apply_refs([&groupingTable](auto& x) {
          PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable);
          return true;
        },
                               task);

        invokeProcessWithArgsGeneric(task, processingFunction, groupingTable, associatedTables);
      }
    }
  }

  template <typename C, typename T, typename G, typename... A>
  static void invokeProcessWithArgsGeneric(C& task, T processingFunction, G g, std::tuple<A...>& at)
  {
    std::invoke(processingFunction, task, g, std::get<A>(at)...);
  }

  template <typename T, typename G, typename... A>
  static void invokeProcessWithArgs(T& task, G g, std::tuple<A...>& at)
  {
    task.process(g, std::get<A>(at)...);
  }
};

namespace
{
template <typename T>
class has_process
{
  template <typename C>
  static std::true_type test(decltype(&C::process));
  template <typename C>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <class T>
inline constexpr bool has_process_v = has_process<T>::value;

template <typename T>
class has_run
{
  template <typename C>
  static std::true_type test(decltype(&C::run));
  template <typename C>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <class T>
inline constexpr bool has_run_v = has_run<T>::value;

template <typename T>
class has_init
{
  template <typename C>
  static std::true_type test(decltype(&C::init));
  template <typename C>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <class T>
inline constexpr bool has_init_v = has_init<T>::value;
} // namespace

struct SetDefaultProcesses {
  std::vector<std::pair<std::string, bool>> map;
};

/// Struct to differentiate task names from possible task string arguments
struct TaskName {
  TaskName(std::string const& name) : value{name} {}
  std::string value;
};

template <typename T, typename... A>
auto getTaskNameSetProcesses(TaskName first, SetDefaultProcesses second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : second.map) {
    homogeneous_apply_refs(
      [&](auto& x) {
        return UpdateProcessSwitches<std::decay_t<decltype(x)>>::set(setting, x);
      },
      *task.get());
  }
  return std::make_tuple(first.value, task);
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(SetDefaultProcesses first, TaskName second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : first.map) {
    homogeneous_apply_refs(
      [&](auto& x) {
        return UpdateProcessSwitches<std::decay_t<decltype(x)>>::set(setting, x);
      },
      *task.get());
  }
  return std::make_tuple(second.value, task);
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(SetDefaultProcesses first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : first.map) {
    homogeneous_apply_refs(
      [&](auto& x) {
        return UpdateProcessSwitches<std::decay_t<decltype(x)>>::set(setting, x);
      },
      *task.get());
  }
  auto type_name_str = type_name<T>();
  std::string name = type_to_task_name(type_name_str);
  return std::make_tuple(name, task);
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(TaskName first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  return std::make_tuple(first.value, task);
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  auto type_name_str = type_name<T>();
  std::string name = type_to_task_name(type_name_str);
  return std::make_tuple(name, task);
}

template <typename T, typename... A>
auto getTaskName(TaskName first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  return std::make_tuple(first.value, task);
}

template <typename T, typename... A>
auto getTaskName(A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  auto type_name_str = type_name<T>();
  std::string name = type_to_task_name(type_name_str);
  return std::make_tuple(name, task);
}

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(ConfigContext const& ctx, Args&&... args)
{
  TH1::AddDirectory(false);

  auto [name_str, task] = getTaskNameSetProcesses<T>(args...);

  auto suffix = ctx.options().get<std::string>("workflow-suffix");
  if (!suffix.empty()) {
    name_str += suffix;
  }
  const char* name = name_str.c_str();

  auto hash = compile_time_hash(name);

  std::vector<OutputSpec> outputs;
  std::vector<InputSpec> inputs;
  std::vector<ConfigParamSpec> options;
  std::vector<ExpressionInfo> expressionInfos;

  /// make sure options and configurables are set before expression infos are created
  homogeneous_apply_refs([&options, &hash](auto& x) { return OptionManager<std::decay_t<decltype(x)>>::appendOption(options, x); }, *task.get());

  /// parse process functions defined by corresponding configurables
  if constexpr (has_process_v<T>) {
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, "default", true, inputs, expressionInfos);
  }
  homogeneous_apply_refs(
    [name = name_str, &expressionInfos, &inputs](auto& x) {
      using D = std::decay_t<decltype(x)>;
      if constexpr (is_base_of_template<ProcessConfigurable, D>::value) {
        // this pushes (argumentIndex,processHash,schemaPtr,nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
        AnalysisDataProcessorBuilder::inputsFromArgs(x.process, (name + "/" + x.name).c_str(), x.value, inputs, expressionInfos);
        return true;
      }
      return false;
    },
    *task.get());

  // avoid self-forwarding if process methods subscribe to same tables
  std::sort(inputs.begin(), inputs.end(), [](InputSpec const& a, InputSpec const& b) { return a.binding < b.binding; });
  auto last = std::unique(inputs.begin(), inputs.end(), [](InputSpec const& a, InputSpec const& b) { return a.binding == b.binding; });
  inputs.erase(last, inputs.end());

  // request base tables for spawnable extended tables
  // this checks for duplications
  homogeneous_apply_refs([&inputs](auto& x) {
    return SpawnManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x);
  },
                         *task.get());

  //request base tables for indices to be built
  homogeneous_apply_refs([&inputs](auto& x) {
    return IndexManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x);
  },
                         *task.get());

  // no static way to check if the task defines any processing, we can only make sure it subscribes to at least something
  if (inputs.empty() == true) {
    LOG(warn) << "Task " << name_str << " has no inputs";
  }

  homogeneous_apply_refs([&outputs, &hash](auto& x) { return OutputManager<std::decay_t<decltype(x)>>::appendOutput(outputs, x, hash); }, *task.get());

  std::vector<ServiceSpec> requiredServices = CommonServices::defaultServices();
  homogeneous_apply_refs([&requiredServices](auto& x) { return ServiceManager<std::decay_t<decltype(x)>>::add(requiredServices, x); }, *task.get());

  auto algo = AlgorithmSpec::InitCallback{[task = task, expressionInfos](InitContext& ic) mutable {
    homogeneous_apply_refs([&ic](auto&& x) { return OptionManager<std::decay_t<decltype(x)>>::prepare(ic, x); }, *task.get());
    homogeneous_apply_refs([&ic](auto&& x) { return ServiceManager<std::decay_t<decltype(x)>>::prepare(ic, x); }, *task.get());

    auto& callbacks = ic.services().get<CallbackService>();
    auto endofdatacb = [task](EndOfStreamContext& eosContext) {
      homogeneous_apply_refs([&eosContext](auto&& x) { return OutputManager<std::decay_t<decltype(x)>>::postRun(eosContext, x); }, *task.get());
      eosContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

    /// update configurables in filters
    homogeneous_apply_refs(
      [&ic](auto& x) -> bool { return FilterManager<std::decay_t<decltype(x)>>::updatePlaceholders(x, ic); },
      *task.get());
    /// update configurables in partitions
    homogeneous_apply_refs(
      [&ic](auto& x) -> bool { PartitionManager<std::decay_t<decltype(x)>>::updatePlaceholders(x, ic); return true; },
      *task.get());
    /// create for filters gandiva trees matched to schemas and store the pointers into expressionInfos
    homogeneous_apply_refs([&expressionInfos](auto& x) {
      return FilterManager<std::decay_t<decltype(x)>>::createExpressionTrees(x, expressionInfos);
    },
                           *task.get());

    if constexpr (has_init_v<T>) {
      task->init(ic);
    }

    return [task, expressionInfos](ProcessingContext& pc) mutable {
      // reset selections for the next dataframe
      for (auto& info : expressionInfos) {
        info.resetSelection = true;
      }
      homogeneous_apply_refs([&pc](auto&& x) { return OutputManager<std::decay_t<decltype(x)>>::prepare(pc, x); }, *task.get());
      if constexpr (has_run_v<T>) {
        task->run(pc);
      }
      if constexpr (has_process_v<T>) {
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos);
      }
      homogeneous_apply_refs(
        [&pc, &expressionInfos, &task](auto& x) mutable {
          if constexpr (is_base_of_template<ProcessConfigurable, std::decay_t<decltype(x)>>::value) {
            if (x.value == true) {
              AnalysisDataProcessorBuilder::invokeProcess(*task.get(), pc.inputs(), x.process, expressionInfos);
              return true;
            }
          }
          return false;
        },
        *task.get());

      homogeneous_apply_refs([&pc](auto&& x) { return OutputManager<std::decay_t<decltype(x)>>::finalize(pc, x); }, *task.get());
    };
  }};

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    outputs,
    algo,
    options,
    requiredServices};
  return spec;
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
