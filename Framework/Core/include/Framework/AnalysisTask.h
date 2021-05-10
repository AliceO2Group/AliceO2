// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_ANALYSIS_TASK_H_
#define FRAMEWORK_ANALYSIS_TASK_H_

#include "../../src/AnalysisManagers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigContext.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "../src/ExpressionHelpers.h"
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

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...

struct AnalysisDataProcessorBuilder {
  template <typename T>
  static ConfigParamSpec getSpec()
  {
    if constexpr (soa::is_type_with_metadata_v<aod::MetadataTrait<T>>) {
      return ConfigParamSpec{aod::MetadataTrait<T>::metadata::tableLabel(), VariantType::String, aod::MetadataTrait<T>::metadata::sourceSpec(), {"\"\""}};
    } else {
      using O1 = framework::pack_element_t<0, typename T::originals>;
      return ConfigParamSpec{aod::MetadataTrait<T>::metadata::tableLabel(), VariantType::String, aod::MetadataTrait<O1>::metadata::sourceSpec(), {"\"\""}};
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
  static void doAppendInputWithMetadata(std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    if constexpr (soa::is_soa_index_table_t<std::decay_t<Arg>>::value) {
      auto inputSources = getIndexSources<std::decay_t<Arg>>();
      std::sort(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name < b.name; });
      auto last = std::unique(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name == b.name; });
      inputSources.erase(last, inputSources.end());
      inputs.push_back(InputSpec{metadata::tableLabel(), metadata::origin(), metadata::description(), Lifetime::Timeframe, inputSources});
    } else {
      inputs.push_back({metadata::tableLabel(), metadata::origin(), metadata::description()});
    }
  }

  template <typename... Args>
  static void doAppendInputWithMetadata(framework::pack<Args...>, std::vector<InputSpec>& inputs)
  {
    (doAppendInputWithMetadata<Args>(inputs), ...);
  }

  template <typename T, int PI, int AI>
  static void appendSomethingWithMetadata(std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    using dT = std::decay_t<T>;
    if constexpr (framework::is_specialization<dT, soa::Filtered>::value) {
      eInfos.push_back({AI, PI, dT::hashes(), o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
    } else if constexpr (soa::is_soa_iterator_t<dT>::value) {
      if constexpr (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
        eInfos.push_back({AI, PI, dT::parent_t::hashes(), o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
      }
    }
    doAppendInputWithMetadata(soa::make_originals_from_type<dT>(), inputs);
  }

  template <typename... T>
  static void inputsFromArgsTuple(std::tuple<T...>& processTuple, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    (inputsFromArgs<o2::framework::has_type_at_v<T>(pack<T...>{})>(std::get<T>(processTuple), inputs, eInfos), ...);
  }

  template <int PI, typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    (appendSomethingWithMetadata<Args, PI, o2::framework::has_type_at_v<Args>(pack<Args...>{})>(inputs, eInfos), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord&, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <int PI, typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const& infos)
  {
    return extractSomethingFromRecord<Grouping, PI, 0>(record, infos);
  }

  template <int PI, typename R, typename C>
  static auto bindGroupingTable(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const&)
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
        table = makeEmptyTable<T>();
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
  static auto extractFilteredFromRecord(InputRecord& record, ExpressionInfo const& info, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    } else {
      return T(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    }
  }

  template <typename T, int PI, int AI>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo> const infos)
  {
    using decayed = std::decay_t<T>;

    if constexpr (soa::is_soa_filtered_t<decayed>::value) {
      return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [](ExpressionInfo const& i) { return (i.processIndex == PI && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
    } else if constexpr (soa::is_soa_iterator_t<decayed>::value) {
      if constexpr (std::is_same_v<typename decayed::policy_t, soa::FilteredIndexPolicy>) {
        return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [](ExpressionInfo const& i) { return (i.processIndex == PI && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
      } else {
        return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
      }
    } else {
      return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <int PI, typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const infos)
  {
    return std::make_tuple(extractSomethingFromRecord<Args, PI, has_type_at_v<Args>(pack<Args...>{}) + 1>(record, infos)...);
  }

  template <int PI, typename R, typename C>
  static auto bindAssociatedTables(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const)
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

      auto getLabelFromType()
      {
        if constexpr (soa::is_soa_index_table_t<std::decay_t<G>>::value) {
          using T = typename std::decay_t<G>::first_t;
          if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
            using O = typename framework::pack_element_t<0, typename std::decay_t<G>::originals>;
            using groupingMetadata = typename aod::MetadataTrait<O>::metadata;
            return std::string("fIndex") + groupingMetadata::tableLabel();
          } else {
            using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
            return std::string("fIndex") + groupingMetadata::tableLabel();
          }
        } else if constexpr (soa::is_type_with_originals_v<std::decay_t<G>>) {
          using T = typename framework::pack_element_t<0, typename std::decay_t<G>::originals>;
          using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
          return std::string("fIndex") + groupingMetadata::tableLabel();
        } else {
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<G>>::metadata;
          return std::string("fIndex") + groupingMetadata::tableLabel();
        }
      }

      GroupSlicerIterator(G& gt, std::tuple<A...>& at)
        : mAt{&at},
          mGroupingElement{gt.begin()},
          position{0}
      {
        if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
          groupSelection = &gt.getSelectedRows();
        }
        auto indexColumnName = getLabelFromType();
        /// prepare slices and offsets for all associated tables that have index
        /// to grouping table
        ///
        auto splitter = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          constexpr auto index = framework::has_type_at_v<std::decay_t<decltype(x)>>(associated_pack_t{});
          if (x.size() != 0 && hasIndexTo<std::decay_t<G>>(typename xt::persistent_columns_t{})) {
            auto result = o2::framework::sliceByColumn(indexColumnName.c_str(),
                                                       x.asArrowTable(),
                                                       static_cast<int32_t>(gt.tableSize()),
                                                       &groups[index],
                                                       &offsets[index]);
            if (result.ok() == false) {
              throw runtime_error("Cannot split collection");
            }
            if (groups[index].size() != gt.tableSize()) {
              throw runtime_error_f("Splitting collection resulted in different group number (%d) than there is rows in the grouping table (%d).", groups[index].size(), gt.tableSize());
            };
          }
        };

        std::apply(
          [&](auto&&... x) -> void {
            (splitter(x), ...);
          },
          at);
        /// extract selections from filtered associated tables
        auto extractor = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          if constexpr (soa::is_soa_filtered_t<xt>::value) {
            constexpr auto index = framework::has_type_at_v<std::decay_t<decltype(x)>>(associated_pack_t{});
            selections[index] = &x.getSelectedRows();
            starts[index] = selections[index]->begin();
            offsets[index].push_back(std::get<xt>(at).tableSize());
          }
        };
        std::apply(
          [&](auto&&... x) -> void {
            (extractor(x), ...);
          },
          at);
      }

      template <typename B, typename... C>
      constexpr bool hasIndexTo(framework::pack<C...>&&)
      {
        return (isIndexTo<B, C>() || ...);
      }

      template <typename B, typename C>
      constexpr bool isIndexTo()
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

      GroupSlicerIterator operator++()
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
        if (std::get<A1>(*mAt).size() != 0 && hasIndexTo<G>(typename std::decay_t<A1>::persistent_columns_t{})) {
          uint64_t pos;
          if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
            pos = (*groupSelection)[position];
          } else {
            pos = position;
          }
          if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);

            // for each grouping element we need to slice the selection vector
            auto start_iterator = std::lower_bound(starts[index], selections[index]->end(), (offsets[index])[pos]);
            auto stop_iterator = std::lower_bound(start_iterator, selections[index]->end(), (offsets[index])[pos + 1]);
            starts[index] = stop_iterator;
            soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
            std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                           [&](int64_t idx) {
                             return idx - static_cast<int64_t>((offsets[index])[pos]);
                           });

            std::decay_t<A1> typedTable{{groupedElementsTable}, std::move(slicedSelection), (offsets[index])[pos]};
            return typedTable;
          } else {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);
            std::decay_t<A1> typedTable{{groupedElementsTable}, (offsets[index])[pos]};
            return typedTable;
          }
        } else {
          return std::get<A1>(*mAt);
        }
        O2_BUILTIN_UNREACHABLE();
      }

      std::tuple<A...>* mAt;
      typename grouping_t::iterator mGroupingElement;
      uint64_t position = 0;
      soa::SelectionVector const* groupSelection = nullptr;

      std::array<std::vector<arrow::Datum>, sizeof...(A)> groups;
      std::array<std::vector<uint64_t>, sizeof...(A)> offsets;
      std::array<soa::SelectionVector const*, sizeof...(A)> selections;
      std::array<soa::SelectionVector::const_iterator, sizeof...(A)> starts;
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
  static void invokeProcessTuple(Task& task, InputRecord& inputs, std::tuple<T...> const& processTuple, std::vector<ExpressionInfo> const& infos)
  {
    (invokeProcess<o2::framework::has_type_at_v<T>(pack<T...>{})>(task, inputs, std::get<T>(processTuple), infos), ...);
  }

  template <int PI, typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*processingFunction)(Grouping, Associated...), std::vector<ExpressionInfo> const& infos)
  {
    auto tupledTask = o2::framework::to_tuple_refs(task);
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable<PI>(inputs, processingFunction, infos);

    // set filtered tables for partitions with grouping
    std::apply([&groupingTable](auto&... x) {
      (PartitionManager<std::decay_t<decltype(x)>>::setPartition(x, groupingTable), ...);
    },
               tupledTask);

    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      std::apply([&groupingTable](auto&... x) {
        (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
        (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
      },
                 tupledTask);
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
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables<PI>(inputs, processingFunction, infos);
      auto binder = [&](auto&& x) {
        x.bindExternalIndices(&groupingTable, &std::get<std::decay_t<Associated>>(associatedTables)...);
        std::apply([&x](auto&... t) {
          (PartitionManager<std::decay_t<decltype(t)>>::setPartition(t, x), ...);
          (PartitionManager<std::decay_t<decltype(t)>>::bindExternalIndices(t, &x), ...);
          (PartitionManager<std::decay_t<decltype(t)>>::getBoundToExternalIndices(t, x), ...);
        },
                   tupledTask);
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
          std::apply([&groupingTable](auto&... x) {
            (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
            (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
          },
                     tupledTask);

          invokeProcessWithArgsGeneric(task, processingFunction, slice.groupingElement(), associatedSlices);
        }
      } else {
        // non-grouping case

        // bind partitions and grouping table
        std::apply([&groupingTable](auto&... x) {
          (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
          (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
        },
                   tupledTask);

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

/// Struct to differentiate task names from possible task string arguments
struct TaskName {
  TaskName(std::string const& name) : value{name} {}
  std::string value;
};

///
template <typename... Ts>
struct Processes {
  Processes(Ts&&... args) : processes{std::make_tuple(args...)}
  {
  }
  std::tuple<Ts...> processes;
};

template <typename T, typename... S, typename... A>
auto getNameTaskProcesses(TaskName first, Processes<S...> second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  return std::make_tuple(first.value, task, second.processes);
}

template <typename T, typename... S, typename... A>
auto getNameTaskProcesses(Processes<S...> first, TaskName second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  return std::make_tuple(second.value, task, first.processes);
}

template <typename T, typename... A>
auto getNameTaskProcesses(TaskName first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  if constexpr (has_process_v<T>) {
    return std::make_tuple(first.value, task, std::make_tuple(&T::process));
  } else {
    return std::make_tuple(first.value, task, std::make_tuple());
  }
}

template <typename T, typename... S, typename... A>
auto getNameTaskProcesses(Processes<S...> first, A... args)
{
  auto type_name_str = type_name<T>();
  std::string name = type_to_task_name(type_name_str);
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  return std::make_tuple(name, task, first.processes);
}

template <typename T>
auto getNameTaskProcesses()
{
  auto type_name_str = type_name<T>();
  std::string name = type_to_task_name(type_name_str);
  auto task = std::make_shared<T>();
  if constexpr (has_process_v<T>) {
    return std::make_tuple(name, task, std::make_tuple(&T::process));
  } else {
    return std::make_tuple(name, task, std::make_tuple());
  }
}

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(ConfigContext const& ctx, Args&&... args)
{
  TH1::AddDirectory(false);

  auto [name_str, task, processTuple] = getNameTaskProcesses<T>(args...);

  auto suffix = ctx.options().get<std::string>("workflow-suffix");
  if (!suffix.empty()) {
    name_str += suffix;
  }
  const char* name = name_str.c_str();

  auto hash = compile_time_hash(name);

  std::vector<OutputSpec> outputs;
  std::vector<ConfigParamSpec> options;

  auto tupledTask = o2::framework::to_tuple_refs(*task.get());
  static_assert(std::tuple_size_v<std::decay_t<decltype(processTuple)>> > 0 || has_run_v<T> || has_init_v<T>,
                "At least one of process(...), T::run(...), init(...) must be defined");

  std::vector<InputSpec> inputs;
  std::vector<ExpressionInfo> expressionInfos;

  /// make sure options and configurables are set before expression infos are created
  std::apply([&options, &hash](auto&... x) { return (OptionManager<std::decay_t<decltype(x)>>::appendOption(options, x), ...); }, tupledTask);

  if constexpr ((std::tuple_size_v<std::decay_t<decltype(processTuple)>>) > 0) {
    // this pushes (argumentIndex,processIndex,schemaPtr,nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
    AnalysisDataProcessorBuilder::inputsFromArgsTuple(processTuple, inputs, expressionInfos);
  }
  // avoid self-forwarding if process methods subscribe to same tables
  std::sort(inputs.begin(), inputs.end(), [](InputSpec const& a, InputSpec const& b) { return a.binding < b.binding; });
  auto last = std::unique(inputs.begin(), inputs.end(), [](InputSpec const& a, InputSpec const& b) { return a.binding == b.binding; });
  inputs.erase(last, inputs.end());

  //request base tables for spawnable extended tables
  std::apply([&inputs](auto&... x) {
    return (SpawnManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x), ...);
  },
             tupledTask);

  //request base tables for indices to be built
  std::apply([&inputs](auto&... x) {
    return (IndexManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x), ...);
  },
             tupledTask);

  std::apply([&outputs, &hash](auto&... x) { return (OutputManager<std::decay_t<decltype(x)>>::appendOutput(outputs, x, hash), ...); }, tupledTask);

  auto algo = AlgorithmSpec::InitCallback{[task = task, processTuple = processTuple, expressionInfos](InitContext& ic) mutable {
    auto tupledTask = o2::framework::to_tuple_refs(*task.get());
    std::apply([&ic](auto&&... x) { return (OptionManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);
    std::apply([&ic](auto&&... x) { return (ServiceManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);

    auto& callbacks = ic.services().get<CallbackService>();
    auto endofdatacb = [task](EndOfStreamContext& eosContext) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&eosContext](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::postRun(eosContext, x), ...); }, tupledTask);
      eosContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

    if constexpr ((std::tuple_size_v<std::decay_t<decltype(processTuple)>>) > 0) {
      /// update configurables in filters
      std::apply(
        [&ic](auto&... x) { return (FilterManager<std::decay_t<decltype(x)>>::updatePlaceholders(x, ic), ...); },
        tupledTask);
      /// update configurables in partitions
      std::apply(
        [&ic](auto&... x) { return (PartitionManager<std::decay_t<decltype(x)>>::updatePlaceholders(x, ic), ...); },
        tupledTask);
      /// create for filters gandiva trees matched to schemas and store the pointers into expressionInfos
      std::apply([&expressionInfos](auto&... x) {
        return (FilterManager<std::decay_t<decltype(x)>>::createExpressionTrees(x, expressionInfos), ...);
      },
                 tupledTask);
    }

    if constexpr (has_init_v<T>) {
      task->init(ic);
    }

    return [task, processTuple, expressionInfos](ProcessingContext& pc) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::prepare(pc, x), ...); }, tupledTask);
      if constexpr (has_run_v<T>) {
        task->run(pc);
      }
      if constexpr ((std::tuple_size_v<std::decay_t<decltype(processTuple)>>) > 0) {
        AnalysisDataProcessorBuilder::invokeProcessTuple(*(task.get()), pc.inputs(), processTuple, expressionInfos);
      }
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::finalize(pc, x), ...); }, tupledTask);
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
    options};
  return spec;
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
