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
#include "Framework/Condition.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/GroupSlicer.h"
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

template <int64_t BEGIN, int64_t END, int64_t STEP = 1>
struct Enumeration {
  static constexpr int64_t begin = BEGIN;
  static constexpr int64_t end = END;
  static constexpr int64_t step = STEP;
};

template <typename T>
static constexpr bool is_enumeration_v = false;

template <int64_t BEGIN, int64_t END, int64_t STEP>
static constexpr bool is_enumeration_v<Enumeration<BEGIN, END, STEP>> = true;

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
  static inline std::vector<ConfigParamSpec> getInputSpecs(framework::pack<T...>)
  {
    return std::vector{getSpec<T>()...};
  }

  template <typename T>
  static inline auto getSources()
  {
    if constexpr (soa::is_soa_index_table_v<T>) {
      return getInputSpecs(typename T::sources_t{});
    } else if constexpr (soa::is_soa_extension_table_v<std::decay_t<T>>) {
      return getInputSpecs(typename aod::MetadataTrait<T>::metadata::sources{});
    } else {
      always_static_assert<T>("Can be only used with index or extension table");
    }
  }

  template <typename T>
  static auto getInputMetadata()
  {
    std::vector<ConfigParamSpec> inputMetadata;
    auto inputSources = getSources<T>();
    std::sort(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name < b.name; });
    auto last = std::unique(inputSources.begin(), inputSources.end(), [](ConfigParamSpec const& a, ConfigParamSpec const& b) { return a.name == b.name; });
    inputSources.erase(last, inputSources.end());
    inputMetadata.insert(inputMetadata.end(), inputSources.begin(), inputSources.end());
    return inputMetadata;
  }

  template <typename Arg>
  static void doAppendInputWithMetadata(const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    std::vector<ConfigParamSpec> inputMetadata;
    inputMetadata.emplace_back(ConfigParamSpec{std::string{"control:"} + name, VariantType::Bool, value, {"\"\""}});
    if constexpr (soa::is_soa_index_table_v<std::decay_t<Arg>> || soa::is_soa_extension_table_v<std::decay_t<Arg>>) {
      auto inputSources = getInputMetadata<std::decay_t<Arg>>();
      inputMetadata.insert(inputMetadata.end(), inputSources.begin(), inputSources.end());
    }
    auto newInput = InputSpec{metadata::tableLabel(), metadata::origin(), metadata::description(), metadata::version(), Lifetime::Timeframe, inputMetadata};
    DataSpecUtils::updateInputList(inputs, std::move(newInput));
  }

  static void doAppendEnumeration(const char* name, int64_t begin, int64_t end, int64_t step, std::vector<InputSpec>& inputs)
  {
    std::vector<ConfigParamSpec> inputMetadata;
    // FIXME: for the moment we do not support begin, end and step.
    auto newInput = InputSpec{"enumeration", "DPL", "ENUM", 0, Lifetime::Enumeration, inputMetadata};
    DataSpecUtils::updateInputList(inputs, std::move(newInput));
  }

  template <typename... Args>
  static void doAppendInputWithMetadata(framework::pack<Args...>, const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    (doAppendInputWithMetadata<Args>(name, value, inputs), ...);
  }

  template <typename T>
  static void appendSomethingWithMetadata(int ai, const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos, size_t hash)
  {
    static_assert(std::is_lvalue_reference_v<T>, "Argument to process needs to be a reference (&).");
    using dT = std::decay_t<T>;
    if constexpr (is_enumeration_v<dT> == false) {
      if constexpr (soa::is_soa_filtered_v<dT>) {
        auto fields = createFieldsFromColumns(typename dT::table_t::persistent_columns_t{});
        eInfos.push_back({ai, hash, dT::hashes(), std::make_shared<arrow::Schema>(fields), nullptr});
      } else if constexpr (soa::is_soa_iterator_v<dT>) {
        auto fields = createFieldsFromColumns(typename dT::parent_t::persistent_columns_t{});
        if constexpr (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
          eInfos.push_back({ai, hash, dT::parent_t::hashes(), std::make_shared<arrow::Schema>(fields), nullptr});
        }
      }
      doAppendInputWithMetadata(soa::make_originals_from_type<dT>(), name, value, inputs);
    } else {
      doAppendEnumeration(name, dT::begin, dT::end, dT::step, inputs);
    }
  }

  template <typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    int ai = 0;
    auto hash = typeHash<R (C::*)(Args...)>();
    (appendSomethingWithMetadata<Args>(ai++, name, value, inputs, eInfos, hash), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord&, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
  {
    auto hash = typeHash<R (C::*)(Grouping, Args...)>();
    return extractSomethingFromRecord<Grouping, 0>(record, infos, hash);
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
    if constexpr (soa::is_soa_iterator_v<T>) {
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
    if constexpr (!o2::soa::is_smallgroups_v<std::decay_t<T>>) {
      if (info.selection == nullptr) {
        throw runtime_error_f("Null selection for %d (arg %d), missing Filter declaration?", info.processHash, info.argumentIndex);
      }
    }
    if constexpr (soa::is_soa_iterator_v<T>) {
      return typename T::parent_t({table}, info.selection);
    } else {
      return T({table}, info.selection);
    }
  }

  template <typename T, int AI>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo>& infos, size_t phash)
  {
    using decayed = std::decay_t<T>;

    if constexpr (is_enumeration_v<decayed> == false) {
      if constexpr (soa::is_soa_filtered_v<decayed>) {
        return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
      } else if constexpr (soa::is_soa_iterator_v<decayed>) {
        if constexpr (std::is_same_v<typename decayed::policy_t, soa::FilteredIndexPolicy>) {
          return extractFilteredFromRecord<decayed>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }), soa::make_originals_from_type<decayed>());
        } else {
          return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
        }
      } else {
        return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
      }
      O2_BUILTIN_UNREACHABLE();
    } else {
      return decayed{};
    }
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
  {
    constexpr auto p = pack<Args...>{};
    auto hash = typeHash<R (C::*)(Grouping, Args...)>();
    return std::make_tuple(extractSomethingFromRecord<Args, has_type_at_v<Args>(p) + 1>(record, infos, hash)...);
  }

  template <typename R, typename C>
  static auto bindAssociatedTables(InputRecord&, R (C::*)(), std::vector<ExpressionInfo>&)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return std::tuple<>{};
  }

  template <typename T, typename C>
  using is_external_index_to_t = std::is_same<typename C::binding_t, T>;

  template <typename Task, typename... T>
  static void invokeProcessTuple(Task& task, InputRecord& inputs, std::tuple<T...> const& processTuple, std::vector<ExpressionInfo>& infos)
  {
    pack<T...> p;
    (invokeProcess<o2::framework::has_type_at_v<T>(p)>(task, inputs, std::get<T>(processTuple), infos), ...);
  }

  template <typename... As>
  static void overwriteInternalIndices(std::tuple<As...>& dest, std::tuple<As...> const& src)
  {
    (std::get<As>(dest).bindInternalIndicesTo(&std::get<As>(src)), ...);
  }

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*processingFunction)(Grouping, Associated...), std::vector<ExpressionInfo>& infos)
  {
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, processingFunction, infos);

    auto presliceTable = [&task](auto& table) {
      homogeneous_apply_refs([&table](auto& x) {
        return PresliceManager<std::decay_t<decltype(x)>>::processTable(x, table);
      },
                             task);
    };
    // pre-slice grouping table if required
    presliceTable(groupingTable);

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
        GroupedCombinationManager<std::decay_t<decltype(x)>>::setGroupedCombination(x, groupingTable);
        return true;
      },
                             task);
      if constexpr (soa::is_soa_iterator_v<G>) {
        for (auto& element : groupingTable) {
          std::invoke(processingFunction, task, *element);
        }
      } else {
        static_assert(soa::is_soa_table_like_v<G> || is_enumeration_v<G>,
                      "Single argument of process() should be a table-like or an iterator");
        std::invoke(processingFunction, task, groupingTable);
      }
    } else {
      // multiple arguments to process
      static_assert(((soa::is_soa_iterator_v<std::decay_t<Associated>> == false) && ...),
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

      // GroupedCombinations bound separately, as they should be set once for all associated tables
      homogeneous_apply_refs([&groupingTable, &associatedTables](auto& t) {
        GroupedCombinationManager<std::decay_t<decltype(t)>>::setGroupedCombination(t, groupingTable, associatedTables);
        return true;
      },
                             task);

      if constexpr (soa::is_soa_iterator_v<std::decay_t<G>>) {
        // grouping case
        // pre-slice associated tables
        std::apply([&presliceTable](auto&... x) {
          (presliceTable(x), ...);
        },
                   associatedTables);

        auto slicer = GroupSlicer(groupingTable, associatedTables);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();
          overwriteInternalIndices(associatedSlices, associatedTables);
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
        // pre-slice associated tables
        std::apply([&presliceTable](auto&... x) {
          (presliceTable(x), ...);
        },
                   associatedTables);

        overwriteInternalIndices(associatedTables, associatedTables);
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
  TaskName(std::string name) : value{std::move(name)} {}
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
  /// extract conditions and append them as inputs
  homogeneous_apply_refs([&inputs](auto& x) { return ConditionManager<std::decay_t<decltype(x)>>::appendCondition(inputs, x); }, *task.get());

  /// parse process functions defined by corresponding configurables
  if constexpr (has_process_v<T>) {
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, "default", true, inputs, expressionInfos);
  }
  homogeneous_apply_refs(
    [name = name_str, &expressionInfos, &inputs](auto& x) {
      using D = std::decay_t<decltype(x)>;
      if constexpr (is_base_of_template_v<ProcessConfigurable, D>) {
        // this pushes (argumentIndex,processHash,schemaPtr,nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
        AnalysisDataProcessorBuilder::inputsFromArgs(x.process, (name + "/" + x.name).c_str(), x.value, inputs, expressionInfos);
        return true;
      }
      return false;
    },
    *task.get());

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
      homogeneous_apply_refs([&eosContext](auto&& x) {
          using X = std::decay_t<decltype(x)>;
          ServiceManager<X>::postRun(eosContext, x);
          return OutputManager<X>::postRun(eosContext, x); },
                             *task.get());
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
      // load the ccdb object from their cache
      homogeneous_apply_refs([&pc](auto&& x) { return ConditionManager<std::decay_t<decltype(x)>>::newDataframe(pc.inputs(), x); }, *task.get());
      // reset partitions once per dataframe
      homogeneous_apply_refs([](auto&& x) { return PartitionManager<std::decay_t<decltype(x)>>::newDataframe(x); }, *task.get());
      // reset selections for the next dataframe
      for (auto& info : expressionInfos) {
        info.resetSelection = true;
      }
      // reset pre-slice for the next dataframe
      homogeneous_apply_refs([](auto& x) { return PresliceManager<std::decay_t<decltype(x)>>::setNewDF(x); }, *(task.get()));
      // prepare outputs
      homogeneous_apply_refs([&pc](auto&& x) { return OutputManager<std::decay_t<decltype(x)>>::prepare(pc, x); }, *task.get());
      // execute run()
      if constexpr (has_run_v<T>) {
        task->run(pc);
      }
      // execture process()
      if constexpr (has_process_v<T>) {
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos);
      }
      // execute optional process()
      homogeneous_apply_refs(
        [&pc, &expressionInfos, &task](auto& x) mutable {
          if constexpr (is_base_of_template_v<ProcessConfigurable, std::decay_t<decltype(x)>>) {
            if (x.value == true) {
              AnalysisDataProcessorBuilder::invokeProcess(*task.get(), pc.inputs(), x.process, expressionInfos);
              return true;
            }
          }
          return false;
        },
        *task.get());
      // finalize outputs
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
