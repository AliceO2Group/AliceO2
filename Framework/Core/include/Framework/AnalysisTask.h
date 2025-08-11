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
#include "Framework/EndOfStreamContext.h"
#include "Framework/GroupSlicer.h"
#include "Framework/StructToTuple.h"
#include "Framework/Traits.h"
#include "Framework/TypeIdHelpers.h"
#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/AnalysisDataModel.h"

#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <gandiva/node.h>
#include <type_traits>
#include <utility>
#include <memory>
#include <tuple> // IWYU pragma: export

namespace o2::framework
{
/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
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

template <typename T>
concept is_enumeration = is_enumeration_v<std::decay_t<T>>;

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...
namespace {
struct AnalysisDataProcessorBuilder {
  template <soa::is_iterator G, typename... Args>
  static void addGroupingCandidates(Cache& bk, Cache& bku, bool enabled)
  {
    [&bk, &bku, enabled]<typename... As>(framework::pack<As...>) mutable {
      auto key = std::string{"fIndex"} + o2::framework::cutString(soa::getLabelFromType<std::decay_t<G>>());
      ([&bk, &bku, &key, enabled]() mutable {
        if constexpr (soa::relatedByIndex<std::decay_t<G>, std::decay_t<As>>()) {
          auto binding = soa::getLabelFromTypeForKey<std::decay_t<As>>(key);
          if constexpr (o2::soa::is_smallgroups<std::decay_t<As>>) {
            framework::updatePairList(bku, binding, key, enabled);
          } else {
            framework::updatePairList(bk, binding, key, enabled);
          }
        }
      }(),
       ...);
    }(framework::pack<Args...>{});
  }

  template <soa::TableRef R>
  static void addOriginalRef(const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    auto spec = soa::tableRef2InputSpec<R>();
    spec.metadata.emplace_back(ConfigParamSpec{std::string{"control:"} + name, VariantType::Bool, value, {"\"\""}});
    DataSpecUtils::updateInputList(inputs, std::move(spec));
  }

  /// helpers to append expression information for a single argument
  template <soa::is_table A>
    requires(!soa::is_filtered_table<std::decay_t<A>>)
  static void addExpression(int, uint32_t, std::vector<ExpressionInfo>&)
  {
  }

  template <soa::is_filtered_table A>
  static void addExpression(int ai, uint32_t hash, std::vector<ExpressionInfo>& eInfos)
  {
    auto fields = soa::createFieldsFromColumns(typename std::decay_t<A>::persistent_columns_t{});
    eInfos.emplace_back(ai, hash, std::decay_t<A>::hashes(), std::make_shared<arrow::Schema>(fields));
  }

  template <soa::is_iterator A>
  static void addExpression(int ai, uint32_t hash, std::vector<ExpressionInfo>& eInfos)
  {
    addExpression<typename std::decay_t<A>::parent_t>(ai, hash, eInfos);
  }

  /// helpers to append InputSpec for a single argument
  template <soa::is_table A>
  static void addInput(const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    [&name, &value, &inputs]<size_t N, std::array<soa::TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) mutable {
      (addOriginalRef<refs[Is]>(name, value, inputs), ...);
    }.template operator()<A::originals.size(), std::decay_t<A>::originals>(std::make_index_sequence<std::decay_t<A>::originals.size()>());
  }

  template <soa::is_iterator A>
  static void addInput(const char* name, bool value, std::vector<InputSpec>& inputs)
  {
    addInput<typename std::decay_t<A>::parent_t>(name, value, inputs);
  }

  /// helper to append the inputs and expression information for normalized arguments
  template <soa::is_table... As>
  static void addInputsAndExpressions(uint32_t hash, const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    int ai = -1;
    ([&ai, &hash, &eInfos, &name, &value, &inputs]() mutable {
      ++ai;
      using T = std::decay_t<As>;
      addExpression<T>(ai, hash, eInfos);
      addInput<T>(name, value, inputs);
    }(),
     ...);
  }

  /// helper to parse the process arguments
  template <typename T>
  inline static bool requestInputsFromArgs(T&, std::string const&, std::vector<InputSpec>&, std::vector<ExpressionInfo>&)
  {
    return false;
  }
  template <is_process_configurable T>
  inline static bool requestInputsFromArgs(T& pc, std::string const& name, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eis)
  {
    AnalysisDataProcessorBuilder::inputsFromArgs(pc.process, (name + "/" + pc.name).c_str(), pc.value, inputs, eis);
    return true;
  }
  template <typename T>
  inline static bool requestCacheFromArgs(T&, Cache&, Cache&)
  {
    return false;
  }
  template <is_process_configurable T>
  inline static bool requestCacheFromArgs(T& pc, Cache& bk, Cache& bku)
  {
    AnalysisDataProcessorBuilder::cacheFromArgs(pc.process, pc.value, bk, bku);
    return true;
  }
  /// 1. enumeration (must be the only argument)
  template <typename R, typename C, is_enumeration A>
  static void inputsFromArgs(R (C::*)(A), const char* /*name*/, bool /*value*/, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>&) //, Cache&, Cache&)
  {
    std::vector<ConfigParamSpec> inputMetadata;
    // FIXME: for the moment we do not support begin, end and step.
    DataSpecUtils::updateInputList(inputs, InputSpec{"enumeration", "DPL", "ENUM", 0, Lifetime::Enumeration, inputMetadata});
  }

  /// 2. 1st argument is an iterator
  template <typename R, typename C, soa::is_iterator A, soa::is_table... Args>
  static void inputsFromArgs(R (C::*)(A, Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos) //, Cache& bk, Cache& bku)
    requires(std::is_lvalue_reference_v<A> && (std::is_lvalue_reference_v<Args> && ...))
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<R (C::*)(A, Args...)>();
    addInputsAndExpressions<typename std::decay_t<A>::parent_t, Args...>(hash, name, value, inputs, eInfos);
  }

  /// 3. generic case
  template <typename R, typename C, soa::is_table... Args>
  static void inputsFromArgs(R (C::*)(Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos) //, Cache&, Cache&)
    requires(std::is_lvalue_reference_v<Args> && ...)
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<R (C::*)(Args...)>();
    addInputsAndExpressions<Args...>(hash, name, value, inputs, eInfos);
  }

  /// 1. enumeration (no grouping)
  template <typename R, typename C, is_enumeration A>
  static void cacheFromArgs(R (C::*)(A), bool, Cache&, Cache&)
  {
  }
  /// 2. iterator (the only grouping case)
  template <typename R, typename C, soa::is_iterator A, soa::is_table... Args>
  static void cacheFromArgs(R (C::*)(A, Args...), bool value, Cache& bk, Cache& bku)
  {
    addGroupingCandidates<A, Args...>(bk, bku, value);
  }
  /// 3. generic case (no grouping)
  template <typename R, typename C, soa::is_table A, soa::is_table... Args>
  static void cacheFromArgs(R (C::*)(A, Args...), bool, Cache&, Cache&)
  {
  }

  template <soa::TableRef R>
  static auto extractTableFromRecord(InputRecord& record)
  {
    auto table = record.get<TableConsumer>(o2::aod::label<R>())->asArrowTable();
    if (table->num_rows() == 0) {
      table = makeEmptyTable<R>();
    }
    return table;
  }

  template <soa::is_table T>
  static auto extractFromRecord(InputRecord& record)
  {
    return T { [&record]<size_t N, std::array<soa::TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) { return std::vector{extractTableFromRecord<refs[Is]>(record)...}; }.template operator()<T::originals.size(), T::originals>(std::make_index_sequence<T::originals.size()>()) };
  }

  template <soa::is_iterator T>
  static auto extractFromRecord(InputRecord& record)
  {
    return typename T::parent_t { [&record]<size_t N, std::array<soa::TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) { return std::vector{extractTableFromRecord<refs[Is]>(record)...}; }.template operator()<T::parent_t::originals.size(), T::parent_t::originals>(std::make_index_sequence<T::parent_t::originals.size()>()) };
  }

  template <soa::is_filtered T>
  static auto extractFilteredFromRecord(InputRecord& record, ExpressionInfo& info)
  {
    std::shared_ptr<arrow::Table> table = nullptr;
    auto joiner = [&record]<size_t N, std::array<soa::TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) { return std::vector{extractTableFromRecord<refs[Is]>(record)...}; };
    if constexpr (soa::is_iterator<T>) {
      table = o2::soa::ArrowHelpers::joinTables(joiner.template operator()<T::parent_t::originals.size(), T::parent_t::originals>(std::make_index_sequence<T::parent_t::originals.size()>()), std::span{T::parent_t::originalLabels});
    } else {
      table = o2::soa::ArrowHelpers::joinTables(joiner.template operator()<T::originals.size(), T::originals>(std::make_index_sequence<T::originals.size()>()), std::span{T::originalLabels});
    }
    expressions::updateFilterInfo(info, table);
    if constexpr (!o2::soa::is_smallgroups<std::decay_t<T>>) {
      if (info.selection == nullptr) {
        soa::missingFilterDeclaration(info.processHash, info.argumentIndex);
      }
    }
    if constexpr (soa::is_iterator<T>) {
      return typename T::parent_t({table}, info.selection);
    } else {
      return T({table}, info.selection);
    }
  }

  template <is_enumeration T, int AI>
  static auto extract(InputRecord&, std::vector<ExpressionInfo>&, size_t)
  {
    return T{};
  }

  template <soa::is_iterator T, int AI>
  static auto extract(InputRecord& record, std::vector<ExpressionInfo>& infos, size_t phash)
  {
    if constexpr (std::same_as<typename T::policy_t, soa::FilteredIndexPolicy>) {
      return extractFilteredFromRecord<T>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }));
    } else {
      return extractFromRecord<T>(record);
    }
  }

  template <soa::is_table T, int AI>
  static auto extract(InputRecord& record, std::vector<ExpressionInfo>& infos, size_t phash)
  {
    if constexpr (soa::is_filtered_table<T>) {
      return extractFilteredFromRecord<T>(record, *std::find_if(infos.begin(), infos.end(), [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }));
    } else {
      return extractFromRecord<T>(record);
    }
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
    requires(!std::same_as<Grouping, void>)
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<R (C::*)(Grouping, Args...)>();
    return extract<std::decay_t<Grouping>, 0>(record, infos, hash);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
    requires(!std::same_as<Grouping, void> && sizeof...(Args) > 0)
  {
    constexpr auto p = pack<Args...>{};
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<R (C::*)(Grouping, Args...)>();
    return std::make_tuple(extract<std::decay_t<Args>, has_type_at_v<Args>(p) + 1>(record, infos, hash)...);
  }

  template <typename... As>
  static void overwriteInternalIndices(std::tuple<As...>& dest, std::tuple<As...> const& src)
  {
    (std::get<As>(dest).bindInternalIndicesTo(&std::get<As>(src)), ...);
  }

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*processingFunction)(Grouping, Associated...), std::vector<ExpressionInfo>& infos, ArrowTableSlicingCache& slices)
  {
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, processingFunction, infos);

    // set filtered tables for partitions with grouping
    homogeneous_apply_refs([&groupingTable](auto& element) {
      analysis_task_parsers::setPartition(element, groupingTable);
      analysis_task_parsers::bindInternalIndicesPartition(element, &groupingTable);
      return true;
    },
                           task);

    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      homogeneous_apply_refs([&groupingTable](auto& element) {
        analysis_task_parsers::bindExternalIndicesPartition(element, &groupingTable);
        analysis_task_parsers::setGroupedCombination(element, groupingTable);
        return true;
      },
                             task);
      if constexpr (soa::is_iterator<G>) {
        for (auto& element : groupingTable) {
          std::invoke(processingFunction, task, *element);
        }
      } else {
        static_assert(soa::is_table<G> || is_enumeration<G>,
                      "Single argument of process() should be a table-like or an iterator");
        std::invoke(processingFunction, task, groupingTable);
      }
    } else {
      // multiple arguments to process
      static_assert(((soa::is_iterator<std::decay_t<Associated>> == false) && ...),
                    "Associated arguments of process() should not be iterators");
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, processingFunction, infos);
      // pre-bind self indices
      std::apply(
        [&task](auto&... t) mutable {
          (homogeneous_apply_refs(
             [&t](auto& p) {
               analysis_task_parsers::bindInternalIndicesPartition(p, &t);
               return true;
             },
             task),
           ...);
        },
        associatedTables);

      auto binder = [&task, &groupingTable, &associatedTables](auto& x) mutable {
        x.bindExternalIndices(&groupingTable, &std::get<std::decay_t<Associated>>(associatedTables)...);
        homogeneous_apply_refs([&x](auto& t) mutable {
          analysis_task_parsers::setPartition(t, x);
          analysis_task_parsers::bindExternalIndicesPartition(t, &x);
          return true;
        },
                               task);
      };
      groupingTable.bindExternalIndices(&std::get<std::decay_t<Associated>>(associatedTables)...);

      // always pre-bind full tables to support index hierarchy
      std::apply(
        [&binder](auto&... x) mutable {
          (binder(x), ...);
        },
        associatedTables);

      // GroupedCombinations bound separately, as they should be set once for all associated tables
      homogeneous_apply_refs([&groupingTable, &associatedTables](auto& t) {
        analysis_task_parsers::setGroupedCombination(t, groupingTable, associatedTables);
        return true;
      },
                             task);
      overwriteInternalIndices(associatedTables, associatedTables);
      if constexpr (soa::is_iterator<std::decay_t<G>>) {
        auto slicer = GroupSlicer(groupingTable, associatedTables, slices);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();
          overwriteInternalIndices(associatedSlices, associatedTables);
          std::apply(
            [&binder](auto&... x) mutable {
              (binder(x), ...);
            },
            associatedSlices);

          // bind partitions and grouping table
          homogeneous_apply_refs([&groupingTable](auto& x) {
            analysis_task_parsers::bindExternalIndicesPartition(x, &groupingTable);
            return true;
          },
                                 task);

          invokeProcessWithArgs(task, processingFunction, slice.groupingElement(), associatedSlices);
        }
      } else {
        // bind partitions and grouping table
        homogeneous_apply_refs([&groupingTable](auto& x) {
          analysis_task_parsers::bindExternalIndicesPartition(x, &groupingTable);
          return true;
        },
                               task);

        invokeProcessWithArgs(task, processingFunction, groupingTable, associatedTables);
      }
    }
  }

  template <typename C, typename T, typename G, typename... A>
  static void invokeProcessWithArgs(C& task, T processingFunction, G g, std::tuple<A...>& at)
  {
    std::invoke(processingFunction, task, g, std::get<A>(at)...);
  }
};
}

struct SetDefaultProcesses {
  std::vector<std::pair<std::string, bool>> map;
};

/// Struct to differentiate task names from possible task string arguments
struct TaskName {
  TaskName(std::string name) : value{std::move(name)} {}
  std::string value;
};

namespace {
template <typename T, typename... A>
auto getTaskNameSetProcesses(std::string& outputName, TaskName first, SetDefaultProcesses second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : second.map) {
    homogeneous_apply_refs(
      [&](auto& element) {
        return analysis_task_parsers::setProcessSwitch(setting, element);
      },
      *task.get());
  }
  outputName = first.value;
  return task;
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(std::string& outputName, SetDefaultProcesses first, TaskName second, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : first.map) {
    homogeneous_apply_refs(
      [&](auto& element) {
        return analysis_task_parsers::setProcessSwitch(setting, element);
      },
      *task.get());
  }
  outputName = second.value;
  return task;
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(std::string& outputName, SetDefaultProcesses first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  for (auto& setting : first.map) {
    homogeneous_apply_refs(
      [&](auto& element) {
        return analysis_task_parsers::setProcessSwitch(setting, element);
      },
      *task.get());
  }
  auto type_name_str = type_name<T>();
  outputName = type_to_task_name(type_name_str);
  return task;
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(std::string& outputName, TaskName first, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  outputName = first.value;
  return task;
}

template <typename T, typename... A>
auto getTaskNameSetProcesses(std::string& outputName, A... args)
{
  auto task = std::make_shared<T>(std::forward<A>(args)...);
  auto type_name_str = type_name<T>();
  outputName = type_to_task_name(type_name_str);
  return task;
}

}

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(ConfigContext const& ctx, Args&&... args)
{
  TH1::AddDirectory(false);

  std::string name_str;
  auto task = getTaskNameSetProcesses<T>(name_str, args...);

  auto suffix = ctx.options().get<std::string>("workflow-suffix");
  if (!suffix.empty()) {
    name_str += suffix;
  }
  const char* name = name_str.c_str();

  auto hash = runtime_hash(name);

  std::vector<OutputSpec> outputs;
  std::vector<InputSpec> inputs;
  std::vector<ConfigParamSpec> options;
  std::vector<ExpressionInfo> expressionInfos;

  /// make sure options and configurables are set before expression infos are created
  homogeneous_apply_refs([&options, &hash](auto& element) { return analysis_task_parsers::appendOption(options, element); }, *task.get());
  /// extract conditions and append them as inputs
  homogeneous_apply_refs([&inputs](auto& element) { return analysis_task_parsers::appendCondition(inputs, element); }, *task.get());

  /// parse process functions defined by corresponding configurables
  if constexpr (requires { &T::process; }) {
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, "default", true, inputs, expressionInfos);
  }
  homogeneous_apply_refs(
    [name = name_str, &expressionInfos, &inputs](auto& x) mutable {
      // this pushes (argumentIndex, processHash, schemaPtr, nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
      return AnalysisDataProcessorBuilder::requestInputsFromArgs(x, name, inputs, expressionInfos);
    },
    *task.get());

  // request base tables for spawnable extended tables and indices to be built
  // this checks for duplications
  homogeneous_apply_refs([&inputs](auto& element) {
    return analysis_task_parsers::requestInputs(inputs, element);
  },
                         *task.get());

  // no static way to check if the task defines any processing, we can only make sure it subscribes to at least something
  if (inputs.empty() == true) {
    LOG(warn) << "Task " << name_str << " has no inputs";
  }

  homogeneous_apply_refs([&outputs, &hash](auto& element) { return analysis_task_parsers::appendOutput(outputs, element, hash); }, *task.get());

  auto requiredServices = CommonServices::defaultServices();
  auto arrowServices = CommonServices::arrowServices();
  requiredServices.insert(requiredServices.end(), arrowServices.begin(), arrowServices.end());
  homogeneous_apply_refs([&requiredServices](auto& element) { return analysis_task_parsers::addService(requiredServices, element); }, *task.get());

  auto algo = AlgorithmSpec::InitCallback{[task = task, expressionInfos](InitContext& ic) mutable {
    Cache bindingsKeys;
    Cache bindingsKeysUnsorted;
    // add preslice declarations to slicing cache definition
    homogeneous_apply_refs([&bindingsKeys, &bindingsKeysUnsorted](auto& element) { return analysis_task_parsers::registerCache(element, bindingsKeys, bindingsKeysUnsorted); }, *task.get());

    homogeneous_apply_refs([&ic](auto&& element) { return analysis_task_parsers::prepareOption(ic, element); }, *task.get());
    homogeneous_apply_refs([&ic](auto&& element) { return analysis_task_parsers::prepareService(ic, element); }, *task.get());

    auto& callbacks = ic.services().get<CallbackService>();
    auto eoscb = [task](EndOfStreamContext& eosContext) {
      homogeneous_apply_refs([&eosContext](auto& element) {
          analysis_task_parsers::postRunService(eosContext, element);
          analysis_task_parsers::postRunOutput(eosContext, element);
          return true; },
                             *task.get());
      eosContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    callbacks.set<CallbackService::Id::EndOfStream>(eoscb);

    /// call the task's init() function first as it may manipulate the task's elements
    if constexpr (requires { task->init(ic); }) {
      task->init(ic);
    }

    /// update configurables in filters and partitions
    homogeneous_apply_refs(
      [&ic](auto& element) -> bool { return analysis_task_parsers::updatePlaceholders(ic, element); },
      *task.get());
    /// create expression trees for filters gandiva trees matched to schemas and store the pointers into expressionInfos
    homogeneous_apply_refs([&expressionInfos](auto& element) {
      return analysis_task_parsers::createExpressionTrees(expressionInfos, element);
    },
                           *task.get());

    /// parse process functions to enable requested grouping caches - note that at this state process configurables have their final values
    if constexpr (requires { &T::process; }) {
      AnalysisDataProcessorBuilder::cacheFromArgs(&T::process, true, bindingsKeys, bindingsKeysUnsorted);
    }
    homogeneous_apply_refs(
      [&bindingsKeys, &bindingsKeysUnsorted](auto& x) mutable {
        return AnalysisDataProcessorBuilder::requestCacheFromArgs(x, bindingsKeys, bindingsKeysUnsorted);
      },
      *task.get());

    ic.services().get<ArrowTableSlicingCacheDef>().setCaches(std::move(bindingsKeys));
    ic.services().get<ArrowTableSlicingCacheDef>().setCachesUnsorted(std::move(bindingsKeysUnsorted));
    // initialize global caches
    homogeneous_apply_refs([&ic](auto& element) {
      return analysis_task_parsers::preInitializeCache(ic, element);
    },
                           *(task.get()));

    return [task, expressionInfos](ProcessingContext& pc) mutable {
      // load the ccdb object from their cache
      homogeneous_apply_refs([&pc](auto& element) { return analysis_task_parsers::newDataframeCondition(pc.inputs(), element); }, *task.get());
      // reset partitions once per dataframe
      homogeneous_apply_refs([](auto& element) { return analysis_task_parsers::newDataframePartition(element); }, *task.get());
      // reset selections for the next dataframe
      for (auto& info : expressionInfos) {
        info.resetSelection = true;
      }
      // reset pre-slice for the next dataframe
      auto slices = pc.services().get<ArrowTableSlicingCache>();
      homogeneous_apply_refs([&pc, &slices](auto& element) {
        return analysis_task_parsers::updateSliceInfo(element, slices);
      },
                             *(task.get()));
      // initialize local caches
      homogeneous_apply_refs([&pc](auto& element) { return analysis_task_parsers::initializeCache(pc, element); }, *(task.get()));
      // prepare outputs
      homogeneous_apply_refs([&pc](auto& element) { return analysis_task_parsers::prepareOutput(pc, element); }, *task.get());
      // execute run()
      if constexpr (requires { task->run(pc); }) {
        task->run(pc);
      }
      // execute process()
      if constexpr (requires { AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos, slices); }) {
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos, slices);
      }
      // execute optional process()
      homogeneous_apply_refs(
        [&pc, &expressionInfos, &task, &slices](auto& x) mutable {
          if constexpr (base_of_template<ProcessConfigurable, std::decay_t<decltype(x)>>) {
            if (x.value == true) {
              AnalysisDataProcessorBuilder::invokeProcess(*task.get(), pc.inputs(), x.process, expressionInfos, slices);
              return true;
            }
          }
          return false;
        },
        *task.get());
      // prepare delayed outputs
      homogeneous_apply_refs([&pc](auto& element) { return analysis_task_parsers::prepareDelayedOutput(pc, element); }, *task.get());
      // finalize outputs
      homogeneous_apply_refs([&pc](auto& element) { return analysis_task_parsers::finalizeOutput(pc, element); }, *task.get());
    };
  }};

  return {
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    outputs,
    algo,
    options,
    requiredServices};
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
