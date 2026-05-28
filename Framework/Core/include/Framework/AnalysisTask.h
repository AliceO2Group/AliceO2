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
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/GroupSlicer.h"
#include "Framework/StructToTuple.h"
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
/// Convert a CamelCase task struct name to snake-case task name
std::string type_to_task_name(std::string_view const& camelCase);

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

template <typename T>
concept is_table_iterator_or_enumeration = soa::is_table_or_iterator<T> || is_enumeration<T>;

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...
namespace
{
struct AnalysisDataProcessorBuilder {
  template <soa::is_iterator G, soa::is_table... Args>
  static void addGroupingCandidates(Cache& bk, Cache& bku, bool enabled)
  {
    []<soa::is_table... As>(framework::pack<As...>, Cache& bk, Cache& bku, bool enabled) {
      auto key = std::string{"fIndex"} + o2::framework::cutString(soa::getLabelFromType<std::decay_t<G>>());
      ([](Cache& bk, Cache& bku, bool enabled, std::string const& key) {
        if constexpr (soa::relatedByIndex<std::decay_t<G>, std::decay_t<As>>()) {
          Entry e{soa::getLabelFromTypeForKey<std::decay_t<As>>(key), soa::getMatcherFromTypeForKey<std::decay_t<As>>(key), key, enabled};
          if constexpr (o2::soa::is_smallgroups<std::decay_t<As>>) {
            framework::updatePairList(bku, e);
          } else {
            framework::updatePairList(bk, e);
          }
        }
      }(bk, bku, enabled, key),
       ...);
    }(framework::pack<Args...>{}, bk, bku, enabled);
  }

  template <soa::TableRef R>
  static void addOriginalRef(const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<InputInfo>& iInfos, int ai, uint32_t hash, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
  {
    auto spec = soa::tableRef2InputSpec<R>(newOrigin);
    if (R.origin_hash != "AOD"_h) {
      spec.metadata.emplace_back(ConfigParamSpec{"aod-origin-replaced", VariantType::Bool, true, {"\"\""}});
    }
    spec.metadata.emplace_back(ConfigParamSpec{std::string{"control:"} + name, VariantType::Bool, value, {"\"\""}});
    auto matcher = DataSpecUtils::asConcreteDataMatcher(spec);
    DataSpecUtils::updateInputList(inputs, std::move(spec));
    auto locate = std::ranges::find_if(iInfos, [&hash](auto const& info) { return info.hash == hash; });
    if (locate == iInfos.end()) {
      iInfos.emplace_back(hash, std::vector{std::pair{ai, matcher}});
    } else {
      if (std::ranges::none_of(locate->matchers, [&ai, &matcher](auto const& match) { return (match.first == ai) && (match.second == matcher); })) {
        locate->matchers.emplace_back(std::pair{ai, matcher});
      }
    }
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
  static void addInput(const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<InputInfo>& iInfos, int ai, uint32_t hash, header::DataOrigin&& newOrigin = header::DataOrigin{"AOD"})
  {
    [&name, &value, &inputs, &iInfos, &ai, &hash, newOrigin = std::move(newOrigin)]<size_t N, std::array<soa::TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) mutable {
      (addOriginalRef<refs[Is]>(name, value, inputs, iInfos, ai, hash, newOrigin), ...);
    }.template operator()<A::originals.size(), std::decay_t<A>::originals>(std::make_index_sequence<std::decay_t<A>::originals.size()>());
  }

  /// helper to append the inputs and expression information for normalized arguments
  template <soa::is_table... As>
  static void addInputsAndExpressions(uint32_t hash, const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos, std::vector<InputInfo>& iInfos, header::DataOrigin&& newOrigin = header::DataOrigin{"AOD"})
  {
    int ai = -1;
    ([&ai, &hash, &eInfos, &name, &value, &inputs, &iInfos, newOrigin]() mutable {
      ++ai;
      using T = std::decay_t<As>;
      addExpression<T>(ai, hash, eInfos);
      addInput<T>(name, value, inputs, iInfos, ai, hash, std::move(newOrigin));
    }(),
     ...);
  }

  /// helper to parse the process arguments
  template <typename T>
  inline static bool requestInputsFromArgs(T&, std::string const&, std::vector<InputSpec>&, std::vector<ExpressionInfo>&, std::vector<InputInfo>&, header::DataOrigin)
  {
    return false;
  }
  template <is_process_configurable T>
  inline static bool requestInputsFromArgs(T& pc, std::string const& name, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eis, std::vector<InputInfo>& iifs, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
  {
    AnalysisDataProcessorBuilder::inputsFromArgs(pc.process, (name + "/" + pc.name).c_str(), pc.value, inputs, eis, iifs, newOrigin);
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
  template <typename C, is_enumeration A>
  static void inputsFromArgs(void (C::*)(A), const char* /*name*/, bool /*value*/, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>&, std::vector<InputInfo>&, header::DataOrigin)
  {
    std::vector<ConfigParamSpec> inputMetadata;
    // FIXME: for the moment we do not support begin, end and step.
    DataSpecUtils::updateInputList(inputs, InputSpec{"enumeration", "DPL", "ENUM", 0, Lifetime::Enumeration, inputMetadata});
  }

  /// 2. 1st argument is an iterator
  template <typename C, soa::is_iterator A, soa::is_table... Args>
  static void inputsFromArgs(void (C::*)(A, Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos, std::vector<InputInfo>& iInfos, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
    requires(std::is_lvalue_reference_v<A> && (std::is_lvalue_reference_v<Args> && ...))
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<void (C::*)(A, Args...)>();
    addInputsAndExpressions<typename std::decay_t<A>::parent_t, Args...>(hash, name, value, inputs, eInfos, iInfos, std::move(newOrigin));
  }

  /// 3. generic case
  template <typename C, soa::is_table... Args>
  static void inputsFromArgs(void (C::*)(Args...), const char* name, bool value, std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos, std::vector<InputInfo>& iInfos, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
    requires(std::is_lvalue_reference_v<Args> && ...)
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<void (C::*)(Args...)>();
    addInputsAndExpressions<Args...>(hash, name, value, inputs, eInfos, iInfos, std::move(newOrigin));
  }

  /// 1. enumeration (no grouping)
  template <typename C, is_enumeration A>
  static void cacheFromArgs(void (C::*)(A), bool, Cache&, Cache&)
  {
  }
  /// 2. iterator (the only grouping case)
  template <typename C, soa::is_iterator A, soa::is_table... Args>
  static void cacheFromArgs(void (C::*)(A, Args...), bool value, Cache& bk, Cache& bku)
  {
    addGroupingCandidates<A, Args...>(bk, bku, value);
  }
  /// 3. generic case (no grouping)
  template <typename C, soa::is_table A, soa::is_table... Args>
  static void cacheFromArgs(void (C::*)(A, Args...), bool, Cache&, Cache&)
  {
  }

  template <std::ranges::input_range R>
  static auto extractTablesFromRecord(InputRecord& record, R matchers)
  {
    std::vector<std::shared_ptr<arrow::Table>> tables;
    std::ranges::transform(matchers, std::back_inserter(tables), [&record](auto const& m) {
      return record.get<TableConsumer>(m.second)->asArrowTable();
    });
    return tables;
  }

  template <soa::is_table T, std::ranges::input_range R>
  static auto extractFromRecord(InputRecord& record, R matchers)
  {
    return T{extractTablesFromRecord(record, matchers)};
  }

  template <soa::is_iterator T, std::ranges::input_range R>
  static auto extractFromRecord(InputRecord& record, R matchers)
  {
    return typename T::parent_t{extractTablesFromRecord(record, matchers)};
  }

  template <soa::is_filtered T, std::ranges::input_range R>
  static auto extractFilteredFromRecord(InputRecord& record, R matchers, ExpressionInfo& info)
  {
    std::shared_ptr<arrow::Table> table = soa::ArrowHelpers::joinTables(extractTablesFromRecord(record, matchers));
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

  template <is_enumeration T, int AI, std::ranges::input_range R>
  static auto extract(InputRecord&, R, std::vector<ExpressionInfo>&, size_t)
  {
    return T{};
  }

  template <soa::is_table_or_iterator T, int AI, std::ranges::input_range R>
  static auto extract(InputRecord& record, R matchers, std::vector<ExpressionInfo>& infos, size_t phash)
  {
    // auto matchers = std::ranges::find_if(iInfos, [&phash](auto const& info) { return info.hash == phash; })->matchers | std::views::filter([](auto const& pair) { return pair.first == AI; });
    if constexpr (soa::is_filtered<T>) {
      return extractFilteredFromRecord<T>(record, matchers, *std::ranges::find_if(infos, [&phash](ExpressionInfo const& i) { return (i.processHash == phash && i.argumentIndex == AI); }));
    } else {
      return extractFromRecord<T>(record, matchers);
    }
  }

  template <std::ranges::input_range R, typename C, is_table_iterator_or_enumeration Grouping, soa::is_table... Args>
  static auto bindGroupingTable(InputRecord& record, R matchers, void (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
    requires(!std::same_as<Grouping, void>)
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<void (C::*)(Grouping, Args...)>();
    return extract<std::decay_t<Grouping>, 0>(record, matchers | std::views::filter([](auto const& pair) { return pair.first == 0; }), infos, hash);
  }

  template <std::ranges::input_range R, typename C, is_table_iterator_or_enumeration Grouping, soa::is_table... Args>
  static auto bindAssociatedTables(InputRecord& record, R matchers, void (C::*)(Grouping, Args...), std::vector<ExpressionInfo>& infos)
    requires(!std::same_as<Grouping, void> && sizeof...(Args) > 0)
  {
    constexpr auto hash = o2::framework::TypeIdHelpers::uniqueId<void (C::*)(Grouping, Args...)>();
    return std::make_tuple(extract<std::decay_t<Args>, has_type_at_v<Args>(pack<Args...>{}) + 1>(record, matchers | std::views::filter([](auto const& pair) { return pair.first == has_type_at_v<Args>(pack<Args...>{}) + 1; }), infos, hash)...);
  }

  template <soa::is_table... As>
  static void overwriteInternalIndices(std::tuple<As...>& dest, std::tuple<As...> const& src)
  {
    (std::get<As>(dest).bindInternalIndicesTo(&std::get<As>(src)), ...);
  }

  template <typename Task, is_table_iterator_or_enumeration Grouping, std::ranges::input_range R, soa::is_table... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R matchers, void (Task::*processingFunction)(Grouping, Associated...), std::vector<ExpressionInfo>& infos, ArrowTableSlicingCache& slices, header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
  {
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, matchers, processingFunction, infos);

    constexpr const int numElements = nested_brace_constructible_size<false, std::decay_t<Task>>() / 10;

    // set filtered tables for partitions with grouping
    homogeneous_apply_refs_sized<numElements>([&groupingTable](auto& element) {
      analysis_task_parsers::setPartition(element, groupingTable);
      analysis_task_parsers::bindInternalIndicesPartition(element, &groupingTable);
      return true;
    },
                                              task);

    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      homogeneous_apply_refs_sized<numElements>([&groupingTable](auto& element) {
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
        std::invoke(processingFunction, task, groupingTable);
      }
    } else {
      // multiple arguments to process
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, matchers, processingFunction, infos);
      // pre-bind self indices
      std::apply(
        [&task](auto&... t) mutable {
          (homogeneous_apply_refs_sized<numElements>(
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
        homogeneous_apply_refs_sized<numElements>([&x](auto& t) mutable {
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
      homogeneous_apply_refs_sized<numElements>([&groupingTable, &associatedTables](auto& t) {
        analysis_task_parsers::setGroupedCombination(t, groupingTable, associatedTables);
        return true;
      },
                                                task);
      overwriteInternalIndices(associatedTables, associatedTables);
      if constexpr (soa::is_iterator<G>) {
        auto slicer = GroupSlicer(groupingTable, associatedTables, slices, newOrigin);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();
          overwriteInternalIndices(associatedSlices, associatedTables);
          std::apply(
            [&binder](auto&... x) mutable {
              (binder(x), ...);
            },
            associatedSlices);

          // bind partitions and grouping table
          homogeneous_apply_refs_sized<numElements>([&groupingTable](auto& x) {
            analysis_task_parsers::bindExternalIndicesPartition(x, &groupingTable);
            return true;
          },
                                                    task);

          [](Task& task, void (Task::*processingFunction)(Grouping, Associated...), Grouping g, std::tuple<std::decay_t<Associated>...>& at) {
            std::invoke(processingFunction, task, g, std::get<std::decay_t<Associated>>(at)...);
          }(task, processingFunction, slice.groupingElement(), associatedSlices);
        }
      } else {
        // bind partitions and grouping table
        homogeneous_apply_refs_sized<numElements>([&groupingTable](auto& x) {
          analysis_task_parsers::bindExternalIndicesPartition(x, &groupingTable);
          return true;
        },
                                                  task);

        [](Task& task, void (Task::*processingFunction)(Grouping, Associated...), Grouping g, std::tuple<std::decay_t<Associated>...>& at) {
          std::invoke(processingFunction, task, g, std::get<std::decay_t<Associated>>(at)...);
        }(task, processingFunction, groupingTable, associatedTables);
      }
    }
  }
};
} // namespace

struct SetDefaultProcesses {
  std::vector<std::pair<std::string, bool>> map;
};

/// Struct to differentiate task names from possible task string arguments
struct TaskName {
  TaskName(std::string name) : value{std::move(name)} {}
  std::string value;
};

namespace
{
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
} // namespace

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
  std::vector<InputInfo> inputInfos;

  std::string newOriginStr;
  header::DataOrigin newOrigin{"AOD"};
  if (ctx.options().hasOption("aod-origin-replace")) {
    newOriginStr = ctx.options().get<std::string>("aod-origin-replace");
    if (newOriginStr.size() > 4UL) {
      wrongOriginReplacement(newOriginStr);
    }
  }
  if (!newOriginStr.empty()) {
    newOrigin.runtimeInit(newOriginStr.c_str(), std::min(newOriginStr.size(), 4UL));
  }

  constexpr const int numElements = nested_brace_constructible_size<false, std::decay_t<T>>() / 10;

  /// make sure options and configurables are set before expression infos are created
  homogeneous_apply_refs_sized<numElements>([&options](auto& element) { return analysis_task_parsers::appendOption(options, element); }, *task.get());
  /// extract conditions and append them as inputs
  homogeneous_apply_refs_sized<numElements>([&inputs](auto& element) { return analysis_task_parsers::appendCondition(inputs, element); }, *task.get());

  /// parse process functions defined by corresponding configurables
  if constexpr (requires { &T::process; }) {
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, "default", true, inputs, expressionInfos, inputInfos, newOrigin);
  }
  homogeneous_apply_refs_sized<numElements>(
    [name = name_str, &expressionInfos, &inputs, &inputInfos, &newOrigin](auto& x) mutable {
      // this pushes (argumentIndex, processHash, schemaPtr, nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
      return AnalysisDataProcessorBuilder::requestInputsFromArgs(x, name, inputs, expressionInfos, inputInfos, newOrigin);
    },
    *task.get());

  // request base tables for spawnable extended tables and indices to be built
  // this checks for duplications
  homogeneous_apply_refs_sized<numElements>([&inputs, &newOrigin](auto& element) {
    return analysis_task_parsers::requestInputs(inputs, element, newOrigin);
  },
                                            *task.get());

  // no static way to check if the task defines any processing, we can only make sure it subscribes to at least something
  if (inputs.empty() == true) {
    LOG(warn) << "Task " << name_str << " has no inputs";
  }

  // update OutputSpecs in output declarations
  homogeneous_apply_refs_sized<numElements>([&newOrigin](auto& element) { return analysis_task_parsers::updateOutputSpec(element, newOrigin); }, *task.get());

  // Auto-register default ccdb: path options from subscribed timestamped-table inputs.
  // This allows tasks to accept --ccdb:fXxx overrides without requiring an explicit
  // ConfigurableCCDBPath<> member for every column in the subscribed table.
  for (auto& input : inputs) {
    for (auto& meta : input.metadata) {
      if (meta.name.starts_with("ccdb:") && meta.name != "ccdb:") {
        ConfigParamsHelper::addOptionIfMissing(options, meta);
      }
    }
  }

  // append outputs
  homogeneous_apply_refs_sized<numElements>([&outputs, &hash](auto& element) { return analysis_task_parsers::appendOutput(outputs, element, hash); }, *task.get());

  // request services
  auto requiredServices = CommonServices::defaultServices();
  auto arrowServices = CommonServices::arrowServices();
  requiredServices.insert(requiredServices.end(), arrowServices.begin(), arrowServices.end());
  homogeneous_apply_refs_sized<numElements>([&requiredServices](auto& element) { return analysis_task_parsers::addService(requiredServices, element); }, *task.get());

  // replace origins in Preslice declarations
  homogeneous_apply_refs_sized<numElements>([&newOrigin](auto& element) { return analysis_task_parsers::replaceOrigin(element, newOrigin); }, *task.get());

  auto algo = AlgorithmSpec::InitCallback{[task = task, expressionInfos, inputInfos, newOrigin, newOriginStr](InitContext& ic) mutable {
    Cache bindingsKeys;
    Cache bindingsKeysUnsorted;
    // add preslice declarations to slicing cache definition
    homogeneous_apply_refs_sized<numElements>([&bindingsKeys, &bindingsKeysUnsorted](auto& element) { return analysis_task_parsers::registerCache(element, bindingsKeys, bindingsKeysUnsorted); }, *task.get());

    homogeneous_apply_refs_sized<numElements>([&ic](auto&& element) { return analysis_task_parsers::prepareOption(ic, element); }, *task.get());
    homogeneous_apply_refs_sized<numElements>([&ic](auto&& element) { return analysis_task_parsers::prepareService(ic, element); }, *task.get());

    auto& callbacks = ic.services().get<CallbackService>();
    auto eoscb = [task](EndOfStreamContext& eosContext) {
      homogeneous_apply_refs_sized<numElements>([&eosContext](auto& element) {
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
    homogeneous_apply_refs_sized<numElements>(
      [&ic](auto& element) -> bool { return analysis_task_parsers::updatePlaceholders(ic, element); },
      *task.get());
    /// create expression trees for filters gandiva trees matched to schemas and store the pointers into expressionInfos
    homogeneous_apply_refs_sized<numElements>([&expressionInfos](auto& element) {
      return analysis_task_parsers::createExpressionTrees(expressionInfos, element);
    },
                                              *task.get());

    /// parse process functions to enable requested grouping caches - note that at this state process configurables have their final values
    if constexpr (requires { &T::process; }) {
      AnalysisDataProcessorBuilder::cacheFromArgs(&T::process, true, bindingsKeys, bindingsKeysUnsorted);
    }
    homogeneous_apply_refs_sized<numElements>(
      [&bindingsKeys, &bindingsKeysUnsorted](auto& x) {
        return AnalysisDataProcessorBuilder::requestCacheFromArgs(x, bindingsKeys, bindingsKeysUnsorted);
      },
      *task.get());

    /// replace origin in slicing caches
    std::ranges::transform(bindingsKeys, bindingsKeys.begin(), [&newOrigin](Entry& entry) {
      if ((entry.matcher.origin == header::DataOrigin{"AOD"}) && (newOrigin != header::DataOrigin{"AOD"})) {
        entry.matcher = replaceOrigin(entry.matcher, newOrigin);
      }
      return entry;
    });
    std::ranges::transform(bindingsKeysUnsorted, bindingsKeysUnsorted.begin(), [&newOrigin](Entry& entry) {
      if ((entry.matcher.origin == header::DataOrigin{"AOD"}) && (newOrigin != header::DataOrigin{"AOD"})) {
        entry.matcher = replaceOrigin(entry.matcher, newOrigin);
      }
      return entry;
    });

    ic.services().get<ArrowTableSlicingCacheDef>().setCaches(std::move(bindingsKeys));
    ic.services().get<ArrowTableSlicingCacheDef>().setCachesUnsorted(std::move(bindingsKeysUnsorted));
    ic.services().get<ArrowTableSlicingCacheDef>().setOrigin(newOrigin);

    return [task, expressionInfos, inputInfos, newOrigin](ProcessingContext& pc) mutable {
      // load the ccdb object from their cache
      homogeneous_apply_refs_sized<numElements>([&pc](auto& element) { return analysis_task_parsers::newDataframeCondition(pc.inputs(), element); }, *task.get());
      // reset partitions once per dataframe
      homogeneous_apply_refs_sized<numElements>([](auto& element) { return analysis_task_parsers::newDataframePartition(element); }, *task.get());
      // reset selections for the next dataframe
      std::ranges::for_each(expressionInfos, [](auto& info) { info.resetSelection = true; });
      // reset pre-slice for the next dataframe
      auto& slices = pc.services().get<ArrowTableSlicingCache>();
      homogeneous_apply_refs_sized<numElements>([&slices](auto& element) {
        return analysis_task_parsers::updateSliceInfo(element, slices);
      },
                                                *(task.get()));
      // initialize local caches
      homogeneous_apply_refs_sized<numElements>([&pc](auto& element) { return analysis_task_parsers::initializeCache(pc, element); }, *(task.get()));
      // prepare outputs
      homogeneous_apply_refs_sized<numElements>([&pc](auto& element) { return analysis_task_parsers::prepareOutput(pc, element); }, *task.get());
      // execute run()
      if constexpr (requires { task->run(pc); }) {
        task->run(pc);
      }
      // execute process()
      if constexpr (requires { &T::process; }) {
        constexpr auto phash = o2::framework::TypeIdHelpers::uniqueId<decltype(&T::process)>();
        auto matchers = std::ranges::find_if(inputInfos, [&phash](auto const& info) { return info.hash == phash; })->matchers;
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), matchers, &T::process, expressionInfos, slices, newOrigin);
      }
      // execute optional process()
      homogeneous_apply_refs_sized<numElements>(
        [&pc, &expressionInfos, &task, &slices, &inputInfos, &newOrigin](auto& x) {
          if constexpr (is_process_configurable<decltype(x)>) {
            if (x.value == true) {
              constexpr auto phash = o2::framework::TypeIdHelpers::uniqueId<decltype(x.process)>();
              auto matchers = std::ranges::find_if(inputInfos, [&phash](auto const& info) { return info.hash == phash; })->matchers;
              AnalysisDataProcessorBuilder::invokeProcess(*task.get(), pc.inputs(), matchers, x.process, expressionInfos, slices, newOrigin);
              return true;
            }
            return false;
          }
          return false;
        },
        *task.get());
      // prepare delayed outputs
      homogeneous_apply_refs_sized<numElements>([&pc](auto& element) { return analysis_task_parsers::prepareDelayedOutput(pc, element); }, *task.get());
      // finalize outputs
      homogeneous_apply_refs_sized<numElements>([&pc](auto& element) { return analysis_task_parsers::finalizeOutput(pc, element); }, *task.get());
    };
  }};

  return {
    name,
    inputs,
    outputs,
    algo,
    options,
    requiredServices};
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
