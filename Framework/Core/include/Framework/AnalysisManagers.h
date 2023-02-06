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

#ifndef FRAMEWORK_ANALYSISMANAGERS_H
#define FRAMEWORK_ANALYSISMANAGERS_H
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/GroupedCombinations.h"
#include "Framework/ASoA.h"
#include "Framework/ProcessingContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigurableHelpers.h"
#include "Framework/Condition.h"
#include "Framework/InitContext.h"
#include "Framework/ConfigContext.h"
#include "Framework/RootConfigParamHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/CommonServices.h"
#include "Framework/Plugins.h"
#include "Framework/RootMessageContext.h"

namespace o2::framework
{

template <typename ANY>
struct GroupedCombinationManager {
  template <typename TG, typename... T2s>
  static void setGroupedCombination(ANY&, TG&, T2s&...)
  {
  }
};

template <typename T1, typename GroupingPolicy, typename BP, typename G, typename... As>
struct GroupedCombinationManager<GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, As...>> {
  template <typename TG, typename... T2s>
  static void setGroupedCombination(GroupedCombinationsGenerator<T1, GroupingPolicy, BP, G, As...>& comb, TG& grouping, std::tuple<T2s...>& associated)
  {
    static_assert(sizeof...(T2s) > 0, "There must be associated tables in process() for a correct pair");
    if constexpr (std::is_same_v<G, TG>) {
      static_assert(std::conjunction_v<framework::has_type<As, pack<T2s...>>...>, "You didn't subscribed to all tables requested for mixing");
      comb.setTables(grouping, associated);
    }
  }
};

template <typename ANY>
struct PartitionManager {
  template <typename... T2s>
  static void setPartition(ANY&, T2s&...)
  {
  }

  template <typename... Ts>
  static void bindExternalIndices(ANY&, Ts*...)
  {
  }

  template <typename E>
  static void bindInternalIndices(ANY&, E*)
  {
  }

  template <typename... Ts>
  static void getBoundToExternalIndices(ANY&, Ts&...)
  {
  }

  static void updatePlaceholders(ANY&, InitContext&)
  {
  }

  static bool newDataframe(ANY&)
  {
    return false;
  }
};

template <typename T>
struct PartitionManager<Partition<T>> {
  template <typename T2>
  static void doSetPartition(Partition<T>& partition, T2& table)
  {
    if constexpr (std::is_same_v<T, T2>) {
      partition.setTable(table);
    }
  }

  template <typename... T2s>
  static void setPartition(Partition<T>& partition, T2s&... tables)
  {
    (doSetPartition(partition, tables), ...);
  }

  template <typename... Ts>
  static void bindExternalIndices(Partition<T>& partition, Ts*... tables)
  {
    partition.bindExternalIndices(tables...);
  }

  template <typename E>
  static void bindInternalIndices(Partition<T>& partition, E* table)
  {
    if constexpr (o2::soa::is_binding_compatible_v<T, std::decay_t<E>>()) {
      partition.bindInternalIndicesTo(table);
    }
  }

  static void updatePlaceholders(Partition<T>& partition, InitContext& context)
  {
    partition.updatePlaceholders(context);
  }

  static bool newDataframe(Partition<T>& partition)
  {
    partition.dataframeChanged = true;
    return true;
  }
};

template <typename ANY>
struct FilterManager {
  static bool createExpressionTrees(ANY&, std::vector<ExpressionInfo>&)
  {
    return false;
  }

  static bool updatePlaceholders(ANY&, InitContext&)
  {
    return false;
  }
};

template <>
struct FilterManager<expressions::Filter> {
  static bool createExpressionTrees(expressions::Filter const& filter, std::vector<ExpressionInfo>& expressionInfos)
  {
    expressions::updateExpressionInfos(filter, expressionInfos);
    return true;
  }

  static bool updatePlaceholders(expressions::Filter& filter, InitContext& ctx)
  {
    expressions::updatePlaceholders(filter, ctx);
    return true;
  }
};

/// A manager which takes care of condition objects
template <typename T>
struct ConditionManager {
  template <typename ANY>
  static bool appendCondition(std::vector<InputSpec>& inputs, ANY& x)
  {
    if constexpr (std::is_base_of_v<ConditionGroup, ANY>) {
      homogeneous_apply_refs<true>([&inputs](auto& y) { return ConditionManager<std::decay_t<decltype(y)>>::appendCondition(inputs, y); }, x);
      return true;
    } else {
      return false;
    }
  }

  template <typename ANY>
  static bool newDataframe(InputRecord& record, ANY& x)
  {
    if constexpr (std::is_base_of_v<ConfigurableGroup, ANY>) {
      homogeneous_apply_refs<true>([&record](auto&& y) { return ConditionManager<std::decay_t<decltype(y)>>::newDataframe(record, y); }, x);
      return true;
    } else {
      return false;
    }
  }
};

template <typename OBJ>
struct ConditionManager<Condition<OBJ>> {
  static bool appendCondition(std::vector<InputSpec>& inputs, Condition<OBJ>& what)
  {
    inputs.emplace_back(InputSpec{what.path, "AODC", compile_time_hash(what.path.c_str()), Lifetime::Condition, ccdbParamSpec(what.path)});
    return true;
  }
  static bool newDataframe(InputRecord& inputs, Condition<OBJ>& what)
  {
    what.instance = (OBJ*)inputs.get<OBJ*>(what.path).get();
    return true;
  }
};

/// SFINAE placeholder
template <typename T>
struct OutputManager {
  template <typename ANY>
  static bool appendOutput(std::vector<OutputSpec>&, ANY&, uint32_t)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(ProcessingContext&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool postRun(EndOfStreamContext&, ANY&)
  {
    return true;
  }

  template <typename ANY>
  static bool finalize(ProcessingContext&, ANY&)
  {
    return true;
  }
};

/// Produces specialization
template <typename TABLE>
struct OutputManager<Produces<TABLE>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Produces<TABLE>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext& context, Produces<TABLE>& what)
  {
    what.resetCursor(context.outputs().make<TableBuilder>(what.ref()));
    return true;
  }
  static bool finalize(ProcessingContext&, Produces<TABLE>& what)
  {
    what.setLabel(o2::aod::MetadataTrait<TABLE>::metadata::tableLabel());
    return true;
  }
  static bool postRun(EndOfStreamContext&, Produces<TABLE>&)
  {
    return true;
  }
};

/// HistogramRegistry specialization
template <>
struct OutputManager<HistogramRegistry> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, HistogramRegistry& what, uint32_t hash)
  {
    what.setHash(hash);
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& context, HistogramRegistry& what)
  {
    context.outputs().snapshot(what.ref(), *(*what));
    return true;
  }
};

/// OutputObj specialization
template <typename T>
struct OutputManager<OutputObj<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, OutputObj<T>& what, uint32_t hash)
  {
    what.setHash(hash);
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& context, OutputObj<T>& what)
  {
    context.outputs().snapshot(what.ref(), *what);
    return true;
  }
};

/// Spawns specializations
template <typename O>
static inline auto extractOriginal(ProcessingContext& pc)
{
  return pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable();
}

template <typename... Os>
static inline std::vector<std::shared_ptr<arrow::Table>> extractOriginals(framework::pack<Os...>, ProcessingContext& pc)
{
  return {extractOriginal<Os>(pc)...};
}

template <typename T>
struct OutputManager<Spawns<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Spawns<T>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }

  static bool prepare(ProcessingContext& pc, Spawns<T>& what)
  {
    auto originalTable = soa::ArrowHelpers::joinTables(extractOriginals(what.sources_pack(), pc));
    if (originalTable->schema()->fields().empty() == true) {
      using base_table_t = typename Spawns<T>::base_table_t;
      originalTable = makeEmptyTable<base_table_t>(aod::MetadataTrait<typename Spawns<T>::extension_t>::metadata::tableLabel());
    }

    what.extension = std::make_shared<typename Spawns<T>::extension_t>(o2::framework::spawner(what.pack(), extractOriginals(what.sources_pack(), pc), aod::MetadataTrait<typename Spawns<T>::extension_t>::metadata::tableLabel()));
    what.table = std::make_shared<typename T::table_t>(soa::ArrowHelpers::joinTables({what.extension->asArrowTable(), originalTable}));
    return true;
  }

  static bool finalize(ProcessingContext& pc, Spawns<T>& what)
  {
    pc.outputs().adopt(what.output(), what.asArrowTable());
    return true;
  }

  static bool postRun(EndOfStreamContext&, Spawns<T>&)
  {
    return true;
  }
};

/// Builds specialization
template <typename... Ts>
static inline auto doExtractOriginal(framework::pack<Ts...>, ProcessingContext& pc)
{
  if constexpr (sizeof...(Ts) == 1) {
    return pc.inputs().get<TableConsumer>(aod::MetadataTrait<framework::pack_element_t<0, framework::pack<Ts...>>>::metadata::tableLabel())->asArrowTable();
  } else {
    return std::vector{pc.inputs().get<TableConsumer>(aod::MetadataTrait<Ts>::metadata::tableLabel())->asArrowTable()...};
  }
}

template <typename O>
static inline auto extractOriginalJoined(ProcessingContext& pc)
{
  return o2::soa::ArrowHelpers::joinTables({doExtractOriginal(soa::make_originals_from_type<O>(), pc)});
}

template <typename... Os>
static inline auto extractOriginalsVector(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::vector{extractOriginalJoined<Os>(pc)...};
}

template <typename T>
struct OutputManager<Builds<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Builds<T>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }

  static bool prepare(ProcessingContext& pc, Builds<T>& what)
  {
    return what.template build<typename T::indexing_t>(what.pack(), what.originals_pack(),
                                                       extractOriginalsVector(what.originals_pack(), pc));
  }

  static bool finalize(ProcessingContext& pc, Builds<T>& what)
  {
    pc.outputs().adopt(what.output(), what.asArrowTable());
    return true;
  }

  static bool postRun(EndOfStreamContext&, Builds<T>&)
  {
    return true;
  }
};

template <typename T>
class has_instance
{
  using one = char;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::instance));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
class has_end_of_stream
{
  using one = char;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::endOfStream));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
inline constexpr bool has_end_of_stream_v = has_end_of_stream<T>::value;

template <typename T>
struct ServiceManager {
  template <typename ANY>
  static bool add(std::vector<ServiceSpec>& specs, ANY& any)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool postRun(EndOfStreamContext&, ANY&)
  {
    return false;
  }
};

template <typename T>
struct ServiceManager<Service<T>> {
  static bool add(std::vector<ServiceSpec>& specs, Service<T>& service)
  {
    if constexpr (o2::framework::is_base_of_template_v<LoadableServicePlugin, T>) {
      T p = T{};
      auto loadableServices = ServiceHelpers::parseServiceSpecString(p.loadSpec.c_str());
      ServiceHelpers::loadFromPlugin(loadableServices, specs);
    }
    return true;
  }

  static bool prepare(InitContext& context, Service<T>& service)
  {
    if constexpr (has_instance<T>::value) {
      service.service = &(T::instance()); // Sigh...
      return true;
    } else {
      service.service = &(context.services().get<T>());
      return true;
    }
    return false;
  }

  /// If a service has a method endOfStream, it is called at the end of the stream.
  static bool postRun(EndOfStreamContext& context, Service<T>& service)
  {
    // FIXME: for the moment we only need endOfStream to be
    // stateless. In the future we might want to pass it EndOfStreamContext
    if constexpr (has_end_of_stream_v<T>) {
      service.service->endOfStream();
      return true;
    }
    return false;
  }
};

template <typename T>
struct OptionManager {
  template <typename ANY>
  static bool appendOption(std::vector<ConfigParamSpec>& options, ANY& x)
  {
    /// Recurse, in case we are brace constructible
    if constexpr (std::is_base_of_v<ConfigurableGroup, ANY>) {
      homogeneous_apply_refs<true>([&options](auto& y) { return OptionManager<std::decay_t<decltype(y)>>::appendOption(options, y); }, x);
      return true;
    } else {
      return false;
    }
  }

  template <typename ANY>
  static bool prepare(InitContext& ic, ANY& x)
  {
    if constexpr (std::is_base_of_v<ConfigurableGroup, ANY>) {
      homogeneous_apply_refs<true>([&ic](auto&& y) { return OptionManager<std::decay_t<decltype(y)>>::prepare(ic, y); }, x);
      return true;
    } else {
      return false;
    }
  }
};

template <typename T, ConfigParamKind K, typename IP>
struct OptionManager<Configurable<T, K, IP>> {
  static bool appendOption(std::vector<ConfigParamSpec>& options, Configurable<T, K, IP>& what)
  {
    return ConfigurableHelpers::appendOption(options, what);
  }

  static bool prepare(InitContext& context, Configurable<T, K, IP>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      what.value = context.options().get<T>(what.name.c_str());
    } else {
      auto pt = context.options().get<boost::property_tree::ptree>(what.name.c_str());
      what.value = RootConfigParamHelpers::as<T>(pt);
    }
    return true;
  }
};

template <typename R, typename T, typename... As>
struct OptionManager<ProcessConfigurable<R, T, As...>> {
  static bool appendOption(std::vector<ConfigParamSpec>& options, ProcessConfigurable<R, T, As...>& what)
  {
    options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<std::decay_t<bool>>, what.value, {what.help}, what.kind});
    return true;
  }

  static bool prepare(InitContext& context, ProcessConfigurable<R, T, As...>& what)
  {
    what.value = context.options().get<bool>(what.name.c_str());
    return true;
  }
};

template <typename ANY>
struct UpdateProcessSwitches {
  static bool set(std::pair<std::string, bool>, ANY&)
  {
    return false;
  }
};

template <typename R, typename T, typename... As>
struct UpdateProcessSwitches<ProcessConfigurable<R, T, As...>> {
  static bool set(std::pair<std::string, bool> setting, ProcessConfigurable<R, T, As...>& what)
  {
    if (what.name == setting.first) {
      what.value = setting.second;
      return true;
    }
    return false;
  }
};

/// Manager template to facilitate extended tables spawning
template <typename T>
struct SpawnManager {
  static bool requestInputs(std::vector<InputSpec>&, T const&) { return false; }
};

template <typename TABLE>
struct SpawnManager<Spawns<TABLE>> {
  static bool requestInputs(std::vector<InputSpec>& inputs, Spawns<TABLE>& spawns)
  {
    auto base_specs = spawns.base_specs();
    for (auto base_spec : base_specs) {
      base_spec.metadata.push_back(ConfigParamSpec{std::string{"control:spawn"}, VariantType::Bool, true, {"\"\""}});
      DataSpecUtils::updateInputList(inputs, std::forward<InputSpec>(base_spec));
    }
    return true;
  }
};

/// Manager template for building index tables
template <typename T>
struct IndexManager {
  static bool requestInputs(std::vector<InputSpec>&, T const&) { return false; };
};

template <typename IDX>
struct IndexManager<Builds<IDX>> {
  static bool requestInputs(std::vector<InputSpec>& inputs, Builds<IDX>& builds)
  {
    auto base_specs = builds.base_specs();
    for (auto base_spec : base_specs) {
      base_spec.metadata.push_back(ConfigParamSpec{std::string{"control:build"}, VariantType::Bool, true, {"\"\""}});
      DataSpecUtils::updateInputList(inputs, std::forward<InputSpec>(base_spec));
    }
    return true;
  }
};

/// Manager template to handle slice caching
template <typename T>
struct PresliceManager {
  template <typename T1>
  static bool processTable(T&, T1&)
  {
    return false;
  }

  static bool setNewDF(T&) { return false; };
};

template <typename T>
struct PresliceManager<Preslice<T>> {
  template <typename T1>
  static bool processTable(Preslice<T>& container, T1& table)
  {
    if constexpr (o2::soa::is_binding_compatible_v<T, std::decay_t<T1>>()) {
      return container.processTable(table.asArrowTable()).ok();
    } else {
      return false;
    }
  }

  static bool setNewDF(Preslice<T>& container)
  {
    container.setNewDF();
    return true;
  }
};
} // namespace o2::framework

#endif // ANALYSISMANAGERS_H
