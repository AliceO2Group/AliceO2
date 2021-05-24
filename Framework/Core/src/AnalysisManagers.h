// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_ANALYSISMANAGERS_H
#define FRAMEWORK_ANALYSISMANAGERS_H
#include "Framework/AnalysisHelpers.h"
#include "Framework/Kernels.h"
#include "Framework/ASoA.h"
#include "Framework/ProcessingContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InitContext.h"
#include "Framework/RootConfigParamHelpers.h"
#include "../src/ExpressionHelpers.h"

namespace o2::framework
{

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

  template <typename... Ts>
  static void getBoundToExternalIndices(ANY&, Ts&...)
  {
  }

  static void updatePlaceholders(ANY&, InitContext&)
  {
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

  template <typename... Ts>
  static void getBoundToExternalIndices(Partition<T>& partition, Ts&... tables)
  {
    partition.getBoundToExternalIndices(tables...);
  }

  static void updatePlaceholders(Partition<T>& partition, InitContext& context)
  {
    partition.updatePlaceholders(context);
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
  static bool finalize(ProcessingContext&, Produces<TABLE>&)
  {
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
static auto extractOriginal(ProcessingContext& pc)
{
  return pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable();
}

template <typename... Os>
static std::vector<std::shared_ptr<arrow::Table>> extractOriginals(framework::pack<Os...>, ProcessingContext& pc)
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
    auto original_table = soa::ArrowHelpers::joinTables(extractOriginals(what.sources_pack(), pc));
    if (original_table->schema()->fields().empty() == true) {
      using base_table_t = typename Spawns<T>::base_table_t;
      original_table = makeEmptyTable<base_table_t>();
    }

    what.extension = std::make_shared<typename Spawns<T>::extension_t>(o2::framework::spawner(what.pack(), original_table.get(), aod::MetadataTrait<typename Spawns<T>::extension_t>::metadata::tableLabel()));
    what.table = std::make_shared<typename T::table_t>(soa::ArrowHelpers::joinTables({what.extension->asArrowTable(), original_table}));
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
template <typename... Os>
static inline auto extractOriginalsVector(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::vector{extractOriginal<Os>(pc)...};
}

template <typename O>
static inline auto extractTypedOriginal(ProcessingContext& pc)
{
  if constexpr (soa::is_type_with_originals_v<O>) {
    return O{extractOriginalsVector(soa::originals_pack_t<O>{}, pc)};
  } else {
    return O{pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable()};
  }
}

template <typename... Os>
static inline auto extractOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

template <typename T, typename P>
struct OutputManager<Builds<T, P>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Builds<T, P>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }

  static bool prepare(ProcessingContext& pc, Builds<T, P>& what)
  {
    return what.build(what.pack(),
                      extractTypedOriginal<typename Builds<T, P>::Key>(pc),
                      extractOriginalsTuple(what.originals_pack(), pc));
  }

  static bool finalize(ProcessingContext& pc, Builds<T, P>& what)
  {
    pc.outputs().adopt(what.output(), what.asArrowTable());
    return true;
  }

  static bool postRun(EndOfStreamContext&, Builds<T, P>&)
  {
    return true;
  }
};

template <typename T>
class has_instance
{
  typedef char one;
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
struct ServiceManager {
  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }
};

template <typename T>
struct ServiceManager<Service<T>> {
  static bool prepare(InitContext& context, Service<T>& service)
  {
    if constexpr (has_instance<T>::value) {
      service.service = &(T::instance()); // Sigh...
      return true;
    } else {
      service.service = context.services().get<T>();
      return true;
    }
    return false;
  }
};

template <typename T>
struct OptionManager {
  template <typename ANY>
  static bool appendOption(std::vector<ConfigParamSpec>&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }
};

template <typename T, ConfigParamKind K, typename IP>
struct OptionManager<Configurable<T, K, IP>> {
  static bool appendOption(std::vector<ConfigParamSpec>& options, Configurable<T, K, IP>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<std::decay_t<T>>, what.value, {what.help}, what.kind});
    } else {
      auto specs = RootConfigParamHelpers::asConfigParamSpecs<T>(what.name, what.value);
      options.insert(options.end(), specs.begin(), specs.end());
    }
    return true;
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
    for (auto& base_spec : base_specs) {
      if (std::find_if(inputs.begin(), inputs.end(), [&](InputSpec const& spec) { return base_spec.binding == spec.binding; }) == inputs.end()) {
        inputs.emplace_back(base_spec);
      }
    }
    return true;
  }
};

/// Manager template for building index tables
template <typename T>
struct IndexManager {
  static bool requestInputs(std::vector<InputSpec>&, T const&) { return false; };
};

template <typename IDX, typename P>
struct IndexManager<Builds<IDX, P>> {
  static bool requestInputs(std::vector<InputSpec>& inputs, Builds<IDX, P>& builds)
  {
    auto base_specs = builds.base_specs();
    for (auto& base_spec : base_specs) {
      if (std::find_if(inputs.begin(), inputs.end(), [&](InputSpec const& spec) { return base_spec.binding == spec.binding; }) == inputs.end()) {
        inputs.emplace_back(base_spec);
      }
    }
    return true;
  }
};
} // namespace o2::framework

#endif // ANALYSISMANAGERS_H
