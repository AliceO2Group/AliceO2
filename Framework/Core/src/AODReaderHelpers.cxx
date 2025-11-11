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

#include "Framework/AODReaderHelpers.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/DataSpecUtils.h"
#include "ExpressionJSONHelpers.h"
#include "Framework/ConfigContext.h"
#include "Framework/AnalysisContext.h"

#include <Monitoring/Monitoring.h>

#include <TGrid.h>
#include <TFile.h>
#include <TTreeCache.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

namespace o2::framework::readers
{
auto setEOSCallback(InitContext& ic)
{
  ic.services().get<CallbackService>().set<CallbackService::Id::EndOfStream>(
    [](EndOfStreamContext& eosc) {
      auto& control = eosc.services().get<ControlService>();
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    });
}

template <size_t N, std::array<soa::TableRef, N> refs>
static inline auto extractOriginals(ProcessingContext& pc)
{
  return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::vector<std::shared_ptr<arrow::Table>> {
    return {pc.inputs().get<TableConsumer>(o2::aod::label<refs[Is]>())->asArrowTable()...};
  }(std::make_index_sequence<refs.size()>());
}
namespace
{
template <typename D>
  requires(D::exclusive)
auto make_build(D metadata, InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = decltype(metadata);
  using Key = typename metadata_t::Key;
  using index_pack_t = typename metadata_t::index_pack_t;
  constexpr auto sources = metadata_t::sources;
  return o2::framework::IndexBuilder<o2::framework::Exclusive>::indexBuilder<Key, sources.size(), sources>(input.binding.c_str(),
                                                                                                           extractOriginals<sources.size(), sources>(pc),
                                                                                                           index_pack_t{});
}

template <typename D>
  requires(!D::exclusive)
auto make_build(D metadata, InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = decltype(metadata);
  using Key = typename metadata_t::Key;
  using index_pack_t = typename metadata_t::index_pack_t;
  constexpr auto sources = metadata_t::sources;
  return o2::framework::IndexBuilder<o2::framework::Sparse>::indexBuilder<Key, sources.size(), sources>(input.binding.c_str(),
                                                                                                        extractOriginals<sources.size(), sources>(pc),
                                                                                                        index_pack_t{});
}
} // namespace

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(std::vector<InputSpec>& requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& /*ic*/) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto&& [origin, description, version] = DataSpecUtils::asConcreteDataMatcher(input);
        if (description == header::DataDescription{"MA_RN2_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN2_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOL_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOL_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOLS_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsExclusiveMultiMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOLS_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsSparseMultiMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedToBCSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_BC_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedToBCExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN2_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedToBCSparseMetadata{}, input, pc));
        } else {
          throw std::runtime_error("Not an index table");
        }
      }
    };
  }};
}

namespace
{
template <o2::aod::is_aod_hash D>
auto make_spawn(InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = o2::aod::MetadataTrait<D>::metadata;
  constexpr auto sources = metadata_t::sources;
  static std::shared_ptr<gandiva::Projector> projector = nullptr;
  static std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(typename metadata_t::expression_pack_t{}));
  static auto projectors = []<typename... C>(framework::pack<C...>) -> std::array<expressions::Projector, sizeof...(C)>
  {
    return {{std::move(C::Projector())...}};
  }
  (typename metadata_t::expression_pack_t{});
  return o2::framework::spawner<D>(extractOriginals<sources.size(), sources>(pc), input.binding.c_str(), projectors.data(), projector, schema);
}

struct Maker {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<std::shared_ptr<gandiva::Expression>> expressions;
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema;

  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  std::shared_ptr<arrow::Table> make(ProcessingContext& pc)
  {
    std::vector<std::shared_ptr<arrow::Table>> originals;
    for (auto const& label : labels) {
      originals.push_back(pc.inputs().get<TableConsumer>(label)->asArrowTable());
    }
    auto fullTable = soa::ArrowHelpers::joinTables(std::move(originals), std::span{labels.begin(), labels.size()});
    if (projector == nullptr) {
      auto s = gandiva::Projector::Make(
        fullTable->schema(),
        expressions,
        &projector);
      if (!s.ok()) {
        throw o2::framework::runtime_error_f("Failed to create projector: %s", s.ToString().c_str());
      }
    }

    return spawnerHelper(fullTable, schema, binding.c_str(), schema->num_fields(), projector);
  }
};

struct Spawnable {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<expressions::Projector> projectors;
  std::vector<std::shared_ptr<gandiva::Expression>> expressions;
  std::shared_ptr<arrow::Schema> outputSchema;
  std::shared_ptr<arrow::Schema> inputSchema;

  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  Spawnable(InputSpec const& spec)
    : binding{spec.binding}
  {
    auto&& [origin_, description_, version_] = DataSpecUtils::asConcreteDataMatcher(spec);
    origin = origin_;
    description = description_;
    version = version_;
    auto loc = std::find_if(spec.metadata.begin(), spec.metadata.end(), [](ConfigParamSpec const& cps) { return cps.name.compare("projectors") == 0; });
    std::stringstream iws(loc->defaultValue.get<std::string>());
    projectors = ExpressionJSONHelpers::read(iws);

    loc = std::find_if(spec.metadata.begin(), spec.metadata.end(), [](ConfigParamSpec const& cps) { return cps.name.compare("schema") == 0; });
    iws.clear();
    iws.str(loc->defaultValue.get<std::string>());
    outputSchema = ArrowJSONHelpers::read(iws);

    for (auto& i : spec.metadata) {
      if (i.name.starts_with("input:")) {
        labels.emplace_back(i.name.substr(6));
      }
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto& p : projectors) {
      expressions::walk(p.node.get(),
                        [&fields](expressions::Node* n) mutable {
                          if (n->self.index() == 1) {
                            auto& b = std::get<expressions::BindingNode>(n->self);
                            if (std::find_if(fields.begin(), fields.end(), [&b](std::shared_ptr<arrow::Field> const& field) { return field->name() == b.name; }) == fields.end()) {
                              fields.emplace_back(std::make_shared<arrow::Field>(b.name, expressions::concreteArrowType(b.type)));
                            }
                          }
                        });
    }
    inputSchema = std::make_shared<arrow::Schema>(fields);

    int i = 0;
    for (auto& p : projectors) {
      expressions.push_back(
        expressions::makeExpression(
          expressions::createExpressionTree(
            expressions::createOperations(p),
            inputSchema),
          outputSchema->field(i)));
      ++i;
    }
  }

  std::shared_ptr<gandiva::Projector> makeProjector()
  {
    return expressions::createProjectorHelper(projectors.size(), projectors.data(), inputSchema, outputSchema->fields());
  }

  Maker createMaker()
  {
    return {
      binding,
      labels,
      expressions,
      nullptr,
      outputSchema,
      origin,
      description,
      version};
  }
};

} // namespace

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(/*std::vector<InputSpec>& requested*/ ConfigContext const& ctx)
{
  auto& ac = ctx.services().get<AnalysisContext>();
  return AlgorithmSpec::InitCallback{[requested = ac.spawnerInputs](InitContext& /*ic*/) {
    std::vector<Spawnable> spawnables;
    for (auto& i : requested) {
      spawnables.emplace_back(i);
    }
    std::vector<Maker> makers;
    for (auto& s : spawnables) {
      makers.push_back(s.createMaker());
    }

    return [makers](ProcessingContext& pc) mutable {
      auto outputs = pc.outputs();
      for (auto& maker : makers) {
        outputs.adopt(Output{maker.origin, maker.description, maker.version}, maker.make(pc));
      }
    };
  }};
}

} // namespace o2::framework::readers
