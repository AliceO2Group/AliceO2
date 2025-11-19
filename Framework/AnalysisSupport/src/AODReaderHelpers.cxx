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

#include "AODReaderHelpers.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/DataSpecUtils.h"
#include "../src/ExpressionJSONHelpers.h"
#include "../src/IndexJSONHelpers.h"
#include "Framework/ConfigContext.h"
#include "Framework/AnalysisContext.h"

namespace o2::framework::readers
{
namespace
{
template <size_t N, std::array<soa::TableRef, N> refs>
static inline auto extractOriginals(ProcessingContext& pc)
{
  return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::vector<std::shared_ptr<arrow::Table>> {
    return {pc.inputs().get<TableConsumer>(o2::aod::label<refs[Is]>())->asArrowTable()...};
  }(std::make_index_sequence<refs.size()>());
}

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

static inline auto extractSources(ProcessingContext& pc, std::vector<std::string> const& labels)
{
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto const& label : labels) {
    tables.emplace_back(pc.inputs().get<TableConsumer>(label.c_str())->asArrowTable());
  }
  return tables;
}

struct Builder {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<o2::soa::IndexRecord> records;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  std::shared_ptr<arrow::Table> build(ProcessingContext& pc) const
  {
    std::shared_ptr<arrow::Table> result;
    auto tables = extractSources(pc, labels);
    return result;
  }

};

struct Buildable {
  std::string binding;
  std::vector<std::string> labels;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;
  std::vector<o2::soa::IndexRecord> records;

  Buildable(InputSpec const& spec)
    : binding{spec.binding}
  {
    auto&& [origin_, description_, version_] = DataSpecUtils::asConcreteDataMatcher(spec);
    origin = origin_;
    description = description_;
    version = version_;

    auto loc = std::find_if(spec.metadata.begin(), spec.metadata.end(), [](ConfigParamSpec const& cps) { return cps.name.compare("index-records") == 0; });
    std::stringstream iws(loc->defaultValue.get<std::string>());
    records = IndexJSONHelpers::read(iws);

    for (auto const& r : records) {
      labels.emplace_back(r.label);
    }
  }

  Builder createBuilder() const
  {
    return Builder{
      binding,
      labels,
      records,
      origin,
      description,
      version
    };
  }

};

} // namespace

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(ConfigContext const& ctx)
{
  auto& ac = ctx.services().get<AnalysisContext>();
  return AlgorithmSpec::InitCallback{[requested = ac.requestedIDXs](InitContext& /*ic*/) {
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
struct Maker {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<std::shared_ptr<gandiva::Expression>> expressions;
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema = nullptr;
  std::shared_ptr<arrow::Schema> inputSchema = nullptr;

  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  std::shared_ptr<arrow::Table> make(ProcessingContext& pc) const
  {
    std::vector<std::shared_ptr<arrow::Table>> originals;
    for (auto const& label : labels) {
      originals.push_back(pc.inputs().get<TableConsumer>(label)->asArrowTable());
    }
    auto fullTable = soa::ArrowHelpers::joinTables(std::move(originals), std::span{labels.begin(), labels.size()});
    if (fullTable->num_rows() == 0) {
      return arrow::Table::MakeEmpty(schema).ValueOrDie();
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

    std::vector<std::shared_ptr<arrow::Schema>> schemas;
    for (auto& i : spec.metadata) {
      if (i.name.starts_with("input-schema:")) {
        labels.emplace_back(i.name.substr(13));
        iws.clear();
        auto json = i.defaultValue.get<std::string>();
        iws.str(json);
        schemas.emplace_back(ArrowJSONHelpers::read(iws));
      }
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto& s : schemas) {
      std::copy(s->fields().begin(), s->fields().end(), std::back_inserter(fields));
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

  std::shared_ptr<gandiva::Projector> makeProjector() const
  {
    std::shared_ptr<gandiva::Projector> p = nullptr;
    auto s = gandiva::Projector::Make(
      inputSchema,
      expressions,
      &p);
    if (!s.ok()) {
      throw o2::framework::runtime_error_f("Failed to create projector: %s", s.ToString().c_str());
    }
    return p;
  }

  Maker createMaker() const
  {
    o2::framework::addLabelToSchema(outputSchema, binding.c_str());
    return {
      binding,
      labels,
      expressions,
      makeProjector(),
      outputSchema,
      inputSchema,
      origin,
      description,
      version};
  }
};

} // namespace

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(ConfigContext const& ctx)
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
