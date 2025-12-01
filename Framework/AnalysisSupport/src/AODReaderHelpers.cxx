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
#include "../src/ExpressionJSONHelpers.h"
#include "../src/IndexJSONHelpers.h"

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ConfigContext.h"
#include "Framework/DanglingEdgesContext.h"

namespace o2::framework::readers
{
namespace
{
struct Buildable {
  bool exclusive = false;
  std::string binding;
  std::vector<std::string> labels;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;
  std::vector<o2::soa::IndexRecord> records;
  std::shared_ptr<arrow::Schema> outputSchema;

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

    loc = std::find_if(spec.metadata.begin(), spec.metadata.end(), [](ConfigParamSpec const& cps) { return cps.name.compare("index-exclusive") == 0; });
    exclusive = loc->defaultValue.get<bool>();

    for (auto const& r : records) {
      labels.emplace_back(r.label);
    }
    outputSchema = std::make_shared<arrow::Schema>([](std::vector<o2::soa::IndexRecord> const& recs) {
                     std::vector<std::shared_ptr<arrow::Field>> fields;
                     for (auto& r : recs) {
                       fields.push_back(r.field());
                     }
                     return fields;
                   }(records))
                     ->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{binding}}));
  }

  framework::Builder createBuilder() const
  {
    return {
      exclusive,
      labels,
      records,
      outputSchema,
      origin,
      description,
      version,
      nullptr};
  }
};

} // namespace

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(ConfigContext const& ctx)
{
  auto& ac = ctx.services().get<DanglingEdgesContext>();
  return AlgorithmSpec::InitCallback{[requested = ac.requestedIDXs](InitContext& /*ic*/) {
    std::vector<Buildable> buildables;
    for (auto& i : requested) {
      buildables.emplace_back(i);
    }
    std::vector<Builder> builders;
    for (auto& b : buildables) {
      builders.push_back(b.createBuilder());
    }
    return [builders](ProcessingContext& pc) mutable {
      auto outputs = pc.outputs();
      for (auto& builder : builders) {
        outputs.adopt(Output{builder.origin, builder.description, builder.version}, builder.materialize(pc));
      }
    };
  }};
}

namespace
{
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
    o2::framework::addLabelToSchema(outputSchema, binding.c_str());

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
    expressions = expressions::materializeProjectors(projectors, inputSchema, outputSchema->fields());
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

  framework::Spawner createMaker() const
  {
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
  auto& ac = ctx.services().get<DanglingEdgesContext>();
  return AlgorithmSpec::InitCallback{[requested = ac.spawnerInputs](InitContext& /*ic*/) {
    std::vector<Spawnable> spawnables;
    for (auto& i : requested) {
      spawnables.emplace_back(i);
    }
    std::vector<Spawner> spawners;
    for (auto& s : spawnables) {
      spawners.push_back(s.createMaker());
    }

    return [spawners](ProcessingContext& pc) mutable {
      auto outputs = pc.outputs();
      for (auto& spawner : spawners) {
        outputs.adopt(Output{spawner.origin, spawner.description, spawner.version}, spawner.materialize(pc));
      }
    };
  }};
}

} // namespace o2::framework::readers
