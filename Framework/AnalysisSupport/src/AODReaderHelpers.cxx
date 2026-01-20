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
#include "Framework/DataSpecViews.h"
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
  std::vector<framework::ConcreteDataMatcher> matchers;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;
  std::vector<o2::soa::IndexRecord> records;
  std::shared_ptr<arrow::Schema> outputSchema;

  explicit Buildable(InputSpec const& spec)
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
      matchers.emplace_back(r.matcher);
    }
    outputSchema = std::make_shared<arrow::Schema>([](std::vector<o2::soa::IndexRecord> const& recs) {
                     std::vector<std::shared_ptr<arrow::Field>> fields;
                     fields.reserve(recs.size());
                     std::ranges::transform(recs, std::back_inserter(fields), [](auto& r) { return r.field(); });
                     return fields;
                   }(records))
                     ->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{binding}}));
  }

  framework::Builder createBuilder() const
  {
    return {
      exclusive,
      labels,
      matchers,
      records,
      outputSchema,
      origin,
      description,
      version,
      nullptr};
  }
};

} // namespace

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(ConfigContext const& /*ctx*/)
{
  return AlgorithmSpec::InitCallback{[](InitContext& ic) {
    auto const& requested = ic.services().get<DanglingEdgesContext>().requestedIDXs;
    std::vector<Builder> builders;
    builders.reserve(requested.size());
    std::ranges::transform(requested, std::back_inserter(builders), [](auto const& i) { return Buildable{i}.createBuilder(); });
    return [builders](ProcessingContext& pc) mutable {
      auto outputs = pc.outputs();
      std::ranges::for_each(builders, [&pc, &outputs](auto& builder) { outputs.adopt(Output{builder.origin, builder.description, builder.version}, builder.materialize(pc)); });
    };
  }};
}

namespace
{
struct Spawnable {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<framework::ConcreteDataMatcher> matchers;
  std::vector<expressions::Projector> projectors;
  std::vector<std::shared_ptr<gandiva::Expression>> expressions;
  std::shared_ptr<arrow::Schema> outputSchema;
  std::shared_ptr<arrow::Schema> inputSchema;

  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  explicit Spawnable(InputSpec const& spec)
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
    for (auto const& i : spec.metadata | views::filter_string_params_starts_with("input-schema:")) {
      labels.emplace_back(i.name.substr(13));
      iws.clear();
      auto json = i.defaultValue.get<std::string>();
      iws.str(json);
      schemas.emplace_back(ArrowJSONHelpers::read(iws));
    }
    std::ranges::transform(spec.metadata |
                             views::filter_string_params_starts_with("input:") |
                             std::ranges::views::transform(
                               [](auto const& param) {
                                 return DataSpecUtils::fromMetadataString(param.defaultValue.template get<std::string>());
                               }),
                           std::back_inserter(matchers), [](auto const& i) { return std::get<ConcreteDataMatcher>(i.matcher); });

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::ranges::for_each(schemas,
                          [&fields](auto const& s) {
                            std::ranges::copy(s->fields(), std::back_inserter(fields));
                          });

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
      matchers,
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

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(ConfigContext const& /*ctx*/)
{
  return AlgorithmSpec::InitCallback{[](InitContext& ic) {
    auto const& requested = ic.services().get<DanglingEdgesContext>().spawnerInputs;
    std::vector<Spawner> spawners;
    spawners.reserve(requested.size());
    std::ranges::transform(requested, std::back_inserter(spawners), [](auto const& i) { return Spawnable{i}.createMaker(); });
    return [spawners](ProcessingContext& pc) mutable {
      auto outputs = pc.outputs();
      std::ranges::for_each(spawners, [&pc, &outputs](auto& spawner) { outputs.adopt(Output{spawner.origin, spawner.description, spawner.version}, spawner.materialize(pc)); });
    };
  }};
}

} // namespace o2::framework::readers
