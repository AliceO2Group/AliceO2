// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AnalysisCCDBHelpers.h"
#include "CCDBFetcherHelper.h"
#include "Framework/DeviceSpec.h"
#include "Framework/TimingInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Output.h"
#include "Framework/Signpost.h"
#include "Framework/DanglingEdgesContext.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigContext.h"
#include <arrow/array/builder_binary.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <fmt/base.h>
#include <ctime>
#include <memory>
#include <unordered_map>

O2_DECLARE_DYNAMIC_LOG(ccdb);

namespace o2::framework
{
// Fill valid routes. Notice that for analysis the timestamps are associated to
// a ATIM table and there might be multiple CCDB objects of the same kind for
// dataframe.
// For this reason rather than matching the Lifetime::Condition, we match the
// origin.
namespace
{
void fillValidRoutes(CCDBFetcherHelper& helper, std::vector<o2::framework::OutputRoute> const& outputRoutes, std::unordered_map<std::string, int>& bindings)
{
  for (auto& route : outputRoutes) {
    auto originMatcher = DataSpecUtils::asConcreteDataMatcher(route.matcher);
    if (originMatcher.origin != header::DataOrigin{"ATIM"}) {
      continue;
    }
    auto specStr = DataSpecUtils::describe(route.matcher);
    if (bindings.find(specStr) != bindings.end()) {
      continue;
    }
    bindings[specStr] = helper.routes.size();
    helper.routes.push_back(route);
    LOGP(info, "The following route needs condition objects {} ", DataSpecUtils::describe(route.matcher));
    for (auto& metadata : route.matcher.metadata) {
      if (metadata.type == VariantType::String) {
        LOGP(info, "- {}: {}", metadata.name, metadata.defaultValue.asString());
      }
    }
  }
}
} // namespace

AlgorithmSpec AnalysisCCDBHelpers::fetchFromCCDB(ConfigContext const& /*ctx*/)
{
  return adaptStateful([](ConfigParamRegistry const& options, DeviceSpec const& spec, InitContext& ic) {
    auto& dec = ic.services().get<DanglingEdgesContext>();
    std::vector<std::shared_ptr<arrow::Schema>> schemas;
    auto schemaMetadata = std::make_shared<arrow::KeyValueMetadata>();

    for (auto& input : dec.analysisCCDBInputs) {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      schemaMetadata->Append("outputRoute", DataSpecUtils::describe(input));
      schemaMetadata->Append("outputBinding", input.binding);

      for (auto& m : input.metadata) {
        // Save the list of input tables
        if (m.name.starts_with("input:")) {
          auto name = m.name.substr(6);
          schemaMetadata->Append("sourceTable", name);
          schemaMetadata->Append("sourceMatcher", DataSpecUtils::describe(std::get<ConcreteDataMatcher>(DataSpecUtils::fromMetadataString(m.defaultValue.get<std::string>()).matcher)));
          continue;
        }
        // Ignore the non ccdb: entries
        if (!m.name.starts_with("ccdb:")) {
          continue;
        }
        // Create the schema of the output
        auto metadata = std::make_shared<arrow::KeyValueMetadata>();
        metadata->Append("url", m.defaultValue.asString());
        auto columnName = m.name.substr(strlen("ccdb:"));
        fields.emplace_back(std::make_shared<arrow::Field>(columnName, arrow::binary_view(), false, metadata));
      }
      schemas.emplace_back(std::make_shared<arrow::Schema>(fields, schemaMetadata));
    }

    std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
    CCDBFetcherHelper::initialiseHelper(*helper, options);
    std::unordered_map<std::string, int> bindings;
    fillValidRoutes(*helper, spec.outputs, bindings);

    return adaptStateless([schemas, bindings, helper](InputRecord& inputs, DataTakingContext& dtc, DataAllocator& allocator, TimingInfo& timingInfo) {
      O2_SIGNPOST_ID_GENERATE(sid, ccdb);
      O2_SIGNPOST_START(ccdb, sid, "fetchFromAnalysisCCDB", "Fetching CCDB objects for analysis%" PRIu64, (uint64_t)timingInfo.timeslice);
      for (auto& schema : schemas) {
        std::vector<CCDBFetcherHelper::FetchOp> ops;
        auto inputBinding = *schema->metadata()->Get("sourceTable");
        auto inputMatcher = DataSpecUtils::fromString(*schema->metadata()->Get("sourceMatcher"));
        auto outRouteDesc = *schema->metadata()->Get("outputRoute");
        std::string outBinding = *schema->metadata()->Get("outputBinding");
        O2_SIGNPOST_EVENT_EMIT_INFO(ccdb, sid, "fetchFromAnalysisCCDB",
                                    "Fetching CCDB objects for %{public}s's columns with timestamps from %{public}s and putting them in route %{public}s",
                                    outBinding.c_str(), inputBinding.c_str(), outRouteDesc.c_str());
        auto table = inputs.get<TableConsumer>(inputMatcher)->asArrowTable();
        // FIXME: make the fTimestamp column configurable.
        auto timestampColumn = table->GetColumnByName("fTimestamp");
        O2_SIGNPOST_EVENT_EMIT_INFO(ccdb, sid, "fetchFromAnalysisCCDB",
                                    "There are %zu bindings available", bindings.size());
        for (auto& binding : bindings) {
          O2_SIGNPOST_EVENT_EMIT_INFO(ccdb, sid, "fetchFromAnalysisCCDB",
                                      "* %{public}s: %d",
                                      binding.first.c_str(), binding.second);
        }
        int outputRouteIndex = bindings.at(outRouteDesc);
        auto& spec = helper->routes[outputRouteIndex].matcher;
        std::vector<std::shared_ptr<arrow::BinaryViewBuilder>> builders;
        for (auto const& _ : schema->fields()) {
          builders.emplace_back(std::make_shared<arrow::BinaryViewBuilder>());
        }

        for (auto ci = 0; ci < timestampColumn->num_chunks(); ++ci) {
          std::shared_ptr<arrow::Array> chunk = timestampColumn->chunk(ci);
          auto const* timestamps = chunk->data()->GetValuesSafe<size_t>(1);

          for (int64_t ri = 0; ri < chunk->data()->length; ri++) {
            ops.clear();
            int64_t timestamp = timestamps[ri];
            for (auto& field : schema->fields()) {
              auto url = *field->metadata()->Get("url");
              // Time to actually populate the blob
              ops.push_back({
                .spec = spec,
                .url = url,
                .timestamp = timestamp,
                .runNumber = 1,
                .runDependent = 0,
                .queryRate = 0,
              });
            }
            auto responses = CCDBFetcherHelper::populateCacheWith(helper, ops, timingInfo, dtc, allocator);
            O2_SIGNPOST_START(ccdb, sid, "handlingResponses",
                              "Got %zu responses from server.",
                              responses.size());
            if (builders.size() != responses.size()) {
              LOGP(fatal, "Not enough responses (expected {}, found {})", builders.size(), responses.size());
            }
            arrow::Status result;
            for (size_t bi = 0; bi < responses.size(); bi++) {
              auto& builder = builders[bi];
              auto& response = responses[bi];
              char const* address = reinterpret_cast<char const*>(response.id.value);
              result &= builder->Append(std::string_view(address, response.size));
            }
            if (!result.ok()) {
              LOGP(fatal, "Error adding results from CCDB");
            }
            O2_SIGNPOST_END(ccdb, sid, "handlingResponses", "Done processing responses");
          }
        }
        arrow::ArrayVector arrays;
        for (auto& builder : builders) {
          arrays.push_back(*builder->Finish());
        }
        auto outTable = arrow::Table::Make(schema, arrays);
        auto concrete = DataSpecUtils::asConcreteDataMatcher(spec);
        allocator.adopt(Output{concrete.origin, concrete.description, concrete.subSpec}, outTable);
      }

      O2_SIGNPOST_END(ccdb, sid, "fetchFromAnalysisCCDB", "Fetching CCDB objects");
    });
  });
}

} // namespace o2::framework
