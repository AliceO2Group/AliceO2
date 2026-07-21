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
#include "Framework/ArrowTypes.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/DeviceSpec.h"
#include "Framework/TimingInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Output.h"
#include "Framework/Signpost.h"
#include "Framework/DanglingEdgesContext.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamsHelper.h"
#include <fairmq/Version.h>
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
    if (std::ranges::none_of(route.matcher.metadata, [](auto const& m) { return m.name.starts_with("ccdb:"); })) {
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
    // The effective default for each ccdb: option was already resolved at topology
    // time by ArrowSupport (consulting task Configurables) and registered on this
    // device's options. Here we just read the final value — honouring any further
    // runtime override supplied via CLI or JSON config.
    std::unordered_map<std::string, std::string> ccdbUrls;
    for (auto& input : dec.analysisCCDBInputs) {
      for (auto& m : input.metadata) {
        if (!m.name.starts_with("ccdb:") || ccdbUrls.count(m.name)) {
          continue;
        }
        std::string url = m.defaultValue.asString();
        if (ConfigParamsHelper::hasOption(spec.options, m.name)) {
          url = options.get<std::string>(m.name.c_str());
        }
        LOGP(info, "CCDB path resolved for {}: {}", m.name, url);
        ccdbUrls.emplace(m.name, std::move(url));
      }
    }
    std::vector<std::shared_ptr<arrow::Schema>> schemas;
    for (auto& input : dec.analysisCCDBInputs) {
      auto schemaMetadata = std::make_shared<arrow::KeyValueMetadata>();
      std::vector<std::shared_ptr<arrow::Field>> fields;
      schemaMetadata->Append("outputRoute", DataSpecUtils::describe(input));
      schemaMetadata->Append("outputBinding", input.binding);
      for (auto& m : input.metadata) {
        if (m.name.starts_with("input:")) {
          auto name = m.name.substr(6);
          schemaMetadata->Append("sourceTable", name);
          schemaMetadata->Append("sourceMatcher", DataSpecUtils::describe(std::get<ConcreteDataMatcher>(DataSpecUtils::fromMetadataString(m.defaultValue.get<std::string>()).matcher)));
          continue;
        }
        if (!m.name.starts_with("ccdb:")) {
          continue;
        }
        auto fieldMetadata = std::make_shared<arrow::KeyValueMetadata>();
        auto it = ccdbUrls.find(m.name);
        fieldMetadata->Append("url", it != ccdbUrls.end() ? it->second : m.defaultValue.asString());
        auto columnName = m.name.substr(strlen("ccdb:"));
#if (FAIRMQ_VERSION_DEC >= 111000)
        fields.emplace_back(std::make_shared<arrow::Field>(columnName, soa::asArrowDataType<int64_t[3]>(), false, fieldMetadata));
#else
        fields.emplace_back(std::make_shared<arrow::Field>(columnName, arrow::binary_view(), false, fieldMetadata));
#endif
      }
      schemas.emplace_back(std::make_shared<arrow::Schema>(fields, schemaMetadata));
    }

#if (FAIRMQ_VERSION_DEC >= 111000)
    std::vector<std::pair<uint32_t, std::shared_ptr<arrow::FixedSizeListBuilder>>> allbuilders;
#else
    std::vector<std::pair<uint32_t, std::shared_ptr<arrow::BinaryViewBuilder>>> allbuilders;
#endif
    allbuilders.resize([&schemas]() { size_t size = 0; for (auto& schema : schemas) { size += schema->num_fields(); }; return size; }());
    auto* pool = arrow::default_memory_pool();

    int idx = 0;
    int sidx = 0;
    for (auto const& schema : schemas) {
      for (auto const& _ : schema->fields()) {
#if (FAIRMQ_VERSION_DEC >= 111000)
        auto value_builder = std::make_shared<arrow::Int64Builder>();
        allbuilders[idx] = std::make_pair(sidx, std::make_shared<arrow::FixedSizeListBuilder>(pool, std::move(value_builder), 3));
#else
        allbuilders[idx] = std::make_pair(sidx, std::make_shared<arrow::BinaryViewBuilder>());
#endif
        ++idx;
      }
      ++sidx;
    }

    std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
    CCDBFetcherHelper::initialiseHelper(*helper, options);
    std::unordered_map<std::string, int> bindings;
    fillValidRoutes(*helper, spec.outputs, bindings);

    return adaptStateless([schemas, bindings, helper, allbuilders](InputRecord& inputs, DataTakingContext& dtc, DataAllocator& allocator, TimingInfo& timingInfo, DataProcessingStats& stats) {
      O2_SIGNPOST_ID_GENERATE(sid, ccdb);
      O2_SIGNPOST_START(ccdb, sid, "fetchFromAnalysisCCDB", "Fetching CCDB objects for analysis%" PRIu64, (uint64_t)timingInfo.timeslice);
      std::ranges::for_each(allbuilders, [](auto& builder) { builder.second->Reset(); });
      for (auto i = 0U; i < schemas.size(); ++i) {
        auto& schema = schemas[i];
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
        auto reserveSize = timestampColumn->length();
        O2_SIGNPOST_EVENT_EMIT_INFO(ccdb, sid, "fetchFromAnalysisCCDB",
                                    "There are %zu bindings available", bindings.size());
        for (auto const& binding : bindings) {
          O2_SIGNPOST_EVENT_EMIT_INFO(ccdb, sid, "fetchFromAnalysisCCDB",
                                      "* %{public}s: %d",
                                      binding.first.c_str(), binding.second);
        }
        int outputRouteIndex = bindings.at(outRouteDesc);
        auto& spec = helper->routes[outputRouteIndex].matcher;
        auto builders = allbuilders | std::views::filter([&i](auto const& builder) { return builder.first == i; });
        unsigned int numBuilders = std::ranges::count_if(allbuilders, [&i](auto const& builder) { return builder.first == i; });
        arrow::Status status;
        std::ranges::for_each(builders, [&status, &reserveSize](auto& builder) {
          if (reserveSize > builder.second->capacity()) {
            status &= builder.second->Reserve(reserveSize - builder.second->capacity());
          }
        });
        if (!status.ok()) {
          throw framework::runtime_error_f("Failed to reserve arrays: ", status.ToString().c_str());
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
            if (numBuilders != responses.size()) {
              LOGP(fatal, "Not enough responses (expected {}, found {})", numBuilders, responses.size());
            }
            arrow::Status result;

            int bi = 0;
            for (auto& builder : builders) {
              auto& response = responses[bi];
#if (FAIRMQ_VERSION_DEC >= 111000)
              result &= builder.second->Append();
              auto* value_builder = dynamic_cast<arrow::Int64Builder*>(builder.second->value_builder());
              result &= value_builder->Append(response.id.handle);
              result &= value_builder->Append(response.id.segment);
              result &= value_builder->Append(response.size);
#else
              char const* address = reinterpret_cast<char const*>(response.id.value);
              result &= builder.second->Append(std::string_view(address, response.size));
#endif
              ++bi;
            }
            if (!result.ok()) {
              LOGP(fatal, "Error adding results from CCDB");
            }
            O2_SIGNPOST_END(ccdb, sid, "handlingResponses", "Done processing responses");
          }
        }
        arrow::ArrayVector arrays;
        std::ranges::for_each(builders, [&arrays](auto& builder) { arrays.push_back(*builder.second->Finish()); });
        auto outTable = arrow::Table::Make(schema, arrays);
        auto concrete = DataSpecUtils::asConcreteDataMatcher(spec);
        allocator.adopt(Output{concrete.origin, concrete.description, concrete.subSpec}, outTable);
      }

      stats.updateStats({(int)ProcessingStatsId::CCDB_CACHE_FETCHED_BYTES, DataProcessingStats::Op::Set, (int64_t)helper->totalFetchedBytes});
      stats.updateStats({(int)ProcessingStatsId::CCDB_CACHE_REQUESTED_BYTES, DataProcessingStats::Op::Set, (int64_t)helper->totalRequestedBytes});
      O2_SIGNPOST_END(ccdb, sid, "fetchFromAnalysisCCDB", "Fetching CCDB objects");
    });
  });
}

} // namespace o2::framework
