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
#include "Framework/AnalysisHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "ExpressionJSONHelpers.h"
#include "IndexJSONHelpers.h"

namespace o2::soa
{
std::vector<framework::IndexColumnBuilder> IndexBuilder::makeBuilders(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::vector<soa::IndexRecord> const& records)
{
  std::vector<framework::IndexColumnBuilder> builders;
  builders.reserve(records.size());
  auto pool = arrow::default_memory_pool();
  builders.emplace_back(IndexKind::IdxSelf, records[0].pos, pool);
  if (records[0].pos >= 0) {
    std::get<framework::SelfBuilder>(builders[0].builder).keyIndex = std::make_unique<framework::ChunkedArrayIterator>(tables[0]->column(records[0].pos));
  }

  for (auto i = 1U; i < records.size(); ++i) {
    builders.emplace_back(records[i].kind, records[i].pos, pool, records[i].pos >= 0 ? tables[i]->column(records[i].pos) : nullptr);
  }

  return builders;
}

void IndexBuilder::resetBuilders(std::vector<framework::IndexColumnBuilder>& builders, std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  for (auto i = 0U; i < builders.size(); ++i) {
    builders[i].reset(builders[i].mColumnPos >= 0 ? tables[i]->column(builders[i].mColumnPos) : nullptr);
  }

  if (builders[0].mColumnPos >= 0) {
    std::get<framework::SelfBuilder>(builders[0].builder).keyIndex = std::make_unique<framework::ChunkedArrayIterator>(tables[0]->column(builders[0].mColumnPos));
  }
}

std::shared_ptr<arrow::Table> IndexBuilder::materialize(std::vector<framework::IndexColumnBuilder>& builders, std::vector<std::shared_ptr<arrow::Table>>&& tables, std::vector<soa::IndexRecord> const& records, std::shared_ptr<arrow::Schema> const& schema, bool exclusive)
{
  auto size = tables[0]->num_rows();
  if (O2_BUILTIN_UNLIKELY(builders.empty())) {
    builders = makeBuilders(std::move(tables), records);
  } else {
    resetBuilders(builders, std::move(tables));
  }

  for (int64_t counter = 0; counter < size; ++counter) {
    int64_t idx = -1;
    if (std::get<framework::SelfBuilder>(builders[0].builder).keyIndex == nullptr) {
      idx = counter;
    } else {
      idx = std::get<framework::SelfBuilder>(builders[0].builder).keyIndex->valueAt(counter);
    }

    bool found = true;
    std::ranges::for_each(builders, [&idx, &found](auto& builder) { found &= builder.find(idx); });

    if (!exclusive || found) {
      builders[0].fill(counter);
      std::ranges::for_each(builders.begin() + 1, builders.end(), [&idx](auto& builder) { builder.fill(idx); });
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(builders.size());
  std::ranges::transform(builders, std::back_inserter(arrays), [](auto& builder) { return builder.result(); });

  return arrow::Table::Make(schema, arrays);
}
} // namespace o2::soa

namespace o2::framework
{
std::shared_ptr<arrow::Table> makeEmptyTableImpl(const char* name, std::shared_ptr<arrow::Schema>& schema)
{
  schema = schema->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{name}}));
  return arrow::Table::MakeEmpty(schema).ValueOrDie();
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> const& fullTable, std::shared_ptr<arrow::Schema> newSchema, size_t nColumns,
                                            expressions::Projector* projectors, const char* name,
                                            std::shared_ptr<gandiva::Projector>& projector)
{
  if (projector == nullptr) {
    projector = framework::expressions::createProjectorHelper(nColumns, projectors, fullTable->schema(), newSchema->fields());
  }

  return spawnerHelper(fullTable, newSchema, name, nColumns, projector);
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> const& fullTable, std::shared_ptr<arrow::Schema> newSchema,
                                            const char* name, size_t nColumns,
                                            std::shared_ptr<gandiva::Projector> const& projector)
{
  arrow::TableBatchReader reader(*fullTable);
  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::ArrayVector v;
  std::vector<arrow::ArrayVector> chunks;
  chunks.resize(nColumns);
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  while (true) {
    auto s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw runtime_error_f("Cannot read batches from the source table to spawn %s: %s", name, s.ToString().c_str());
    }
    if (batch == nullptr) {
      break;
    }
    try {
      s = projector->Evaluate(*batch, arrow::default_memory_pool(), &v);
      if (!s.ok()) {
        throw runtime_error_f("Cannot apply projector to the source table of %s: %s", name, s.ToString().c_str());
      }
    } catch (std::exception& e) {
      throw runtime_error_f("Cannot apply projector to the source table of %s: exception caught: %s", name, e.what());
    }

    for (auto i = 0U; i < nColumns; ++i) {
      chunks[i].emplace_back(v.at(i));
    }
  }

  arrays.reserve(nColumns);
  std::ranges::transform(chunks, std::back_inserter(arrays), [](auto&& chunk) { return std::make_shared<arrow::ChunkedArray>(chunk); });

  return arrow::Table::Make(newSchema, arrays);
}

void initializePartitionCaches(std::set<uint32_t> const& hashes, std::shared_ptr<arrow::Schema> const& schema, expressions::Filter const& filter, gandiva::NodePtr& tree, gandiva::FilterPtr& gfilter)
{
  if (tree == nullptr) {
    expressions::Operations ops = createOperations(filter);
    if (isTableCompatible(hashes, ops)) {
      tree = createExpressionTree(ops, schema);
    } else {
      throw std::runtime_error("Partition filter does not match declared table type");
    }
  }
  if (gfilter == nullptr) {
    gfilter = framework::expressions::createFilter(schema, framework::expressions::makeCondition(tree));
  }
}

std::string serializeProjectors(std::vector<framework::expressions::Projector>& projectors)
{
  std::stringstream osm;
  ExpressionJSONHelpers::write(osm, projectors);
  return osm.str();
}

std::string serializeSchema(std::shared_ptr<arrow::Schema> schema)
{
  std::stringstream osm;
  ArrowJSONHelpers::write(osm, schema);
  return osm.str();
}

std::string serializeIndexRecords(std::vector<o2::soa::IndexRecord>& irs)
{
  std::stringstream osm;
  IndexJSONHelpers::write(osm, irs);
  return osm.str();
}

std::vector<std::shared_ptr<arrow::Table>> extractSources(ProcessingContext& pc, std::vector<ConcreteDataMatcher> const& matchers)
{
  std::vector<std::shared_ptr<arrow::Table>> tables;
  tables.reserve(matchers.size());
  std::ranges::transform(matchers, std::back_inserter(tables), [&pc](auto const& matcher) { return pc.inputs().get<TableConsumer>(matcher)->asArrowTable(); });
  return tables;
}

std::shared_ptr<arrow::Table> Spawner::materialize(ProcessingContext& pc) const
{
  auto tables = extractSources(pc, matchers);
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables), std::span{labels.begin(), labels.size()});
  if (fullTable->num_rows() == 0) {
    return arrow::Table::MakeEmpty(schema).ValueOrDie();
  }

  return spawnerHelper(fullTable, schema, binding.c_str(), schema->num_fields(), projector);
}

std::shared_ptr<arrow::Table> Builder::materialize(ProcessingContext& pc)
{
  if (builders == nullptr) {
    builders = std::make_shared<std::vector<framework::IndexColumnBuilder>>();
    builders->reserve(records.size());
  }
  std::shared_ptr<arrow::Table> result;
  auto tables = extractSources(pc, matchers);
  result = o2::soa::IndexBuilder::materialize(*builders.get(), std::move(tables), records, outputSchema, exclusive);
  return result;
}
} // namespace o2::framework
