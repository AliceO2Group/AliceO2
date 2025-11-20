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

namespace o2::soa {
std::shared_ptr<arrow::Table> IndexBuilder::materialize(const char* label, std::vector<std::shared_ptr<arrow::Table>>&& tables, std::vector<soa::IndexRecord> const& records, bool exclusive)
{
  auto pool = arrow::default_memory_pool();
  std::vector<std::shared_ptr<framework::SelfIndexColumnBuilder>> builders;
  framework::SelfIndexColumnBuilder self{records[0].columnLabel.c_str(), pool};
  std::unique_ptr<framework::ChunkedArrayIterator> keyIndex = nullptr;
  if (records[0].kind != soa::IndexKind::IdxSelf) {
    keyIndex = std::make_unique<framework::ChunkedArrayIterator>(tables[0]->column(records[0].pos));
  }

  for (auto i = 1U; i < records.size(); ++i) {
    if (records[i].kind == soa::IndexKind::IdxSelf) {
      builders.emplace_back(std::make_shared<framework::SelfIndexColumnBuilder>(records[i].columnLabel.c_str(), pool));
    } else {
      builders.emplace_back(std::make_shared<framework::IndexColumnBuilder>(tables[i]->column(records[i].pos), records[i].columnLabel.c_str(), listSize(records[i].kind), pool));
    }
  }

  std::vector<bool> finds;
  finds.resize(builders.size());
  for (int64_t counter = 0; counter < tables[0]->num_rows(); ++counter) {
    int64_t idx = -1;
    if (keyIndex == nullptr) {
      idx = counter;
    } else {
      idx = keyIndex->valueAt(counter);
    }
    for (auto i = 0U; i < builders.size(); ++i) {
      if (records[i+1].kind == soa::IndexKind::IdxSelf) {
        finds[i] = builders[i]->find(idx);
      } else {
        finds[i] = std::static_pointer_cast<framework::IndexColumnBuilder>(builders[i])->find(idx);
      }
    }
    if (exclusive) {
      if (std::none_of(finds.begin(), finds.end(), [](bool const x) { return x == false; })) {
        for (auto i = 0U; i < builders.size(); ++i) {
          if (records[i+1].kind == soa::IndexKind::IdxSelf) {
            builders[i]->fill(idx);
          } else {
            std::static_pointer_cast<framework::IndexColumnBuilder>(builders[i])->fill(idx);
          }
        }
        self.fill(counter);
      }
    } else {
      for (auto i = 0U; i < builders.size(); ++i) {
        if (records[i+1].kind == soa::IndexKind::IdxSelf) {
          builders[i]->fill(idx);
        } else {
          std::static_pointer_cast<framework::IndexColumnBuilder>(builders[i])->fill(idx);
        }
      }
      self.fill(counter);
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(records.size());
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(records.size());
  arrays.push_back(self.result());
  fields.push_back(self.field());
  for (auto i = 0U; i < builders.size(); ++i) {
    if (records[i+1].kind == soa::IndexKind::IdxSelf) {
      arrays.push_back(builders[i]->result());
      fields.push_back(builders[i]->field());
    } else {
      arrays.push_back(std::static_pointer_cast<framework::IndexColumnBuilder>(builders[i])->result());
      fields.push_back(std::static_pointer_cast<framework::IndexColumnBuilder>(builders[i])->field());
    }
  }

  return framework::makeArrowTable(label, std::move(arrays), std::move(fields));
}
} // namespace o2::soa

namespace o2::framework
{
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

std::vector<std::shared_ptr<arrow::Table>> extractSources(ProcessingContext& pc, std::vector<std::string> const& labels)
{
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto const& label : labels) {
    tables.emplace_back(pc.inputs().get<TableConsumer>(label.c_str())->asArrowTable());
  }
  return tables;
}

std::shared_ptr<arrow::Table> Spawner::materialize(ProcessingContext& pc) const
{
  auto tables = extractSources(pc, labels);
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables), std::span{labels.begin(), labels.size()});
  if (fullTable->num_rows() == 0) {
    return arrow::Table::MakeEmpty(schema).ValueOrDie();
  }

  return spawnerHelper(fullTable, schema, binding.c_str(), schema->num_fields(), projector);
}

std::shared_ptr<arrow::Table> Builder::materialize(ProcessingContext& pc) const
{
  std::shared_ptr<arrow::Table> result;
  auto tables = extractSources(pc, labels);
  result = o2::soa::IndexBuilder::materialize(binding.c_str(), std::move(tables), records, exclusive);
  return result;
}
} // namespace o2::framework
