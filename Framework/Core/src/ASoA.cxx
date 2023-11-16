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

#include "Framework/ASoA.h"
#include "ArrowDebugHelpers.h"
#include "Framework/RuntimeError.h"
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/config.h>

namespace o2::soa
{
SelectionVector selectionToVector(gandiva::Selection const& sel)
{
  SelectionVector rows;
  rows.resize(sel->GetNumSlots());
  for (auto i = 0; i < sel->GetNumSlots(); ++i) {
    rows[i] = sel->GetIndex(i);
  }
  return rows;
}

std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  for (auto i = 0U; i < tables.size() - 1; ++i) {
    if (tables[i]->num_rows() != tables[i + 1]->num_rows()) {
      throw o2::framework::runtime_error_f("Tables %s and %s have different sizes (%d vs %d) and cannot be joined!",
                                           tables[i]->schema()->metadata()->Get("label").ValueOrDie().c_str(),
                                           tables[i + 1]->schema()->metadata()->Get("label").ValueOrDie().c_str(),
                                           tables[i]->num_rows(),
                                           tables[i + 1]->num_rows());
    }
  }
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

  for (auto& t : tables) {
    auto tf = t->fields();
    std::copy(tf.begin(), tf.end(), std::back_inserter(fields));
  }

  auto schema = std::make_shared<arrow::Schema>(fields);

  if (tables[0]->num_rows() != 0) {
    for (auto& t : tables) {
      auto tc = t->columns();
      std::copy(tc.begin(), tc.end(), std::back_inserter(columns));
    }
  }
  return arrow::Table::Make(schema, columns);
}

std::shared_ptr<arrow::Table> ArrowHelpers::concatTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  assert(tables.size() > 1);
  std::vector<std::shared_ptr<arrow::Field>> resultFields = tables[0]->schema()->fields();
  auto compareFields = [](std::shared_ptr<arrow::Field> const& f1, std::shared_ptr<arrow::Field> const& f2) {
    // Let's do this with stable sorting.
    return (!f1->Equals(f2)) && (f1->name() < f2->name());
  };
  for (size_t i = 1; i < tables.size(); ++i) {
    auto& fields = tables[i]->schema()->fields();
    std::vector<std::shared_ptr<arrow::Field>> intersection;

    std::set_intersection(resultFields.begin(), resultFields.end(),
                          fields.begin(), fields.end(),
                          std::back_inserter(intersection), compareFields);
    resultFields.swap(intersection);
  }

  for (auto& field : resultFields) {
    arrow::ArrayVector chunks;
    for (auto& table : tables) {
      auto ci = table->schema()->GetFieldIndex(field->name());
      if (ci == -1) {
        throw std::runtime_error("Unable to find field " + field->name());
      }
      auto column = table->column(ci);
      auto otherChunks = column->chunks();
      chunks.insert(chunks.end(), otherChunks.begin(), otherChunks.end());
    }
    columns.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
  }

  auto result = arrow::Table::Make(std::make_shared<arrow::Schema>(resultFields), columns);
  return result;
}

arrow::ChunkedArray* getIndexFromLabel(arrow::Table* table, const char* label)
{
  auto index = table->schema()->GetAllFieldIndices(label);
  if (index.empty()) {
    o2::framework::throw_error(o2::framework::runtime_error_f("Unable to find column with label %s", label));
  }
  return table->column(index[0]).get();
}

void notBoundTable(const char* tableName)
{
  throw o2::framework::runtime_error_f("Index pointing to %s is not bound! Did you subscribe to the table?", tableName);
}

void notFoundColumn(const char* label, const char* key){
  throw o2::framework::runtime_error_f(R"(Preslice not valid: table "%s" (or join based on it) does not have column "%s"")", label, key);
}

} // namespace o2::soa

namespace o2::framework
{
std::string cutString(std::string&& str)
{
  auto pos = str.find('_');
  if (pos != std::string::npos) {
    str.erase(pos);
  }
  return str;
}

void sliceByColumnGeneric(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  ListVector* groups,
  ListVector* unassigned)
{
  groups->resize(fullSize);
  auto column = input->GetColumnByName(key);
  int32_t row = 0;
  for (auto iChunk = 0; iChunk < column->num_chunks(); ++iChunk) {
    auto chunk = static_cast<arrow::NumericArray<arrow::Int32Type>>(column->chunk(iChunk)->data());
    for (auto iElement = 0; iElement < chunk.length(); ++iElement) {
      auto v = chunk.Value(iElement);
      if (v >= 0) {
        if (v >= groups->size()) {
          throw runtime_error_f("Table %s has an entry with index (%d) that is larger than the grouping table size (%d)", target, v, fullSize);
        }
        (*groups)[v].push_back(row);
      } else if (unassigned != nullptr) {
        auto av = std::abs(v);
        if (unassigned->size() < av + 1) {
          unassigned->resize(av + 1);
        }
        (*unassigned)[av].push_back(row);
      }
      ++row;
    }
  }
}
} // namespace o2::framework
