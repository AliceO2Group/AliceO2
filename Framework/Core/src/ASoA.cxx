// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ASoA.h"
#include "ArrowDebugHelpers.h"

namespace o2::soa
{

std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  for (auto i = 0u; i < tables.size() - 1; ++i) {
    assert(tables[i]->num_rows() == tables[i + 1]->num_rows());
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
  if (index.empty() == true) {
    o2::framework::throw_error(o2::framework::runtime_error_f("Unable to find column with label %s", label));
  }
  return table->column(index[0]).get();
}

arrow::Status getSliceFor(int value, char const* key, std::shared_ptr<arrow::Table> const& input, std::shared_ptr<arrow::Table>& output, uint64_t& offset)
{
  arrow::Datum value_counts;
  auto options = arrow::compute::CountOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {input->GetColumnByName(key)},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  auto values = static_cast<arrow::NumericArray<arrow::Int32Type>>(pair.field(0)->data());
  auto counts = static_cast<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());

  int slice;
  for (slice = 0; slice < values.length(); ++slice) {
    if (values.Value(slice) == value) {
      offset = slice;
      output = input->Slice(slice, counts.Value(slice));
      return arrow::Status::OK();
    }
  }
  output = input->Slice(0, 0);
  return arrow::Status::OK();
}

} // namespace o2::soa
