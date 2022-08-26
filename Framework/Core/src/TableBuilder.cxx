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

#include "Framework/TableBuilder.h"
#include <memory>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/status.h>
#include <arrow/util/key_value_metadata.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace arrow;

namespace
{

// FIXME: Dummy schema, to compile.
template <typename TYPE, typename C_TYPE>
void ArrayFromVector(const std::vector<C_TYPE>& values, std::shared_ptr<arrow::Array>* out)
{
  typename arrow::TypeTraits<TYPE>::BuilderType builder;
  for (size_t i = 0; i < values.size(); ++i) {
    auto status = builder.Append(values[i]);
    assert(status.ok());
  }
  auto status = builder.Finish(out);
  assert(status.ok());
}

} // namespace

namespace o2::framework
{
void addLabelToSchema(std::shared_ptr<arrow::Schema>& schema, const char* label)
{
  schema = schema->WithMetadata(
    std::make_shared<arrow::KeyValueMetadata>(
      std::vector{std::string{"label"}},
      std::vector{std::string{label}}));
}

std::shared_ptr<arrow::Table>
  TableBuilder::finalize()
{
  bool status = mFinalizer(mSchema, mArrays, mHolders);
  if (status == false) {
    throwError(runtime_error("Unable to finalize"));
  }
  assert(mSchema->num_fields() > 0 && "Schema needs to be non-empty");
  return arrow::Table::Make(mSchema, mArrays);
}

void TableBuilder::throwError(RuntimeErrorRef const& ref)
{
  throw ref;
}

void TableBuilder::validate() const
{
  if (mHolders != nullptr) {
    throwError(runtime_error("TableBuilder::persist can only be invoked once per instance"));
  }
}

void TableBuilder::setLabel(const char* label)
{
  mSchema = mSchema->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{label}}));
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> fullTable, std::shared_ptr<arrow::Schema> newSchema, size_t nColumns,
                                            expressions::Projector* projectors, std::vector<std::shared_ptr<arrow::Field>> const& fields, const char* name)
{
  auto mergedProjectors = framework::expressions::createProjectorHelper(nColumns, projectors, fullTable->schema(), fields);

  arrow::TableBatchReader reader(*fullTable);
  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::ArrayVector v;
  std::vector<arrow::ArrayVector> chunks;
  chunks.resize(nColumns);
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  while (true) {
    auto s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw runtime_error_f("Cannot read batches from source table to spawn %s: %s", name, s.ToString().c_str());
    }
    if (batch == nullptr) {
      break;
    }
    try {
      s = mergedProjectors->Evaluate(*batch, arrow::default_memory_pool(), &v);
      if (!s.ok()) {
        throw runtime_error_f("Cannot apply projector to source table of %s: %s", name, s.ToString().c_str());
      }
    } catch (std::exception& e) {
      throw runtime_error_f("Cannot apply projector to source table of %s: exception caught: %s", name, e.what());
    }

    for (auto i = 0u; i < nColumns; ++i) {
      chunks[i].emplace_back(v.at(i));
    }
  }

  for (auto i = 0u; i < nColumns; ++i) {
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(chunks[i]));
  }

  addLabelToSchema(newSchema, name);
  return arrow::Table::Make(newSchema, arrays);
}

} // namespace o2::framework
