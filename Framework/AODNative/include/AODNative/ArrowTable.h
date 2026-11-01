// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_AODNATIVE_ARROWTABLE_H_
#define O2_FRAMEWORK_AODNATIVE_ARROWTABLE_H_

#include "AODNative/Decoder.h"
#include "AODNative/Objects.h"
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/type.h>
#include <memory>
#include <vector>

namespace o2::aod::native
{

/// The physical schema of a tree, exactly as the TTree plugin's Inspect builds
/// it: one field per branch, in branch order. A branch X_size becomes an int32
/// field and must be immediately followed by X, which becomes list(T).
std::shared_ptr<arrow::Schema> physicalSchema(TreeInfo const& tree);

/// The physical schema without the *_size fields, as DataInputDirector derives it.
std::shared_ptr<arrow::Schema> datasetSchema(std::shared_ptr<arrow::Schema> const& physical);

/// One buffer to fill for a column of the dataset schema. The byte sizes are
/// those of the TTree plugin, so that the IPC stream is byte-identical.
struct ColumnOp {
  enum struct Kind { Values,
                     Booleans,
                     VLA };
  Kind kind;
  BranchInfo const* branch;
  int datasetField;
  size_t byteSize;
  /// For VLA columns: the offsets, decoded eagerly because they size the values.
  std::shared_ptr<arrow::Buffer> offsets;
};

/// Plans the reading of @a schema (a subset of the dataset schema) from @a tree,
/// decoding VLA offsets into buffers from @a pool. Checks the row counts with
/// the same errors as the TTree plugin. Returns the number of rows in @a rows.
std::vector<ColumnOp> planColumns(TreeInfo const& tree, std::shared_ptr<arrow::Schema> const& schema,
                                  BranchDecoder& decoder, arrow::MemoryPool* pool, int64_t& rows);

/// Fills the buffer of @a op, which is exactly op.byteSize bytes.
void executeColumnOp(BranchDecoder& decoder, ColumnOp const& op, uint8_t* dst);

/// Wraps the buffers of @a ops (one per op, in order) into a record batch.
std::shared_ptr<arrow::RecordBatch> makeBatch(std::shared_ptr<arrow::Schema> const& schema, std::vector<ColumnOp> const& ops,
                                              std::vector<std::shared_ptr<arrow::Buffer>> const& buffers, int64_t rows);

/// Reads a whole tree into heap memory: plan, execute, wrap.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> readTree(ByteSource& source, TreeInfo const& tree, Options const& options,
                                                            arrow::MemoryPool* pool = arrow::default_memory_pool());

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_ARROWTABLE_H_
