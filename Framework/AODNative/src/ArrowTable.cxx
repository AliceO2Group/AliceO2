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

#include "AODNative/ArrowTable.h"
#include "AODNative/Errors.h"
#include <arrow/array/array_nested.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <cstdio>
#include <cstring>

namespace o2::aod::native
{

namespace
{
std::shared_ptr<arrow::DataType> valueType(BranchInfo const& b)
{
  switch (b.leafClass) {
    case 'O':
      return arrow::boolean();
    case 'B':
      return b.isUnsigned ? arrow::uint8() : arrow::int8();
    case 'S':
      return b.isUnsigned ? arrow::uint16() : arrow::int16();
    case 'I':
      return b.isUnsigned ? arrow::uint32() : arrow::int32();
    case 'L':
      return b.isUnsigned ? arrow::uint64() : arrow::int64();
    case 'F':
      return arrow::float32();
    case 'D':
      return arrow::float64();
  }
  throw UnsupportedLayout(std::string("leaf class TLeaf") + b.leafClass + " of branch " + b.name);
}

BranchInfo const& branchNamed(TreeInfo const& tree, std::string const& name)
{
  for (auto& b : tree.branches) {
    if (b.name == name) {
      return b;
    }
  }
  throw FormatError("tree " + tree.name + " has no branch " + name);
}

std::string format(char const* fmt, auto... args)
{
  auto n = std::snprintf(nullptr, 0, fmt, args...);
  std::string s(n, '\0');
  std::snprintf(s.data(), n + 1, fmt, args...);
  return s;
}
} // namespace

std::shared_ptr<arrow::Schema> physicalSchema(TreeInfo const& tree)
{
  std::vector<std::shared_ptr<arrow::Field>> fields;
  bool prevIsSize = false;
  for (auto& b : tree.branches) {
    if (prevIsSize && fields.back()->name() != b.name + "_size") {
      throw UnsupportedLayout(format("Unexpected layout for VLA container %s.", b.name.c_str()));
    }
    if (b.name.ends_with("_size")) {
      fields.push_back(std::make_shared<arrow::Field>(b.name, arrow::int32()));
      prevIsSize = true;
      continue;
    }
    if (prevIsSize) {
      fields.push_back(std::make_shared<arrow::Field>(b.name, arrow::list(valueType(b))));
    } else if (!b.countBranch.empty()) {
      throw UnsupportedLayout("variable-length branch " + b.name + " is not preceded by its size branch");
    } else if (b.fixedLen > 1) {
      fields.push_back(std::make_shared<arrow::Field>(b.name, arrow::fixed_size_list(valueType(b), b.fixedLen)));
    } else {
      fields.push_back(std::make_shared<arrow::Field>(b.name, valueType(b)));
    }
    prevIsSize = false;
  }
  if (!fields.empty() && fields.back()->name().ends_with("_size")) {
    throw UnsupportedLayout(format("Missing values for VLA indices %s.", fields.back()->name().c_str()));
  }
  return std::make_shared<arrow::Schema>(fields);
}

std::shared_ptr<arrow::Schema> datasetSchema(std::shared_ptr<arrow::Schema> const& physical)
{
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& f : physical->fields()) {
    if (!f->name().ends_with("_size")) {
      fields.push_back(f);
    }
  }
  return std::make_shared<arrow::Schema>(fields);
}

std::vector<ColumnOp> planColumns(TreeInfo const& tree, std::shared_ptr<arrow::Schema> const& schema,
                                  BranchDecoder& decoder, arrow::MemoryPool* pool, int64_t& rows)
{
  std::vector<ColumnOp> ops;
  rows = -1;
  for (int fi = 0; fi < schema->num_fields(); ++fi) {
    auto const& field = schema->field(fi);
    auto const& branch = branchNamed(tree, field->name());
    auto type = field->type();
    ColumnOp op{ColumnOp::Kind::Values, &branch, fi, 0, nullptr};
    int64_t opRows = branch.entries;
    if (type->id() == arrow::Type::LIST) {
      auto const& sizeBranch = branchNamed(tree, branch.name + "_size");
      if (branch.entries != sizeBranch.entries) {
        throw CorruptData(decoder.url(), 0, format("Branch %s has %lld entries, but its size branch %s has %lld.", branch.name.c_str(), (long long)branch.entries, sizeBranch.name.c_str(), (long long)sizeBranch.entries));
      }
      auto offsets = decoder.decodeOffsets(sizeBranch);
      auto buffer = arrow::AllocateBuffer(offsets.size() * sizeof(int32_t), pool).ValueOrDie();
      std::memcpy(buffer->mutable_data(), offsets.data(), offsets.size() * sizeof(int32_t));
      op.offsets = std::move(buffer);
      op.kind = ColumnOp::Kind::VLA;
      op.byteSize = (size_t)offsets.back() * leafElementSize(branch.leafClass);
      opRows = sizeBranch.entries;
    } else if (type->id() == arrow::Type::BOOL) {
      op.kind = ColumnOp::Kind::Booleans;
      op.byteSize = (branch.entries + 7) / 8;
    } else if (type->id() == arrow::Type::FIXED_SIZE_LIST && type->field(0)->type()->id() == arrow::Type::BOOL) {
      // Not the minimal size, but the TTree plugin's, for byte identity.
      op.kind = ColumnOp::Kind::Booleans;
      op.byteSize = (branch.entries * branch.fixedLen) / 8 + 1;
    } else {
      op.byteSize = branch.entries * branch.fixedLen * leafElementSize(branch.leafClass);
    }
    if (rows == -1) {
      rows = opRows;
    } else if (rows != opRows) {
      throw CorruptData(decoder.url(), 0, format("Unmatching number of rows for branch %s. Expected %lli, found %lli", branch.name.c_str(), (long long)rows, (long long)opRows));
    }
    ops.push_back(std::move(op));
  }
  if (rows == -1) {
    rows = 0;
  }
  return ops;
}

void executeColumnOp(BranchDecoder& decoder, ColumnOp const& op, uint8_t* dst)
{
  switch (op.kind) {
    case ColumnOp::Kind::Values:
      decoder.decodeValues(*op.branch, dst, op.byteSize);
      break;
    case ColumnOp::Kind::Booleans:
      decoder.decodeBooleans(*op.branch, dst, op.byteSize);
      break;
    case ColumnOp::Kind::VLA: {
      auto const* offsets = (int32_t const*)op.offsets->data();
      decoder.decodeVLAValues(*op.branch, {offsets, (size_t)op.branch->entries + 1}, dst, op.byteSize);
    } break;
  }
}

std::shared_ptr<arrow::RecordBatch> makeBatch(std::shared_ptr<arrow::Schema> const& schema, std::vector<ColumnOp> const& ops,
                                              std::vector<std::shared_ptr<arrow::Buffer>> const& buffers, int64_t rows)
{
  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto const& op = ops[i];
    auto type = schema->field(op.datasetField)->type();
    auto entries = op.branch->entries;
    if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
      auto vdata = std::make_shared<arrow::ArrayData>(type->field(0)->type(), entries * op.branch->fixedLen,
                                                      std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, buffers[i]});
      columns.push_back(std::make_shared<arrow::FixedSizeListArray>(type, entries, arrow::MakeArray(vdata)));
    } else if (op.kind == ColumnOp::Kind::VLA) {
      auto total = ((int32_t const*)op.offsets->data())[entries];
      auto vdata = std::make_shared<arrow::ArrayData>(type->field(0)->type(), total,
                                                      std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, buffers[i]});
      columns.push_back(std::make_shared<arrow::ListArray>(type, entries, op.offsets, arrow::MakeArray(vdata)));
    } else {
      auto data = std::make_shared<arrow::ArrayData>(type, entries, std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, buffers[i]});
      columns.push_back(arrow::MakeArray(data));
    }
  }
  return arrow::RecordBatch::Make(schema, rows, columns);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> readTree(ByteSource& source, TreeInfo const& tree, Options const& options, arrow::MemoryPool* pool)
{
  auto schema = datasetSchema(physicalSchema(tree));
  BranchDecoder decoder{source, options};
  int64_t rows;
  auto ops = planColumns(tree, schema, decoder, pool, rows);
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (auto& op : ops) {
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> buffer, arrow::AllocateBuffer(op.byteSize, pool));
    executeColumnOp(decoder, op, buffer->mutable_data());
    buffers.push_back(std::move(buffer));
  }
  return makeBatch(schema, ops, buffers, rows);
}

} // namespace o2::aod::native
