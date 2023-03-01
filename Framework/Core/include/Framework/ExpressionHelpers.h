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
#ifndef O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
#define O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
#include "Framework/Expressions.h"

#include <vector>
#include <iosfwd>
#include <fmt/format.h>

namespace o2::framework::expressions
{

struct DatumSpec {
  /// datum spec either contains an index, a value of a literal or a binding label
  using datum_t = std::variant<std::monostate, size_t, LiteralNode::var_t, std::string>;
  datum_t datum = std::monostate{};
  size_t hash = 0;
  atype::type type = atype::NA;

  explicit DatumSpec(size_t index, atype::type type_) : datum{index}, type{type_} {}
  explicit DatumSpec(LiteralNode::var_t literal, atype::type type_) : datum{literal}, type{type_} {}
  explicit DatumSpec(std::string binding, size_t hash_, atype::type type_) : datum{binding}, hash{hash_}, type{type_} {}
  DatumSpec() = default;
  DatumSpec(DatumSpec const&) = default;
  DatumSpec(DatumSpec&&) = default;
  DatumSpec& operator=(DatumSpec const&) = default;
  DatumSpec& operator=(DatumSpec&&) = default;
};

bool operator==(DatumSpec const& lhs, DatumSpec const& rhs);

std::ostream& operator<<(std::ostream& os, DatumSpec const& spec);

struct ColumnOperationSpec {
  size_t index = 0;
  BasicOp op;
  DatumSpec left;
  DatumSpec right;
  DatumSpec condition;
  DatumSpec result;
  atype::type type = atype::NA;
  ColumnOperationSpec() = default;
  explicit ColumnOperationSpec(BasicOp op_, size_t index_ = 0)
    : index{index_},
      op{op_},
      left{},
      right{},
      condition{},
      result{}
  {
    switch (op) {
      case BasicOp::LogicalOr:
      case BasicOp::LogicalAnd:
      case BasicOp::LessThan:
      case BasicOp::LessThanOrEqual:
      case BasicOp::GreaterThan:
      case BasicOp::GreaterThanOrEqual:
      case BasicOp::Equal:
      case BasicOp::NotEqual:
        type = atype::BOOL;
        break;
      case BasicOp::Division:
        type = atype::FLOAT;
      default:
        type = atype::NA;
    }
    result.type = type;
  }
};

/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  size_t index = 0;
  explicit NodeRecord(Node* node_, size_t index_) : node_ptr(node_), index{index_} {}
  bool operator!=(NodeRecord const& rhs)
  {
    return this->node_ptr != rhs.node_ptr;
  }
};
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
