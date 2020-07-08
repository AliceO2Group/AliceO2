// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
/// a map between BasicOp and gandiva node definitions
/// note that logical 'and' and 'or' are created separately
static std::array<std::string, BasicOp::Abs + 1> binaryOperationsMap = {
  "and",
  "or",
  "add",
  "subtract",
  "divide",
  "multiply",
  "less_than",
  "less_than_or_equal_to",
  "greater_than",
  "greater_than_or_equal_to",
  "equal",
  "not_equal",
  "powerf",
  "sqrtf",
  "expf",
  "logf",
  "log10f",
  "sinf",
  "cosf",
  "tanf",
  "asinf",
  "acosf",
  "atanf",
  "absf"};

struct DatumSpec {
  /// datum spec either contains an index, a value of a literal or a binding label
  using datum_t = std::variant<std::monostate, size_t, LiteralNode::var_t, std::string>;
  datum_t datum = std::monostate{};
  atype::type type = atype::NA;
  explicit DatumSpec(size_t index, atype::type type_) : datum{index}, type{type_} {}
  explicit DatumSpec(LiteralNode::var_t literal, atype::type type_) : datum{literal}, type{type_} {}
  explicit DatumSpec(std::string binding, atype::type type_) : datum{binding}, type{type_} {}
  DatumSpec() = default;
  DatumSpec(DatumSpec const&) = default;
  DatumSpec(DatumSpec&&) = default;
  DatumSpec& operator=(DatumSpec const&) = default;
  DatumSpec& operator=(DatumSpec&&) = default;
};

bool operator==(DatumSpec const& lhs, DatumSpec const& rhs);

std::ostream& operator<<(std::ostream& os, DatumSpec const& spec);

struct ColumnOperationSpec {
  BasicOp op;
  DatumSpec left;
  DatumSpec right;
  DatumSpec result;
  atype::type type = atype::NA;
  ColumnOperationSpec() = default;
  // TODO: extend this to support unary ops seamlessly
  explicit ColumnOperationSpec(BasicOp op_) : op{op_},
                                              left{},
                                              right{},
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

template <typename... C>
std::shared_ptr<gandiva::Projector> createProjectors(framework::pack<C...>, gandiva::SchemaPtr schema)
{
  std::shared_ptr<gandiva::Projector> projector;
  auto s = gandiva::Projector::Make(
    schema,
    {makeExpression(
      framework::expressions::createExpressionTree(
        framework::expressions::createOperations(C::Projector()),
        schema),
      C::asArrowField())...},
    &projector);
  if (s.ok()) {
    return projector;
  } else {
    throw std::runtime_error(fmt::format("Failed to create projector: {}", s.ToString()));
  }
}
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
