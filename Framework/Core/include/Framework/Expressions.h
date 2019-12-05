// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_EXPRESSIONS_H_
#define O2_FRAMEWORK_EXPRESSIONS_H_

#include "Framework/BasicOps.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/FunctionalHelpers.h"
#include <arrow/type.h>
#include <arrow/table.h>
#include <gandiva/selection_vector.h>
#include <gandiva/node.h>
#include "gandiva/filter.h"
#include <variant>
#include <string>
#include <memory>

using atype = arrow::Type;
using SchemaInfo = std::pair<std::string, gandiva::SchemaPtr>;
using ExpressionInfo = std::pair<std::string, gandiva::NodePtr>;

namespace o2::framework::expressions
{
template <typename... T>
struct LiteralStorage {
  using stored_type = std::variant<T...>;
  using stored_pack = framework::pack<T...>;
};

using LiteralValue = LiteralStorage<int, bool, float, double>;

template <typename T>
constexpr auto selectArrowType()
{
  if constexpr (std::is_same_v<T, int>) {
    return atype::INT32;
  } else if constexpr (std::is_same_v<T, bool>) {
    return atype::BOOL;
  } else if constexpr (std::is_same_v<T, float>) {
    return atype::FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return atype::DOUBLE;
  } else {
    return atype::NA;
  }
  O2_BUILTIN_UNREACHABLE();
}

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type);

/// An expression tree node corresponding to a literal value
struct LiteralNode {
  template <typename T>
  LiteralNode(T v) : value{v}, type{selectArrowType<T>()}
  {
  }
  using var_t = LiteralValue::stored_type;
  var_t value;
  atype::type type = atype::NA;
};

/// An expression tree node corresponding to a column binding
struct BindingNode {
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  BindingNode(std::string const& name_, atype::type type_) : name{name_}, type{type_} {}
  std::string name;
  atype::type type;
};

/// An expression tree node corresponding to binary operation
struct BinaryOpNode {
  BinaryOpNode(BasicOp op_) : op{op_} {}
  BasicOp op;
};

/// A generic tree node
struct Node {
  Node(LiteralNode v) : self{v}, left{nullptr}, right{nullptr}
  {
  }

  Node(Node&& n) : self{n.self}, left{std::move(n.left)}, right{std::move(n.right)}
  {
  }

  Node(BindingNode n) : self{n}, left{nullptr}, right{nullptr}
  {
  }

  Node(BinaryOpNode op, Node&& l, Node&& r)
    : self{op},
      left{std::make_unique<Node>(std::move(l))},
      right{std::make_unique<Node>(std::move(r))} {}

  /// variant with possible nodes
  using self_t = std::variant<LiteralNode, BindingNode, BinaryOpNode>;
  self_t self;
  /// pointers to children
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
};

/// overloaded operators to build the tree from an expression

/// literal comparisons
template <typename T>
inline Node operator>(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::GreaterThan}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator<(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::LessThan}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator>=(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::GreaterThanOrEqual}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator<=(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::LessThanOrEqual}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator==(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::Equal}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator!=(Node left, T rightValue)
{
  return Node{BinaryOpNode{BasicOp::NotEqual}, std::move(left), LiteralNode{rightValue}};
}

/// node comparisons
inline Node operator>(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::GreaterThan}, std::move(left), std::move(right)};
}

inline Node operator<(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::LessThan}, std::move(left), std::move(right)};
}

inline Node operator>=(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::GreaterThanOrEqual}, std::move(left), std::move(right)};
}

inline Node operator<=(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::LessThanOrEqual}, std::move(left), std::move(right)};
}

inline Node operator==(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::Equal}, std::move(left), std::move(right)};
}

inline Node operator!=(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::NotEqual}, std::move(left), std::move(right)};
}

/// logical operations
inline Node operator&&(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::LogicalAnd}, std::move(left), std::move(right)};
}

inline Node operator||(Node left, Node right)
{
  return Node{BinaryOpNode{BasicOp::LogicalOr}, std::move(left), std::move(right)};
}

/// arithmetical operations between node and literal
template <typename T>
inline Node operator*(Node left, T right)
{
  return Node{BinaryOpNode{BasicOp::Multiplication}, std::move(left), LiteralNode{right}};
}

template <typename T>
inline Node operator/(Node left, T right)
{
  return Node{BinaryOpNode{BasicOp::Division}, std::move(left), LiteralNode{right}};
}

template <typename T>
inline Node operator/(T left, Node right)
{
  return Node{BinaryOpNode{BasicOp::Division}, LiteralNode{left}, std::move(right)};
}

template <typename T>
inline Node operator+(Node left, T right)
{
  return Node{BinaryOpNode{BasicOp::Addition}, std::move(left), LiteralNode{right}};
}

template <typename T>
inline Node operator-(Node left, T right)
{
  return Node{BinaryOpNode{BasicOp::Subtraction}, std::move(left), LiteralNode{right}};
}

/// A struct, containing the root of the expression tree
struct Filter {
  Filter(Node&& node_) : node{std::make_unique<Node>(std::move(node_))} {}

  std::unique_ptr<Node> node;
};

using Selection = std::shared_ptr<gandiva::SelectionVector>;
Selection createSelection(std::shared_ptr<arrow::Table> table, Filter const& expression);
Selection createSelection(std::shared_ptr<arrow::Table> table, std::shared_ptr<gandiva::Filter> gfilter);

struct ColumnOperationSpec;
using Operations = std::vector<ColumnOperationSpec>;

Operations createOperations(Filter const& expression);
bool isSchemaCompatible(gandiva::SchemaPtr const& Schema, Operations const& opSpecs);
gandiva::NodePtr createExpressionTree(Operations const& opSpecs,
                                      gandiva::SchemaPtr const& Schema);
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              gandiva::ConditionPtr condition);
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              Operations const& opSpecs);
std::vector<ExpressionInfo> createExpressionInfos(std::vector<SchemaInfo> const& infos, expressions::Filter const& filter);
gandiva::ConditionPtr createCondition(gandiva::NodePtr node);
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
