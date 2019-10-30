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

#include "Framework/Kernels.h"

#include <variant>
#include <string>
#include <memory>

namespace o2::framework::expressions
{

/// A helper type for an expression tree node corresponding to a literal value
struct LiteralNode {
  template <typename T>
  LiteralNode(T v) : value{v}
  {
  }
  std::variant<int, bool, float, double> value;
};

/// An expression tree node corresponding to a column binding
struct BindingNode {
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  BindingNode(std::string const& name_) : name{name_} {}
  std::string name;
};

/// A helper type for an expression tree node corresponding to binary operation
struct BinaryOpNode {
  enum Op : unsigned int {
    LogicalAnd,
    LogicalOr,
    Addition,
    Subtraction,
    Division,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal
  };
  BinaryOpNode(Op op_) : op{op_} {}
  Op op;
};

struct ArrowKernelSpec {
  std::unique_ptr<arrow::compute::OpKernel> kernel = nullptr;
  std::shared_ptr<arrow::compute::Datum> left = nullptr;
  std::shared_ptr<arrow::compute::Datum> right = nullptr;
  std::shared_ptr<arrow::compute::Datum> result = nullptr;
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
template <typename T>
inline Node operator>(Node left, T rightValue)
{
  return Node{BinaryOpNode{BinaryOpNode::GreaterThan}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator<(Node left, T rightValue)
{
  return Node{BinaryOpNode{BinaryOpNode::LessThan}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator>=(Node left, T rightValue)
{
  return Node{BinaryOpNode{BinaryOpNode::GreaterThanOrEqual}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator<=(Node left, T rightValue)
{
  return Node{BinaryOpNode{BinaryOpNode::LessThanOrEqual}, std::move(left), LiteralNode{rightValue}};
}

template <typename T>
inline Node operator==(Node left, T rightValue)
{
  return Node{BinaryOpNode{BinaryOpNode::Equal}, std::move(left), LiteralNode{rightValue}};
}

inline Node operator&&(Node left, Node right)
{
  return Node{BinaryOpNode{BinaryOpNode::LogicalAnd}, std::move(left), std::move(right)};
}

inline Node operator||(Node left, Node right)
{
  return Node{BinaryOpNode{BinaryOpNode::LogicalOr}, std::move(left), std::move(right)};
}

/// A struct, containing the root of the expression tree
struct Filter {
  Filter(Node&& node_) : node{std::make_unique<Node>(std::move(node_))} {}

  std::unique_ptr<Node> node;
};

} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
