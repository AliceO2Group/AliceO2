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

#include <variant>
#include <string>
#include <memory>

namespace o2::framework::expressions
{

template <typename T>
struct LiteralNode {
  T value;
};

struct BindingNode {
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  BindingNode(std::string const& name_) : name{name_} {}
  std::string name;
};

struct BinaryOpNode {
};

struct GreaterThanOp : BinaryOpNode {
};

struct LessThanOp : BinaryOpNode {
};

struct AndOp : BinaryOpNode {
};

template <>
struct LiteralNode<bool> {
  LiteralNode(bool v) : value(v) {}
  bool value;
};

template <>
struct LiteralNode<int> {
  LiteralNode(int v) : value(v) {}
  int value;
};

struct Node;

struct Node {
  template <typename T>
  Node(T v) : self{LiteralNode<T>(v)}, left{nullptr}, right{nullptr}
  {
  }

  template <typename T>
  Node(LiteralNode<T> v) : self{v}, left{nullptr}, right{nullptr}
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

  std::variant<LiteralNode<bool>, LiteralNode<int>, BindingNode, BinaryOpNode> self;
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
};

template <typename T>
Node operator>(Node left, T rightValue)
{
  return Node{GreaterThanOp{}, std::move(left), LiteralNode<T>{rightValue}};
}

template <typename T>
Node operator<(Node left, T rightValue)
{
  return Node{LessThanOp{}, std::move(left), LiteralNode<T>{rightValue}};
}

inline Node operator&&(Node left, Node right)
{
  return Node{AndOp{}, std::move(left), std::move(right)};
}

struct Filter {
  template <typename T>
  Filter(T&& v) : node{v}
  {
  }

  Filter(Node&& node_) : node{std::move(node_)}
  {
  }

  Node node;
};

} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
