// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Expressions.h"
#include "Framework/VariantHelpers.h"
#include <stack>

using namespace arrow::compute;
using namespace o2::framework;

namespace o2::framework::expressions
{
// dummy function until arrow has actual scalars...
struct LiteralNodeHelper {
  ArrowDatumSpec operator()(LiteralNode node) const
  {
    return ArrowDatumSpec{node.value};
  }
};

// dummy function: needs to be bound to a table column
struct BindingNodeHelper {
  ArrowDatumSpec operator()(BindingNode node) const
  {
    return ArrowDatumSpec{node.name};
  }
};

struct BinaryOpNodeHelper {
  ArrowKernelSpec operator()(BinaryOpNode node) const
  {
    switch (node.op) {
      case BinaryOpNode::LogicalAnd:
      case BinaryOpNode::LogicalOr:
      case BinaryOpNode::LessThan:
      case BinaryOpNode::GreaterThan:
      case BinaryOpNode::LessThanOrEqual:
      case BinaryOpNode::GreaterThanOrEqual:
      case BinaryOpNode::Equal:
      case BinaryOpNode::Addition:
      case BinaryOpNode::Subtraction:
      case BinaryOpNode::Division:
        break;
    }
    return ArrowKernelSpec{};
  }
};

bool operator==(ArrowDatumSpec const& lhs, ArrowDatumSpec const& rhs)
{
  return (lhs.datum == rhs.datum);
}

std::ostream& operator<<(std::ostream& os, ArrowDatumSpec const& spec)
{
  std::visit(
    overloaded{
      [&os](LiteralNode::var_t&& arg) -> void {
        std::visit(
          [&os](auto&& arg) { os << arg; },
          arg);
      },
      [&os](size_t&& arg) { os << arg; },
      [&os](std::string&& arg) { os << arg; },
      [](auto&&) {}},
    spec.datum);
  return os;
}

/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  size_t index = 0;
  explicit NodeRecord(Node* node_, size_t index_) : node_ptr(node_), index{index_} {}
};

std::vector<ArrowKernelSpec> createKernelsFromFilter(Filter const& filter)
{
  std::vector<ArrowKernelSpec> kernelSpecs;
  std::stack<NodeRecord> path;
  auto isLeaf = [](Node const* const node) {
    return ((node->left == nullptr) && (node->right == nullptr));
  };

  size_t index = 0;
  // insert the top node into stack
  path.emplace(filter.node.get(), index++);

  // while the stack is not empty
  while (path.empty() == false) {
    auto& top = path.top();
    //if the top node is not a leaf node
    // check the children and create datums if needed
    auto left = top.node_ptr->left.get();
    auto right = top.node_ptr->right.get();
    bool leftLeaf = isLeaf(left);
    bool rightLeaf = isLeaf(right);

    auto processLeaf = [](Node const* const node) {
      return std::visit(
        overloaded{
          [lh = LiteralNodeHelper{}](LiteralNode node) { return lh(node); },
          [bh = BindingNodeHelper{}](BindingNode node) { return bh(node); },
          [](auto&&) { return ArrowDatumSpec{}; }},
        node->self);
    };

    // create kernel spec, pop the node and add its children
    auto&& kernel =
      std::visit(
        overloaded{
          [bh = BinaryOpNodeHelper{}](BinaryOpNode node) { return bh(node); },
          [](auto&&) { return ArrowKernelSpec{}; }},
        top.node_ptr->self);
    kernel.result = ArrowDatumSpec{top.index};
    path.pop();

    size_t li = 0;
    size_t ri = 0;

    if (leftLeaf) {
      kernel.left = processLeaf(left);
    } else {
      li = index;
      kernel.left = ArrowDatumSpec{index++};
    }

    if (rightLeaf) {
      kernel.right = processLeaf(right);
    } else {
      ri = index;
      kernel.right = ArrowDatumSpec{index++};
    }

    kernelSpecs.push_back(std::move(kernel));
    if (!leftLeaf)
      path.emplace(left, li);
    if (!rightLeaf)
      path.emplace(right, ri);
  }
  return kernelSpecs;
}

} // namespace o2::framework::expressions
