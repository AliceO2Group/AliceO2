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

/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  explicit NodeRecord(Node* node_) : node_ptr(node_) {}
};

std::vector<ArrowKernelSpec> createKernelsFromFilter(Filter const& filter)
{
  std::vector<ArrowDatumSpec> datums;
  std::vector<ArrowKernelSpec> kernelSpecs;
  std::stack<NodeRecord> path;
  auto isLeaf = [](Node const* const node) {
    return ((node->left == nullptr) && (node->right == nullptr));
  };

  size_t index = 0;
  // create and put output datum
  datums.emplace_back(ArrowDatumSpec{index++});
  // insert the top node into stack
  path.emplace(filter.node.get());

  // while the stack is not empty
  while (path.empty() == false) {
    auto& top = path.top();
    //if the top node is not a leaf node
    // check the children and create datums if needed
    auto left = top.node_ptr->left.get();
    auto right = top.node_ptr->right.get();
    bool leftLeaf = isLeaf(left);
    bool rightLeaf = isLeaf(right);

    size_t li = 0;
    size_t ri = 0;
    size_t ti = index;

    auto processLeaf = [&datums](Node const* const node) {
      datums.push_back(
        std::visit(
          overloaded{
            [lh = LiteralNodeHelper{}](LiteralNode node) { return lh(node); },
            [bh = BindingNodeHelper{}](BindingNode node) { return bh(node); },
            [](auto&&) { return ArrowDatumSpec{}; }},
          node->self));
    };

    if (leftLeaf) {
      processLeaf(left);
    } else {
      datums.emplace_back(ArrowDatumSpec{index++});
    }
    li = datums.size() - 1;

    if (rightLeaf) {
      processLeaf(right);
    } else {
      datums.emplace_back(ArrowDatumSpec{index++});
    }
    ri = datums.size() - 1;

    // create kernel spec, pop the node and add its children
    auto&& kernel =
      std::visit(
        overloaded{
          [bh = BinaryOpNodeHelper{}](BinaryOpNode node) { return bh(node); },
          [](auto&&) { return ArrowKernelSpec{}; }},
        top.node_ptr->self);
    kernel.left = ArrowDatumSpec{li};
    kernel.right = ArrowDatumSpec{ri};
    kernel.result = ArrowDatumSpec{ti};
    kernelSpecs.push_back(std::move(kernel));
    path.pop();
    if (!leftLeaf)
      path.emplace(left);
    if (!rightLeaf)
      path.emplace(right);
  }
  return kernelSpecs;
}

} // namespace o2::framework::expressions
