// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "../src/ExpressionHelpers.h"
#include "Framework/VariantHelpers.h"

#include <arrow/table.h>
#include <gandiva/selection_vector.h>
#include <stack>
#include <iostream>

using namespace o2::framework;

namespace o2::framework::expressions
{
struct LiteralNodeHelper {
  DatumSpec operator()(LiteralNode node) const
  {
    return DatumSpec{node.value};
  }
};

struct BindingNodeHelper {
  DatumSpec operator()(BindingNode node) const
  {
    return DatumSpec{node.name};
  }
};

struct BinaryOpNodeHelper {
  ColumnOperationSpec operator()(BinaryOpNode node) const
  {
    return ColumnOperationSpec{node.op};
  }
};

bool operator==(DatumSpec const& lhs, DatumSpec const& rhs)
{
  return (lhs.datum == rhs.datum);
}

std::ostream& operator<<(std::ostream& os, DatumSpec const& spec)
{
  std::visit(
    overloaded{
      [&os](LiteralNode::var_t&& arg) {
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

std::vector<ColumnOperationSpec> createKernelsFromFilter(Filter const& filter)
{
  std::vector<ColumnOperationSpec> columnOperationSpecs;
  std::stack<NodeRecord> path;
  auto isLeaf = [](Node const* const node) {
    return ((node->left == nullptr) && (node->right == nullptr));
  };

  auto processLeaf = [](Node const* const node) {
    return std::visit(
      overloaded{
        [lh = LiteralNodeHelper{}](LiteralNode node) { return lh(node); },
        [bh = BindingNodeHelper{}](BindingNode node) { return bh(node); },
        [](auto&&) { return DatumSpec{}; }},
      node->self);
  };

  size_t index = 0;
  // insert the top node into stack
  path.emplace(filter.node.get(), index++);

  // while the stack is not empty
  while (path.empty() == false) {
    auto& top = path.top();

    auto left = top.node_ptr->left.get();
    auto right = top.node_ptr->right.get();
    bool leftLeaf = isLeaf(left);
    bool rightLeaf = isLeaf(right);

    // create kernel spec, pop the node and add its children
    auto&& kernel =
      std::visit(
        overloaded{
          [bh = BinaryOpNodeHelper{}](BinaryOpNode node) { return bh(node); },
          [](auto&&) { return ColumnOperationSpec{}; }},
        top.node_ptr->self);
    kernel.result = DatumSpec{top.index};
    path.pop();

    size_t li = 0;
    size_t ri = 0;

    if (leftLeaf) {
      kernel.left = processLeaf(left);
    } else {
      li = index;
      kernel.left = DatumSpec{index++};
    }

    if (rightLeaf) {
      kernel.right = processLeaf(right);
    } else {
      ri = index;
      kernel.right = DatumSpec{index++};
    }

    columnOperationSpecs.push_back(std::move(kernel));
    if (!leftLeaf)
      path.emplace(left, li);
    if (!rightLeaf)
      path.emplace(right, ri);
  }
  return columnOperationSpecs;
}

Selection createSelection(std::shared_ptr<arrow::Table> table, Filter const& expr)
{
  std::shared_ptr<gandiva::SelectionVector> result;
  auto status = gandiva::SelectionVector::MakeInt32(table->num_rows(), arrow::default_memory_pool(), &result);
  if (status.ok() == false) {
    throw std::runtime_error("Unable to create selection");
  }
  return result;
}

} // namespace o2::framework::expressions
