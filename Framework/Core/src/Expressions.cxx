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
  Datum* operator()(LiteralNode) const
  {
    return new Datum{};
  }
};

// dummy function: needs to be bound to a table column
struct BindingNodeHelper {
  Datum* operator()(BindingNode) const
  {
    return new Datum{};
  }
};

struct BinaryOpNodeHelper {
  ArrowKernelSpec operator()(BinaryOpNode) const { return ArrowKernelSpec{}; }
};

/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  /// index of the assigned output in the datum array
  size_t index = 0;
  explicit NodeRecord(Node* node_, size_t di) : node_ptr(node_), index{di} {}
};

std::vector<ArrowKernelSpec> createKernelsFromFilter(Filter const& filter)
{
  std::vector<std::shared_ptr<Datum>> datums;
  std::vector<ArrowKernelSpec> kernelSpecs;
  std::stack<NodeRecord> path;
  auto isLeaf = [](Node const* const node) {
    return ((node->left == nullptr) && (node->right == nullptr));
  };

  // create and put output datum
  datums.emplace_back();
  // insert the top node into stack
  path.emplace(filter.node.get(), datums.size() - 1);

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

    auto processLeaf = [&datums](Node const* const node) {
      auto nodeDatum =
        std::visit(
          overloaded{
            [lh = LiteralNodeHelper{}](LiteralNode node) { return lh(node); },
            [bh = BindingNodeHelper{}](BindingNode node) { return bh(node); },
            [](auto &&)
              -> arrow::compute::Datum* { return nullptr; }},
          node->self);
      datums.push_back(std::shared_ptr<arrow::compute::Datum>(nodeDatum));
    };

    if (leftLeaf) {
      processLeaf(left);
    } else {
      datums.emplace_back();
    }
    li = datums.size() - 1;

    if (rightLeaf) {
      processLeaf(right);
    } else {
      datums.emplace_back();
    }
    ri = datums.size() - 1;

    // create kernel spec, pop the node and add its children
    auto&& kernel =
      std::visit(
        overloaded{
          [bh = BinaryOpNodeHelper{}](BinaryOpNode node) { return bh(node); },
          [](auto&&) { return ArrowKernelSpec{}; }},
        top.node_ptr->self);
    kernel.left = datums[li];
    kernel.right = datums[ri];
    kernel.result = datums[top.index];
    kernelSpecs.push_back(std::move(kernel));
    path.pop();
    if (!leftLeaf)
      path.emplace(left, li);
    if (!rightLeaf)
      path.emplace(right, ri);
  }
  return kernelSpecs;
}

} // namespace o2::framework::expressions
