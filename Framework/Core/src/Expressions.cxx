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
#include "Framework/Logger.h"
#include "gandiva/tree_expr_builder.h"
#include "arrow/table.h"
#include "fmt/format.h"
#include <stack>
#include <iostream>
#include <unordered_map>
#include <set>
#include <algorithm>

using namespace o2::framework;

namespace o2::framework::expressions
{
namespace
{
struct LiteralNodeHelper {
  DatumSpec operator()(LiteralNode node) const
  {
    return DatumSpec{node.value, node.type};
  }
};

struct BindingNodeHelper {
  DatumSpec operator()(BindingNode node) const
  {
    return DatumSpec{node.name, node.type};
  }
};

struct OpNodeHelper {
  ColumnOperationSpec operator()(OpNode node) const
  {
    return ColumnOperationSpec{node.op};
  }
};
} // namespace

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type)
{
  if (type == atype::INT32)
    return arrow::int32();
  if (type == atype::FLOAT)
    return arrow::float32();
  if (type == atype::DOUBLE)
    return arrow::float64();
  if (type == atype::BOOL)
    return arrow::boolean();
  return nullptr;
}

bool operator==(DatumSpec const& lhs, DatumSpec const& rhs)
{
  return (lhs.datum == rhs.datum) && (lhs.type == rhs.type);
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

namespace
{
/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  size_t index = 0;
  explicit NodeRecord(Node* node_, size_t index_) : node_ptr(node_), index{index_} {}
};
} // namespace

Operations createOperations(Filter const& expression)
{
  Operations OperationSpecs;
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
  path.emplace(expression.node.get(), index++);

  // while the stack is not empty
  while (path.empty() == false) {
    auto& top = path.top();

    // create operation spec, pop the node and add its children
    auto operationSpec =
      std::visit(
        overloaded{
          [](OpNode node) { return ColumnOperationSpec{node.op}; },
          [](auto&&) { return ColumnOperationSpec{}; }},
        top.node_ptr->self);

    operationSpec.result = DatumSpec{top.index, operationSpec.type};
    path.pop();

    auto left = top.node_ptr->left.get();
    bool leftLeaf = isLeaf(left);
    size_t li = 0;
    if (leftLeaf) {
      operationSpec.left = processLeaf(left);
    } else {
      li = index;
      operationSpec.left = DatumSpec{index++, atype::NA};
    }

    decltype(left) right = nullptr;
    if (top.node_ptr->right != nullptr)
      right = top.node_ptr->right.get();
    bool rightLeaf = true;
    if (right != nullptr)
      rightLeaf = isLeaf(right);
    size_t ri = 0;
    auto isUnary = false;
    if (top.node_ptr->right == nullptr) {
      operationSpec.right = DatumSpec{};
      isUnary = true;
    } else {
      if (rightLeaf) {
        operationSpec.right = processLeaf(right);
      } else {
        ri = index;
        operationSpec.right = DatumSpec{index++, atype::NA};
      }
    }

    OperationSpecs.push_back(std::move(operationSpec));
    if (!leftLeaf)
      path.emplace(left, li);
    if (!isUnary && !rightLeaf)
      path.emplace(right, ri);
  }
  // at this stage the operations vector is created, but the field types are
  // only set for the logical operations and leaf nodes
  std::vector<atype::type> resultTypes;
  resultTypes.resize(OperationSpecs.size());

  auto inferResultType = [&resultTypes](DatumSpec& left, DatumSpec& right) {
    // if the left datum is monostate (error)
    if (left.datum.index() == 0) {
      throw std::runtime_error("Malformed operation spec: empty left datum");
    }

    // check if the datums are references
    if (left.datum.index() == 1)
      left.type = resultTypes[std::get<size_t>(left.datum)];

    if (right.datum.index() == 1)
      right.type = resultTypes[std::get<size_t>(right.datum)];

    auto t1 = left.type;
    auto t2 = right.type;
    // if the right datum is monostate (unary op)
    if (right.datum.index() == 0) {
      if (t1 == atype::DOUBLE)
        return atype::DOUBLE;
      return atype::FLOAT;
    }

    if (t1 == t2)
      return t1;

    if (t1 == atype::INT32) {
      if (t2 == atype::FLOAT)
        return atype::FLOAT;
      if (t2 == atype::DOUBLE)
        return atype::DOUBLE;
    }
    if (t1 == atype::FLOAT) {
      if (t2 == atype::INT32)
        return atype::FLOAT;
      if (t2 == atype::DOUBLE)
        return atype::DOUBLE;
    }
    if (t1 == atype::DOUBLE) {
      return atype::DOUBLE;
    }
    throw std::runtime_error(fmt::format("Invalid combination of argument types {} and {}", t1, t2));
  };

  for (auto it = OperationSpecs.rbegin(); it != OperationSpecs.rend(); ++it) {
    auto type = inferResultType(it->left, it->right);
    if (it->type == atype::NA) {
      it->type = type;
    }
    it->result.type = it->type;
    resultTypes[std::get<size_t>(it->result.datum)] = it->type;
  }

  return OperationSpecs;
}

gandiva::ConditionPtr createCondition(gandiva::NodePtr node)
{
  return gandiva::TreeExprBuilder::MakeCondition(node);
}

std::shared_ptr<gandiva::Filter>
  createFilter(gandiva::SchemaPtr const& Schema, Operations const& opSpecs)
{
  std::shared_ptr<gandiva::Filter> filter;
  auto s = gandiva::Filter::Make(Schema,
                                 createCondition(createExpressionTree(opSpecs, Schema)),
                                 &filter);
  if (s.ok())
    return filter;
  throw std::runtime_error(fmt::format("Failed to create filter: {}", s));
}

std::shared_ptr<gandiva::Filter>
  createFilter(gandiva::SchemaPtr const& Schema, gandiva::ConditionPtr condition)
{
  std::shared_ptr<gandiva::Filter> filter;
  auto s = gandiva::Filter::Make(Schema,
                                 condition,
                                 &filter);
  if (s.ok())
    return filter;
  throw std::runtime_error(fmt::format("Failed to create filter: {}", s));
}

Selection createSelection(std::shared_ptr<arrow::Table> table, std::shared_ptr<gandiva::Filter> gfilter)
{
  Selection selection;
  auto s = gandiva::SelectionVector::MakeInt64(table->num_rows(),
                                               arrow::default_memory_pool(),
                                               &selection);
  if (!s.ok())
    throw std::runtime_error(fmt::format("Cannot allocate selection vector {}", s));
  arrow::TableBatchReader reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw std::runtime_error(fmt::format("Cannot read batches from table {}", s));
    }
    if (batch == nullptr) {
      break;
    }
    s = gfilter->Evaluate(*batch, selection);
    if (!s.ok())
      throw std::runtime_error(fmt::format("Cannot apply filter {}", s));
  }

  return selection;
}

Selection createSelection(std::shared_ptr<arrow::Table> table,
                          Filter const& expression)
{
  return createSelection(table, createFilter(table->schema(), createOperations(expression)));
}

gandiva::NodePtr createExpressionTree(Operations const& opSpecs,
                                      gandiva::SchemaPtr const& Schema)
{
  std::vector<gandiva::NodePtr> opNodes;
  opNodes.resize(opSpecs.size());
  std::fill(opNodes.begin(), opNodes.end(), nullptr);
  std::unordered_map<std::string, gandiva::NodePtr> fieldNodes;

  auto datumNode = [Schema, &opNodes, &fieldNodes](DatumSpec const& spec) {
    if (spec.datum.index() == 1) {
      return opNodes[std::get<size_t>(spec.datum)];
    }

    if (spec.datum.index() == 2) {
      auto content = std::get<LiteralNode::var_t>(spec.datum);
      if (content.index() == 0)
        return gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(std::get<int>(content)));
      if (content.index() == 1)
        return gandiva::TreeExprBuilder::MakeLiteral(std::get<bool>(content));
      if (content.index() == 2)
        return gandiva::TreeExprBuilder::MakeLiteral(std::get<float>(content));
      if (content.index() == 3)
        return gandiva::TreeExprBuilder::MakeLiteral(std::get<double>(content));
      throw std::runtime_error("Malformed LiteralNode");
    }

    if (spec.datum.index() == 3) {
      auto name = std::get<std::string>(spec.datum);
      auto lookup = fieldNodes.find(name);
      if (lookup != fieldNodes.end())
        return lookup->second;
      auto node = gandiva::TreeExprBuilder::MakeField(Schema->GetFieldByName(name));
      fieldNodes.insert({name, node});
      return node;
    }
    throw std::runtime_error("Malformed DatumSpec");
  };

  gandiva::NodePtr tree = nullptr;
  for (auto it = opSpecs.rbegin(); it != opSpecs.rend(); ++it) {
    auto leftNode = datumNode(it->left);
    auto rightNode = datumNode(it->right);
    switch (it->op) {
      case BasicOp::LogicalOr:
        tree = gandiva::TreeExprBuilder::MakeOr({leftNode, rightNode});
        break;
      case BasicOp::LogicalAnd:
        tree = gandiva::TreeExprBuilder::MakeAnd({leftNode, rightNode});
        break;
      default:
        if (it->op <= BasicOp::Exp) {
          tree = gandiva::TreeExprBuilder::MakeFunction(binaryOperationsMap[it->op], {leftNode, rightNode}, concreteArrowType(it->type));
        } else {
          tree = gandiva::TreeExprBuilder::MakeFunction(binaryOperationsMap[it->op], {leftNode}, concreteArrowType(it->type));
        }
        break;
    }
    opNodes[std::get<size_t>(it->result.datum)] = tree;
  }

  return tree;
}

bool isSchemaCompatible(gandiva::SchemaPtr const& Schema, Operations const& opSpecs)
{
  std::set<std::string> opFieldNames;
  for (auto& spec : opSpecs) {
    if (spec.left.datum.index() == 3)
      opFieldNames.insert(std::get<std::string>(spec.left.datum));
    if (spec.right.datum.index() == 3)
      opFieldNames.insert(std::get<std::string>(spec.right.datum));
  }

  std::set<std::string> schemaFieldNames;
  for (auto& field : Schema->fields()) {
    schemaFieldNames.insert(field->name());
  }

  return std::includes(schemaFieldNames.begin(), schemaFieldNames.end(),
                       opFieldNames.begin(), opFieldNames.end());
}

void updateExpressionInfos(expressions::Filter const& filter, std::vector<ExpressionInfo>& eInfos)
{
  if (eInfos.empty()) {
    throw std::runtime_error("Empty expression info vector.");
  }
  Operations ops = createOperations(filter);
  for (auto& info : eInfos) {
    if (isSchemaCompatible(info.schema, ops)) {
      /// FIXME: check if there is already a tree assigned for an entry and
      ///        and if so merge the new tree into it with 'and' node
      info.tree = createExpressionTree(ops, info.schema);
    }
  }
}

} // namespace o2::framework::expressions
