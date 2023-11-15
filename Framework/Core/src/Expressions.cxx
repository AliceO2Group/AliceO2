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

#include "Framework/ExpressionHelpers.h"
#include "Framework/RuntimeError.h"
#include "Framework/VariantHelpers.h"
#include "arrow/table.h"
#include "gandiva/tree_expr_builder.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <stack>
#include <unordered_map>

using namespace o2::framework;

namespace o2::framework::expressions
{

/// a map between BasicOp and gandiva node definitions
/// note that logical 'and' and 'or' are created separately
static const std::array<std::string, BasicOp::Conditional + 1> basicOperationsMap = {
  "and",
  "or",
  "add",
  "subtract",
  "divide",
  "multiply",
  "bitwise_and",
  "bitwise_or",
  "bitwise_xor",
  "less_than",
  "less_than_or_equal_to",
  "greater_than",
  "greater_than_or_equal_to",
  "equal",
  "not_equal",
  "atan2f",
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
  "absf",
  "round",
  "bitwise_not",
  "if"};

size_t Filter::designateSubtrees(Node* node, size_t index)
{
  std::stack<NodeRecord> path;
  auto local_index = index;
  path.emplace(node, 0);

  while (!path.empty()) {
    auto& top = path.top();
    top.node_ptr->index = local_index;
    path.pop();
    if (top.node_ptr->condition != nullptr) {
      // start new subtrees
      index = designateSubtrees(top.node_ptr->left.get(), local_index + 1);
      index = designateSubtrees(top.node_ptr->condition.get(), index + 1);
      index = designateSubtrees(top.node_ptr->right.get(), index + 1);
    } else {
      // continue current subtree
      if (top.node_ptr->left != nullptr) {
        path.emplace(top.node_ptr->left.get(), 0);
      }
      if (top.node_ptr->right != nullptr) {
        path.emplace(top.node_ptr->right.get(), 0);
      }
    }
  }

  return index;
}

namespace
{
struct LiteralNodeHelper {
  DatumSpec operator()(LiteralNode const& node) const
  {
    return DatumSpec{node.value, node.type};
  }
};

struct BindingNodeHelper {
  DatumSpec operator()(BindingNode const& node) const
  {
    return DatumSpec{node.name, node.hash, node.type};
  }
};

struct OpNodeHelper {
  ColumnOperationSpec operator()(OpNode const& node) const
  {
    return ColumnOperationSpec{node.op};
  }
};

struct PlaceholderNodeHelper {
  DatumSpec operator()(PlaceholderNode const& node) const
  {
    return DatumSpec{node.value, node.type};
  }
};
} // namespace

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type)
{
  switch (type) {
    case atype::UINT8:
      return arrow::uint8();
    case atype::INT8:
      return arrow::int8();
    case atype::INT16:
      return arrow::int16();
    case atype::UINT16:
      return arrow::uint16();
    case atype::INT32:
      return arrow::int32();
    case atype::UINT32:
      return arrow::uint32();
    case atype::INT64:
      return arrow::int64();
    case atype::UINT64:
      return arrow::uint64();
    case atype::FLOAT:
      return arrow::float32();
    case atype::DOUBLE:
      return arrow::float64();
    case atype::BOOL:
      return arrow::boolean();
    default:
      return nullptr;
  }
}

std::string upcastTo(atype::type f)
{
  switch (f) {
    case atype::INT32:
      return "castINT";
    case atype::INT64:
      return "castBIGINT";
    case atype::FLOAT:
      return "castFLOAT4";
    case atype::DOUBLE:
      return "castFLOAT8";
    default:
      throw runtime_error_f("Do not know how to cast to %d", f);
  }
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

void updatePlaceholders(Filter& filter, InitContext& context)
{
  std::stack<NodeRecord> path;

  // insert the top node into stack
  path.emplace(filter.node.get(), 0);

  auto updateNode = [&](Node* node) {
    if (node->self.index() == 3) {
      std::get_if<3>(&node->self)->reset(context);
    }
  };

  // while the stack is not empty
  while (!path.empty()) {
    auto& top = path.top();
    updateNode(top.node_ptr);

    auto* leftp = top.node_ptr->left.get();
    auto* rightp = top.node_ptr->right.get();
    auto* condp = top.node_ptr->condition.get();
    path.pop();

    if (leftp != nullptr) {
      path.emplace(leftp, 0);
    }
    if (rightp != nullptr) {
      path.emplace(rightp, 0);
    }
    if (condp != nullptr) {
      path.emplace(condp, 0);
    }
  }
}

const char* stringType(atype::type t)
{
  switch (t) {
    case atype::BOOL:
      return "bool";
    case atype::DOUBLE:
      return "double";
    case atype::FLOAT:
      return "float";
    case atype::INT8:
      return "int8";
    case atype::INT16:
      return "int16";
    case atype::INT32:
      return "int32";
    case atype::INT64:
      return "int64";
    case atype::UINT8:
      return "uint8";
    case atype::UINT16:
      return "uint16";
    case atype::UINT32:
      return "uint32";
    case atype::UINT64:
      return "uint64";
    default:
      return "unsupported";
  }
  O2_BUILTIN_UNREACHABLE();
}

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
        [lh = LiteralNodeHelper{}](LiteralNode const& node) { return lh(node); },
        [bh = BindingNodeHelper{}](BindingNode const& node) { return bh(node); },
        [ph = PlaceholderNodeHelper{}](PlaceholderNode const& node) { return ph(node); },
        [](auto&&) { return DatumSpec{}; }},
      node->self);
  };

  size_t index = 0;
  // insert the top node into stack
  path.emplace(expression.node.get(), index++);

  // while the stack is not empty
  while (!path.empty()) {
    auto& top = path.top();

    // create operation spec, pop the node and add its children
    auto operationSpec =
      std::visit(
        overloaded{
          [&](OpNode node) { return ColumnOperationSpec{node.op, top.node_ptr->index}; },
          [&](ConditionalNode) { return ColumnOperationSpec{BasicOp::Conditional, top.node_ptr->index}; },
          [](auto&&) { return ColumnOperationSpec{}; }},
        top.node_ptr->self);

    operationSpec.result = DatumSpec{top.index, operationSpec.type};
    path.pop();

    auto* left = top.node_ptr->left.get();
    bool leftLeaf = isLeaf(left);
    size_t li = 0;
    if (leftLeaf) {
      operationSpec.left = processLeaf(left);
    } else {
      li = index;
      operationSpec.left = DatumSpec{index++, atype::NA};
    }

    decltype(left) right = nullptr;
    if (top.node_ptr->right != nullptr) {
      right = top.node_ptr->right.get();
    }
    bool rightLeaf = true;
    if (right != nullptr) {
      rightLeaf = isLeaf(right);
    }
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

    decltype(left) condition = nullptr;
    if (top.node_ptr->condition != nullptr) {
      condition = top.node_ptr->condition.get();
    }
    bool condleaf = condition != nullptr ? isLeaf(condition) : true;
    size_t ci = 0;
    if (condition != nullptr) {
      if (condleaf) {
        operationSpec.condition = processLeaf(condition);
      } else {
        ci = index;
        operationSpec.condition = DatumSpec{index++, atype::BOOL};
      }
    } else {
      operationSpec.condition = DatumSpec{};
    }

    OperationSpecs.push_back(std::move(operationSpec));
    if (!leftLeaf) {
      path.emplace(left, li);
    }
    if (!isUnary && !rightLeaf) {
      path.emplace(right, ri);
    }
    if (!condleaf) {
      path.emplace(condition, ci);
    }
  }
  // at this stage the operations vector is created, but the field types are
  // only set for the logical operations and leaf nodes
  std::vector<atype::type> resultTypes;
  resultTypes.resize(OperationSpecs.size());

  auto inferResultType = [&resultTypes](DatumSpec& left, DatumSpec& right) {
    // if the left datum is monostate (error)
    if (left.datum.index() == 0) {
      throw runtime_error("Malformed operation spec: empty left datum");
    }

    // check if the datums are references
    if (left.datum.index() == 1) {
      left.type = resultTypes[std::get<size_t>(left.datum)];
    }

    if (right.datum.index() == 1) {
      right.type = resultTypes[std::get<size_t>(right.datum)];
    }

    auto t1 = left.type;
    auto t2 = right.type;
    // if the right datum is monostate (unary op)
    if (right.datum.index() == 0) {
      if (t1 == atype::DOUBLE) {
        return atype::DOUBLE;
      }
      return atype::FLOAT;
    }

    if (t1 == t2) {
      return t1;
    }

    auto isIntType = [](auto t) {
      return (t == atype::UINT8) || (t == atype::INT8) || (t == atype::UINT16) || (t == atype::INT16) || (t == atype::UINT32) || (t == atype::INT32) || (t == atype::UINT64) || (t == atype::INT64);
    };

    if (isIntType(t1)) {
      if (t2 == atype::FLOAT) {
        return atype::FLOAT;
      }
      if (t2 == atype::DOUBLE) {
        return atype::DOUBLE;
      }
      if (isIntType(t2)) {
        if (t1 > t2) {
          return t1;
        }
        return t2;
      }
    }
    if (t1 == atype::FLOAT) {
      if (isIntType(t2)) {
        return atype::FLOAT;
      }
      if (t2 == atype::DOUBLE) {
        return atype::DOUBLE;
      }
    }
    if (t1 == atype::DOUBLE) {
      return atype::DOUBLE;
    }
    throw runtime_error_f("Invalid combination of argument types %s and %s", stringType(t1), stringType(t2));
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

gandiva::ConditionPtr makeCondition(gandiva::NodePtr node)
{
  return gandiva::TreeExprBuilder::MakeCondition(std::move(node));
}

gandiva::ExpressionPtr makeExpression(gandiva::NodePtr node, gandiva::FieldPtr result)
{
  return gandiva::TreeExprBuilder::MakeExpression(std::move(node), std::move(result));
}

std::shared_ptr<gandiva::Filter>
  createFilter(gandiva::SchemaPtr const& Schema, Operations const& opSpecs)
{
  std::shared_ptr<gandiva::Filter> filter;
  auto s = gandiva::Filter::Make(Schema,
                                 makeCondition(createExpressionTree(opSpecs, Schema)),
                                 &filter);
  if (!s.ok()) {
    throw runtime_error_f("Failed to create filter: %s", s.ToString().c_str());
  }
  return filter;
}

std::shared_ptr<gandiva::Filter>
  createFilter(gandiva::SchemaPtr const& Schema, gandiva::ConditionPtr condition)
{
  std::shared_ptr<gandiva::Filter> filter;
  auto s = gandiva::Filter::Make(Schema,
                                 std::move(condition),
                                 &filter);
  if (!s.ok()) {
    throw runtime_error_f("Failed to create filter: %s", s.ToString().c_str());
  }
  return filter;
}

std::shared_ptr<gandiva::Projector>
  createProjector(gandiva::SchemaPtr const& Schema, Operations const& opSpecs, gandiva::FieldPtr result)
{
  std::shared_ptr<gandiva::Projector> projector;
  auto s = gandiva::Projector::Make(Schema,
                                    {makeExpression(createExpressionTree(opSpecs, Schema), std::move(result))},
                                    &projector);
  if (!s.ok()) {
    throw runtime_error_f("Failed to create projector: %s", s.ToString().c_str());
  }
  return projector;
}

std::shared_ptr<gandiva::Projector>
  createProjector(gandiva::SchemaPtr const& Schema, Projector&& p, gandiva::FieldPtr result)
{
  return createProjector(Schema, createOperations(p), std::move(result));
}

std::shared_ptr<gandiva::Projector> createProjectorHelper(size_t nColumns, expressions::Projector* projectors,
                                                          std::shared_ptr<arrow::Schema> schema,
                                                          std::vector<std::shared_ptr<arrow::Field>> const& fields)
{
  std::vector<gandiva::ExpressionPtr> expressions;

  for (size_t ci = 0; ci < nColumns; ++ci) {
    expressions.push_back(
      makeExpression(
        framework::expressions::createExpressionTree(
          framework::expressions::createOperations(projectors[ci]),
          schema),
        fields[ci]));
  }

  std::shared_ptr<gandiva::Projector> projector;
  auto s = gandiva::Projector::Make(
    schema,
    expressions,
    &projector);
  if (s.ok()) {
    return projector;
  }
  throw o2::framework::runtime_error_f("Failed to create projector: %s", s.ToString().c_str());
}

gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table, std::shared_ptr<gandiva::Filter> const& gfilter)
{
  gandiva::Selection selection;
  auto s = gandiva::SelectionVector::MakeInt64(table->num_rows(),
                                               arrow::default_memory_pool(),
                                               &selection);
  if (!s.ok()) {
    throw runtime_error_f("Cannot allocate selection vector %s", s.ToString().c_str());
  }
  if (table->num_rows() == 0) {
    return selection;
  }
  arrow::TableBatchReader reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw runtime_error_f("Cannot read batches from table %s", s.ToString().c_str());
    }
    if (batch == nullptr) {
      break;
    }
    s = gfilter->Evaluate(*batch, selection);
    if (!s.ok()) {
      throw runtime_error_f("Cannot apply filter %s", s.ToString().c_str());
    }
  }

  return selection;
}

gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table,
                                   Filter const& expression)
{
  return createSelection(table, createFilter(table->schema(), createOperations(std::move(expression))));
}

auto createProjection(std::shared_ptr<arrow::Table> const& table, std::shared_ptr<gandiva::Projector> const& gprojector)
{
  arrow::TableBatchReader reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  std::shared_ptr<arrow::ArrayVector> v;
  while (true) {
    auto s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw runtime_error_f("Cannot read batches from table %s", s.ToString().c_str());
    }
    if (batch == nullptr) {
      break;
    }
    s = gprojector->Evaluate(*batch, arrow::default_memory_pool(), v.get());
    if (!s.ok()) {
      throw runtime_error_f("Cannot apply projector %s", s.ToString().c_str());
    }
  }
  return v;
}

gandiva::NodePtr createExpressionTree(Operations const& opSpecs,
                                      gandiva::SchemaPtr const& Schema)
{
  std::vector<gandiva::NodePtr> opNodes;
  opNodes.resize(opSpecs.size());
  std::fill(opNodes.begin(), opNodes.end(), nullptr);
  std::unordered_map<std::string, gandiva::NodePtr> fieldNodes;
  std::unordered_map<size_t, gandiva::NodePtr> subtrees;

  auto datumNode = [Schema, &opNodes, &fieldNodes](DatumSpec const& spec) {
    if (spec.datum.index() == 0) {
      return gandiva::NodePtr(nullptr);
    }
    if (spec.datum.index() == 1) {
      return opNodes[std::get<size_t>(spec.datum)];
    }

    if (spec.datum.index() == 2) {
      auto content = std::get<LiteralNode::var_t>(spec.datum);
      switch (content.index()) {
        case 0: // int
          return gandiva::TreeExprBuilder::MakeLiteral(static_cast<int32_t>(std::get<int>(content)));
        case 1: // bool
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<bool>(content));
        case 2: // float
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<float>(content));
        case 3: // double
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<double>(content));
        case 4: // uint8
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<uint8_t>(content));
        case 5: // int64
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<int64_t>(content));
        case 6: // int16
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<int16_t>(content));
        case 7: // uint16
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<uint16_t>(content));
        case 8: // int8
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<int8_t>(content));
        case 9: // uint32
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<uint32_t>(content));
        case 10: // uint64
          return gandiva::TreeExprBuilder::MakeLiteral(std::get<uint64_t>(content));
        default:
          throw runtime_error("Malformed LiteralNode");
      }
    }

    if (spec.datum.index() == 3) {
      auto name = std::get<std::string>(spec.datum);
      auto lookup = fieldNodes.find(name);
      if (lookup != fieldNodes.end()) {
        return lookup->second;
      }
      auto field = Schema->GetFieldByName(name);
      if (field == nullptr) {
        throw runtime_error_f("Cannot find field \"%s\"", name.c_str());
      }
      auto node = gandiva::TreeExprBuilder::MakeField(field);
      fieldNodes.insert({name, node});
      return node;
    }
    throw runtime_error("Malformed DatumSpec");
  };

  gandiva::NodePtr tree = nullptr;
  for (auto it = opSpecs.rbegin(); it != opSpecs.rend(); ++it) {
    auto leftNode = datumNode(it->left);
    auto rightNode = datumNode(it->right);
    auto condNode = datumNode(it->condition);

    auto insertUpcastNode = [&](gandiva::NodePtr node, atype::type t) {
      if (t != it->type) {
        auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(it->type), {node}, concreteArrowType(it->type));
        node = upcast;
      }
      return node;
    };

    auto insertEqualizeUpcastNode = [&](gandiva::NodePtr& node1, gandiva::NodePtr& node2, atype::type t1, atype::type t2) {
      if (t2 > t1) {
        auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(t2), {node1}, concreteArrowType(t2));
        node1 = upcast;
      } else if (t1 > t2) {
        auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(t1), {node2}, concreteArrowType(t1));
        node2 = upcast;
      }
    };

    gandiva::NodePtr temp_node;

    switch (it->op) {
      case BasicOp::LogicalOr:
        temp_node = gandiva::TreeExprBuilder::MakeOr({leftNode, rightNode});
        break;
      case BasicOp::LogicalAnd:
        temp_node = gandiva::TreeExprBuilder::MakeAnd({leftNode, rightNode});
        break;
      case BasicOp::Conditional:
        temp_node = gandiva::TreeExprBuilder::MakeIf(condNode, leftNode, rightNode, concreteArrowType(it->type));
        break;
      default:
        if (it->op < BasicOp::Sqrt) {
          if (it->type != atype::BOOL) {
            leftNode = insertUpcastNode(leftNode, it->left.type);
            rightNode = insertUpcastNode(rightNode, it->right.type);
          } else if (it->op == BasicOp::Equal || it->op == BasicOp::NotEqual) {
            insertEqualizeUpcastNode(leftNode, rightNode, it->left.type, it->right.type);
          }
          temp_node = gandiva::TreeExprBuilder::MakeFunction(basicOperationsMap[it->op], {leftNode, rightNode}, concreteArrowType(it->type));
        } else {
          leftNode = insertUpcastNode(leftNode, it->left.type);
          temp_node = gandiva::TreeExprBuilder::MakeFunction(basicOperationsMap[it->op], {leftNode}, concreteArrowType(it->type));
        }
        break;
    }
    if (it->index == 0) {
      tree = temp_node;
    } else {
      auto subtree = subtrees.find(it->index);
      if (subtree == subtrees.end()) {
        subtrees.insert({it->index, temp_node});
      } else {
        subtree->second = temp_node;
      }
    }
    opNodes[std::get<size_t>(it->result.datum)] = temp_node;
  }

  return tree;
}

bool isTableCompatible(std::set<size_t> const& hashes, Operations const& specs)
{
  std::set<size_t> opHashes;
  for (auto const& spec : specs) {
    if (spec.left.datum.index() == 3) {
      opHashes.insert(spec.left.hash);
    }
    if (spec.right.datum.index() == 3) {
      opHashes.insert(spec.right.hash);
    }
  }

  return std::includes(hashes.begin(), hashes.end(),
                       opHashes.begin(), opHashes.end());
}

bool isSchemaCompatible(gandiva::SchemaPtr const& Schema, Operations const& opSpecs)
{
  std::set<std::string> opFieldNames;
  for (auto const& spec : opSpecs) {
    if (spec.left.datum.index() == 3) {
      opFieldNames.insert(std::get<std::string>(spec.left.datum));
    }
    if (spec.right.datum.index() == 3) {
      opFieldNames.insert(std::get<std::string>(spec.right.datum));
    }
  }

  std::set<std::string> schemaFieldNames;
  for (auto const& field : Schema->fields()) {
    schemaFieldNames.insert(field->name());
  }

  return std::includes(schemaFieldNames.begin(), schemaFieldNames.end(),
                       opFieldNames.begin(), opFieldNames.end());
}

void updateExpressionInfos(expressions::Filter const& filter, std::vector<ExpressionInfo>& eInfos)
{
  if (eInfos.empty()) {
    throw runtime_error("Empty expression info vector.");
  }
  Operations ops = createOperations(filter);
  for (auto& info : eInfos) {
    if (isTableCompatible(info.hashes, ops)) {
      auto tree = createExpressionTree(ops, info.schema);
      /// If the tree is already set, add a new tree to it with logical 'and'
      if (info.tree != nullptr) {
        info.tree = gandiva::TreeExprBuilder::MakeAnd({info.tree, tree});
      } else {
        info.tree = tree;
      }
    }
  }
}

} // namespace o2::framework::expressions
