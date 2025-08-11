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
#include "CommonConstants/MathConstants.h"

using namespace o2::framework;

namespace o2::framework::expressions
{
void unknownParameterUsed(const char* name)
{
  runtime_error_f("Unknown parameter used in expression: %s", name);
}

/// a map between BasicOp and tokens in string expressions
constexpr std::array<std::string_view, BasicOp::Conditional + 1> mapping{
  "&&",
  "||",
  "+",
  "-",
  "/",
  "*",
  "&",
  "|",
  "^",
  "<",
  "<=",
  ">",
  ">=",
  "==",
  "!=",
  "natan2",
  "npow",
  "nsqrt",
  "nexp",
  "nlog",
  "nlog10",
  "nsin",
  "ncos",
  "ntan",
  "nasin",
  "nacos",
  "natan",
  "nabs",
  "nround",
  "nbitwise_not",
  "ifnode"};

/// math constants to recognize in string expressions
constexpr std::array<std::string_view, 9> mathConstants{
  "Almost0",
  "Epsilon",
  "Almost1",
  "VeryBig",
  "PI",
  "TwoPI",
  "PIHalf",
  "PIThird",
  "PIQuarter"};

/// values of math constants to substiture
constexpr std::array<float, 9> mathConstantsValues{
  o2::constants::math::Almost0,
  o2::constants::math::Epsilon,
  o2::constants::math::Almost1,
  o2::constants::math::VeryBig,
  o2::constants::math::PI,
  o2::constants::math::TwoPI,
  o2::constants::math::PIHalf,
  o2::constants::math::PIThird,
  o2::constants::math::PIQuarter};

/// a map between BasicOp and gandiva node definitions
/// note that logical 'and' and 'or' are created separately
constexpr std::array<const char*, BasicOp::Conditional + 1> basicOperationsMap = {
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
    auto top = path.top();
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

void Filter::parse()
{
  node = std::make_unique<Node>(Parser::parse(input));
  (void)designateSubtrees(node.get());
}

template <typename T>
constexpr inline auto makeDatum(T const&)
{
  return DatumSpec{};
}

template <is_literal_like T>
constexpr inline auto makeDatum(T const& node)
{
  return DatumSpec{node.value, node.type};
}

template <is_binding T>
constexpr inline auto makeDatum(T const& node)
{
  return DatumSpec{node.name, node.hash, node.type};
}

template <typename T>
constexpr inline auto makeOp(T const&, size_t const&)
{
  return ColumnOperationSpec{};
}

template <is_operation T>
constexpr inline auto makeOp(T const& node, size_t const& index)
{
  return ColumnOperationSpec{node.op, index};
}

template <is_conditional T>
constexpr inline auto makeOp(T const&, size_t const& index)
{
  return ColumnOperationSpec{BasicOp::Conditional, index};
}

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
      throw runtime_error_f("Do not know how to cast to %s", stringType(f));
  }
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
  if (filter.node == nullptr && !filter.input.empty()) {
    filter.parse();
  }
  expressions::walk(filter.node.get(), [&](Node* node) {
    if (node->self.index() == 3) {
      std::get_if<3>(&node->self)->reset(context);
    }
  });
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
      [](auto const& n) { return makeDatum(n); },
      node->self);
  };

  size_t index = 0;
  // insert the top node into stack
  path.emplace(expression.node.get(), index++);

  // while the stack is not empty
  while (!path.empty()) {
    auto top = path.top();

    // create operation spec, pop the node and add its children
    auto operationSpec =
      std::visit(
        [&](auto const& n) { return makeOp(n, top.node_ptr->index); },
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

  auto inferResultType = [&resultTypes](BasicOp op, DatumSpec& left, DatumSpec& right) {
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

    auto isBitwiseOp = [](auto o) {
      return ((o == BasicOp::BitwiseAnd) || (o == BasicOp::BitwiseNot) || (o == BasicOp::BitwiseOr) || (o == BasicOp::BitwiseXor));
    };

    if (isIntType(t1)) {
      if (t2 == atype::FLOAT && !isBitwiseOp(op)) {
        return atype::FLOAT;
      }
      if (t2 == atype::DOUBLE && !isBitwiseOp(op)) {
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
      if (isIntType(t2) && !isBitwiseOp(op)) {
        return atype::FLOAT;
      }
      if (t2 == atype::DOUBLE) {
        return atype::DOUBLE;
      }
    }
    if (t1 == atype::DOUBLE) {
      return atype::DOUBLE;
    }

    if (isIntType(t1) && isBitwiseOp(op)) {
      return t1;
    }
    if (isIntType(t2) && isBitwiseOp(op)) {
      return t2;
    }

    throw runtime_error_f("Invalid combination of argument types %s and %s", stringType(t1), stringType(t2));
  };

  for (auto it = OperationSpecs.rbegin(); it != OperationSpecs.rend(); ++it) {
    auto type = inferResultType(it->op, it->left, it->right);
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
                                 condition,
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

  auto insertUpcastNode = [](gandiva::NodePtr node, atype::type t0, atype::type t) {
    if (t != t0) {
      auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(t0), {node}, concreteArrowType(t0));
      node = upcast;
    }
    return node;
  };

  auto insertEqualizeUpcastNode = [](gandiva::NodePtr& node1, gandiva::NodePtr& node2, atype::type t1, atype::type t2) {
    if (t2 > t1) {
      auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(t2), {node1}, concreteArrowType(t2));
      node1 = upcast;
    } else if (t1 > t2) {
      auto upcast = gandiva::TreeExprBuilder::MakeFunction(upcastTo(t1), {node2}, concreteArrowType(t1));
      node2 = upcast;
    }
  };

  auto isBitwiseOp = [](auto o) {
    return ((o == BasicOp::BitwiseAnd) || (o == BasicOp::BitwiseNot) || (o == BasicOp::BitwiseOr) || (o == BasicOp::BitwiseXor));
  };

  gandiva::NodePtr tree = nullptr;
  for (auto it = opSpecs.rbegin(); it != opSpecs.rend(); ++it) {
    auto leftNode = datumNode(it->left);
    auto rightNode = datumNode(it->right);
    auto condNode = datumNode(it->condition);

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
          if (it->type != atype::BOOL && !isBitwiseOp(it->op)) {
            leftNode = insertUpcastNode(leftNode, it->type, it->left.type);
            rightNode = insertUpcastNode(rightNode, it->type, it->right.type);
          } else if (it->op == BasicOp::Equal || it->op == BasicOp::NotEqual) {
            insertEqualizeUpcastNode(leftNode, rightNode, it->left.type, it->right.type);
          }
          temp_node = gandiva::TreeExprBuilder::MakeFunction(basicOperationsMap[it->op], {leftNode, rightNode}, concreteArrowType(it->type));
        } else {
          if (!isBitwiseOp(it->op)) {
            leftNode = insertUpcastNode(leftNode, it->type, it->left.type);
          }
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

bool isTableCompatible(std::set<uint32_t> const& hashes, Operations const& specs)
{
  std::set<uint32_t> opHashes;
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

void updateFilterInfo(ExpressionInfo& info, std::shared_ptr<arrow::Table>& table)
{
  if (info.tree != nullptr && info.filter == nullptr) {
    info.filter = framework::expressions::createFilter(table->schema(), framework::expressions::makeCondition(info.tree));
  }
  if (info.tree != nullptr && info.filter != nullptr && info.resetSelection == true) {
    info.selection = framework::expressions::createSelection(table, info.filter);
    info.resetSelection = false;
  }
}

/// String parsing
Tokenizer::Tokenizer(std::string const& input)
  : source{input},
    IdentifierStr{""},
    StrValue{""},
    IntegerValue{0},
    FloatValue{0.f}
{
  LastChar = ' ';
  if (!source.empty()) {
    source.erase(std::remove_if(source.begin(), source.end(), ::isspace), source.end());
  }
  current = source.begin();
}

void Tokenizer::reset(std::string const& input)
{
  LastChar = ' ';
  IdentifierStr = "";
  StrValue = "";
  IntegerValue = 0;
  FloatValue = 0.f;
  source = input;
  if (!source.empty()) {
    source.erase(std::remove_if(source.begin(), source.end(), ::isspace), source.end());
  }
  current = source.begin();
  currentToken = Token::Unexpected;
}

int Tokenizer::nextToken()
{
  // skip initial space
  if (isspace(LastChar)) {
    pop();
  }
  // logical or bitwise OR
  if (LastChar == '|') {
    BinaryOpStr = LastChar;
    if (peek() == '|') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    }
  }
  // logical or bitwise AND
  if (LastChar == '&') {
    BinaryOpStr = LastChar;
    if (peek() == '&') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    }
  }
  // less than or less or equal than
  if (LastChar == '<') {
    BinaryOpStr = LastChar;
    if (peek() == '=') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    }
  }
  // greater than or greater or equal than
  if (LastChar == '>') {
    BinaryOpStr = LastChar;
    if (peek() == '=') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    }
  }
  // equal or error
  if (LastChar == '=') {
    BinaryOpStr = LastChar;
    if (peek() == '=') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::Unexpected;
      return currentToken;
    }
  }
  // not equal or error
  if (LastChar == '!') {
    BinaryOpStr = LastChar;
    if (peek() == '=') {
      pop();
      BinaryOpStr += LastChar;
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    } else {
      pop();
      TokenStr = BinaryOpStr;
      currentToken = Token::BinaryOp;
      return currentToken;
    }
  }
  // unambiguous single-character binary operations: addition, multiplication, subtraction, division, bitwise XOR
  if (LastChar == '+' || LastChar == '*' || (LastChar == '-' && (currentToken != Token::BinaryOp && currentToken != '(' && currentToken != Token::Unexpected)) || LastChar == '/' || LastChar == '^') {
    BinaryOpStr = LastChar;
    pop();
    TokenStr = BinaryOpStr;
    currentToken = Token::BinaryOp;
    return currentToken;
  }
  // identifier: column, function, constant
  if (isalpha(LastChar)) {
    // identifier
    IdentifierStr = LastChar;
    pop();
    while (isalnum(LastChar) || (LastChar == '_') || (LastChar == ':')) {
      IdentifierStr += LastChar;
      pop();
    }
    TokenStr = IdentifierStr;
    currentToken = Token::Identifier;
    return currentToken;
  }
  // number: integer, unsigned integer or float
  if (isdigit(LastChar) || LastChar == '.' || (LastChar == '-' && isdigit(peek()))) {
    // number
    StrValue = "";
    bool isFloat = false;
    bool isUnsigned = false;
    do {
      StrValue += LastChar;
      pop();
    } while (isdigit(LastChar) || LastChar == '.');
    if (LastChar == 'f') {
      isFloat = true;
      pop();
    }
    if (LastChar == 'u') {
      isUnsigned = true;
      pop();
    }
    if (std::find(StrValue.begin(), StrValue.end(), '.') == StrValue.end() && !isFloat) {
      if (!isUnsigned) {
        IntegerValue = atoi(StrValue.c_str());
      } else {
        IntegerValue = static_cast<unsigned int>(atoi(StrValue.c_str()));
      }
      TokenStr = StrValue;
      currentToken = Token::IntegerNumber;
      return currentToken;
    }
    if (isFloat) {
      FloatValue = strtof(StrValue.c_str(), nullptr);
    } else {
      FloatValue = strtod(StrValue.c_str(), nullptr);
    }
    TokenStr = StrValue;
    currentToken = Token::FloatNumber;
    return currentToken;
  }
  // end-of-line
  if (LastChar == '\0') {
    TokenStr = LastChar;
    currentToken = Token::EoL;
    return currentToken;
  }
  // generic character
  currentToken = LastChar;
  TokenStr = LastChar;
  pop();
  return currentToken;
}

void Tokenizer::pop()
{
  if (current != source.end()) {
    LastChar = *current;
    ++current;
  } else {
    LastChar = '\0';
  }
}

char Tokenizer::peek()
{
  if (current != source.end()) {
    return *current;
  } else {
    return '\0';
  }
}

Node Parser::parse(std::string const& input)
{
  auto tk = Tokenizer(input);
  tk.nextToken();
  auto node = parsePrimary(tk);
  if (tk.currentToken != Token::EoL) {
    throw runtime_error_f("Unexpected token after expression: %s", tk.TokenStr.c_str());
  }
  return *node.get();
}

std::unique_ptr<Node> Parser::parsePrimary(Tokenizer& tk)
{
  auto root = parseTier1(tk);
  while (tk.TokenStr == "||") {
    auto opnode = std::make_unique<Node>(OpNode{BasicOp::LogicalOr}, std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier1(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier1(Tokenizer& tk)
{
  auto root = parseTier2(tk);
  while (tk.TokenStr == "&&") {
    auto opnode = std::make_unique<Node>(OpNode{BasicOp::LogicalAnd}, std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier2(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier2(Tokenizer& tk)
{
  auto root = parseTier3(tk);
  while (tk.TokenStr == "|") {
    auto opnode = std::make_unique<Node>(OpNode{BasicOp::BitwiseOr}, std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier3(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier3(Tokenizer& tk)
{
  auto root = parseTier4(tk);
  while (tk.TokenStr == "^") {
    auto opnode = std::make_unique<Node>(OpNode{BasicOp::BitwiseXor}, std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier4(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier4(Tokenizer& tk)
{
  auto root = parseTier5(tk);
  while (tk.TokenStr == "&") {
    auto opnode = std::make_unique<Node>(OpNode{BasicOp::BitwiseAnd}, std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier5(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier5(Tokenizer& tk)
{
  auto root = parseTier6(tk);
  while (tk.TokenStr == "==" || tk.TokenStr == "!=") {
    auto opnode = std::make_unique<Node>(opFromToken(tk.TokenStr), std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier6(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier6(Tokenizer& tk)
{
  auto root = parseTier7(tk);
  while (tk.TokenStr == "<" || tk.TokenStr == "<=" || tk.TokenStr == "=>" || tk.TokenStr == ">") {
    auto opnode = std::make_unique<Node>(opFromToken(tk.TokenStr), std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier7(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier7(Tokenizer& tk)
{
  auto root = parseTier8(tk);
  while (tk.TokenStr == "+" || tk.TokenStr == "-") {
    auto opnode = std::make_unique<Node>(opFromToken(tk.TokenStr), std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseTier8(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseTier8(Tokenizer& tk)
{
  auto root = parseBase(tk);
  while (tk.TokenStr == "*" || tk.TokenStr == "/") {
    auto opnode = std::make_unique<Node>(opFromToken(tk.TokenStr), std::move(root), LiteralNode{-1});
    root.swap(opnode);
    tk.nextToken();
    root->right = parseBase(tk);
  }
  return root;
}

std::unique_ptr<Node> Parser::parseBase(Tokenizer& tk)
{
  // parentheses
  if (tk.currentToken == '(') {
    tk.nextToken();
    auto node = parsePrimary(tk);
    if (tk.currentToken != ')') {
      throw runtime_error_f("Expected \")\" got %s", tk.TokenStr.c_str());
    }
    tk.nextToken();
    return node;
  }

  // identifier or function call
  if (tk.currentToken == Token::Identifier) {
    std::string id = tk.IdentifierStr;
    tk.nextToken();
    if (tk.currentToken != '(') { // binding node or a constant
      std::string binding = id;
      auto posc = std::find(mathConstants.begin(), mathConstants.end(), id);
      if (posc != mathConstants.end()) { // constant
        return std::make_unique<Node>(LiteralNode{mathConstantsValues[std::distance(mathConstants.begin(), posc)]});
      }
      // binding node
      auto pos = binding.rfind(':');
      binding.erase(0, pos + 1);
      binding[0] = std::toupper(binding[0]);
      binding.insert(binding.begin(), 'f');
      return std::make_unique<Node>(BindingNode{runtime_hash(id.c_str()), atype::FLOAT}, binding);
    }

    // function call
    if (id == "ifnode") { // conditional, 3 args
      auto node = std::make_unique<Node>(ConditionalNode{}, LiteralNode{-1}, LiteralNode{-1}, LiteralNode{-1});
      int args = 0;
      while (tk.currentToken != ')') {
        do {
          tk.nextToken();
          if (args == 0) {
            node->condition = parsePrimary(tk);
          } else if (args == 1) {
            node->left = parsePrimary(tk);
          } else if (args == 2) {
            node->right = parsePrimary(tk);
          } else {
            throw runtime_error_f("Extra argument in a conditional: %s", tk.TokenStr.c_str());
          }
          ++args;
        } while (tk.currentToken == ',');
      }
      tk.nextToken();
      return node;
    } else { // normal function
      auto node = std::make_unique<Node>(opFromToken(id), LiteralNode{-1}, LiteralNode{-1});
      int args = 0;
      while (tk.currentToken != ')') {
        do {
          tk.nextToken();
          if (args == 0) {
            node->left = parsePrimary(tk);
          } else if (args == 1) {
            node->right = parsePrimary(tk);
          } else {
            throw runtime_error_f("Extra argument in a function call: %s", tk.TokenStr.c_str());
          }
          ++args;
        } while (tk.currentToken == ',');
      }
      if (args == 1) {
        node->right = nullptr;
      }
      tk.nextToken();
      return node;
    }
  }

  // number
  if (tk.currentToken == Token::FloatNumber) {
    tk.nextToken();
    switch (tk.FloatValue.index()) {
      case 0:
        return std::make_unique<Node>(LiteralNode{get<0>(tk.FloatValue)});
      case 1:
        return std::make_unique<Node>(LiteralNode{get<1>(tk.FloatValue)});
    }
  }
  if (tk.currentToken == Token::IntegerNumber) {
    tk.nextToken();
    switch (tk.IntegerValue.index()) {
      case 0:
        return std::make_unique<Node>(LiteralNode{get<0>(tk.IntegerValue)});
      case 1:
        return std::make_unique<Node>(LiteralNode{get<1>(tk.IntegerValue)});
      case 2:
        return std::make_unique<Node>(LiteralNode{get<2>(tk.IntegerValue)});
      case 3:
        return std::make_unique<Node>(LiteralNode{get<3>(tk.IntegerValue)});
    }
  }

  // error
  throw runtime_error_f("Unexpected token %s in operand", tk.TokenStr.c_str());
}

OpNode Parser::opFromToken(std::string const& token)
{
  auto locate = std::find(mapping.begin(), mapping.end(), token);
  if (locate == mapping.end()) {
    throw runtime_error_f("No operation \"%s\" defined", token.c_str());
  }
  return OpNode{static_cast<BasicOp>(std::distance(mapping.begin(), locate))};
}

} // namespace o2::framework::expressions
