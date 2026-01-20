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
#ifndef O2_FRAMEWORK_EXPRESSIONS_H_
#define O2_FRAMEWORK_EXPRESSIONS_H_

#include "Framework/BasicOps.h"
#include "Framework/Pack.h"
#include "Framework/Configurable.h"
#include "Framework/Variant.h"
#include "Framework/InitContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonConstants/MathConstants.h"
#include <arrow/type_fwd.h>
#include <gandiva/gandiva_aliases.h>
#include <arrow/type.h>
#include <gandiva/arrow.h>
#if !defined(__CLING__) && !defined(__ROOTCLING__)
#include <arrow/table.h>
#include <gandiva/selection_vector.h>
#include <gandiva/node.h>
#include <gandiva/filter.h>
#include <gandiva/projector.h>
#else
namespace gandiva
{
class SelectionVector;
class Filter;
class Projector;
} // namespace gandiva
#endif
#include <variant>
#include <string>
#include <memory>
#include <set>
#include <stack>
namespace gandiva
{
using Selection = std::shared_ptr<gandiva::SelectionVector>;
using FilterPtr = std::shared_ptr<gandiva::Filter>;
} // namespace gandiva

using atype = arrow::Type;
struct ExpressionInfo {
  ExpressionInfo(int ai, size_t hash, std::set<uint32_t>&& hs, gandiva::SchemaPtr sc)
    : argumentIndex(ai),
      processHash(hash),
      hashes(hs),
      schema(sc)
  {
  }
  int argumentIndex;
  size_t processHash;
  std::set<uint32_t> hashes;
  gandiva::SchemaPtr schema;
  gandiva::NodePtr tree = nullptr;
  gandiva::FilterPtr filter = nullptr;
  gandiva::Selection selection = nullptr;
  bool resetSelection = false;
};

namespace o2::framework::expressions
{
void unknownParameterUsed(const char* name);
const char* stringType(atype::type t);

template <typename... T>
struct LiteralStorage {
  using stored_type = std::variant<T...>;
  using stored_pack = framework::pack<T...>;
};

using LiteralValue = LiteralStorage<int, bool, float, double, uint8_t, int64_t, int16_t, uint16_t, int8_t, uint32_t, uint64_t>;

template <typename T>
constexpr auto selectArrowType()
{
  return atype::NA;
}

#define SELECT_ARROW_TYPE(_Ctype_, _Atype_) \
  template <typename T>                     \
    requires std::same_as<T, _Ctype_>       \
  constexpr auto selectArrowType()          \
  {                                         \
    return atype::_Atype_;                  \
  }

SELECT_ARROW_TYPE(bool, BOOL);
SELECT_ARROW_TYPE(float, FLOAT);
SELECT_ARROW_TYPE(double, DOUBLE);
SELECT_ARROW_TYPE(uint8_t, UINT8);
SELECT_ARROW_TYPE(int8_t, INT8);
SELECT_ARROW_TYPE(uint16_t, UINT16);
SELECT_ARROW_TYPE(int16_t, INT16);
SELECT_ARROW_TYPE(uint32_t, UINT32);
SELECT_ARROW_TYPE(int32_t, INT32);
SELECT_ARROW_TYPE(uint64_t, UINT64);
SELECT_ARROW_TYPE(int64_t, INT64);

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type);
std::string upcastTo(atype::type f);

/// An expression tree node corresponding to a literal value
struct LiteralNode {
  using var_t = LiteralValue::stored_type;

  LiteralNode()
    : value{-1},
      type{atype::INT32}
  {
  }
  template <typename T>
  LiteralNode(T v) : value{v}, type{selectArrowType<T>()}
  {
  }

  LiteralNode(var_t v, atype::type t)
    : value{v},
      type{t}
  {
  }

  var_t value;
  atype::type type = atype::NA;
};

/// An expression tree node corresponding to a column binding
struct BindingNode {
  constexpr BindingNode()
    : name{nullptr},
      hash{0},
      type{atype::FLOAT}
  {
  }
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  constexpr BindingNode(const char* name_, uint32_t hash_, atype::type type_) : name{name_}, hash{hash_}, type{type_} {}
  constexpr BindingNode(uint32_t hash_, atype::type type_) : name{nullptr}, hash{hash_}, type{type_} {}
  const char* name;
  uint32_t hash;
  atype::type type;
};

/// An expression tree node corresponding to binary or unary operation
struct OpNode {
  OpNode() : op{BasicOp::Abs} {}
  OpNode(BasicOp op_) : op{op_} {}
  BasicOp op;
};

/// A placeholder node for simple type configurable
struct PlaceholderNode : LiteralNode {
  template <typename T>
    requires(variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown)
  PlaceholderNode(Configurable<T> const& v) : LiteralNode{v.value}, name{v.name}
  {
    retrieve = [](InitContext& context, char const* name) { return LiteralNode::var_t{context.options().get<T>(name)}; };
  }

  template <typename T, typename AT>
    requires((std::convertible_to<T, AT>) && (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown))
  PlaceholderNode(Configurable<T> const& v, AT*) : LiteralNode{static_cast<AT>(v.value)}, name{v.name}
  {
    retrieve = [](InitContext& context, char const* name) { return LiteralNode::var_t{static_cast<AT>(context.options().get<T>(name))}; };
  }

  template <typename T>
  PlaceholderNode(T defaultValue, std::string&& path)
    : LiteralNode{defaultValue},
      stored_name{path},
      name{stored_name}
  {
    retrieve = [](InitContext& context, char const* name) { return LiteralNode::var_t{context.options().get<T>(name)}; };
  }

  void reset(InitContext& context)
  {
    value = retrieve(context, stored_name.empty() ? name.data() : stored_name.data());
  }

  std::string stored_name;
  std::string const& name;
  LiteralNode::var_t (*retrieve)(InitContext&, char const*);
};

template <typename AT, typename T>
PlaceholderNode as(Configurable<T> const& v)
{
  return PlaceholderNode(v, (AT*)nullptr);
}

/// A placeholder node for parameters taken from an array
struct ParameterNode : LiteralNode {
  ParameterNode(int index_ = -1)
    : LiteralNode((float)0),
      index{index_}
  {
  }

  template <typename T>
  void reset(T value_, int index_ = -1)
  {
    (*static_cast<LiteralNode*>(this)) = LiteralNode(value_);
    if (index_ > 0) {
      index = index_;
    }
  }

  int index;
};

/// A conditional node
struct ConditionalNode {
};

/// concepts
template <typename T>
concept is_literal_like = std::same_as<T, LiteralNode> || std::same_as<T, PlaceholderNode> || std::same_as<T, ParameterNode>;

template <typename T>
concept is_binding = std::same_as<T, BindingNode>;

template <typename T>
concept is_operation = std::same_as<T, OpNode>;

template <typename T>
concept is_conditional = std::same_as<T, ConditionalNode>;

/// A generic tree node
struct Node {
  Node(LiteralNode&& v) : self{std::forward<LiteralNode>(v)}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(PlaceholderNode&& v) : self{std::forward<PlaceholderNode>(v)}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(Node&& n) : self{std::forward<self_t>(n.self)}, left{std::forward<std::unique_ptr<Node>>(n.left)}, right{std::forward<std::unique_ptr<Node>>(n.right)}, condition{std::forward<std::unique_ptr<Node>>(n.condition)}, binding{std::forward<std::string>(n.binding)}
  {
  }

  Node(BindingNode const& n) : self{n}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(BindingNode const& n, std::string binding_) : self{n}, left{nullptr}, right{nullptr}, condition{nullptr}, binding{binding_}
  {
    get<BindingNode>(self).name = binding.c_str();
  }

  Node(ParameterNode&& p) : self{std::forward<ParameterNode>(p)}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(ConditionalNode op, Node&& then_, Node&& else_, Node&& condition_)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(then_))},
      right{std::make_unique<Node>(std::forward<Node>(else_))},
      condition{std::make_unique<Node>(std::forward<Node>(condition_))} {}

  Node(ConditionalNode op, Node&& then_, std::unique_ptr<Node>&& else_, Node&& condition_)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(then_))},
      right{std::forward<std::unique_ptr<Node>>(else_)},
      condition{std::make_unique<Node>(std::forward<Node>(condition_))} {}

  Node(OpNode op, Node&& l, Node&& r)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(l))},
      right{std::make_unique<Node>(std::forward<Node>(r))},
      condition{nullptr} {}

  Node(OpNode op, std::unique_ptr<Node>&& l, Node&& r)
    : self{op},
      left{std::forward<std::unique_ptr<Node>>(l)},
      right{std::make_unique<Node>(std::forward<Node>(r))},
      condition{nullptr} {}

  Node(OpNode op, Node&& l)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(l))},
      right{nullptr},
      condition{nullptr} {}

  Node(Node const& other)
    : self{other.self},
      index{other.index}
  {
    if (other.left != nullptr) {
      left = std::make_unique<Node>(*other.left);
    }
    if (other.right != nullptr) {
      right = std::make_unique<Node>(*other.right);
    }
    if (other.condition != nullptr) {
      condition = std::make_unique<Node>(*other.condition);
    }
    binding = other.binding;
    if (!binding.empty()) {
      get<BindingNode>(self).name = binding.c_str();
    }
  }

  /// variant with possible nodes
  using self_t = std::variant<LiteralNode, BindingNode, OpNode, PlaceholderNode, ConditionalNode, ParameterNode>;
  self_t self;
  size_t index = 0;
  /// pointers to children
  std::unique_ptr<Node> left = nullptr;
  std::unique_ptr<Node> right = nullptr;
  std::unique_ptr<Node> condition = nullptr;

  /// buffer for dynamic binding
  std::string binding;
};

/// helper struct used to parse trees
struct NodeRecord {
  /// pointer to the actual tree node
  Node* node_ptr = nullptr;
  size_t index = 0;
  explicit NodeRecord(Node* node_, size_t index_) : node_ptr(node_), index{index_} {}
  bool operator!=(NodeRecord const& rhs)
  {
    return this->node_ptr != rhs.node_ptr;
  }
};

/// Tree-walker helper
template <typename L>
void walk(Node* head, L&& pred)
{
  std::stack<NodeRecord> path;
  path.emplace(head, 0);
  while (!path.empty()) {
    auto& top = path.top();
    pred(top.node_ptr);

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

/// helper concepts
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

/// overloaded operators to build the tree from an expression

#define BINARY_OP_NODES(_operator_, _operation_)                                                    \
  inline Node operator _operator_(Node&& left, Node&& right)                                        \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), std::forward<Node>(right)}; \
  }                                                                                                 \
  template <arithmetic T>                                                                           \
  inline Node operator _operator_(Node&& left, T right)                                             \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), LiteralNode{right}};        \
  }                                                                                                 \
  template <arithmetic T>                                                                           \
  inline Node operator _operator_(T left, Node&& right)                                             \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, LiteralNode{left}, std::forward<Node>(right)};        \
  }                                                                                                 \
  template <typename T>                                                                             \
  inline Node operator _operator_(Node&& left, Configurable<T> const& right)                        \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), PlaceholderNode{right}};    \
  }                                                                                                 \
  template <typename T>                                                                             \
  inline Node operator _operator_(Configurable<T> const& left, Node&& right)                        \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, std::forward<Node>(right)};    \
  }                                                                                                 \
  inline Node operator _operator_(BindingNode const& left, BindingNode const& right)                \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, left, right};                                         \
  }                                                                                                 \
  inline Node operator _operator_(BindingNode const& left, Node&& right)                            \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, left, std::forward<Node>(right)};                     \
  }                                                                                                 \
  inline Node operator _operator_(Node&& left, BindingNode const& right)                            \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), right};                     \
  }                                                                                                 \
  template <typename T>                                                                             \
  inline Node operator _operator_(Configurable<T> const& left, BindingNode const& right)            \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, right};                        \
  }                                                                                                 \
  template <typename T>                                                                             \
  inline Node operator _operator_(BindingNode const& left, Configurable<T> const& right)            \
  {                                                                                                 \
    return Node{OpNode{BasicOp::_operation_}, left, PlaceholderNode{right}};                        \
  }

BINARY_OP_NODES(&, BitwiseAnd);
BINARY_OP_NODES(^, BitwiseXor);
BINARY_OP_NODES(|, BitwiseOr);
BINARY_OP_NODES(+, Addition);
BINARY_OP_NODES(-, Subtraction);
BINARY_OP_NODES(*, Multiplication);
BINARY_OP_NODES(/, Division);
BINARY_OP_NODES(>, GreaterThan);
BINARY_OP_NODES(>=, GreaterThanOrEqual);
BINARY_OP_NODES(<, LessThan);
BINARY_OP_NODES(<=, LessThanOrEqual);
BINARY_OP_NODES(==, Equal);
BINARY_OP_NODES(!=, NotEqual);
BINARY_OP_NODES(&&, LogicalAnd);
BINARY_OP_NODES(||, LogicalOr);

/// functions
template <arithmetic T>
inline Node npow(Node&& left, T right)
{
  return Node{OpNode{BasicOp::Power}, std::forward<Node>(left), LiteralNode{right}};
}

#define BINARY_FUNC_NODES(_func_, _node_)                                                      \
  template <arithmetic L, arithmetic R>                                                        \
  inline Node _node_(L left, R right)                                                          \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, LiteralNode{left}, LiteralNode{right}};               \
  }                                                                                            \
                                                                                               \
  inline Node _node_(Node&& left, Node&& right)                                                \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), std::forward<Node>(right)}; \
  }                                                                                            \
                                                                                               \
  inline Node _node_(Node&& left, BindingNode const& right)                                    \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), right};                     \
  }                                                                                            \
                                                                                               \
  inline Node _node_(BindingNode const& left, BindingNode const& right)                        \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, left, right};                                         \
  }                                                                                            \
                                                                                               \
  inline Node _node_(BindingNode const& left, Node&& right)                                    \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, left, std::forward<Node>(right)};                     \
  }                                                                                            \
                                                                                               \
  template <typename T>                                                                        \
  inline Node _node_(Node&& left, Configurable<T> const& right)                                \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), PlaceholderNode{right}};    \
  }                                                                                            \
                                                                                               \
  template <typename T>                                                                        \
  inline Node _node_(Configurable<T> const& left, Node&& right)                                \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, std::forward<Node>(right)};    \
  }                                                                                            \
                                                                                               \
  template <typename T>                                                                        \
  inline Node _node_(BindingNode const& left, Configurable<T> const& right)                    \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, left, PlaceholderNode{right}};                        \
  }                                                                                            \
                                                                                               \
  template <typename T>                                                                        \
  inline Node _node_(Configurable<T> const& left, BindingNode const& right)                    \
  {                                                                                            \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, right};                        \
  }

BINARY_FUNC_NODES(Atan2, natan2);
#define ncheckbit(_node_, _bit_) ((_node_ & _bit_) == _bit_)

/// unary functions on nodes
#define UNARY_FUNC_NODES(_func_, _node_)                           \
  inline Node _node_(Node&& arg)                                   \
  {                                                                \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(arg)}; \
  }

UNARY_FUNC_NODES(Round, nround);
UNARY_FUNC_NODES(Sqrt, nsqrt);
UNARY_FUNC_NODES(Exp, nexp);
UNARY_FUNC_NODES(Log, nlog);
UNARY_FUNC_NODES(Log10, nlog10);
UNARY_FUNC_NODES(Abs, nabs);
UNARY_FUNC_NODES(Sin, nsin);
UNARY_FUNC_NODES(Cos, ncos);
UNARY_FUNC_NODES(Tan, ntan);
UNARY_FUNC_NODES(Asin, nasin);
UNARY_FUNC_NODES(Acos, nacos);
UNARY_FUNC_NODES(Atan, natan);
UNARY_FUNC_NODES(BitwiseNot, nbitwise_not);

/// conditionals
inline Node ifnode(Node&& condition_, Node&& then_, Node&& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <arithmetic L>
inline Node ifnode(Node&& condition_, Node&& then_, L else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), LiteralNode{else_}, std::forward<Node>(condition_)};
}

template <arithmetic L>
inline Node ifnode(Node&& condition_, L then_, Node&& else_)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <arithmetic L1, arithmetic L2>
inline Node ifnode(Node&& condition_, L1 then_, L2 else_)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, LiteralNode{else_}, std::forward<Node>(condition_)};
}

template <typename T>
inline Node ifnode(Configurable<T> const& condition_, Node&& then_, Node&& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), std::forward<Node>(else_), PlaceholderNode{condition_}};
}

template <typename L>
inline Node ifnode(Node&& condition_, Node&& then_, Configurable<L> const& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), PlaceholderNode{else_}, std::forward<Node>(condition_)};
}

template <typename L>
inline Node ifnode(Node&& condition_, Configurable<L> const& then_, Node&& else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <typename L1, typename L2>
inline Node ifnode(Node&& condition_, Configurable<L1> const& then_, Configurable<L2> const& else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, PlaceholderNode{else_}, std::forward<Node>(condition_)};
}

/// Parameters
inline Node par(int index)
{
  return Node{ParameterNode{index}};
}

/// binned functional
template <typename T>
inline Node binned(std::vector<T> const& binning, std::vector<T> const& parameters, Node&& binned, Node&& pexp, Node&& out)
{
  int bins = binning.size() - 1;
  const auto binned_copy = binned;
  const auto out_copy = out;
  auto root = ifnode(Node{binned_copy} < binning[0], Node{out_copy}, LiteralNode{-1});
  auto* current = &root;
  for (auto i = 0; i < bins; ++i) {
    current->right = std::make_unique<Node>(ifnode(Node{binned_copy} < binning[i + 1], updateParameters(pexp, bins, parameters, i), LiteralNode{-1}));
    current = current->right.get();
  }
  current->right = std::make_unique<Node>(out);
  return root;
}

template <typename T>
inline Node updateParameters(Node const& pexp, int bins, std::vector<T> const& parameters, int bin)
{
  Node result{pexp};
  walk(&result, [&bins, &parameters, &bin](Node* node) {
    if (node->self.index() == 5) {
      auto* n = std::get_if<5>(&node->self);
      n->reset(parameters[n->index * bins + bin]);
    }
  });
  return result;
}

/// clamping functional
template <typename T>
inline Node clamp(Node&& expr, T low, T hi)
{
  auto copy = expr;
  return ifnode(Node{copy} < LiteralNode{low}, LiteralNode{low}, ifnode(Node{copy} > LiteralNode{hi}, LiteralNode{hi}, Node{copy}));
}

/// division by 0 protector
inline Node protect0(Node&& expr)
{
  auto copy = expr;
  return ifnode(nabs(Node{copy}) < o2::constants::math::Almost0, o2::constants::math::Almost0, Node{copy});
}

/// context-independent configurable
template <typename T>
inline Node ncfg(T defaultValue, std::string path)
{
  return PlaceholderNode(defaultValue, path);
}

/// A struct, containing the root of the expression tree
struct Filter {
  Filter() = default;

  Filter(std::unique_ptr<Node>&& ptr)
  {
    node = std::move(ptr);
    (void)designateSubtrees(node.get());
  }

  Filter(Node&& node_) : node{std::make_unique<Node>(std::forward<Node>(node_))}
  {
    (void)designateSubtrees(node.get());
  }

  Filter(Filter&& other) : node{std::forward<std::unique_ptr<Node>>(other.node)}
  {
  }

  Filter(std::string const& input_) : input{input_} {}

  Filter& operator=(Filter&& other) noexcept
  {
    node = std::move(other.node);
    input = std::move(other.input);
    return *this;
  }

  Filter& operator=(std::string const& input_)
  {
    input = input_;
    if (node != nullptr) {
      node = nullptr;
    }
    return *this;
  }

  std::unique_ptr<Node> node = nullptr;
  std::string input;

  size_t designateSubtrees(Node* node, size_t index = 0);
  void parse();
};

template <typename T>
concept is_filter = std::same_as<T, Filter>;

using Projector = Filter;

/// Function for creating gandiva selection from our internal filter tree
gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table, Filter const& expression);
/// Function for creating gandiva selection from prepared gandiva expressions tree
gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table, std::shared_ptr<gandiva::Filter> const& gfilter);

struct ColumnOperationSpec;
using Operations = std::vector<ColumnOperationSpec>;

/// Function to create an internal operation sequence from a filter tree
Operations createOperations(Filter const& expression);

/// Function to check compatibility of a given arrow schema with operation sequence
bool isTableCompatible(std::set<uint32_t> const& hashes, Operations const& specs);
/// Function to create gandiva expression tree from operation sequence
gandiva::NodePtr createExpressionTree(Operations const& opSpecs,
                                      gandiva::SchemaPtr const& Schema);
/// Function to create gandiva filter from gandiva condition
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              gandiva::ConditionPtr condition);
/// Function to create gandiva filter from operation sequence
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              Operations const& opSpecs);
/// Function to create gandiva projector from operation sequence
std::shared_ptr<gandiva::Projector> createProjector(gandiva::SchemaPtr const& Schema,
                                                    Operations const& opSpecs,
                                                    gandiva::FieldPtr result);
/// Function to create gandiva projector directly from expression
std::shared_ptr<gandiva::Projector> createProjector(gandiva::SchemaPtr const& Schema,
                                                    Projector&& p,
                                                    gandiva::FieldPtr result);
/// Function for attaching gandiva filters to to compatible task inputs
void updateExpressionInfos(expressions::Filter const& filter, std::vector<ExpressionInfo>& eInfos);
/// Function to create gandiva condition expression from generic gandiva expression tree
gandiva::ConditionPtr makeCondition(gandiva::NodePtr node);
/// Function to create gandiva projecting expression from generic gandiva expression tree
gandiva::ExpressionPtr makeExpression(gandiva::NodePtr node, gandiva::FieldPtr result);
/// Update placeholder nodes from context
void updatePlaceholders(Filter& filter, InitContext& context);

std::shared_ptr<gandiva::Projector> createProjectorHelper(size_t nColumns, expressions::Projector* projectors,
                                                          std::shared_ptr<arrow::Schema> schema,
                                                          std::vector<std::shared_ptr<arrow::Field>> const& fields);

std::vector<std::shared_ptr<gandiva::Expression>> materializeProjectors(std::vector<expressions::Projector> const& projectors, std::shared_ptr<arrow::Schema> const& inputSchema, std::vector<std::shared_ptr<arrow::Field>> const& outputFields);

template <typename... C>
std::shared_ptr<gandiva::Projector> createProjectors(framework::pack<C...>, std::vector<std::shared_ptr<arrow::Field>> const& fields, gandiva::SchemaPtr schema)
{
  std::array<expressions::Projector, sizeof...(C)> projectors{{std::move(C::Projector())...}};

  return createProjectorHelper(sizeof...(C), projectors.data(), schema, fields);
}

void updateFilterInfo(ExpressionInfo& info, std::shared_ptr<arrow::Table>& table);

/*
 * The formal grammar for framework expressions.
 * Operations are in the order of increasing priority.
 * Identifier includes namespaces, e.g. o2::aod::track::pt.
 *
 * top ::= primary
 *
 * primary ::= tier1 ('||' tier1)*
 * tier1 ::= tier2 ('&&' tier2)*
 * tier2 ::= tier3 ('|' tier3)*
 * tier3 ::= tier4 ('^' tier4)*
 * tier4 ::= tier5 ('&' tier5)*
 * tier5 ::= tier6 (('=='|'!=') tier6)*
 * tier6 ::= tier7 (('<'|'>'|'<='|'>=') tier7)*
 * tier7 ::= tier8 (('+'|'-') tier8)*
 * tier8 ::= base (('*'|'/') base)*
 *
 * base ::= identifier
 *  | number
 *  | function_call
 *  | '(' primary ')'
 *
 * number ::= -?[0-9]+(\.[0-9]*)?([uf])?
 * identifier ::= [a-zA-Z][a-zA-Z0-9_]* ('::' [a-zA-Z][a-zA-Z0-9_]*)*
 * function_call ::= identifier '(' (primary (',' primary)*)? ')'
 */

/// String parsing
enum Token : int {
  EoL = -1,
  Identifier = -2,
  IntegerNumber = -3,
  FloatNumber = -4,
  BinaryOp = -5,
  Unexpected = -100
};

struct Tokenizer {
  std::string source;
  std::string::iterator current;
  std::string IdentifierStr;
  std::string BinaryOpStr;
  std::string StrValue;
  std::string TokenStr;
  std::variant<uint32_t, int32_t, uint64_t, int64_t> IntegerValue;
  std::variant<float, double> FloatValue;
  char LastChar;
  int currentToken = Token::Unexpected;

  Tokenizer(std::string const& input = "");
  void reset(std::string const& input);
  [[maybe_unused]] int nextToken();
  void pop();
  char peek();
};

struct Parser {
  static Node parse(std::string const& input);
  static std::unique_ptr<Node> parsePrimary(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier1(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier2(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier3(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier4(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier5(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier6(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier7(Tokenizer& tk);
  static std::unique_ptr<Node> parseTier8(Tokenizer& tk);
  static std::unique_ptr<Node> parseBase(Tokenizer& tk);

  static OpNode opFromToken(std::string const& token);
};

} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
