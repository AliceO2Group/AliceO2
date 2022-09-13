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
#include "Framework/CompilerBuiltins.h"
#include "Framework/Pack.h"
#include "Framework/CheckTypes.h"
#include "Framework/Configurable.h"
#include "Framework/Variant.h"
#include "Framework/InitContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RootConfigParamHelpers.h"
#include "Framework/RuntimeError.h"
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
} // namespace gandiva
#endif
#include <variant>
#include <string>
#include <memory>
#include <typeinfo>
#include <set>
namespace gandiva
{
using Selection = std::shared_ptr<gandiva::SelectionVector>;
using FilterPtr = std::shared_ptr<gandiva::Filter>;
} // namespace gandiva

using atype = arrow::Type;
struct ExpressionInfo {
  int argumentIndex;
  size_t processHash;
  std::set<size_t> hashes;
  gandiva::SchemaPtr schema;
  gandiva::NodePtr tree;
  gandiva::FilterPtr filter;
  gandiva::Selection selection;
  bool resetSelection = false;
};

namespace o2::framework::expressions
{
template <typename... T>
struct LiteralStorage {
  using stored_type = std::variant<T...>;
  using stored_pack = framework::pack<T...>;
};

using LiteralValue = LiteralStorage<int, bool, float, double, uint8_t, int64_t, int16_t, uint16_t, int8_t, uint32_t, uint64_t>;

template <typename T>
constexpr auto selectArrowType()
{
  if constexpr (std::is_same_v<T, int>) {
    return atype::INT32;
  } else if constexpr (std::is_same_v<T, bool>) {
    return atype::BOOL;
  } else if constexpr (std::is_same_v<T, float>) {
    return atype::FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return atype::DOUBLE;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return atype::UINT8;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return atype::INT8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return atype::UINT16;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return atype::INT16;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return atype::INT64;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return atype::UINT32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return atype::UINT64;
  } else {
    return atype::NA;
  }
  O2_BUILTIN_UNREACHABLE();
}

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type);
std::string upcastTo(atype::type f);

/// An expression tree node corresponding to a literal value
struct LiteralNode {
  template <typename T>
  LiteralNode(T v) : value{v}, type{selectArrowType<T>()}
  {
  }

  using var_t = LiteralValue::stored_type;
  var_t value;
  atype::type type = atype::NA;
};

/// An expression tree node corresponding to a column binding
struct BindingNode {
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  BindingNode(std::string const& name_, std::size_t hash_, atype::type type_) : name{name_}, hash{hash_}, type{type_} {}
  std::string name;
  std::size_t hash;
  atype::type type;
};

/// An expression tree node corresponding to binary or unary operation
struct OpNode {
  OpNode(BasicOp op_) : op{op_} {}
  BasicOp op;
};

/// A placeholder node for simple type configurable
struct PlaceholderNode : LiteralNode {
  template <typename T>
  PlaceholderNode(Configurable<T> const& v) : LiteralNode{v.value}, name{v.name}
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      retrieve = [](InitContext& context, std::string const& name) { return LiteralNode::var_t{context.options().get<T>(name.c_str())}; };
    } else {
      runtime_error("Unknown parameter used in expression.");
    }
  }

  void reset(InitContext& context)
  {
    value = retrieve(context, name);
  }

  std::string name;
  LiteralNode::var_t (*retrieve)(InitContext&, std::string const& name);
};

/// A conditional node
struct ConditionalNode {
};

/// A generic tree node
struct Node {
  Node(LiteralNode v) : self{v}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(PlaceholderNode v) : self{v}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(Node&& n) : self{n.self}, left{std::move(n.left)}, right{std::move(n.right)}, condition{std::move(n.condition)}
  {
  }

  Node(BindingNode n) : self{n}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(ConditionalNode op, Node&& then_, Node&& else_, Node&& condition_)
    : self{op},
      left{std::make_unique<Node>(std::move(then_))},
      right{std::make_unique<Node>(std::move(else_))},
      condition{std::make_unique<Node>(std::move(condition_))} {}

  Node(OpNode op, Node&& l, Node&& r)
    : self{op},
      left{std::make_unique<Node>(std::move(l))},
      right{std::make_unique<Node>(std::move(r))},
      condition{nullptr} {}

  Node(OpNode op, Node&& l)
    : self{op},
      left{std::make_unique<Node>(std::move(l))},
      right{nullptr},
      condition{nullptr} {}

  /// variant with possible nodes
  using self_t = std::variant<LiteralNode, BindingNode, OpNode, PlaceholderNode, ConditionalNode>;
  self_t self;
  size_t index = 0;
  /// pointers to children
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
  std::unique_ptr<Node> condition;
};

/// overloaded operators to build the tree from an expression

#define BINARY_OP_NODES(_operator_, _operation_)                                        \
  template <typename T>                                                                 \
  inline Node operator _operator_(Node left, T right)                                   \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::move(left), LiteralNode{right}};     \
  }                                                                                     \
  template <typename T>                                                                 \
  inline Node operator _operator_(T left, Node right)                                   \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, LiteralNode{left}, std::move(right)};     \
  }                                                                                     \
  template <typename T>                                                                 \
  inline Node operator _operator_(Node left, Configurable<T> right)                     \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::move(left), PlaceholderNode{right}}; \
  }                                                                                     \
  template <typename T>                                                                 \
  inline Node operator _operator_(Configurable<T> left, Node right)                     \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, std::move(right)}; \
  }                                                                                     \
  inline Node operator _operator_(Node left, Node right)                                \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::move(left), std::move(right)};       \
  }                                                                                     \
  inline Node operator _operator_(BindingNode left, BindingNode right)                  \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, right};                             \
  }                                                                                     \
  template <>                                                                           \
  inline Node operator _operator_(BindingNode left, Node right)                         \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, std::move(right)};                  \
  }                                                                                     \
  template <>                                                                           \
  inline Node operator _operator_(Node left, BindingNode right)                         \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::move(left), right};                  \
  }                                                                                     \
                                                                                        \
  template <typename T>                                                                 \
  inline Node operator _operator_(Configurable<T> left, BindingNode right)              \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, right};            \
  }                                                                                     \
  template <typename T>                                                                 \
  inline Node operator _operator_(BindingNode left, Configurable<T> right)              \
  {                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, PlaceholderNode{right}};            \
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
template <typename T>
inline Node npow(Node left, T right)
{
  return Node{OpNode{BasicOp::Power}, std::move(left), LiteralNode{right}};
}

#define BINARY_FUNC_NODES(_func_, _node_)                                          \
  template <typename L, typename R>                                                \
  inline Node _node_(L left, R right)                                              \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, LiteralNode{left}, LiteralNode{right}};   \
  }                                                                                \
                                                                                   \
  template <>                                                                      \
  inline Node _node_(Node left, Node right)                                        \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::move(left), std::move(right)};       \
  }                                                                                \
                                                                                   \
  template <>                                                                      \
  inline Node _node_(Node left, BindingNode right)                                 \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::move(left), right};                  \
  }                                                                                \
                                                                                   \
  template <>                                                                      \
  inline Node _node_(BindingNode left, BindingNode right)                          \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, right};                             \
  }                                                                                \
                                                                                   \
  template <>                                                                      \
  inline Node _node_(BindingNode left, Node right)                                 \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, std::move(right)};                  \
  }                                                                                \
                                                                                   \
  template <typename T>                                                            \
  inline Node _node_(Node left, Configurable<T> right)                             \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::move(left), PlaceholderNode{right}}; \
  }                                                                                \
                                                                                   \
  template <typename T>                                                            \
  inline Node _node_(Configurable<T> left, Node right)                             \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, std::move(right)}; \
  }                                                                                \
                                                                                   \
  template <typename T>                                                            \
  inline Node _node_(BindingNode left, Configurable<T> right)                      \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, PlaceholderNode{right}};            \
  }                                                                                \
                                                                                   \
  template <typename T>                                                            \
  inline Node _node_(Configurable<T> left, BindingNode right)                      \
  {                                                                                \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, right};            \
  }

BINARY_FUNC_NODES(Atan2, natan2);

/// unary functions on nodes
inline Node nsqrt(Node left)
{
  return Node{OpNode{BasicOp::Sqrt}, std::move(left)};
}

inline Node nexp(Node left)
{
  return Node{OpNode{BasicOp::Exp}, std::move(left)};
}

inline Node nlog(Node left)
{
  return Node{OpNode{BasicOp::Log}, std::move(left)};
}

inline Node nlog10(Node left)
{
  return Node{OpNode{BasicOp::Log10}, std::move(left)};
}

inline Node nabs(Node left)
{
  return Node{OpNode{BasicOp::Abs}, std::move(left)};
}

inline Node nsin(Node left)
{
  return Node{OpNode{BasicOp::Sin}, std::move(left)};
}

inline Node ncos(Node left)
{
  return Node{OpNode{BasicOp::Cos}, std::move(left)};
}

inline Node ntan(Node left)
{
  return Node{OpNode{BasicOp::Tan}, std::move(left)};
}

inline Node nasin(Node left)
{
  return Node{OpNode{BasicOp::Asin}, std::move(left)};
}

inline Node nacos(Node left)
{
  return Node{OpNode{BasicOp::Acos}, std::move(left)};
}

inline Node natan(Node left)
{
  return Node{OpNode{BasicOp::Atan}, std::move(left)};
}

inline Node nbitwise_not(Node left)
{
  return Node{OpNode{BasicOp::BitwiseNot}, std::move(left)};
}

/// conditionals
template <typename C, typename T, typename E>
inline Node ifnode(C condition_, T then_, E else_)
{
  return Node{ConditionalNode{}, std::move(then_), std::move(else_), std::move(condition_)};
}

template <>
inline Node ifnode(Node condition_, Node then_, Node else_)
{
  return Node{ConditionalNode{}, std::move(then_), std::move(else_), std::move(condition_)};
}

template <typename L, std::enable_if_t<std::is_integral<L>::value || std::is_floating_point<L>::value, bool> = true>
inline Node ifnode(Node condition_, Node then_, L else_)
{
  return Node{ConditionalNode{}, std::move(then_), LiteralNode{else_}, std::move(condition_)};
}

template <typename L, std::enable_if_t<std::is_integral<L>::value || std::is_floating_point<L>::value, bool> = true>
inline Node ifnode(Node condition_, L then_, Node else_)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, std::move(else_), std::move(condition_)};
}

template <typename L1, typename L2, std::enable_if_t<(std::is_integral<L1>::value || std::is_floating_point<L1>::value) && (std::is_integral<L2>::value || std::is_floating_point<L2>::value), bool> = true>
inline Node ifnode(Node condition_, L1 then_, L2 else_)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, LiteralNode{else_}, std::move(condition_)};
}

template <typename T>
inline Node ifnode(Configurable<T> condition_, Node then_, Node else_)
{
  return Node{ConditionalNode{}, std::move(then_), std::move(else_), PlaceholderNode{condition_}};
}

template <typename L>
inline Node ifnode(Node condition_, Node then_, Configurable<L> else_)
{
  return Node{ConditionalNode{}, std::move(then_), PlaceholderNode{else_}, std::move(condition_)};
}

template <typename L>
inline Node ifnode(Node condition_, Configurable<L> then_, Node else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, std::move(else_), std::move(condition_)};
}

template <typename L1, typename L2>
inline Node ifnode(Node condition_, Configurable<L1> then_, Configurable<L2> else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, PlaceholderNode{else_}, std::move(condition_)};
}

/// A struct, containing the root of the expression tree
struct Filter {
  Filter(Node&& node_) : node{std::make_unique<Node>(std::move(node_))}
  {
    (void)designateSubtrees(node.get());
  }

  Filter(Filter&& other) : node{std::move(other.node)}
  {
    (void)designateSubtrees(node.get());
  }
  std::unique_ptr<Node> node;

  size_t designateSubtrees(Node* node, size_t index = 0);
};

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
bool isSchemaCompatible(gandiva::SchemaPtr const& Schema, Operations const& opSpecs);
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

template <typename... C>
std::vector<expressions::Projector> makeProjectors(framework::pack<C...>)
{
  return {C::Projector()...};
}

std::shared_ptr<gandiva::Projector> createProjectorHelper(size_t nColumns, expressions::Projector* projectors,
                                                          std::shared_ptr<arrow::Schema> schema,
                                                          std::vector<std::shared_ptr<arrow::Field>> const& fields);

template <typename... C>
std::shared_ptr<gandiva::Projector> createProjectors(framework::pack<C...> columns, std::vector<std::shared_ptr<arrow::Field>> const& fields, gandiva::SchemaPtr schema)
{
  std::array<expressions::Projector, sizeof...(C)> projectors{{std::move(C::Projector())...}};

  return createProjectorHelper(sizeof...(C), projectors.data(), schema, fields);
}
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
