// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ExpressionJSONHelpers.h"

#include <rapidjson/reader.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/error/en.h>

#include <stack>
#include <iostream>
#include "Framework/VariantHelpers.h"

namespace o2::framework
{
namespace
{
using nodes = expressions::Node::self_t;
enum struct Nodes : int {
  NLITERAL = 0,
  NBINDING = 1,
  NOP = 2,
  NNPH = 3,
  NCOND = 4,
  NPAR = 5
};

enum struct ToWrite {
  FULL,
  LEFT,
  RIGHT,
  COND,
  POP
};

struct Entry {
  expressions::Node* ptr = nullptr;
  ToWrite toWrite = ToWrite::FULL;
};

std::array<std::string_view, 11> validKeys{
  "projectors",
  "kind",
  "binding",
  "index",
  "arrow_type",
  "value",
  "hash",
  "operation",
  "left",
  "right",
  "condition"};

struct ExpressionReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ExpressionReader> {
  using Ch = rapidjson::UTF8<>::Ch;
  using SizeType = rapidjson::SizeType;

  enum struct State {
    IN_START,            // global start
    IN_LIST,             // opening brace of the list
    IN_ROOT,             // after encountering the opening of the expression object
    IN_LEFT,             // in "left" key - subexpression
    IN_RIGHT,            // in "right" key - subexpression
    IN_COND,             // in "condition" key - subexpression
    IN_NODE_LITERAL,     // in literal node
    IN_NODE_BINDING,     // in binding node
    IN_NODE_OP,          // in operation node
    IN_NODE_CONDITIONAL, // in conditional node
    IN_ERROR             // generic error state
  };

  std::stack<State> states;
  std::stack<Entry> path;
  std::ostringstream debug;

  std::vector<expressions::Projector> result;

  std::unique_ptr<expressions::Node> rootNode = nullptr;
  std::unique_ptr<expressions::Node> node = nullptr;
  expressions::LiteralValue::stored_type value;
  atype::type type;
  Nodes kind;
  std::string binding;
  BasicOp operation;
  uint32_t hash;
  size_t index;

  std::string previousKey;
  std::string currentKey;

  ExpressionReader()
  {
    debug << ">>> Start" << std::endl;
    states.push(State::IN_START);
  }

  bool StartArray()
  {
    debug << "StartArray()" << std::endl;
    if (states.top() == State::IN_START) {
      states.push(State::IN_LIST);
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool EndArray(SizeType)
  {
    debug << "EndArray()" << std::endl;
    if (states.top() == State::IN_LIST) {
      states.pop();
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool Key(const Ch* str, SizeType, bool)
  {
    debug << "Key(" << str << ")" << std::endl;
    previousKey = currentKey;
    currentKey = str;
    if (std::find(validKeys.begin(), validKeys.end(), currentKey) == validKeys.end()) {
      states.push(State::IN_ERROR);
      return false;
    }

    if (states.top() == State::IN_START) {
      if (currentKey.compare("projectors") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_ROOT) {
      if (currentKey.compare("kind") == 0) {
        return true;
      } else {
        states.push(State::IN_ERROR); // should start from root node
        return false;
      }
    }

    if (states.top() == State::IN_LEFT || states.top() == State::IN_RIGHT || states.top() == State::IN_COND) {
      if (currentKey.compare("kind") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_NODE_LITERAL || states.top() == State::IN_NODE_OP || states.top() == State::IN_NODE_BINDING || states.top() == State::IN_NODE_CONDITIONAL) {
      if (currentKey.compare("index") == 0) {
        return true;
      }
      if (currentKey.compare("left") == 0) {
        // this is the point where the node header is parsed and we can create it
        // create a new node instance here and set a pointer to it in a parent (current stack top), based on its state
        // push the new node into the stack with LEFT state
        switch (states.top()) {
          case State::IN_NODE_LITERAL:
            node = std::make_unique<expressions::Node>(expressions::LiteralNode{value, type});
            break;
          case State::IN_NODE_BINDING:
            node = std::make_unique<expressions::Node>(expressions::BindingNode{hash, type}, binding);
            break;
          case State::IN_NODE_OP:
            node = std::make_unique<expressions::Node>(expressions::OpNode{operation}, expressions::LiteralNode{-1});
            break;
          case State::IN_NODE_CONDITIONAL:
            node = std::make_unique<expressions::Node>(expressions::ConditionalNode{}, expressions::LiteralNode{-1}, expressions::LiteralNode{-1}, expressions::LiteralNode{true});
            break;
          default:
            states.push(State::IN_ERROR);
            return false;
        }

        if (path.empty()) {
          rootNode = std::move(node);
          path.emplace(rootNode.get(), ToWrite::LEFT);
        } else {
          auto* n = path.top().ptr;
          switch (path.top().toWrite) {
            case ToWrite::LEFT:
              n->left = std::move(node);
              path.top().toWrite = ToWrite::RIGHT;
              path.emplace(n->left.get(), ToWrite::LEFT);
              break;
            case ToWrite::RIGHT:
              n->right = std::move(node);
              path.top().toWrite = ToWrite::COND;
              path.emplace(n->right.get(), ToWrite::LEFT);
              break;
            case ToWrite::COND:
              n->condition = std::move(node);
              path.pop();
              path.emplace(n->condition.get(), ToWrite::LEFT);
              break;
            default:
              states.push(State::IN_ERROR);
              return false;
          }
        }

        states.push(State::IN_LEFT);
        return true;
      }
      if (currentKey.compare("right") == 0) {
        if (states.top() == State::IN_LEFT) {
          states.pop();
        }
        // move the stack state of the node to RIGHT state
        path.top().toWrite = ToWrite::RIGHT;
        states.push(State::IN_RIGHT);
        return true;
      }
      if (currentKey.compare("condition") == 0) {
        if (states.top() == State::IN_RIGHT) {
          states.pop();
        }
        // move the stack state of the node to COND state
        path.top().toWrite = ToWrite::COND;
        states.push(State::IN_COND);
        return true;
      }
    }

    if (states.top() == State::IN_NODE_LITERAL) {
      if (currentKey.compare("arrow_type") == 0 || currentKey.compare("value") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_NODE_BINDING) {
      if (currentKey.compare("binding") == 0 || currentKey.compare("hash") == 0 || currentKey.compare("arrow_type") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_NODE_OP) {
      if (currentKey.compare("operation") == 0) {
        return true;
      }
    }

    debug << ">>> Unrecognized" << std::endl;
    states.push(State::IN_ERROR);
    return false;
  }

  bool StartObject()
  {
    // opening brace encountered
    debug << "StartObject()" << std::endl;
    // the first opening brace in the input
    if (states.top() == State::IN_START) {
      return true;
    }
    // the opening of an expression
    if (states.top() == State::IN_LIST) {
      states.push(State::IN_ROOT);
      return true;
    }
    // if we are looking at subexpression
    if (states.top() == State::IN_LEFT || states.top() == State::IN_RIGHT || states.top() == State::IN_COND) { // ready to start a new node
      return true;
    }
    // no other object starts are expected
    states.push(State::IN_ERROR);
    return false;
  }

  bool EndObject(SizeType)
  {
    // closing brace encountered
    debug << "EndObject()" << std::endl;
    // we are closing up an expression
    if (states.top() == State::IN_NODE_LITERAL || states.top() == State::IN_NODE_OP || states.top() == State::IN_NODE_BINDING || states.top() == State::IN_NODE_CONDITIONAL) { // finalize node
      // finalize the current node and pop it from the stack (the pointers should be already set
      states.pop();
      // subexpression
      if (states.top() == State::IN_LEFT || states.top() == State::IN_RIGHT || states.top() == State::IN_COND) {
        states.pop();
        return true;
      }

      // expression
      if (states.top() == State::IN_ROOT) {
        result.emplace_back(std::move(rootNode));
        states.pop();
        return true;
      }
    }

    // we are closing the list
    if (states.top() == State::IN_START) {
      return true;
    }
    // no other object ends are expectedd
    states.push(State::IN_ERROR);
    return false;
  }

  bool Null()
  {
    // null value
    debug << "Null()" << std::endl;
    // the subexpression can be empty
    if (states.top() == State::IN_LEFT || states.top() == State::IN_RIGHT || states.top() == State::IN_COND) {
      // empty node, nothing to do
      // move the path state to the next
      if (path.top().toWrite == ToWrite::LEFT) {
        path.top().toWrite = ToWrite::RIGHT;
      } else if (path.top().toWrite == ToWrite::RIGHT) {
        path.top().toWrite = ToWrite::COND;
      } else if (path.top().toWrite == ToWrite::COND) {
        path.pop();
      }

      states.pop();
      return true;
    }
    states.push(State::IN_ERROR); // no other contexts allow null
    return false;
  }

  bool Bool(bool b)
  {
    debug << "Bool(" << b << ")" << std::endl;
    // can be a value in a literal node
    if (states.top() == State::IN_NODE_LITERAL && currentKey.compare("value") == 0) {
      value = b;
      return true;
    }
    states.push(State::IN_ERROR); // no other contexts allow booleans
    return false;
  }

  bool Int(int i)
  {
    debug << "Int(" << i << ")" << std::endl;
    // can be a value in a literal node
    if (states.top() == State::IN_NODE_LITERAL && currentKey.compare("value") == 0) { // literal
      switch (type) {
        case atype::INT8:
          value = (int8_t)i;
          break;
        case atype::INT16:
          value = (int16_t)i;
          break;
        case atype::INT32:
          value = i;
          break;
        case atype::UINT8:
          value = (uint8_t)i;
          break;
        case atype::UINT16:
          value = (uint16_t)i;
          break;
        case atype::UINT32:
          value = (uint32_t)i;
          break;
        default:
          states.push(State::IN_ERROR);
          return false;
      }
      return true;
    }
    // can be a node kind designator
    if (states.top() == State::IN_ROOT || states.top() == State::IN_LEFT || states.top() == State::IN_RIGHT || states.top() == State::IN_COND) {
      if (currentKey.compare("kind") == 0) {
        kind = (Nodes)i;
        switch (kind) {
          case Nodes::NLITERAL:
          case Nodes::NNPH:
          case Nodes::NPAR: {
            states.push(State::IN_NODE_LITERAL);
            debug << ">>> Literal node" << std::endl;
            return true;
          }
          case Nodes::NBINDING: {
            states.push(State::IN_NODE_BINDING);
            debug << ">>> Binding node" << std::endl;
            return true;
          }
          case Nodes::NOP: {
            states.push(State::IN_NODE_OP);
            debug << ">>> Operation node" << std::endl;
            return true;
          }
          case Nodes::NCOND: {
            states.push(State::IN_NODE_CONDITIONAL);
            debug << ">>> Conditional node" << std::endl;
            return true;
          }
        }
      }
    }
    // can be node index
    if (states.top() == State::IN_NODE_BINDING || states.top() == State::IN_NODE_CONDITIONAL || states.top() == State::IN_NODE_LITERAL || states.top() == State::IN_NODE_OP) {
      if (currentKey.compare("index") == 0) {
        index = (size_t)i;
        return true;
      }
    }
    // can be a node type designator
    if (states.top() == State::IN_NODE_LITERAL || states.top() == State::IN_NODE_BINDING) {
      if (currentKey.compare("arrow_type") == 0) {
        type = (atype::type)i;
        return true;
      }
    }
    // can be a node operation designato
    if (states.top() == State::IN_NODE_OP && currentKey.compare("operation") == 0) {
      operation = (BasicOp)i;
      return true;
    }
    states.push(State::IN_ERROR); // no other contexts allow ints
    return false;
  }

  bool Uint(unsigned i)
  {
    debug << "Uint(" << i << ")" << std::endl;
    // can be node hash
    if (states.top() == State::IN_NODE_BINDING && currentKey.compare("hash") == 0) {
      hash = i;
      return true;
    }
    // any positive value will be first read as unsigned, however the actual type is determined by node's arrow_type
    debug << ">> falling back to Int" << std::endl;
    return Int(i);
  }

  bool Int64(int64_t i)
  {
    debug << "Int64(" << i << ")" << std::endl;
    // can only be a literal node value
    if (states.top() == State::IN_NODE_LITERAL && currentKey.compare("value") == 0) {
      switch (type) {
        case atype::UINT64:
          value = (uint64_t)i;
          break;
        case atype::INT64:
          value = (int64_t)i;
          break;
        default:
          states.push(State::IN_ERROR);
          return false;
      }
      return true;
    }
    states.push(State::IN_ERROR); // no other contexts allow int64s
    return false;
  }

  bool Uint64(uint64_t i)
  {
    debug << "Uint64(" << i << ")" << std::endl;
    // any positive value will be first read as unsigned, however the actual type is determined by node's arrow_type
    debug << ">> falling back to Int64" << std::endl;
    return Int64(i);
  }

  bool Double(double d)
  {
    debug << "Double(" << d << ")" << std::endl;
    // can only be a literal node value
    if (states.top() == State::IN_NODE_LITERAL) {
      switch (type) {
        case atype::FLOAT:
          value = (float)d;
          break;
        case atype::DOUBLE:
          value = d;
          break;
        default:
          states.push(State::IN_ERROR);
          return false;
      }
      return true;
    }
    states.push(State::IN_ERROR); // no other contexts allow doubles
    return false;
  }

  bool String(const Ch* str, SizeType, bool)
  {
    debug << "String(" << str << ")" << std::endl;
    // can only be a binding node
    if (states.top() == State::IN_NODE_BINDING && currentKey.compare("binding") == 0) {
      binding = str;
      return true;
    }
    states.push(State::IN_ERROR); // no strings are expected
    return false;
  }
};
} // namespace

std::vector<expressions::Projector> o2::framework::ExpressionJSONHelpers::read(std::istream& s)
{
  rapidjson::Reader reader;
  rapidjson::IStreamWrapper isw(s);
  ExpressionReader ereader;
  bool ok = reader.Parse(isw, ereader);

  if (!ok) {
    throw framework::runtime_error_f("Cannot parse serialized Expression, error: %s at offset: %d", rapidjson::GetParseError_En(reader.GetParseErrorCode()), reader.GetErrorOffset());
  }
  return std::move(ereader.result);
}

namespace
{
void writeNodeHeader(rapidjson::Writer<rapidjson::OStreamWrapper>& w, expressions::Node const* node)
{
  w.Key("kind");
  w.Int((int)node->self.index());
  w.Key("index");
  w.Uint64(node->index);
  std::visit(overloaded{
               [&w](expressions::LiteralNode const& node) {
                 w.Key("arrow_type");
                 w.Int(node.type);
                 w.Key("value");
                 std::visit(overloaded{
                              [&w](bool v) { w.Bool(v); },
                              [&w](float v) { w.Double(v); },
                              [&w](double v) { w.Double(v); },
                              [&w](uint8_t v) { w.Uint(v); },
                              [&w](uint16_t v) { w.Uint(v); },
                              [&w](uint32_t v) { w.Uint(v); },
                              [&w](uint64_t v) { w.Uint64(v); },
                              [&w](int8_t v) { w.Int(v); },
                              [&w](int16_t v) { w.Int(v); },
                              [&w](int v) { w.Int(v); },
                              [&w](int64_t v) { w.Int64(v); }},
                            node.value);
               },
               [&w](expressions::BindingNode const& node) {
                 w.Key("binding");
                 w.String(node.name);
                 w.Key("hash");
                 w.Uint(node.hash);
                 w.Key("arrow_type");
                 w.Int(node.type);
               },
               [&w](expressions::OpNode const& node) {
                 w.Key("operation");
                 w.Int(node.op);
               },
               [](expressions::ConditionalNode const&) {
               }},
             node->self);
}

void writeExpression(rapidjson::Writer<rapidjson::OStreamWrapper>& w, expressions::Node* n)
{
  std::stack<Entry> path;
  path.emplace(n, ToWrite::FULL);
  while (!path.empty()) {
    auto& top = path.top();

    if (top.toWrite == ToWrite::FULL) {
      w.StartObject();
      writeNodeHeader(w, top.ptr);
      top.toWrite = ToWrite::LEFT;
      continue;
    }

    if (top.toWrite == ToWrite::LEFT) {
      w.Key("left");
      top.toWrite = ToWrite::RIGHT;
      auto* left = top.ptr->left.get();
      if (left != nullptr) {
        path.emplace(left, ToWrite::FULL);
      } else {
        w.Null();
      }
      continue;
    }

    if (top.toWrite == ToWrite::RIGHT) {
      w.Key("right");
      top.toWrite = ToWrite::COND;
      auto* right = top.ptr->right.get();
      if (right != nullptr) {
        path.emplace(right, ToWrite::FULL);
      } else {
        w.Null();
      }
      continue;
    }

    if (top.toWrite == ToWrite::COND) {
      w.Key("condition");
      top.toWrite = ToWrite::POP;
      auto* cond = top.ptr->condition.get();
      if (cond != nullptr) {
        path.emplace(cond, ToWrite::FULL);
      } else {
        w.Null();
      }
      continue;
    }

    if (top.toWrite == ToWrite::POP) {
      w.EndObject();
      path.pop();
      continue;
    }
  }
}
} // namespace

void o2::framework::ExpressionJSONHelpers::write(std::ostream& o, std::vector<o2::framework::expressions::Projector>& projectors)
{
  rapidjson::OStreamWrapper osw(o);
  rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);
  w.StartObject();
  w.Key("projectors");
  w.StartArray();
  for (auto& p : projectors) {
    writeExpression(w, p.node.get());
  }
  w.EndArray();
  w.EndObject();
}

namespace
{
struct SchemaReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, SchemaReader> {
  using Ch = rapidjson::UTF8<>::Ch;
  using SizeType = rapidjson::SizeType;

  enum struct State {
    IN_START,
    IN_LIST,
    IN_FIELD,
    IN_ERROR
  };

  std::stack<State> states;
  std::ostringstream debug;

  std::shared_ptr<arrow::Schema> schema = nullptr;
  std::vector<std::shared_ptr<arrow::Field>> fields;

  std::string currentKey;

  std::string name;
  atype::type type;

  SchemaReader()
  {
    debug << ">>> Start" << std::endl;
    states.push(State::IN_START);
  }

  bool StartArray()
  {
    debug << "Starting array" << std::endl;
    if (states.top() == State::IN_START && currentKey.compare("fields") == 0) {
      states.push(State::IN_LIST);
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool EndArray(SizeType)
  {
    debug << "Ending array" << std::endl;
    if (states.top() == State::IN_LIST) {
      // finalize schema
      schema = std::make_shared<arrow::Schema>(fields);
      states.pop();
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool Key(const Ch* str, SizeType, bool)
  {
    debug << "Key(" << str << ")" << std::endl;
    currentKey = str;
    if (states.top() == State::IN_START) {
      if (currentKey.compare("fields") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_FIELD) {
      if (currentKey.compare("name") == 0) {
        return true;
      }
      if (currentKey.compare("type") == 0) {
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool StartObject()
  {
    debug << "StartObject()" << std::endl;
    if (states.top() == State::IN_START) {
      return true;
    }

    if (states.top() == State::IN_LIST) {
      states.push(State::IN_FIELD);
      return true;
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool EndObject(SizeType)
  {
    debug << "EndObject()" << std::endl;
    if (states.top() == State::IN_FIELD) {
      states.pop();
      // add a field
      fields.emplace_back(std::make_shared<arrow::Field>(name, expressions::concreteArrowType(type)));
      return true;
    }

    if (states.top() == State::IN_START) {
      return true;
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool Uint(unsigned i)
  {
    debug << "Uint(" << i << ")" << std::endl;
    if (states.top() == State::IN_FIELD) {
      if (currentKey.compare("type") == 0) {
        type = (atype::type)i;
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool String(const Ch* str, SizeType, bool)
  {
    debug << "String(" << str << ")" << std::endl;
    if (states.top() == State::IN_FIELD) {
      if (currentKey.compare("name") == 0) {
        name = str;
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool Int(int i)
  {
    debug << "Int(" << i << ")" << std::endl;
    return Uint(i);
  }
};
} // namespace

std::shared_ptr<arrow::Schema> o2::framework::ArrowJSONHelpers::read(std::istream& s)
{
  rapidjson::Reader reader;
  rapidjson::IStreamWrapper isw(s);
  SchemaReader sreader;

  bool ok = reader.Parse(isw, sreader);

  if (!ok) {
    throw framework::runtime_error_f("Cannot parse serialized Expression, error: %s at offset: %d", rapidjson::GetParseError_En(reader.GetParseErrorCode()), reader.GetErrorOffset());
  }
  return sreader.schema;
}

namespace
{
void writeSchema(rapidjson::Writer<rapidjson::OStreamWrapper>& w, arrow::Schema* schema)
{
  for (auto& f : schema->fields()) {
    w.StartObject();
    w.Key("name");
    w.String(f->name().c_str());
    w.Key("type");
    w.Int(f->type()->id());
    w.EndObject();
  }
}
} // namespace

void o2::framework::ArrowJSONHelpers::write(std::ostream& o, std::shared_ptr<arrow::Schema>& schema)
{
  rapidjson::OStreamWrapper osw(o);
  rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);
  w.StartObject();
  w.Key("fields");
  w.StartArray();
  writeSchema(w, schema.get());
  w.EndArray();
  w.EndObject();
}

} // namespace o2::framework
