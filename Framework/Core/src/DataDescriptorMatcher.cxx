// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompilerBuiltins.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataMatcherWalker.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/VariantHelpers.h"
#include <iostream>

namespace o2
{
namespace framework
{
namespace data_matcher
{

ContextElement::Value const& VariableContext::get(size_t pos) const
{
  // First we check if there is any pending update
  for (size_t i = 0; i < mPerformedUpdates; ++i) {
    if (mUpdates[i].position == pos) {
      return mUpdates[i].newValue;
    }
  }
  // Otherwise we return the element.
  return mElements.at(pos).value;
}

void VariableContext::commit()
{
  for (size_t i = 0; i < mPerformedUpdates; ++i) {
    mElements[mUpdates[i].position].value = mUpdates[i].newValue;
  }
  mPerformedUpdates = 0;
}

void VariableContext::reset()
{
  mPerformedUpdates = 0;
  for (auto& element : mElements) {
    element.value = None{};
  }
}

bool OriginValueMatcher::match(header::DataHeader const& header, VariableContext& context) const
{
  if (auto ref = std::get_if<ContextRef>(&mValue)) {
    auto& variable = context.get(ref->index);
    if (auto value = std::get_if<std::string>(&variable)) {
      return strncmp(header.dataOrigin.str, value->c_str(), 4) == 0;
    }
    auto maxSize = strnlen(header.dataOrigin.str, 4);
    context.put({ ref->index, std::string(header.dataOrigin.str, maxSize) });
    return true;
  } else if (auto s = std::get_if<std::string>(&mValue)) {
    return strncmp(header.dataOrigin.str, s->c_str(), 4) == 0;
  }
  throw std::runtime_error("Mismatching type for variable");
}

bool DescriptionValueMatcher::match(header::DataHeader const& header, VariableContext& context) const
{
  if (auto ref = std::get_if<ContextRef>(&mValue)) {
    auto& variable = context.get(ref->index);
    if (auto value = std::get_if<std::string>(&variable)) {
      return strncmp(header.dataDescription.str, value->c_str(), 16) == 0;
    }
    auto maxSize = strnlen(header.dataDescription.str, 16);
    context.put({ ref->index, std::string(header.dataDescription.str, maxSize) });
    return true;
  } else if (auto s = std::get_if<std::string>(&this->mValue)) {
    return strncmp(header.dataDescription.str, s->c_str(), 16) == 0;
  }
  throw std::runtime_error("Mismatching type for variable");
}

bool SubSpecificationTypeValueMatcher::match(header::DataHeader const& header, VariableContext& context) const
{
  if (auto ref = std::get_if<ContextRef>(&mValue)) {
    auto& variable = context.get(ref->index);
    if (auto value = std::get_if<header::DataHeader::SubSpecificationType>(&variable)) {
      return header.subSpecification == *value;
    }
    context.put({ ref->index, header.subSpecification });
    return true;
  } else if (auto v = std::get_if<header::DataHeader::SubSpecificationType>(&mValue)) {
    return header.subSpecification == *v;
  }
  throw std::runtime_error("Mismatching type for variable");
}

/// This will match the timing information which is currently in
/// the DataProcessingHeader. Notice how we apply the scale to the
/// actual values found.
bool StartTimeValueMatcher::match(DataProcessingHeader const& dph, VariableContext& context) const
{
  if (auto ref = std::get_if<ContextRef>(&mValue)) {
    auto& variable = context.get(ref->index);
    if (auto value = std::get_if<uint64_t>(&variable)) {
      return (dph.startTime / mScale) == *value;
    }
    context.put({ ref->index, dph.startTime / mScale });
    return true;
  } else if (auto v = std::get_if<uint64_t>(&mValue)) {
    return (dph.startTime / mScale) == *v;
  }
  throw std::runtime_error("Mismatching type for variable");
}

DataDescriptorMatcher::DataDescriptorMatcher(DataDescriptorMatcher const& other)
  : mOp{ other.mOp },
    mLeft{ ConstantValueMatcher{ false } },
    mRight{ ConstantValueMatcher{ false } }
{
  if (auto pval0 = std::get_if<OriginValueMatcher>(&other.mLeft)) {
    mLeft = *pval0;
  } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&other.mLeft)) {
    mLeft = *pval1;
  } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&other.mLeft)) {
    mLeft = *pval2;
  } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&other.mLeft)) {
    mLeft = std::move(std::make_unique<DataDescriptorMatcher>(*pval3->get()));
  } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&other.mLeft)) {
    mLeft = *pval4;
  } else if (auto pval5 = std::get_if<StartTimeValueMatcher>(&other.mLeft)) {
    mLeft = *pval5;
  } else {
    std::cerr << (other.mLeft.index() == std::variant_npos) << std::endl;
    O2_BUILTIN_UNREACHABLE();
  }

  if (auto pval0 = std::get_if<OriginValueMatcher>(&other.mRight)) {
    mRight = *pval0;
  } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&other.mRight)) {
    mRight = *pval1;
  } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&other.mRight)) {
    mRight = *pval2;
  } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&other.mRight)) {
    mRight = std::move(std::make_unique<DataDescriptorMatcher>(*pval3->get()));
  } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&other.mRight)) {
    mRight = *pval4;
  } else if (auto pval5 = std::get_if<StartTimeValueMatcher>(&other.mRight)) {
    mRight = *pval5;
  } else {
    O2_BUILTIN_UNREACHABLE();
  }
}

DataDescriptorMatcher& DataDescriptorMatcher::operator=(DataDescriptorMatcher const& other)
{
  return *this = DataDescriptorMatcher(other);
}

/// Unary operator on a node
DataDescriptorMatcher::DataDescriptorMatcher(Op op, Node&& lhs, Node&& rhs)
  : mOp{ op },
    mLeft{ std::move(lhs) },
    mRight{ std::move(rhs) }
{
}

/// @return true if the (sub-)query associated to this matcher will
/// match the provided @a spec, false otherwise.
bool DataDescriptorMatcher::match(ConcreteDataMatcher const& matcher, VariableContext& context) const
{
  header::DataHeader dh;
  dh.dataOrigin = matcher.origin;
  dh.dataDescription = matcher.description;
  dh.subSpecification = matcher.subSpec;
  DataProcessingHeader dph;
  dph.startTime = 0;
  header::Stack s{ dh, dph };

  return this->match(reinterpret_cast<char const*>(s.data()), context);
}

bool DataDescriptorMatcher::match(header::DataHeader const& header, VariableContext& context) const
{
  return this->match(reinterpret_cast<char const*>(&header), context);
}

bool DataDescriptorMatcher::match(header::Stack const& stack, VariableContext& context) const
{
  return this->match(reinterpret_cast<char const*>(stack.data()), context);
}

// actual polymorphic matcher which is able to cast the pointer to the correct
// kind of header.
bool DataDescriptorMatcher::match(char const* d, VariableContext& context) const
{
  bool leftValue = false, rightValue = false;

  // FIXME: Using std::visit is not API compatible due to a new
  // exception being thrown. This is the ABI compatible version.
  // Replace with:
  //
  // auto eval = [&d](auto&& arg) -> bool {
  //   using T = std::decay_t<decltype(arg)>;
  //   if constexpr (std::is_same_v<T, std::unique_ptr<DataDescriptorMatcher>>) {
  //     return arg->match(d, context);
  //   if constexpr (std::is_same_v<T, ConstantValueMatcher>) {
  //     return arg->match(d);
  //   } else {
  //     return arg.match(d, context);
  //   }
  // };
  // switch (mOp) {
  //   case Op::Or:
  //     return std::visit(eval, mLeft) || std::visit(eval, mRight);
  //   case Op::And:
  //     return std::visit(eval, mLeft) && std::visit(eval, mRight);
  //   case Op::Xor:
  //     return std::visit(eval, mLeft) ^ std::visit(eval, mRight);
  //   case Op::Just:
  //     return std::visit(eval, mLeft);
  // }
  //  When we drop support for macOS 10.13
  if (auto pval0 = std::get_if<OriginValueMatcher>(&mLeft)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    if (dh == nullptr) {
      throw std::runtime_error("Cannot find DataHeader");
    }
    leftValue = pval0->match(*dh, context);
  } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mLeft)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    if (dh == nullptr) {
      throw std::runtime_error("Cannot find DataHeader");
    }
    leftValue = pval1->match(*dh, context);
  } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mLeft)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    if (dh == nullptr) {
      throw std::runtime_error("Cannot find DataHeader");
    }
    leftValue = pval2->match(*dh, context);
  } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mLeft)) {
    leftValue = (*pval3)->match(d, context);
  } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&mLeft)) {
    leftValue = pval4->match();
  } else if (auto pval5 = std::get_if<StartTimeValueMatcher>(&mLeft)) {
    auto dph = o2::header::get<DataProcessingHeader*>(d);
    if (dph == nullptr) {
      throw std::runtime_error("Cannot find DataProcessingHeader");
    }
    leftValue = pval5->match(*dph, context);
  } else {
    throw std::runtime_error("Bad parsing tree");
  }
  // Common speedup.
  if (mOp == Op::And && leftValue == false) {
    return false;
  }
  if (mOp == Op::Or && leftValue == true) {
    return true;
  }
  if (mOp == Op::Just) {
    return leftValue;
  }

  if (auto pval0 = std::get_if<OriginValueMatcher>(&mRight)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    rightValue = pval0->match(*dh, context);
  } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mRight)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    rightValue = pval1->match(*dh, context);
  } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mRight)) {
    auto dh = o2::header::get<header::DataHeader*>(d);
    rightValue = pval2->match(*dh, context);
  } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mRight)) {
    rightValue = (*pval3)->match(d, context);
  } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&mRight)) {
    rightValue = pval4->match();
  } else if (auto pval5 = std::get_if<StartTimeValueMatcher>(&mRight)) {
    auto dph = o2::header::get<DataProcessingHeader*>(d);
    rightValue = pval5->match(*dph, context);
  }
  // There are cases in which not having a rightValue might be legitimate,
  // so we do not throw an exception.
  switch (mOp) {
    case Op::Or:
      return leftValue || rightValue;
    case Op::And:
      return leftValue && rightValue;
    case Op::Xor:
      return leftValue ^ rightValue;
    case Op::Just:
      return leftValue;
  }
  throw std::runtime_error("Bad parsing tree");
};

bool DataDescriptorMatcher::operator==(DataDescriptorMatcher const& other) const
{
  if (mOp != this->mOp) {
    return false;
  }

  bool leftValue = false;

  {
    auto v1 = std::get_if<OriginValueMatcher>(&this->mLeft);
    auto v2 = std::get_if<OriginValueMatcher>(&other.mLeft);
    if (v1 && v2 && *v1 == *v2) {
      leftValue = true;
    }
  }

  {
    auto v1 = std::get_if<DescriptionValueMatcher>(&this->mLeft);
    auto v2 = std::get_if<DescriptionValueMatcher>(&other.mLeft);
    if (v1 && v2 && *v1 == *v2) {
      leftValue = true;
    }
  }

  {
    auto v1 = std::get_if<SubSpecificationTypeValueMatcher>(&this->mLeft);
    auto v2 = std::get_if<SubSpecificationTypeValueMatcher>(&other.mLeft);
    if (v1 && v2 && *v1 == *v2) {
      leftValue = true;
    }
  }

  {
    auto v1 = std::get_if<ConstantValueMatcher>(&this->mLeft);
    auto v2 = std::get_if<ConstantValueMatcher>(&other.mLeft);
    if (v1 && v2 && *v1 == *v2) {
      leftValue = true;
    }
  }

  {
    auto v1 = std::get_if<StartTimeValueMatcher>(&this->mLeft);
    auto v2 = std::get_if<StartTimeValueMatcher>(&other.mLeft);
    if (v1 && v2 && *v1 == *v2) {
      leftValue = true;
    }
  }

  {
    auto v1 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&this->mLeft);
    auto v2 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&other.mLeft);
    if (v1 && v2 && v1->get() && v2->get() && (**v1 == **v2)) {
      leftValue = true;
    }
  }

  // Shortcut the fact that the left side is different.
  if (leftValue == false) {
    return false;
  }

  if (mOp == Op::Just) {
    return true;
  }

  {
    auto v1 = std::get_if<OriginValueMatcher>(&this->mRight);
    auto v2 = std::get_if<OriginValueMatcher>(&other.mRight);
    if (v1 && v2 && *v1 == *v2) {
      return true;
    }
  }

  {
    auto v1 = std::get_if<DescriptionValueMatcher>(&this->mRight);
    auto v2 = std::get_if<DescriptionValueMatcher>(&other.mRight);
    if (v1 && v2 && *v1 == *v2) {
      return true;
    }
  }

  {
    auto v1 = std::get_if<SubSpecificationTypeValueMatcher>(&this->mRight);
    auto v2 = std::get_if<SubSpecificationTypeValueMatcher>(&other.mRight);
    if (v1 && v2 && *v1 == *v2) {
      return true;
    }
  }

  {
    auto v1 = std::get_if<ConstantValueMatcher>(&this->mRight);
    auto v2 = std::get_if<ConstantValueMatcher>(&other.mRight);
    if (v1 && v2 && *v1 == *v2) {
      return true;
    }
  }

  {
    auto v1 = std::get_if<StartTimeValueMatcher>(&this->mRight);
    auto v2 = std::get_if<StartTimeValueMatcher>(&other.mRight);
    if (v1 && v2 && *v1 == *v2) {
      return true;
    }
  }

  {
    auto v1 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&this->mRight);
    auto v2 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&other.mRight);
    if (v1 && v2 && v1->get() && v2->get() && (**v1 == **v2)) {
      return true;
    }
  }
  // We alredy know the left side is true.
  return false;
}

std::ostream& operator<<(std::ostream& os, DataDescriptorMatcher const& matcher)
{
  auto edgeWalker = overloaded{
    [&os](EdgeActions::EnterNode action) {
      os << "(" << action.node->mOp;
      if (action.node->mOp == DataDescriptorMatcher::Op::Just) {
        return ChildAction::VisitLeft;
      }
      return ChildAction::VisitBoth;
    },
    [&os](EdgeActions::EnterLeft) { os << " "; },
    [&os](EdgeActions::ExitLeft) { os << " "; },
    [&os](EdgeActions::EnterRight) { os << " "; },
    [&os](EdgeActions::ExitRight) { os << " "; },
    [&os](EdgeActions::ExitNode) { os << ")"; },
    [&os](auto) {}
  };
  auto leafWalker = overloaded{
    [&os](OriginValueMatcher const& origin) { os << "origin:" << origin; },
    [&os](DescriptionValueMatcher const& description) { os << "description:" << description; },
    [&os](SubSpecificationTypeValueMatcher const& subSpec) { os << "subSpec:" << subSpec; },
    [&os](StartTimeValueMatcher const& startTime) { os << "startTime:" << startTime; },
    [&os](ConstantValueMatcher const& constant) {},
    [&os](auto t) { os << "not implemented " << typeid(decltype(t)).name(); }
  };
  DataMatcherWalker::walk(matcher,
                          edgeWalker,
                          leafWalker);

  return os;
}

std::ostream& operator<<(std::ostream& os, DataDescriptorMatcher::Op const& op)
{
  switch (op) {
    case DataDescriptorMatcher::Op::And:
      os << "and";
      break;
    case DataDescriptorMatcher::Op::Or:
      os << "or";
      break;
    case DataDescriptorMatcher::Op::Just:
      os << "just";
      break;
    case DataDescriptorMatcher::Op::Xor:
      os << "xor";
      break;
  }
  return os;
}

} // namespace data_matcher
} // namespace framework
} // namespace o2
