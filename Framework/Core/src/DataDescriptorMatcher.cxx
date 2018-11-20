// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataDescriptorMatcher.h"
#include <iostream>

namespace o2
{
namespace framework
{
namespace data_matcher
{

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
    assert(false);
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
    assert(false);
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

  return this->match(reinterpret_cast<char const*>(&dh), context);
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
    auto v1 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&this->mRight);
    auto v2 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&other.mRight);
    if (v1 && v2 && v1->get() && v2->get() && (**v1 == **v2)) {
      return true;
    }
  }
  // We alredy know the left side is true.
  return false;
}

} // namespace data_matcher
} // namespace framework
} // namespace o2
