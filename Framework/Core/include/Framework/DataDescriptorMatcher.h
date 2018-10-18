// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataDescriptorMatcher_H_INCLUDED
#define o2_framework_DataDescriptorMatcher_H_INCLUDED

#include "Headers/DataHeader.h"

#include <cstdint>
#include <string>
#include <variant>

namespace o2
{
namespace framework
{

/// Something which can be matched against a header::DataOrigin
class OriginValueMatcher
{
 public:
  OriginValueMatcher(std::string const& s)
    : mValue{ s }
  {
  }

  bool match(header::DataHeader const& header) const
  {
    return strncmp(header.dataOrigin.str, mValue.c_str(), 4) == 0;
  }

 private:
  std::string mValue;
};

/// Something which can be matched against a header::DataDescription
class DescriptionValueMatcher
{
 public:
  DescriptionValueMatcher(std::string const& s)
    : mValue{ s }
  {
  }

  bool match(header::DataHeader const& header) const
  {
    return strncmp(header.dataDescription.str, mValue.c_str(), 8) == 0;
  }

 private:
  std::string mValue;
};

/// Something which can be matched against a header::SubSpecificationType
class SubSpecificationTypeValueMatcher
{
 public:
  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  SubSpecificationTypeValueMatcher(std::string const& s)
  {
    mValue = strtoull(s.c_str(), nullptr, 10);
  }

  SubSpecificationTypeValueMatcher(uint64_t v)
  {
    mValue = v;
  }
  bool match(header::DataHeader const& header) const
  {
    return header.subSpecification == mValue;
  }

 private:
  uint64_t mValue;
};

/// Something which can be matched against a header::SubSpecificationType
class ConstantValueMatcher
{
 public:
  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  ConstantValueMatcher(bool value)
  {
    mValue = value;
  }
  bool match(header::DataHeader const& header) const
  {
    return mValue;
  }

 private:
  bool mValue;
};

template <typename DESCRIPTOR>
struct DescriptorMatcherTrait {
};

template <>
struct DescriptorMatcherTrait<header::DataOrigin> {
  using Matcher = framework::OriginValueMatcher;
};

template <>
struct DescriptorMatcherTrait<header::DataDescription> {
  using Matcher = DescriptionValueMatcher;
};

template <>
struct DescriptorMatcherTrait<header::DataHeader::SubSpecificationType> {
  using Matcher = SubSpecificationTypeValueMatcher;
};

class DataDescriptorMatcher;
using Node = std::variant<OriginValueMatcher, DescriptionValueMatcher, SubSpecificationTypeValueMatcher, std::unique_ptr<DataDescriptorMatcher>, ConstantValueMatcher>;

// A matcher for a given O2 Data Model descriptor.  We use a variant to hold
// the different kind of matchers so that we can have a hierarchy or
// DataDescriptionMatcher in the future (e.g. to handle OR / AND clauses) or we
// can apply it to the whole DataHeader.
class DataDescriptorMatcher
{
 public:
  enum struct Op { Just,
                   Or,
                   And,
                   Xor };

  /// Unary operator on a node
  DataDescriptorMatcher(Op op, Node&& lhs, Node&& rhs = std::move(ConstantValueMatcher{ false }))
    : mOp{ op },
      mLeft{ std::move(lhs) },
      mRight{ std::move(rhs) }
  {
  }

  inline ~DataDescriptorMatcher() = default;

  bool match(header::DataHeader const& d) const
  {
    auto eval = [&d](auto&& arg) -> bool {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::unique_ptr<DataDescriptorMatcher>>) {
        return arg->match(d);
      } else {
        return arg.match(d);
      }
    };

    bool leftValue = false, rightValue = false;

    // FIXME: Using std::visit is not API compatible due to a new
    // exception being thrown. This is the ABI compatible version.
    // Replace with:
    //
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
      leftValue = pval0->match(d);
    } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mLeft)) {
      leftValue = pval1->match(d);
    } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mLeft)) {
      leftValue = pval2->match(d);
    } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mLeft)) {
      leftValue = (*pval3)->match(d);
    } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&mLeft)) {
      leftValue = pval4->match(d);
    } else {
      throw std::runtime_error("Bad parsing tree");
    }

    if (auto pval0 = std::get_if<OriginValueMatcher>(&mRight)) {
      rightValue = pval0->match(d);
    } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mRight)) {
      rightValue = pval1->match(d);
    } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mRight)) {
      rightValue = pval2->match(d);
    } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mRight)) {
      rightValue = (*pval3)->match(d);
    } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&mRight)) {
      rightValue = pval4->match(d);
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

 private:
  Op mOp;
  Node mLeft;
  Node mRight;
};

} // naemspace framework
} // namespace o2

#endif // o2_framework_DataDescriptorMatcher_H_INCLUDED
