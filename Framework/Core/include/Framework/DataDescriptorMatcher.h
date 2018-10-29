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

#include "Framework/InputSpec.h"
#include "Headers/DataHeader.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace o2
{
namespace framework
{
namespace data_matcher
{

/// Marks an empty item in the context
struct None {
};

/// A typesafe reference to an element of the context.
struct ContextRef {
  size_t index;
};

/// An element of the matching context. Context itself is really a vector of
/// those. It's up to the matcher builder to build the vector in a suitable way.
/// We do not have any float in the value, because AFAICT there is no need for
/// it in the O2 DataHeader, however we could add it later on.
struct ContextElement {
  std::string label;                               /// The name of the variable contained in this element.
  std::variant<uint64_t, std::string, None> value; /// The actual contents of the element.
};

/// Something which can be matched against a header::DataOrigin
class OriginValueMatcher
{
 public:
  /// Initialise the matcher with an actual value
  OriginValueMatcher(std::string const& s)
    : mValue{ s }
  {
  }

  /// This means that the matcher will fill a variable in the context if
  /// the ref points to none or use the dereferenced value, if not.
  OriginValueMatcher(ContextRef variableId)
    : mValue{ variableId }
  {
  }

  bool match(header::DataHeader const& header, std::vector<ContextElement>& context) const
  {
    if (auto ref = std::get_if<ContextRef>(&mValue)) {
      auto& variable = context.at(ref->index);
      if (auto value = std::get_if<std::string>(&variable.value)) {
        return strncmp(header.dataOrigin.str, value->c_str(), 4) == 0;
      }
      auto maxSize = strnlen(header.dataOrigin.str, 4);
      variable.value = std::string(header.dataOrigin.str, maxSize);
      return true;
    } else if (auto s = std::get_if<std::string>(&mValue)) {
      return strncmp(header.dataOrigin.str, s->c_str(), 4) == 0;
    }
    throw std::runtime_error("Mismatching type for variable");
  }

 private:
  std::variant<std::string, ContextRef> mValue;
};

/// Something which can be matched against a header::DataDescription
class DescriptionValueMatcher
{
 public:
  DescriptionValueMatcher(std::string const& s)
    : mValue{ s }
  {
  }

  /// This means that the matcher will fill a variable in the context if
  /// the ref points to none or use the dereferenced value, if not.
  DescriptionValueMatcher(ContextRef ref)
    : mValue{ ref }
  {
  }

  bool match(header::DataHeader const& header, std::vector<ContextElement>& context) const
  {
    if (auto ref = std::get_if<ContextRef>(&mValue)) {
      auto& variable = context.at(ref->index);
      if (auto value = std::get_if<std::string>(&variable.value)) {
        return strncmp(header.dataDescription.str, value->c_str(), 16) == 0;
      }
      auto maxSize = strnlen(header.dataDescription.str, 16);
      variable.value = std::string(header.dataDescription.str, maxSize);
      return true;
    } else if (auto s = std::get_if<std::string>(&mValue)) {
      return strncmp(header.dataDescription.str, s->c_str(), 16) == 0;
    }
    throw std::runtime_error("Mismatching type for variable");
  }

 private:
  std::variant<std::string, ContextRef> mValue;
};

/// Something which can be matched against a header::SubSpecificationType
class SubSpecificationTypeValueMatcher
{
 public:
  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  SubSpecificationTypeValueMatcher(std::string const& s)
    : SubSpecificationTypeValueMatcher(strtoull(s.c_str(), nullptr, 10))
  {
  }

  /// This means that the matcher is looking for a constant.
  SubSpecificationTypeValueMatcher(uint64_t v)
    : mValue{ v }
  {
  }

  /// This means that the matcher will fill a variable in the context if
  /// the ref points to none or use the dereferenced value, if not.
  SubSpecificationTypeValueMatcher(ContextRef ref)
    : mValue{ ref }
  {
  }

  bool match(header::DataHeader const& header, std::vector<ContextElement>& context) const
  {
    if (auto ref = std::get_if<ContextRef>(&mValue)) {
      auto& variable = context.at(ref->index);
      if (auto value = std::get_if<uint64_t>(&variable.value)) {
        return header.subSpecification == *value;
      }
      variable.value = header.subSpecification;
      return true;
    } else if (auto v = std::get_if<uint64_t>(&mValue)) {
      return header.subSpecification == *v;
    }
    throw std::runtime_error("Mismatching type for variable");
  }

 private:
  std::variant<uint64_t, ContextRef> mValue;
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
  using Matcher = OriginValueMatcher;
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

  /// We treat all the nodes as values, hence we copy the
  /// contents mLeft and mRight into a new unique_ptr, if
  /// needed.
  DataDescriptorMatcher(DataDescriptorMatcher const& other)
   : mOp{other.mOp},
     mLeft{ConstantValueMatcher{false}},
     mRight{ConstantValueMatcher{false}}
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
    }
  }

  /// Unary operator on a node
  DataDescriptorMatcher(Op op, Node&& lhs, Node&& rhs = std::move(ConstantValueMatcher{ false }))
    : mOp{ op },
      mLeft{ std::move(lhs) },
      mRight{ std::move(rhs) }
  {
  }

  inline ~DataDescriptorMatcher() = default;

  /// @return true if the (sub-)query associated to this matcher will
  /// match the provided @a spec, false otherwise.
  bool match(InputSpec const& spec, std::vector<ContextElement>& context) const
  {
    header::DataHeader dh;
    dh.dataOrigin = spec.origin;
    dh.dataDescription = spec.description;
    dh.subSpecification = spec.subSpec;

    return this->match(dh, context);
  }

  bool match(header::DataHeader const& d, std::vector<ContextElement>& context) const
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
      leftValue = pval0->match(d, context);
    } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mLeft)) {
      leftValue = pval1->match(d, context);
    } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mLeft)) {
      leftValue = pval2->match(d, context);
    } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mLeft)) {
      leftValue = (*pval3)->match(d, context);
    } else if (auto pval4 = std::get_if<ConstantValueMatcher>(&mLeft)) {
      leftValue = pval4->match(d);
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
      rightValue = pval0->match(d, context);
    } else if (auto pval1 = std::get_if<DescriptionValueMatcher>(&mRight)) {
      rightValue = pval1->match(d, context);
    } else if (auto pval2 = std::get_if<SubSpecificationTypeValueMatcher>(&mRight)) {
      rightValue = pval2->match(d, context);
    } else if (auto pval3 = std::get_if<std::unique_ptr<DataDescriptorMatcher>>(&mRight)) {
      rightValue = (*pval3)->match(d, context);
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

  Node const& getLeft() const { return mLeft; };
  Node const& getRight() const { return mRight; };
  Op getOp() const { return mOp; };

 private:
  Op mOp;
  Node mLeft;
  Node mRight;
};

} // namespace data_matcher
} // namespace framework
} // namespace o2

#endif // o2_framework_DataDescriptorMatcher_H_INCLUDED
