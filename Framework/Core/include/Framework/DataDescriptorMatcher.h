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

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include <array>
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

  /// Two context refs are the same if they point to the
  /// same element in the context
  bool operator==(ContextRef const& other) const
  {
    return index == other.index;
  }
};

/// An element of the matching context. Context itself is really a vector of
/// those. It's up to the matcher builder to build the vector in a suitable way.
/// We do not have any float in the value, because AFAICT there is no need for
/// it in the O2 DataHeader, however we could add it later on.
struct ContextElement {
  using Value = std::variant<uint64_t, std::string, None>;
  std::string label;                               /// The name of the variable contained in this element.
  Value value = None{};                            /// The actual contents of the element.
};

struct ContextUpdate {
  size_t position;
  ContextElement::Value newValue;
};

constexpr int MAX_MATCHING_VARIABLE = 16;
constexpr int MAX_UPDATES_PER_QUERY = 16;

class VariableContext
{
 public:
  VariableContext()
    : mPerformedUpdates{ 0 }
  {
  }

  ContextElement::Value const& get(size_t pos) const
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

  void put(ContextUpdate&& update)
  {
    mUpdates[mPerformedUpdates++] = std::move(update);
  }

  /// Use this after a query to actually commit the matched fields.  Notice the
  /// old matches remain there, but we do not need to clean them up as we have
  /// reset the counter. Use this after a successful query to persist matches
  /// variables and speedup subsequent lookups.
  void commit()
  {
    for (size_t i = 0; i < mPerformedUpdates; ++i) {
      mElements[mUpdates[i].position].value = mUpdates[i].newValue;
    }
    mPerformedUpdates = 0;
  }

  /// Discard the updates. Use this after a failed query if you do not want to
  /// retain partial matches.
  void discard()
  {
    mPerformedUpdates = 0;
  }

  /// Reset the all the variables and updates, without having to
  /// tear down the context.
  void reset()
  {
    mPerformedUpdates = 0;
    for (auto& element : mElements) {
      element.value = None{};
    }
  }

 private:
  /* We make this class fixed size to avoid memory churning while 
     matching as much as posible when doing the matching, as that might become
     performance critical. Given we will have only a few of these (one per
     cacheline of the input messages) it should not be critical memory wise.
   */
  std::array<ContextElement, MAX_MATCHING_VARIABLE> mElements;
  std::array<ContextUpdate, MAX_UPDATES_PER_QUERY> mUpdates;
  int mPerformedUpdates;
};

/// Can hold either an actual value of type T or a reference to
/// a variable of the same type in the Context.
template <typename T>
class ValueHolder
{
 public:
  ValueHolder(T const& s)
    : mValue{ s }
  {
  }
  /// This means that the matcher will fill a variable in the context if
  /// the ref points to none or use the dereferenced value, if not.
  ValueHolder(ContextRef variableId)
    : mValue{ variableId }
  {
  }

  bool operator==(ValueHolder<T> const& other) const
  {
    auto s1 = std::get_if<T>(&mValue);
    auto s2 = std::get_if<T>(&other.mValue);

    if (s1 && s2) {
      return *s1 == *s2;
    }

    auto c1 = std::get_if<ContextRef>(&mValue);
    auto c2 = std::get_if<ContextRef>(&other.mValue);
    if (c1 && c2) {
      return *c1 == *c2;
    }

    return false;
  }

 protected:
  std::variant<T, ContextRef> mValue;
};

/// Something which can be matched against a header::DataOrigin
class OriginValueMatcher : public ValueHolder<std::string>
{
 public:
  OriginValueMatcher(std::string const& s)
    : ValueHolder{ s }
  {
  }

  OriginValueMatcher(ContextRef variableId)
    : ValueHolder{ variableId }
  {
  }

  bool match(header::DataHeader const& header, VariableContext& context) const
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
};

/// Something which can be matched against a header::DataDescription
class DescriptionValueMatcher : public ValueHolder<std::string>
{
 public:
  DescriptionValueMatcher(std::string const& s)
    : ValueHolder{ s }
  {
  }

  DescriptionValueMatcher(ContextRef variableId)
    : ValueHolder{ variableId }
  {
  }

  bool match(header::DataHeader const& header, VariableContext& context) const
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
};

/// Something which can be matched against a header::SubSpecificationType
class SubSpecificationTypeValueMatcher : public ValueHolder<uint64_t>
{
 public:
  SubSpecificationTypeValueMatcher(ContextRef variableId)
    : ValueHolder{ variableId }
  {
  }

  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  SubSpecificationTypeValueMatcher(std::string const& s)
    : ValueHolder<uint64_t>{ strtoull(s.c_str(), nullptr, 10) }
  {
  }

  /// This means that the matcher is looking for a constant.
  SubSpecificationTypeValueMatcher(uint64_t v)
    : ValueHolder<uint64_t>{ v }
  {
  }

  bool match(header::DataHeader const& header, VariableContext& context) const
  {
    if (auto ref = std::get_if<ContextRef>(&mValue)) {
      auto& variable = context.get(ref->index);
      if (auto value = std::get_if<uint64_t>(&variable)) {
        return header.subSpecification == *value;
      }
      context.put({ ref->index, header.subSpecification });
      return true;
    } else if (auto v = std::get_if<uint64_t>(&mValue)) {
      return header.subSpecification == *v;
    }
    throw std::runtime_error("Mismatching type for variable");
  }
};

/// Matcher on actual time, as reported in the DataProcessingHeader
class StartTimeValueMatcher : public ValueHolder<uint64_t>
{
 public:
  StartTimeValueMatcher(ContextRef variableId, uint64_t scale = 1)
    : ValueHolder{ variableId },
      mScale{ scale }
  {
  }

  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  StartTimeValueMatcher(std::string const& s, uint64_t scale = 1)
    : ValueHolder<uint64_t>{ strtoull(s.c_str(), nullptr, 10) },
      mScale{ scale }
  {
  }

  /// This means that the matcher is looking for a constant.
  /// We will divide the input by scale so that we can map
  /// quantities with different granularities to the same record.
  StartTimeValueMatcher(uint64_t v, uint64_t scale = 1)
    : ValueHolder<uint64_t>{ v / scale },
      mScale{ scale }
  {
  }

  /// This will match the timing information which is currently in
  /// the DataProcessingHeader. Notice how we apply the scale to the
  /// actual values found.
  bool match(DataProcessingHeader const& dph, VariableContext& context) const
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

 private:
  uint64_t mScale;
};

class ConstantValueMatcher
{
 public:
  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  ConstantValueMatcher(bool value)
  {
    mValue = value;
  }

  bool match() const
  {
    return mValue;
  }

  bool operator==(ConstantValueMatcher const& other) const
  {
    return mValue == other.mValue;
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
using Node = std::variant<OriginValueMatcher, DescriptionValueMatcher, SubSpecificationTypeValueMatcher, std::unique_ptr<DataDescriptorMatcher>, ConstantValueMatcher, StartTimeValueMatcher>;

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
  DataDescriptorMatcher(DataDescriptorMatcher const& other);
  //DataDescriptorMatcher(DataDescriptorMatcher&& other) noexcept;
  DataDescriptorMatcher& operator=(DataDescriptorMatcher const& other);
  //DataDescriptorMatcher &operator=(DataDescriptorMatcher&& other) noexcept = default;

  /// Unary operator on a node
  DataDescriptorMatcher(Op op, Node&& lhs, Node&& rhs = std::move(ConstantValueMatcher{ false }));

  inline ~DataDescriptorMatcher() = default;

  /// @return true if the (sub-)query associated to this matcher will
  /// match the provided @a spec, false otherwise.
  /// FIXME: these are not really part of the DataDescriptorMatcher API
  /// and should really be relegated to external helpers...
  bool match(ConcreteDataMatcher const& matcher, VariableContext& context) const;
  bool match(header::DataHeader const& header, VariableContext& context) const;
  bool match(header::Stack const& stack, VariableContext& context) const;

  // actual polymorphic matcher which is able to cast the pointer to the correct
  // kind of header.
  bool match(char const* d, VariableContext& context) const;

  bool operator==(DataDescriptorMatcher const& other) const;

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
