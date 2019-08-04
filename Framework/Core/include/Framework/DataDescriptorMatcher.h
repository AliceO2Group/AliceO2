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
#include <iosfwd>
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
  inline bool operator==(ContextRef const& other) const;
};

/// An element of the matching context. Context itself is really a vector of
/// those. It's up to the matcher builder to build the vector in a suitable way.
/// We do not have any float in the value, because AFAICT there is no need for
/// it in the O2 DataHeader, however we could add it later on.
struct ContextElement {
  using Value = std::variant<uint32_t, uint64_t, std::string, None>;
  std::string label;    /// The name of the variable contained in this element.
  Value value = None{}; /// The actual contents of the element.
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
  inline VariableContext();

  ContextElement::Value const& get(size_t pos) const;

  inline void put(ContextUpdate&& update);

  /// Use this after a query to actually commit the matched fields.  Notice the
  /// old matches remain there, but we do not need to clean them up as we have
  /// reset the counter. Use this after a successful query to persist matches
  /// variables and speedup subsequent lookups.
  void commit();

  /// Discard the updates. Use this after a failed query if you do not want to
  /// retain partial matches.
  inline void discard();

  /// Reset the all the variables and updates, without having to
  /// tear down the context.
  void reset();

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
  inline ValueHolder(T const& s);

  /// This means that the matcher will fill a variable in the context if
  /// the ref points to none or use the dereferenced value, if not.
  inline ValueHolder(ContextRef variableId);

  inline bool operator==(ValueHolder<T> const& other) const;

  template <typename V>
  friend std::ostream& operator<<(std::ostream& os, ValueHolder<V> const& holder);

  template <typename VISITOR>
  decltype(auto) visit(VISITOR visitor) const
  {
    return std::visit(visitor, mValue);
  }

 protected:
  std::variant<T, ContextRef> mValue;
};

/// Something which can be matched against a header::DataOrigin
class OriginValueMatcher : public ValueHolder<std::string>
{
 public:
  inline OriginValueMatcher(std::string const& s);
  inline OriginValueMatcher(ContextRef variableId);

  bool match(header::DataHeader const& header, VariableContext& context) const;
};

/// Something which can be matched against a header::DataDescription
class DescriptionValueMatcher : public ValueHolder<std::string>
{
 public:
  inline DescriptionValueMatcher(std::string const& s);

  inline DescriptionValueMatcher(ContextRef variableId);

  bool match(header::DataHeader const& header, VariableContext& context) const;
};

/// Something which can be matched against a header::SubSpecificationType
class SubSpecificationTypeValueMatcher : public ValueHolder<header::DataHeader::SubSpecificationType>
{
 public:
  inline SubSpecificationTypeValueMatcher(ContextRef variableId);

  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  inline SubSpecificationTypeValueMatcher(std::string const& s);

  /// This means that the matcher is looking for a constant.
  inline SubSpecificationTypeValueMatcher(header::DataHeader::SubSpecificationType v);

  bool match(header::DataHeader const& header, VariableContext& context) const;
};

/// Matcher on actual time, as reported in the DataProcessingHeader
class StartTimeValueMatcher : public ValueHolder<uint64_t>
{
 public:
  inline StartTimeValueMatcher(ContextRef variableId, uint64_t scale = 1);

  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  inline StartTimeValueMatcher(std::string const& s, uint64_t scale = 1);

  /// This means that the matcher is looking for a constant.
  /// We will divide the input by scale so that we can map
  /// quantities with different granularities to the same record.
  inline StartTimeValueMatcher(uint64_t v, uint64_t scale = 1);

  /// This will match the timing information which is currently in
  /// the DataProcessingHeader. Notice how we apply the scale to the
  /// actual values found.
  bool match(DataProcessingHeader const& dph, VariableContext& context) const;

 private:
  uint64_t mScale;
};

class ConstantValueMatcher
{
 public:
  /// The passed string @a s is the expected numerical value for
  /// the SubSpecification type.
  inline ConstantValueMatcher(bool value);

  inline bool match() const;

  inline bool operator==(ConstantValueMatcher const& other) const;

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
  DataDescriptorMatcher(Op op, Node&& lhs, Node&& rhs = std::move(ConstantValueMatcher{false}));

  inline ~DataDescriptorMatcher() = default;

  /// @return true if the (sub-)query associated to this matcher will
  /// match the provided @a spec, false otherwise.
  /// FIXME: these are not really part of the DataDescriptorMatcher API
  /// and should really be relegated to external helpers...
  bool match(ConcreteDataMatcher const& matcher, VariableContext& context) const;
  bool match(ConcreteDataTypeMatcher const& matcher, VariableContext& context) const;
  bool match(header::DataHeader const& header, VariableContext& context) const;
  bool match(header::Stack const& stack, VariableContext& context) const;

  // actual polymorphic matcher which is able to cast the pointer to the correct
  // kind of header.
  bool match(char const* d, VariableContext& context) const;

  bool operator==(DataDescriptorMatcher const& other) const;

  friend std::ostream& operator<<(std::ostream& os, DataDescriptorMatcher const& matcher);
  friend std::ostream& operator<<(std::ostream& os, Op const& matcher);

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

// This is to work around CLING issues when parsing
// GCC 7.3.0 std::variant implementation as described by:
// https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=877838
#ifndef __CLING__
#include "DataDescriptorMatcher.inc"
#endif

#endif // o2_framework_DataDescriptorMatcher_H_INCLUDED
