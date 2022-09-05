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

namespace o2::framework::data_matcher
{

inline bool ContextRef::operator==(ContextRef const& other) const
{
  return index == other.index;
}

inline void VariableContext::put(ContextUpdate&& update)
{
  mUpdates[mPerformedUpdates++] = std::move(update);
}

inline void VariableContext::discard()
{
  mPerformedUpdates = 0;
}

inline VariableContext::VariableContext()
  : mPerformedUpdates{0}
{
}

template <typename T>
inline ValueHolder<T>::ValueHolder(T const& s)
  : mValue{s}
{
}

template <typename T>
inline ValueHolder<T>::ValueHolder(ContextRef variableId)
  : mValue{variableId}
{
}

template <typename T>
inline bool ValueHolder<T>::operator==(ValueHolder<T> const& other) const
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

template <typename V>
std::ostream& operator<<(std::ostream& os, ValueHolder<V> const& holder)
{
  if (auto value = std::get_if<V>(&holder.mValue)) {
    os << *value;
  } else if (auto context = std::get_if<ContextRef>(&holder.mValue)) {
    os << "$" << context->index;
  }
  return os;
}

inline OriginValueMatcher::OriginValueMatcher(std::string const& s)
  : ValueHolder{s}
{
  if (s.size() > header::gSizeDataOriginString) {
    throw runtime_error("Header DataDescription too long");
  }
}

inline OriginValueMatcher::OriginValueMatcher(ContextRef variableId)
  : ValueHolder{variableId}
{
}

template <std::size_t L>
inline constexpr OriginValueMatcher::OriginValueMatcher(const char (&s)[L])
  : ValueHolder{header::DataOrigin(s).as<std::string>()}
{
}

inline DescriptionValueMatcher::DescriptionValueMatcher(std::string const& s)
  : ValueHolder{s}
{
  if (s.size() > header::gSizeDataDescriptionString) {
    throw runtime_error("Header DataDescription too long");
  }
}

inline DescriptionValueMatcher::DescriptionValueMatcher(ContextRef variableId)
  : ValueHolder{variableId}
{
}

template <std::size_t L>
inline constexpr DescriptionValueMatcher::DescriptionValueMatcher(const char (&s)[L])
  : ValueHolder{header::DataDescription(s).as<std::string>()}
{
}

inline SubSpecificationTypeValueMatcher::SubSpecificationTypeValueMatcher(ContextRef variableId)
  : ValueHolder{variableId}
{
}

inline SubSpecificationTypeValueMatcher::SubSpecificationTypeValueMatcher(std::string const& s)
  : ValueHolder<header::DataHeader::SubSpecificationType>{o2::header::strtoui<header::DataHeader::SubSpecificationType>(s.c_str(), nullptr, 10)}
{
}

inline SubSpecificationTypeValueMatcher::SubSpecificationTypeValueMatcher(header::DataHeader::SubSpecificationType v)
  : ValueHolder<header::DataHeader::SubSpecificationType>{v}
{
}

inline StartTimeValueMatcher::StartTimeValueMatcher(ContextRef variableId, uint64_t scale)
  : ValueHolder{variableId},
    mScale{scale}
{
}

inline StartTimeValueMatcher::StartTimeValueMatcher(std::string const& s, uint64_t scale)
  : ValueHolder<uint64_t>{strtoull(s.c_str(), nullptr, 10)},
    mScale{scale}
{
}

inline StartTimeValueMatcher::StartTimeValueMatcher(uint64_t v, uint64_t scale)
  : ValueHolder<uint64_t>{v / scale},
    mScale{scale}
{
}

inline ConstantValueMatcher::ConstantValueMatcher(bool value)
{
  mValue = value;
}

inline bool ConstantValueMatcher::match() const
{
  return mValue;
}

inline bool ConstantValueMatcher::operator==(ConstantValueMatcher const& other) const
{
  return mValue == other.mValue;
}

} // namespace o2::framework::data_matcher
