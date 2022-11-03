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
#ifndef O2_FRAMEWORK_SERIALIZATIONMETHODS_H_
#define O2_FRAMEWORK_SERIALIZATIONMETHODS_H_

/// @file SerializationMethods.h
/// @brief Type wrappers for enfording a specific serialization method

#include "Framework/TypeTraits.h"

namespace o2::framework
{

/// @class ROOTSerialized
/// Enforce ROOT serialization for a type
///
/// Usage: (with 'output' being the DataAllocator of the ProcessingContext)
///   SomeType object;
///   ROOTSerialized<decltype(object)> ref(object);
///   output.snapshot(Output{}, ref);
///     - or -
///   output.snapshot(Output{}, ROOTSerialized<decltype(object)>(object));
///
/// The existence of the ROOT dictionary for the wrapped type can not be
/// checked at compile time, a runtime check must be performed in the
/// substitution for the ROOTSerialized type.
///
/// An optional hint can be passed to point to the class info, supported types
/// are TClass or const char. A pointer of the hint can be passed to the
/// constructor in addition to the reference. In the first case, the TClass
/// instance will be used directly (faster) while in the latter the TClass registry
/// is searched by name.
///   TClass* classinfo = ...;
///   ROOTSerialized<decltype(object), TClass>(object, classinfo));
///     - or -
///   ROOTSerialized<decltype(object), const char>(object, "classname"));
template <typename T, typename HintType = void>
class ROOTSerialized
{
 public:
  using non_messageable = o2::framework::MarkAsNonMessageable;
  using wrapped_type = T;
  using hint_type = HintType;

  static_assert(std::is_pointer<T>::value == false, "wrapped type can not be a pointer");
  static_assert(std::is_pointer<HintType>::value == false, "hint type can not be a pointer");

  ROOTSerialized() = delete;
  ROOTSerialized(wrapped_type& ref, hint_type* hint = nullptr) : mRef(ref), mHint(hint) {}

  T& operator()() { return mRef; }
  T const& operator()() const { return mRef; }

  hint_type* getHint() const { return mHint; }

 private:
  wrapped_type& mRef;
  hint_type* mHint; // optional hint e.g. class info or class name
};

template <typename T, typename HintType = void>
class CCDBSerialized
{
 public:
  using non_messageable = o2::framework::MarkAsNonMessageable;
  using wrapped_type = T;
  using hint_type = HintType;

  static_assert(std::is_pointer<T>::value == false, "wrapped type can not be a pointer");
  static_assert(std::is_pointer<HintType>::value == false, "hint type can not be a pointer");

  CCDBSerialized() = delete;
  CCDBSerialized(wrapped_type& ref, hint_type* hint = nullptr) : mRef(ref), mHint(hint) {}

  T& operator()() { return mRef; }
  T const& operator()() const { return mRef; }

  hint_type* getHint() const { return mHint; }

 private:
  wrapped_type& mRef;
  hint_type* mHint; // optional hint e.g. class info or class name
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_SERIALIZATIONMETHODS_H_
