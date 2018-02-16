// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SERIALIZATIONMETHODS_H
#define FRAMEWORK_SERIALIZATIONMETHODS_H

/// @file SerializationMethods.h
/// @brief Type wrappers for enfording a specific serialization method

#include "Framework/TypeTraits.h"

namespace o2
{
namespace framework
{

/// @class ROOTSerialized
/// Enforce ROOT serialization for a type
///
/// Usage: (with 'output' being the DataAllocator of the ProcessingContext)
///   SomeType object;
///   ROOTSerialized<decltype(object)> ref(object);
///   output.snapshot(OutputSpec{}, ref);
///     - or -
///   output.snapshot(OutputSpec{}, ROOTSerialized<decltype(object)>(object));
///
/// The existence of the ROOT dictionary for the wrapped type can not be
/// checked at compile time, a runtime check must be performed in the
/// substitution for the ROOTSerialized type.
template <typename T>
class ROOTSerialized
{
 public:
  using non_messageable = o2::framework::MarkAsNonMessageable;
  using wrapped_type = T;
  ROOTSerialized() = delete;
  ROOTSerialized(wrapped_type& ref) : mRef(ref) {}

  T& operator()() { return mRef; }
  T const& operator()() const { return mRef; }

 private:
  wrapped_type& mRef;
};
}
}

#endif // FRAMEWORK_SERIALIZATIONMETHODS_H
