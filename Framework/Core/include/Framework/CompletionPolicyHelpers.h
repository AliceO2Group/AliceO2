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
#ifndef O2_FRAMEWORK_COMPLETIONPOLICYHELPERS_H_
#define O2_FRAMEWORK_COMPLETIONPOLICYHELPERS_H_

#include "Framework/ChannelSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CompletionPolicy.h"
#include "Headers/DataHeader.h"

#include <functional>
#include <string>
#include <type_traits>

namespace o2::framework
{

/// Helper class which holds commonly used policies.
struct CompletionPolicyHelpers {
  /// Default Completion policy. When all the parts of a record have arrived, consume them.
  static CompletionPolicy consumeWhenAll(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy consumeWhenAll(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return consumeWhenAll("consume-all", matcher);
  }

  /// as consumeWhenAll, but ensures that records are processed with incremental timeSlice (DataHeader::startTime)
  static CompletionPolicy consumeWhenAllOrdered(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy consumeWhenAllOrdered(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return consumeWhenAllOrdered("consume-all-ordered", matcher);
  }
  static CompletionPolicy consumeWhenAllOrdered(std::string matchName);

  /// When any of the parts of the record have been received, consume them.
  static CompletionPolicy consumeWhenAny(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy consumeWhenAny(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return consumeWhenAny("consume-any", matcher);
  }
  static CompletionPolicy consumeWhenAny(std::string matchName);

  /// When any of the parts of the record have been received, consume them.
  static CompletionPolicy consumeWhenAnyWithAllConditions(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy consumeWhenAnyWithAllConditions(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return consumeWhenAnyWithAllConditions("consume-any-all-conditions", matcher);
  }
  static CompletionPolicy consumeWhenAnyWithAllConditions(std::string matchName);

  /// When any of the parts of the record have been received, process the existing and free the associated payloads.
  /// This allows freeing things as early as possible, while still being able to wait
  /// all the parts before disposing the timeslice completely
  static CompletionPolicy consumeExistingWhenAny(const char* name, CompletionPolicy::Matcher matcher);

  /// When any of the parts of the record have been received, process them,
  /// without actually consuming them.
  static CompletionPolicy processWhenAny(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy processWhenAny(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return processWhenAny("process-any", matcher);
  }
  /// Attach a given @a op to a device matching @name.
  static CompletionPolicy defineByName(std::string const& name, CompletionPolicy::CompletionOp op);
  /// Attach a given @a op to a device matching @name, check message of origin @origin is available
  static CompletionPolicy defineByNameOrigin(std::string const& name, std::string const& origin, CompletionPolicy::CompletionOp op);
  /// Get a specific header from the input
  template <typename T, typename U>
  static auto getHeader(U const& input)
  {
    // DataHeader interface requires to specify header pointer type, need to check if the template parameter
    // is already pointer type, and add pointer if not
    using return_type = typename std::conditional<std::is_pointer<T>::value, T, typename std::add_pointer<T>::type>::type;
    return o2::header::get<return_type>(input.header);
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_COMPLETIONPOLICYHELPERS_H_
