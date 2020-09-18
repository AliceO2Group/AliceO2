// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_COMPLETIONPOLICYHELPERS_H
#define FRAMEWORK_COMPLETIONPOLICYHELPERS_H

#include "Framework/ChannelSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CompletionPolicy.h"
#include "Headers/DataHeader.h"

#include <functional>
#include <string>
#include <type_traits>

namespace o2
{
namespace framework
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
  /// When any of the parts of the record have been received, consume them.
  static CompletionPolicy consumeWhenAny(const char* name, CompletionPolicy::Matcher matcher);
  /// Default matcher applies for all devices
  static CompletionPolicy consumeWhenAny(CompletionPolicy::Matcher matcher = [](auto const&) -> bool { return true; })
  {
    return consumeWhenAny("consume-any", matcher);
  }
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

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_COMPLETIONPOLICYHELPERS_H
