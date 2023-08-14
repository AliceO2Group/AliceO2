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
#ifndef O2_FRAMEWORK_CHECKTYPES_H_
#define O2_FRAMEWORK_CHECKTYPES_H_

#include <algorithm>
#include <type_traits>
#include "CompilerBuiltins.h"

namespace o2::framework
{

/// Helper to understand if a given type is complete (declared fully) or not (forward declared).
/// See also: https://devblogs.microsoft.com/oldnewthing/20190710-00/?p=102678
template <typename, typename = void>
constexpr bool is_type_complete_v = false;

template <typename T>
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;

/// Helper which will invoke @a onDefined if the type T is actually available
/// or @a onUndefined if the type T is a forward declaration.
/// Can be used to check for existence or not of a given type.
template <typename T, typename TDefined, typename TUndefined>
void call_if_defined_full(TDefined&& onDefined, TUndefined&& onUndefined)
{
  if constexpr (is_type_complete_v<T>) {
    onDefined(static_cast<T*>(nullptr));
  } else {
    onUndefined();
  }
}

/// Helper which will invoke @a onDefined if the type T is actually available
/// or @a onUndefined if the type T is a forward declaration.
/// Can be used to check for existence or not of a given type.
template <typename T, typename TDefined, typename TUndefined>
T call_if_defined_full_forward(TDefined&& onDefined, TUndefined&& onUndefined)
{
  if constexpr (is_type_complete_v<T>) {
    return std::move(onDefined(static_cast<T*>(nullptr)));
  } else {
    return onUndefined();
  }
}

template <typename T, typename TDefined>
void call_if_defined(TDefined&& onDefined)
{
  call_if_defined_full<T>(onDefined, []() -> void {});
}

template <typename T, typename TDefined>
T call_if_defined_forward(TDefined&& onDefined)
{
  return std::move(call_if_defined_full_forward<T>(onDefined, []() -> T&& { O2_BUILTIN_UNREACHABLE(); }));
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHECKTYPES_H_
