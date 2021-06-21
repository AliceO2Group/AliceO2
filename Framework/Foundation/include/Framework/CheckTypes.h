// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CHECKTYPES_H_
#define O2_FRAMEWORK_CHECKTYPES_H_

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

template <typename T, typename TDefined>
void call_if_defined(TDefined&& onDefined)
{
  call_if_defined_full<T>(onDefined, []() -> void {});
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHECKTYPES_H_
